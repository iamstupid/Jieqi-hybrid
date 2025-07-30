import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import json
import startpos
from collections import defaultdict

import jieqi_game
from factory import load_leela_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PIMCTSConfig:
    """Configuration for PIMCTS self-play"""
    # PIMCTS parameters
    n_samples: int = 1          # Number of determinization samples
    n_mcts_steps: int = 10      # MCTS steps per sample
    n_eval_batch_size: int = 256  # NN evaluation batch size
    cpuct: float = 1.25          # UCB exploration constant
    
    # Game parameters
    max_game_length: int = 512   # Maximum moves per game
    temperature_moves: int = 0  # Use temperature for first N moves
    temperature: float = 1.0     # Sampling temperature
    
    # Self-play control
    games_per_iteration: int = 100
    save_frequency: int = 10     # Save training data every N games
    
    # Paths
    model_path: str = "eval_small.safetensors"
    config_path: str = "eval_small.json"
    training_data_path: str = "training_data"


@dataclass
class TrainingSample:
    """Single training sample for neural network"""
    position_encoding: np.ndarray    # [90, 167] - root node encoding
    policy_target: np.ndarray        # [policy_size] - aggregated visit distribution
    value_target: np.ndarray              # Game outcome from current player's perspective
    game_id: int
    move_number: int


class NeuralNetworkEvaluator:
    """Handles neural network evaluation with caching"""
    
    def __init__(self, model_path: str, config_path: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on device: {device}")
        
        self.model = load_leela_model(model_path, config_path, device)
        self.device = device
        self.model.eval()
    
    def evaluate_batch(self, batch_encodings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of positions
        
        Args:
            batch_encodings: List of [90, 167] position encodings
            
        Returns:
            policies: [batch_size, policy_size] policy probabilities
            values: [batch_size, 3] WDL probabilities or [batch_size, 1] single values
        """
        # Stack and convert to tensor
        batch_tensor = torch.tensor(
            batch_encodings,
            dtype=torch.float32,
            device=self.device
        )
        
        with torch.no_grad():
            policy_logits, value_output = self.model(batch_tensor)
            
            # Convert to probabilities
            policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            values = value_output.cpu().numpy()
        
        return policies, values


class PIMCTSSearcher:
    """Manages PIMCTS search with multiple determinization samples"""
    
    def __init__(self, config: PIMCTSConfig, evaluator: NeuralNetworkEvaluator):
        self.config = config
        self.evaluator = evaluator
    
    def run_pimcts_search(self, position_history: jieqi_game.PositionHistory) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run PIMCTS search with multiple determinization samples
        
        Args:
            position_history: Current game state
            
        Returns:
            root_encoding: [90, 167] root position encoding
            aggregated_policy: [policy_size] aggregated policy distribution
            aggregated_value: Aggregated value estimate
        """
        # Create MCTS for this sample
        mcts = jieqi_game.MCTS(
            position_history, 
            self.config.n_eval_batch_size, 
            self.config.cpuct
        )

        root_eval = None
        all_sample_stats = []
        
        for sample_idx in range(self.config.n_samples):
            # logger.debug(f"Running PIMCTS sample {sample_idx + 1}/{self.config.n_samples}")
            
            # Redeterminize for this sample
            mcts.redeterminize()
            root_encoding = np.array(mcts.run_search()).reshape(1,90,167)
            if root_eval == None:
                root_eval = self.evaluator.evaluate_batch(root_encoding)
            
            mcts.apply_evaluations(root_eval[0], root_eval[1])

            # n_evals = 1
            
            # Run MCTS search steps
            for step in range(self.config.n_mcts_steps):
                # Get batch of positions to evaluate
                encodings = mcts.run_search_batch()
                
                if len(encodings)==0:
                    break  # No more positions to evaluate
                
                # Reshape encodings
                batch_encodings = np.array(encodings).reshape(-1, 90, 167)

                # n_evals += batch_encodings.shape[0]
                
                # Evaluate batch
                policies, values = self.evaluator.evaluate_batch(batch_encodings)
                
                # Apply evaluations to MCTS
                mcts.apply_evaluations(policies, values)
            
            # Collect statistics from this sample
            sample_stats = mcts.get_root_move_evaluations()
            all_sample_stats.append(sample_stats)
        
        # Aggregate statistics across all samples
        aggregated_policy = self._aggregate_sample_statistics(all_sample_stats)
        
        return root_encoding, aggregated_policy
    
    def _aggregate_sample_statistics(self, all_sample_stats: List[List]) -> Tuple[np.ndarray, float]:
        """
        Aggregate PIMCTS statistics across all samples
        
        Args:
            all_sample_stats: List of MoveEvaluation lists from each sample
            
        Returns:
            aggregated_policy: [policy_size] normalized visit count distribution
            aggregated_value: Weighted average value
        """
        # Collect move statistics across all samples
        move_visit_counts = defaultdict(int)
        move_wdl_stats = defaultdict(lambda: {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'visits': 0})
        total_visits = 0
        
        for sample_stats in all_sample_stats:
            for move_eval in sample_stats:
                move_idx = move_eval.move.as_nn_index(0)
                visits = move_eval.visit_count
                
                # Accumulate visit counts for policy
                move_visit_counts[move_idx] += visits
                total_visits += visits
                
                # Accumulate WDL stats for value (weighted by visit count)
                move_wdl_stats[move_idx]['win'] += move_eval.win_prob * visits
                move_wdl_stats[move_idx]['draw'] += move_eval.draw_prob * visits
                move_wdl_stats[move_idx]['loss'] += move_eval.loss_prob * visits
                move_wdl_stats[move_idx]['visits'] += visits
        
        # Create aggregated policy distribution
        policy_size = 2550  # Adjust based on your policy size
        aggregated_policy = np.zeros(policy_size, dtype=np.float32)
        
        if total_visits > 0:
            for move_idx, visits in move_visit_counts.items():
                aggregated_policy[move_idx] = visits / total_visits
        """        
        # Calculate aggregated value (weighted average of win_prob - loss_prob)
        aggregated_value = 0.0
        total_value_weight = 0.0
        
        for move_idx, stats in move_wdl_stats.items():
            if stats['visits'] > 0:
                # Value from current player's perspective: win_prob - loss_prob
                move_value = (stats['win'] - stats['loss']) / stats['visits']
                weight = stats['visits']
                
                aggregated_value += move_value * weight
                total_value_weight += weight
        
        if total_value_weight > 0:
            aggregated_value /= total_value_weight
        """
        
        return aggregated_policy


class PIMCTSSelfPlay:
    """Main PIMCTS self-play engine with training data collection"""
    
    def __init__(self, config: PIMCTSConfig):
        self.config = config
        
        # Initialize neural network evaluator
        self.evaluator = NeuralNetworkEvaluator(config.model_path, config.config_path)
        
        # Initialize PIMCTS searcher
        self.searcher = PIMCTSSearcher(config, self.evaluator)
        
        # Training data storage
        self.training_samples = []
        self.game_counter = 0
        
        # Statistics
        self.stats = {
            'games_played': 0,
            'total_positions': 0,
            'avg_game_length': 0.0,
            'avg_search_time': 0.0,
            'win_rates': {'WHITE': 0, 'BLACK': 0, 'DRAW': 0}
        }
        
        # Ensure output directory exists
        Path(config.training_data_path).mkdir(parents=True, exist_ok=True)
    
    def select_move_with_temperature(self, policy: np.ndarray, temperature: float = 1.0) -> int:
        """Select move using temperature sampling"""
        if temperature == 0.0:
            return np.argmax(policy)
        
        # Apply temperature
        log_probs = np.log(policy + 1e-10)
        probs = np.exp(log_probs / temperature)
        probs /= np.sum(probs)
        
        # Sample from distribution
        return np.random.choice(len(probs), p=probs)
    
    def play_single_game(self) -> Dict:
        """
        Play a single self-play game and collect training data
        
        Returns:
            Game statistics and training samples
        """
        game_start_time = time.time()
        self.game_counter += 1
        
        logger.info(f"Starting PIMCTS game {self.game_counter}")
        
        # Initialize judge game (ground truth)
        position_history = jieqi_game.PositionHistory(startpos.GenerateStartingPosition())

        logger.info(f"{jieqi_game.GetExtFen(position_history.Last())}")
        
        judge_game = jieqi_game.DeterminizedGame(position_history)
        judge_game.determinize()  # Set ground truth hidden piece assignment
        
        game_samples = []
        search_times = []
        
        # Game loop
        for move_number in range(self.config.max_game_length):
            # Check for game termination
            game_result = position_history.ComputeGameResult()
            if game_result != jieqi_game.GameResult.UNDECIDED:
                logger.info(f"Game {self.game_counter} ended: {game_result} after {move_number} moves")
                break
            
            # Run PIMCTS search
            search_start_time = time.time()
            
            root_encoding, aggregated_policy = self.searcher.run_pimcts_search(
                position_history
            )
            
            search_time = time.time() - search_start_time
            search_times.append(search_time)
            
            # Create training sample
            sample = TrainingSample(
                position_encoding=root_encoding,
                policy_target=aggregated_policy,
                value_target=np.array([0,0,0]),  # Will be updated with game outcome
                game_id=self.game_counter,
                move_number=position_history.Last().GetGamePly()
            )
            game_samples.append(sample)
            
            # Select move
            temperature = self.config.temperature if move_number < self.config.temperature_moves else 0.0
            move_idx = self.select_move_with_temperature(aggregated_policy, temperature)
            
            # Convert to move object
            selected_move = jieqi_game.MoveFromNNIndex(move_idx, 0)
            
            # Apply move to judge game
            judge_game.append(selected_move)
            
            # Log progress
            #if move_number % 10 == 0:
            FEN = jieqi_game.GetExtFen(position_history.Last())
            logger.info(f"{FEN} FEN"
                       f"Game {self.game_counter}, Move {move_number}, "
                       f"Search time: {search_time:.2f}s, "
                       f"Move: {selected_move.as_string()}")
        
        # Get final game result
        final_result = position_history.ComputeGameResult()
        game_time = time.time() - game_start_time
        
        # Update training samples with final game outcome
        for sample in game_samples:
            # Determine outcome from current player's perspective
            is_black_to_move = (sample.move_number % 2) == 1
            
            if final_result == jieqi_game.GameResult.WHITE_WON:
                sample.value_target[2 if is_black_to_move else 0] = 1.0
            elif final_result == jieqi_game.GameResult.BLACK_WON:
                sample.value_target[0 if is_black_to_move else 2] = 1.0
            else:  # Draw
                sample.value_target[1] = 1.0
        
        # Add samples to training data
        self.training_samples.extend(game_samples)
        
        game_stats = {
            'game_id': self.game_counter,
            'result': final_result,
            'moves_played': len(game_samples),
            'game_time': game_time,
            'avg_search_time': np.mean(search_times) if search_times else 0.0,
            'samples_collected': len(game_samples)
        }
        
        logger.info(f"Game {self.game_counter} completed: {final_result}, "
                   f"{len(game_samples)} moves, {game_time:.1f}s")
        
        return game_stats
    
    def save_training_data(self, filename: Optional[str] = None) -> None:
        """Save training samples to disk"""
        if not self.training_samples:
            return
        
        if filename is None:
            filename = f"pimcts_training_{int(time.time())}.npz"
        
        filepath = Path(self.config.training_data_path) / filename
        
        # Prepare data arrays
        position_encodings = np.array([s.position_encoding for s in self.training_samples])
        policy_targets = np.array([s.policy_target for s in self.training_samples])
        value_targets = np.array([s.value_target for s in self.training_samples])
        game_ids = np.array([s.game_id for s in self.training_samples])
        move_numbers = np.array([s.move_number for s in self.training_samples])
        
        np.savez_compressed(
            filepath,
            position_encodings=position_encodings,
            policy_targets=policy_targets,
            value_targets=value_targets,
            game_ids=game_ids,
            move_numbers=move_numbers,
            config=self.config.__dict__,
            stats=self.stats
        )
        
        logger.info(f"Saved {len(self.training_samples)} training samples to {filepath}")
    
    def update_stats(self, game_stats: Dict) -> None:
        """Update running statistics"""
        self.stats['games_played'] += 1
        self.stats['total_positions'] += game_stats['samples_collected']
        
        # Update running averages
        n = self.stats['games_played']
        self.stats['avg_game_length'] = (
            (self.stats['avg_game_length'] * (n-1) + game_stats['moves_played']) / n
        )
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (n-1) + game_stats['avg_search_time']) / n
        )
        
        # Update win rates
        result = game_stats['result']
        if result == jieqi_game.GameResult.WHITE_WON:
            self.stats['win_rates']['WHITE'] += 1
        elif result == jieqi_game.GameResult.BLACK_WON:
            self.stats['win_rates']['BLACK'] += 1
        else:
            self.stats['win_rates']['DRAW'] += 1
    
    def run_self_play(self, num_games: Optional[int] = None) -> None:
        """Run PIMCTS self-play for specified number of games"""
        if num_games is None:
            num_games = self.config.games_per_iteration
        
        logger.info(f"Starting PIMCTS self-play for {num_games} games")
        logger.info(f"Config: {self.config.n_samples} samples, {self.config.n_mcts_steps} steps/sample")
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            try:
                # Play game
                game_stats = self.play_single_game()
                
                # Update statistics
                self.update_stats(game_stats)
                
                # Save training data periodically
                if (game_idx + 1) % self.config.save_frequency == 0:
                    self.save_training_data(f"pimcts_batch_{game_idx + 1}.npz")
                    # Clear samples to save memory
                    self.training_samples.clear()
                
                # Log progress
                if (game_idx + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    games_per_hour = (game_idx + 1) / elapsed * 3600
                    logger.info(f"Progress: {game_idx + 1}/{num_games} games, "
                               f"{games_per_hour:.1f} games/hour")
                    logger.info(f"Stats: {self.stats}")
            
            except Exception as e:
                logger.error(f"Error in game {game_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

            logger.info(f"Node mem used:{jieqi_game.GetNodeMem()}")
        
        # Save final training data
        if self.training_samples:
            self.save_training_data("pimcts_final.npz")
        
        # Save final statistics
        stats_file = Path(self.config.training_data_path) / "pimcts_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        total_time = time.time() - start_time
        logger.info(f"PIMCTS self-play completed: {num_games} games in {total_time/3600:.1f} hours")
        logger.info(f"Final stats: {self.stats}")


def main():
    """Main function to run PIMCTS self-play"""
    
    # Configuration
    config = PIMCTSConfig(
        n_samples=1,              # 16 determinization samples
        n_mcts_steps=5,          # 10 MCTS steps per sample
        n_eval_batch_size=256,      # 64 positions per NN batch
        cpuct=1.25,               # UCB exploration
        temperature_moves=100,      # Temperature for first 0 moves
        temperature=0.2,          # Sampling temperature
        games_per_iteration=100,   # Play 100 games
        save_frequency=10,        # Save every 10 games
        model_path="eval_small.safetensors",
        config_path="eval_small.json",
        training_data_path="pimcts_training_data"
    )
    
    try:
        # Initialize and run self-play
        selfplay_engine = PIMCTSSelfPlay(config)
        selfplay_engine.run_self_play()
        
    except Exception as e:
        logger.error(f"PIMCTS self-play failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()