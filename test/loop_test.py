import torch
import numpy as np
import logging
from typing import List, Tuple

# Assuming these imports from your modules
import jieqi_game  # Your C++ binding
from factory import load_leela_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNeuralNetEvaluator:
    """Simple neural network evaluator for testing"""
    
    def __init__(self, model_path: str, config_path: str):
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on device: {device}")
        
        self.model = load_leela_model(model_path, config_path, device)
        self.device = device
        self.model.eval()
    
    def evaluate_position(self, position_features: np.ndarray) -> Tuple[np.ndarray, float]:
        batch_input = torch.Tensor(position_features).to(self.device)
        
        with torch.no_grad():
            policy_logits, value_output = self.model(batch_input)
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            value_probs = value_output.cpu().numpy()
            return policy_probs, value_probs

class SimpleSelfPlay:
    """Simple self-play implementation for testing"""
    
    def __init__(self, model_path: str, config_path: str):
        self.evaluator = SimpleNeuralNetEvaluator(model_path, config_path)
        self.games_played = 0
    
    def play_single_game(self, max_moves: int = 200) -> dict:
        """
        Play a single game and return basic statistics
        
        Returns:
            dict with game statistics
        """
        self.games_played += 1
        logger.info(f"Starting game {self.games_played}")
        
        # Initialize game
        position_history = jieqi_game.PositionHistory()
        position_history.Reset(jieqi_game.ChessBoard.hStartposBoard, 0, 0)
        determinized_game = jieqi_game.DeterminizedGame(position_history)
        
        moves_played = 0

        policy, value = None, None
        
        # Simple game loop
        for move_num in range(max_moves):
            # Check if game is over
            game_result = position_history.ComputeGameResult()
            if game_result != jieqi_game.GameResult.UNDECIDED:
                logger.info(f"Game {self.games_played} ended: {game_result} after {move_num} moves")
                break
            
            # Simple MCTS search (just 1 sample, few steps for testing)
            mcts = jieqi_game.MCTS(position_history, batch_size=32, cpuct=1.25)
            
            # Run a few MCTS steps
            for step in range(100):  # Just 10 steps for testing
                try:
                    # Get positions to evaluate
                    encodings, batch_positions = mcts.run_search_batch(), None
                    if len(encodings)!=0:
                        batch_positions = np.array(encodings).reshape((-1,90,167))
                        
                        # For simplicity, just use the NN evaluation directly
                        # In reality, you'd evaluate the batch_positions
                        policy, value = self.evaluator.evaluate_position(batch_positions)
                        mcts.apply_evaluations(policy, value)
                    
                except Exception as e:
                    logger.warning(f"MCTS step failed: {e}")
                    raise e
                    break
            
            # Get best move from MCTS
            try:
                best_move = mcts.get_best_move()
                if not best_move:  # Empty move
                    logger.warning("No valid move found, picking random legal move")
                    # Fallback: get legal moves and pick one
                    legal_moves = position_history.Last().GetBoard().GenerateLegalMoves()
                    if legal_moves:
                        best_move = legal_moves[0]  # Just pick first legal move
                    else:
                        logger.error("No legal moves available!")
                        break
                
                # Apply the move
                determinized_game.append(best_move)
                moves_played += 1
                
                if move_num % 10 == 0:
                    logger.info(f"Game {self.games_played}, Move {move_num}, "
                               f"Value: {value}, Move: {best_move.as_string()}")
                
            except Exception as e:
                logger.error(f"Error applying move: {e}")
                break
        
        # Final result
        final_result = position_history.ComputeGameResult()
        
        return {
            'game_id': self.games_played,
            'result': final_result,
            'moves_played': moves_played,
            'final_position_fen': jieqi_game.GetFen(position_history.Last())
        }
    
    def run_simple_selfplay(self, num_games: int = 5):
        """Run simple self-play for testing"""
        logger.info(f"Starting simple self-play for {num_games} games")
        
        game_results = []
        
        for game_idx in range(num_games):
            try:
                result = self.play_single_game()
                game_results.append(result)
                
                logger.info(f"Game {game_idx + 1} completed: "
                           f"Result: {result['result']}, "
                           f"Moves: {result['moves_played']}")
                
            except Exception as e:
                logger.error(f"Game {game_idx + 1} failed: {e}")
                raise e
                continue
        
        # Print summary
        logger.info("=" * 50)
        logger.info("SELF-PLAY SUMMARY")
        logger.info("=" * 50)
        
        for result in game_results:
            logger.info(f"Game {result['game_id']}: {result['result']} "
                       f"({result['moves_played']} moves)")
        
        # Count results
        result_counts = {}
        total_moves = 0
        for result in game_results:
            game_result = result['result']
            result_counts[game_result] = result_counts.get(game_result, 0) + 1
            total_moves += result['moves_played']
        
        if game_results:
            avg_moves = total_moves / len(game_results)
            logger.info(f"Average game length: {avg_moves:.1f} moves")
            logger.info(f"Results: {result_counts}")
        
        return game_results


def test_basic_components():
    """Test basic components individually"""
    logger.info("Testing basic components...")
    batch = None
    
    # Test 1: Can we create a position history?
    try:
        position_history = jieqi_game.PositionHistory()
        position_history.Reset(jieqi_game.ChessBoard.hStartposBoard, 0, 0)
        logger.info("✓ Position history creation works")
        
        # Test initial position
        initial_result = position_history.ComputeGameResult()
        logger.info(f"✓ Initial position result: {initial_result}")
        
        # Test legal moves
        legal_moves = position_history.Last().GetBoard().GenerateLegalMoves()
        logger.info(f"✓ Found {len(legal_moves)} legal moves in starting position")
        
    except Exception as e:
        logger.error(f"✗ Position history test failed: {e}")
        return False
    
    # Test 2: Can we create MCTS?
    try:
        mcts = jieqi_game.MCTS(position_history, batch_size=1, cpuct=1.25)
        logger.info("✓ MCTS creation works")
        
        # Test a single search batch call
        batch = mcts.run_search_batch()
        logger.info(f"✓ MCTS search batch returned {len(batch)} array")
        
    except Exception as e:
        logger.error(f"✗ MCTS test failed: {e}")
        return False
    
    logger.info("All basic component tests passed!")
    return True


def main():
    """Main function to test the simple self-play"""
    
    # Configuration - adjust these paths to your actual model files
    model_path = "eval_small.safetensors"  # Path to your model
    config_path = "eval_small.json"        # Path to your model config
    
    # Test basic components first
    if not test_basic_components():
        logger.error("Basic component tests failed. Cannot proceed with self-play.")
        return
    
    try:
        # Test neural network loading
        logger.info("Testing neural network loading...")
        evaluator = SimpleNeuralNetEvaluator(model_path, config_path)
        
        # Test a single evaluation
        dummy_features = np.random.randn(1,90,167).astype(np.float32)
        policy, value = evaluator.evaluate_position(dummy_features)
        logger.info(f"✓ Neural network evaluation works: "
                   f"policy shape {policy.shape}, value shape {value.shape}")
        
        # Run simple self-play
        logger.info("Starting simple self-play test...")
        selfplay = SimpleSelfPlay(model_path, config_path)
        selfplay.run_simple_selfplay(num_games=3)  # Just 3 games for testing
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.info("Please make sure model.safetensors and model.json exist")
        
    except Exception as e:
        logger.error(f"Self-play test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()