import torch
import numpy as np
import logging
from typing import List, Tuple

# Assuming these imports from your modules
import jieqi_game  # Your C++ binding
from factory import load_leela_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class VerboseNeuralNetEvaluator:
    """Neural network evaluator with verbose output"""
    
    def __init__(self, model_path: str, config_path: str):
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Loading neural network model on {device}")
        
        self.model = load_leela_model(model_path, config_path, device)
        self.device = device
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  - Policy output size: {self.model.num_possible_policies}")
        print(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def evaluate_batch(self, position_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions"""
        batch_input = torch.Tensor(position_batch).to(self.device)
        
        with torch.no_grad():
            policy_logits, value_output = self.model(batch_input)
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            value_probs = value_output.cpu().numpy()
            
        return policy_probs, value_probs


def print_board_info(position_history: jieqi_game.PositionHistory, move_number: int):
    """Print detailed board information"""
    current_pos = position_history.Last()
    board = current_pos.GetBoard()
    
    #print(f"\n{'='*60}")
    #print(f"📋 MOVE {move_number} - {'BLACK' if current_pos.IsBlackToMove() else 'WHITE'} TO MOVE")
    #print(f"{'='*60}")
    
    # Print FEN
    fen = jieqi_game.GetFen(current_pos)
    print(f"FEN: {fen}")
    """
    # Print board visual representation
    print(f"\nBoard state:")
    print(board.DebugString())
    
    # Game statistics
    print(f"Game ply: {current_pos.GetGamePly()}")
    print(f"Rule 50 ply: {current_pos.GetRule50Ply()}")
    print(f"Repetitions: {current_pos.GetRepetitions()}")
    
    # Legal moves
    legal_moves = board.GenerateLegalMoves()
    print(f"Legal moves ({len(legal_moves)}): ", end="")
    if len(legal_moves) <= 10:
        print(", ".join([move.as_string() for move in legal_moves]))
    else:
        print(", ".join([move.as_string() for move in legal_moves[:10]]) + f"... (+{len(legal_moves)-10} more)")
    
    # Check status
    if board.IsUnderCheck():
        print("⚠️  King is in CHECK!")
    
    # Dark pieces info
    our_dark = current_pos.our_dark()
    their_dark = current_pos.their_dark()
    print(f"Our dark pieces remaining: {our_dark.nleft}")
    print(f"Their dark pieces remaining: {their_dark.nleft}")
    """


def print_mcts_stats(mcts: jieqi_game.MCTS, step: int, batch_size: int, 
                    policy_stats: np.ndarray, value_stats: np.ndarray):
    """Print MCTS step statistics"""
    if step % 20 == 0 or step < 5:  # Print first few steps and every 20th step
        avg_policy_entropy = -np.mean(np.sum(policy_stats * np.log(policy_stats + 1e-10), axis=1))
        avg_value = np.mean(value_stats)
        
        print(f"  🔍 MCTS Step {step:3d}: batch_size={batch_size:2d}, "
              f"avg_value={avg_value:+.3f}, policy_entropy={avg_policy_entropy:.3f}")


def print_move_selection(mcts: jieqi_game.MCTS, selected_move: jieqi_game.Move):
    """Print move selection details"""
    print(f"\n🎯 MOVE SELECTION:")
    
    # Get detailed move evaluations
    move_evaluations = list(mcts.get_root_move_evaluations())
    
    # Sort by visit count
    move_evaluations.sort(key=lambda x: x.visit_count, reverse=True)
    
    # Print top moves
    print(f"Top moves by visit count:")
    for i, eval in enumerate(move_evaluations[:8]):  # Top 8 moves
        is_selected = eval.move.as_string() == selected_move.as_string()
        marker = "👉" if is_selected else "  "
        print(f"{marker} {i+1:2d}. {eval.move.as_string():6s} "
              f"visits:{eval.visit_count:4d} "
              f"policy:{eval.policy_prior:.3f} "
              f"value:{eval.win_prob - eval.loss_prob:+.3f} "
              f"(W:{eval.win_prob:.2f} D:{eval.draw_prob:.2f} L:{eval.loss_prob:.2f})")
    
    total_visits = sum(eval.visit_count for eval in move_evaluations)
    print(f"Total MCTS visits: {total_visits}")
    
    print(f"\n✅ Selected move: {selected_move.as_string()}")


class VerboseSelfPlay:
    """Self-play with detailed logging and board display"""
    
    def __init__(self, model_path: str, config_path: str):
        self.evaluator = VerboseNeuralNetEvaluator(model_path, config_path)
        print(f"\n🚀 Verbose self-play engine initialized")
    
    def play_single_game(self, max_moves: int = 200, mcts_steps: int = 100) -> dict:
        """Play a single game with detailed logging"""
        
        print(f"\n🎮 STARTING NEW GAME")
        print(f"📋 Configuration:")
        print(f"  - Max moves: {max_moves}")
        print(f"  - MCTS steps per move: {mcts_steps}")
        print(f"  - MCTS batch size: 32")
        print(f"  - CPUCT: 1.25")
        
        # Initialize game
        position_history = jieqi_game.PositionHistory()
        position_history.Reset(jieqi_game.ChessBoard.hStartposBoard, 0, 0)
        determinized_game = jieqi_game.DeterminizedGame(position_history)
        determinized_game.determinize()
        
        # Print initial position
        print_board_info(position_history, 0)
        
        moves_played = 0
        
        # Game loop
        for move_num in range(1, max_moves + 1):
            # Check if game is over
            game_result = position_history.ComputeGameResult()
            if game_result != jieqi_game.GameResult.UNDECIDED:
                print(f"\n🏁 GAME ENDED: {game_result} after {move_num-1} moves")
                break
            
            # print(f"\n🤖 Running MCTS search for move {move_num}...")
            
            # Initialize MCTS
            mcts = jieqi_game.MCTS(position_history, batch_size=32, cpuct=1.25)
            
            # Run MCTS steps
            for step in range(mcts_steps):
                try:
                    # Get positions to evaluate
                    encodings = mcts.run_search_batch()
                    
                    if len(encodings) == 0:
                        if step < 5:  # Only warn for early steps
                            print(f"  ⚠️  No positions to evaluate at step {step}")
                        break
                    
                    # Reshape encodings: [batch_size * 14030] -> [batch_size, 90, 167]
                    batch_positions = np.array(encodings).reshape((-1, 90, 167))
                    batch_size = batch_positions.shape[0]
                    
                    # Evaluate with neural network
                    policy, value = self.evaluator.evaluate_batch(batch_positions)
                    
                    # Apply evaluations to MCTS
                    mcts.apply_evaluations(policy, value)
                    
                    # Print progress
                    # print_mcts_stats(mcts, step, batch_size, policy, value)
                    
                except Exception as e:
                    print(f"  ❌ MCTS step {step} failed: {e}")
                    raise e
            
            # Get best move
            try:
                best_move = mcts.get_best_move()
                
                if not best_move:  # Empty move
                    print(f"  ⚠️  No valid move found from MCTS, using fallback")
                    legal_moves = position_history.Last().GetBoard().GenerateLegalMoves()
                    if legal_moves:
                        best_move = legal_moves[0]
                        print(f"  🔄 Using first legal move: {best_move.as_string()}")
                    else:
                        print(f"  💀 No legal moves available!")
                        break
                
                # Print move selection details
                # print_move_selection(mcts, best_move)
                
                # Apply the move
                determinized_game.append(best_move)
                moves_played += 1
                
                # Print board after move
                print_board_info(position_history, move_num)
                
                # Wait for user input to continue (optional)
                if move_num <= 3:  # Pause for first few moves
                    input()
                    #input(f"\n⏸️  Press Enter to continue to move {move_num + 1}...")
                
            except Exception as e:
                print(f"❌ Error applying move: {e}")
                raise e
        
        # Final game result
        final_result = position_history.ComputeGameResult()
        final_fen = jieqi_game.GetFen(position_history.Last())
        
        #print(f"\n🏆 FINAL GAME RESULT")
        #print(f"{'='*60}")
        #print(f"Result: {final_result}")
        print(f"Moves played: {moves_played}")
        print(f"Final FEN: {final_fen}")
        #print(f"{'='*60}")
        
        return {
            'result': final_result,
            'moves_played': moves_played,
            'final_fen': final_fen
        }


def main():
    """Main function to run verbose self-play"""
    
    # Configuration
    model_path = "eval_small.safetensors"
    config_path = "eval_small.json"
    """
    print("🎯 VERBOSE SELF-PLAY DEMONSTRATION")
    print("="*60)
    print("This program will play 1 complete game and show:")
    print("- Board state after each move (visual + FEN)")
    print("- MCTS search progress and statistics")
    print("- Move selection details with visit counts")
    print("- Game progression and final result")
    print("="*60)
    """
    try:
        # Initialize verbose self-play
        selfplay = VerboseSelfPlay(model_path, config_path)
        
        # Play one game with detailed output
        result = selfplay.play_single_game(
            max_moves=100,   # Reasonable game length
            mcts_steps=50    # Moderate MCTS depth for demo
        )
        print(f"\n✨ GAME COMPLETED SUCCESSFULLY!")
        print(f"Final result: {result}")
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please ensure eval_small.safetensors and eval_small.json exist")
        
    except Exception as e:
        print(f"❌ Self-play failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()