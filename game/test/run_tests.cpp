#include <iostream>
#include <iomanip>
#include <vector>
#include <span>
#include <memory>

#include "chess/position.h"
#include "PIMCTS.h"

void test_mcts_basic() {
    using namespace lczero;

    std::cout << "Starting MCTS test..." << std::endl;

    // Create position from FEN
    Position pos = Position::FromFen("2h3n2/1c3k3/b1a1p1b2/6N2/1Cp3p2/p7R/7A1/2B6/P2CP4/4K1B1r b - - 7 46");

    // Create position history
    PositionHistory ph;
    ph.Reset(pos.GetBoard(), pos.GetRule50Ply(), pos.GetGamePly());

    std::cout << "Position initialized. Game ply: " << pos.GetGamePly() << std::endl;
    std::cout << "Black to move: " << (pos.IsBlackToMove() ? "true" : "false") << std::endl;

    // Create MCTS instance
    MCTS mcts(ph, 8, 1.25f);  // batch_size=20, cpuct=1.25

    std::cout << "MCTS initialized." << std::endl;

    // Run search iterations
    const int num_iterations = 200;
    for (int i = 0; i < num_iterations; ++i) {
        // Get batch of encoded positions
        std::vector<float> encoded_batch = mcts.RunSearchBatch();

        // Calculate batch dimensions
        size_t batch_size = encoded_batch.size() / (90 * 167);

        if (batch_size == 0) {
            std::cout << "No more positions to evaluate at iteration " << i << std::endl;
            break;
        }

        // Create dummy policy (uniform distribution)
        std::vector<float> policy_data(batch_size * 2550);
        float policy_value = 1.0f / 2550.0f;
        std::fill(policy_data.begin(), policy_data.end(), policy_value);

        // Create dummy value (all draws)
        std::vector<float> value_data(batch_size * 3, 0.0f);
        for (size_t j = 1; j < value_data.size(); j += 3) {
            value_data[j] = 1.0f;  // Set draw probability to 1.0
        }

        // Apply evaluations
        std::span<const float> policy_span(policy_data);
        std::span<const float> value_span(value_data);
        mcts.ApplyEvaluations(policy_span, value_span);

        // Print progress every 10 iterations
        if ((i + 1) % 10 == 0) {
            std::cout << "Completed iteration " << (i + 1) << "/" << num_iterations
                      << ", batch size: " << batch_size << std::endl;
        }
    }

    // Get move evaluations
    std::vector<MoveEvaluation> move_evals = mcts.GetRootMoveEvaluations();

    std::cout << "\nMove evaluations:" << std::endl;
    std::cout << "Found " << move_evals.size() << " moves" << std::endl;

    // Sort by visit count (descending)
    std::sort(move_evals.begin(), move_evals.end(),
              [](const MoveEvaluation& a, const MoveEvaluation& b) {
                  return a.visit_count > b.visit_count;
              });

    // Display top moves
    std::cout << "\nTop moves:" << std::endl;
    std::cout << "Move     Visits  Policy   Win%    Draw%   Loss%" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    const size_t max_display = std::min(size_t(10), move_evals.size());
    for (size_t i = 0; i < max_display; ++i) {
        const auto& eval = move_evals[i];
        std::cout << eval.move.as_string() << "     "
                  << eval.visit_count << "       "
                  << std::fixed << std::setprecision(4) << eval.policy_prior << "   "
                  << std::setprecision(1) << (eval.win_prob * 100) << "%     "
                  << (eval.draw_prob * 100) << "%     "
                  << (eval.loss_prob * 100) << "%" << std::endl;
    }

    // Get best move
    if (!move_evals.empty()) {
        Move best_move = mcts.GetBestMove();
        std::cout << "\nBest move: " << best_move.as_string() << std::endl;
    }

    std::cout << "MCTS test completed successfully!" << std::endl;
}

void test_mcts_full_game() {
    using namespace lczero;

    std::cout << "\nStarting full game simulation..." << std::endl;

    // Create position from FEN
    Position pos = Position::FromFen("2h3n2/1c3k3/b1a1p1b2/9/1Cp3p1N/p7R/7A1/2B6/P2CP4/4K1B1r w - - 6 46");

    // Create position history
    PositionHistory ph;
    ph.Reset(pos.GetBoard(), pos.GetRule50Ply(), pos.GetGamePly());

    const int max_moves = 20;  // Limit for testing
    int move_count = 0;

    while (move_count < max_moves) {
        const Position& current_pos = ph.Last();

        // Check if game is over
        GameResult result = ph.ComputeGameResult();
        if (result != GameResult::UNDECIDED) {
            std::cout << "Game over! Result: ";
            switch (result) {
                case GameResult::WHITE_WON: std::cout << "White wins"; break;
                case GameResult::BLACK_WON: std::cout << "Black wins"; break;
                case GameResult::DRAW: std::cout << "Draw"; break;
                default: std::cout << "Unknown"; break;
            }
            std::cout << std::endl;
            break;
        }

        // Check legal moves
        MoveList legal_moves = current_pos.GetBoard().GenerateLegalMoves();
        if (legal_moves.empty()) {
            std::cout << "No legal moves available!" << std::endl;
            break;
        }

        std::cout << "\nMove " << (move_count + 1) << ": "
                  << (current_pos.IsBlackToMove() ? "Black" : "White")
                  << " (" << legal_moves.size() << " legal moves)" << std::endl;

        // Create MCTS and run search
        MCTS mcts(ph, 10, 1.25f);

        // Run fewer iterations for speed
        const int iterations = 200;
        for (int i = 0; i < iterations; ++i) {
            std::vector<float> encoded_batch = mcts.RunSearchBatch();
            size_t batch_size = encoded_batch.size() / (90 * 167);

            if (batch_size == 0) break;

            // Dummy evaluations
            std::vector<float> policy_data(batch_size * 2550, 1.0f / 2550.0f);
            std::vector<float> value_data(batch_size * 3, 0.0f);
            for (size_t j = 1; j < value_data.size(); j += 3) {
                value_data[j] = 1.0f;  // Draw
            }

            mcts.ApplyEvaluations(std::span<const float>(policy_data),
                                  std::span<const float>(value_data));
        }

        // Get and display top moves
        std::vector<MoveEvaluation> move_evals = mcts.GetRootMoveEvaluations();
        if (move_evals.empty()) {
            std::cout << "No move evaluations!" << std::endl;
            break;
        }

        std::sort(move_evals.begin(), move_evals.end(),
                  [](const MoveEvaluation& a, const MoveEvaluation& b) {
                      return a.visit_count > b.visit_count;
                  });

        std::cout << "Top 3 moves:" << std::endl;
        const size_t top_moves = std::min(size_t(3), move_evals.size());
        for (size_t i = 0; i < top_moves; ++i) {
            std::cout << "  " << (i + 1) << ". " << move_evals[i].move.as_string()
                      << " (visits: " << move_evals[i].visit_count << ")" << std::endl;
        }

        // Play best move
        Move best_move = move_evals[0].move;
        std::cout << "Playing: " << best_move.as_string() << std::endl;

        ph.Append(best_move);
        move_count++;
    }

    std::cout << "Game simulation completed after " << move_count << " moves." << std::endl;
}

// Add this to your run_tests.cpp main function:
int main() {
    try {
        // Initialize magic bitboards (required for move generation)
        lczero::InitializeMagicBitboards();

        // Run the basic MCTS test
        test_mcts_basic();

        // Run the full game simulation
        // test_mcts_full_game();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}