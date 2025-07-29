#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

// Include the necessary headers from your project
#include "board.h"
#include "position.h"
#include "PIMCTS.h"
#include "nn/encoder.h" // Needed for NN input dimensions

/**
 * @brief Mocks the behavior of a neural network for testing purposes.
 * @param nn_input The flattened batch of encoded game states from MCTS.
 * @param batch_size The number of game states in the batch.
 * @return A pair of float vectors: {policy_output, value_output}.
 */
std::pair<std::vector<float>, std::vector<float>> simulate_nn_evaluation(int batch_size) {
    // A C++ random number generator for creating mock data
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // 1. Mock Policy Head Output
    // The policy head has 2062 possible move outputs.
    const int num_possible_moves = 2062;
    std::vector<float> policy_output(batch_size * num_possible_moves);
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        // Get start of policy for this batch item
        auto policy_start = policy_output.begin() + i * num_possible_moves;
        // Fill with random values and find the sum for softmax
        for (int j = 0; j < num_possible_moves; ++j) {
            float val = dist(gen);
            *(policy_start + j) = val;
            sum += val;
        }
        // Normalize to create a probability distribution
        for (int j = 0; j < num_possible_moves; ++j) {
            *(policy_start + j) /= sum;
        }
    }

    // 2. Mock Value Head Output
    // The value head has 3 outputs: (Loss, Draw, Win)
    std::vector<float> value_output(batch_size * 3);
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        // Get start of value for this batch item
        auto value_start = value_output.begin() + i * 3;
        for (int j = 0; j < 3; ++j) {
            float val = dist(gen);
            *(value_start + j) = val;
            sum += val;
        }
        // Normalize
        for (int j = 0; j < 3; ++j) {
            *(value_start + j) /= sum;
        }
    }

    return {policy_output, value_output};
}


/**
 * @brief Main test function to run the MCTS search.
 */
void run_mcts_test() {
    // 1. Set up a non-trivial game position
    lczero::PositionHistory history;
    history.Reset(lczero::ChessBoard::hStartposBoard, 0, 0); //

    // Play a few moves to get an interesting position
    for (int i = 0; i < 5; ++i) {
        auto moves = history.Last().GetBoard().GenerateLegalMoves(); //
        if (moves.empty()) break;
        history.Append(moves.front()); //
    }

    std::cout << "--- Testing MCTS on the following position ---" << std::endl;
    std::cout << "FEN: " << lczero::GetFen(history.Last()) << std::endl;
    std::cout << "Player to move: " << (history.IsBlackToMove() ? "Black" : "White") << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 2. Initialize MCTS search
    const int batch_size = 8;
    const int num_simulations = 400;
    lczero::MCTS mcts(history, batch_size, 1.5); //

    // 3. Main search loop
    std::cout << "Running " << num_simulations << " simulations with batch size " << batch_size << "..." << std::endl;
    for (int i = 0; i < num_simulations / batch_size; ++i) {
        // Get a batch of encoded leaf nodes for the NN.
        std::vector<float> nn_input = mcts.RunSearchBatch();

        if (nn_input.empty()) {
            std::cout << "Search tree fully explored, stopping early." << std::endl;
            break;
        }

        // Determine batch size from the NN input vector dimensions.
        const int current_batch_size = nn_input.size() / (lczero::nn_enc_token_count * lczero::nn_enc_channel_per_token);

        // Get mock NN predictions.
        auto [mock_policy, mock_value] = simulate_nn_evaluation(current_batch_size);

        // Apply the mock evaluations to the tree.
        mcts.ApplyEvaluations(mock_policy, mock_value);
    }
    std::cout << "Search complete." << std::endl;
    std::cout << "------------------------------------------------------------------------------" << std::endl;

    // 4. Print results
    std::cout << "--- MCTS Root Move Evaluations ---" << std::endl;

    auto move_evals = mcts.GetRootMoveEvaluations(); //

    // Sort by visit count in descending order
    std::sort(move_evals.begin(), move_evals.end(), [](const auto& a, const auto& b) {
        return a.visit_count > b.visit_count;
    });

    // Print a formatted table header
    std::cout << std::left << std::setw(10) << "Move" << " | "
              << std::right << std::setw(8) << "Visits" << " | "
              << std::right << std::setw(8) << "Policy" << " | "
              << std::right << std::setw(7) << "Win%" << " | "
              << std::right << std::setw(7) << "Draw%" << " | "
              << std::right << std::setw(7) << "Loss%" << std::endl;
    std::cout << "----------+----------+----------+---------+---------+---------" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    for (const auto& eval : move_evals) {
        std::cout << std::left << std::setw(10) << eval.move.as_string() << " | "
                  << std::right << std::setw(8) << eval.visit_count << " | "
                  << std::fixed << std::setprecision(4) << std::right << std::setw(8) << eval.policy_prior << " | "
                  << std::fixed << std::setprecision(2)
                  << std::right << std::setw(6) << eval.win_prob * 100.0 << "% | "
                  << std::right << std::setw(6) << eval.draw_prob * 100.0 << "% | "
                  << std::right << std::setw(6) << eval.loss_prob * 100.0 << "%" << std::endl;
    }

    auto best_move = mcts.GetBestMove(); //
    std::cout << "------------------------------------------------------------------------------" << std::endl;
    std::cout << "🏆 Best Move according to search: " << best_move.as_string() << std::endl;
}

int main() {
    std::cout << "Debugger entry point started. Set breakpoints and step through." << std::endl;

    lczero::InitializeMagicBitboards(); //

    run_mcts_test();

    std::cout << "\n--- Execution paused. Press Enter to exit. ---" << std::endl;
    std::cin.get();

    return 0;
}