// pimcts.h

#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <cmath>
#include <span>
#include <optional>
#include <utility>

#include "chess/position.h"
#include "determinized_game.h"
#include "nn/encoder.h"
#include "nn/policy_map.h"
#include "utils/pool_allocator.h" // Assumes your corrected pool_allocator is here

namespace lczero {

// Forward declarations
    class MCTS;

    struct Node;

// --- MCTS Tree Node and Manager ---

// Convenience type alias for a unique_ptr that uses the static pool.
    template<typename T>
    using PoolUniqPtr = std::unique_ptr<T, StaticPoolDeleter<T>>;

/**
 * @brief Represents a single node in the MCTS tree.
 */
    struct Node {
        Node *parent = nullptr;
        PoolUniqPtr<Node> first_child = nullptr;
        PoolUniqPtr<Node> next_sibling = nullptr;

        Move move;

        double outcome_values[3] = {0.0, 0.0, 0.0}; // Loss, Draw, Win

        int visit_count = 0, sign = 1;
        float policy_value = 0.0f;
        std::atomic<int> virtual_loss_count = 0;
        GameResult result = GameResult::UNDECIDED;
        bool is_expanded = false;

        Node(Node *p, Move m, float policy)
                : parent(p), move(m), policy_value(policy) {}

        Node *SelectChild(float cpuct) const;

        void AddVirtualLoss();

        void Backpropagate(std::span<const double> outcomes, const int virtual_loss=1);

        double GetQValue() const;

        // --- MODIFIED: Expand signature ---
        // Expand now only creates child nodes with a default policy value.
        void Expand(const MoveList &legal_moves);

        // --- NEW: FillPolicy method ---
        // This new method fills in the policy priors on already-existing children.
        void FillPolicy(std::span<const float> policy_vector);
    };

    struct MoveEvaluation {
        Move move;
        int visit_count = 0;
        float policy_prior = 0.0f;
        double win_prob = 0.0;
        double draw_prob = 0.0;
        double loss_prob = 0.0;
    };

    class MCTS {
    public:
        MCTS(const PositionHistory &history, int batch_size, float cpuct);

        // RunSearchBatch will now perform the expansion internally.
        std::vector<float> RunSearchBatch();
        std::vector<float> RunSearch(); // usually used for getting root encoding

        // ApplyEvaluations will now call FillPolicy and Backpropagate.
        void ApplyEvaluations(std::span<const float> policy_data, std::span<const float> value_data);

        Move GetBestMove() const;

        /**
         * @brief Resets the search tree and re-runs the determinization with a new random sample.
         * This is useful for running multiple independent PIMCTS searches on the same position.
         */
        void Redeterminize();

        /**
         * @brief Gets the aggregated search statistics for each legal move from the root.
         * @return A vector of MoveEvaluation structs, one for each child of the root node.
         */
        std::vector<MoveEvaluation> GetRootMoveEvaluations() const;

        const DeterminizedGame& GetDeterminizedGame() const;

    private:
        PoolUniqPtr<Node> root_;
        PositionHistory initial_history_;
        DeterminizedGame determinized_game_;
        std::vector<Node *> pending_batch_;
        int batch_size_;
        float cpuct_;

        Node *SelectLeaf();

        PositionHistory ReconstructHistory(Node *node) const;
    };


} // namespace lczero