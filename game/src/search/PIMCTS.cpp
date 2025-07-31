#include "pimcts.h"
#include <algorithm>
#include <map>
#include <vector>

namespace lczero {

// --- Node Method Implementations ---

    double Node::GetQValue() const {
        if (visit_count == 0) {
            return 0.0;
        }
        double score = outcome_values[2] - outcome_values[0];
        int vlc = virtual_loss_count.load();
        int vabs = vlc * sign;
        double q = (score - static_cast<double>(vlc)) / static_cast<double>(visit_count+vabs);
        return q;
    }

    Node *Node::SelectChild(float cpuct) const {
        // --- OPTIMIZATION: Prioritize unvisited children ---
        // This is a common MCTS optimization. Before computing PUCT scores,
        // if we find any child that has never been explored, we pick it immediately.
        for (auto *child = first_child.get(); child != nullptr; child = child->next_sibling.get()) {
            if (child->visit_count == 0) {
                return child;
            }
        }

        // If all children have been visited, proceed with PUCT calculation.
        Node *best_child = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        double sqrt_total_visits = std::sqrt(static_cast<double>(visit_count));

        for (auto *child = first_child.get(); child != nullptr; child = child->next_sibling.get()) {
            double uct_score =
                    cpuct * child->policy_value * (sqrt_total_visits / (1.0 + child->visit_count))
                    - child->GetQValue(); // select child with smallest Q (I.E. worst situation for opponent)

            if (uct_score > best_score) {
                best_score = uct_score;
                best_child = child;
            }
        }
        return best_child;
    }

    void Node::AddVirtualLoss() {
        Node *current = this;
        while (current != nullptr) {
            current->virtual_loss_count += sign;
            current = current->parent;
        }
    }

    void Node::Backpropagate(std::span<const double> outcomes,const int virtual_loss) {
        double current_outcomes[3] = {outcomes[0], outcomes[1], outcomes[2]};
        Node *current = this;
        while (current != nullptr) {
            current->virtual_loss_count -= sign * virtual_loss;
            current->visit_count++;
            current->outcome_values[0] += current_outcomes[0];
            current->outcome_values[1] += current_outcomes[1];
            current->outcome_values[2] += current_outcomes[2];
            std::swap(current_outcomes[0], current_outcomes[2]);
            current = current->parent;
        }
    }

    void Node::Expand(const MoveList &legal_moves) {
        if (is_expanded) return;
        PoolUniqPtr<Node> *current_child_ptr = &first_child;
        for (const auto &move: legal_moves) {
            Node *raw_node = StaticPool<Node>::New(this, move, 0.0f);
            raw_node -> sign = -sign;
            *current_child_ptr = PoolUniqPtr<Node>(raw_node);
            current_child_ptr = &((*current_child_ptr)->next_sibling);
        }
        is_expanded = true;
    }

    void Node::FillPolicy(std::span<const float> policy_vector) {
        // A single loop that iterates through the children is sufficient.
        for (auto* child = first_child.get(); child != nullptr; child = child->next_sibling.get()) {
            // Get the neural network index for the child's move.
            uint16_t nn_index = child->move.as_nn_index(0);
            child->policy_value = policy_vector[nn_index];
        }
    }

// --- MCTS Method Implementations ---

    MCTS::MCTS(const PositionHistory &history, int batch_size, float cpuct)
            : initial_history_(history),
              determinized_game_(initial_history_),
              batch_size_(batch_size),
              cpuct_(cpuct) {
        initial_history_.SetVP(initial_history_.IsBlackToMove() ? Position::vp_black : Position::vp_red);
        // determinized_game_.determinize();
        Node *raw_root = StaticPool<Node>::New(nullptr, Move(), 1.0f);
        root_ = PoolUniqPtr<Node>(raw_root);
    }

    std::vector<float> MCTS::RunSearch() {
        auto rbatch = batch_size_;
        batch_size_ = 1;
        auto enc = RunSearchBatch();
        batch_size_ = rbatch;
        return enc;
    }

    std::vector<float> MCTS::RunSearchBatch() {
        pending_batch_.clear();
        std::vector<float> flat_encoded_states;
        // Reserve space for the flattened vector to reduce reallocations.
        // This assumes an average case; it will grow if needed.
        flat_encoded_states.reserve(
                batch_size_ * nn_enc_token_count * nn_enc_channel_per_token); // batch_size * tokens * channels

        int attempts = 0;
        const int max_attempts_per_batch = batch_size_ * 2;

        while (pending_batch_.size() < batch_size_ && attempts < max_attempts_per_batch) {
            attempts++;

            Node *current_node = root_.get();
            std::vector<Move> path;

            // Selection loop manipulates initial_history_ via the determinized_game_ reference
            while (current_node->is_expanded && current_node->result == GameResult::UNDECIDED) {
                Node *next_node = current_node->SelectChild(cpuct_);
                if (!next_node) break;

                path.push_back(next_node->move);
                determinized_game_.Append(next_node->move);
                current_node = next_node;
            }

            Node *leaf = current_node;

            if (leaf->result == GameResult::UNDECIDED) {
                leaf->result = initial_history_.ComputeGameResult();
            }

            if (leaf->result != GameResult::UNDECIDED) {
                double outcomes[3] = {0.0, 0.0, 0.0}; // L, D, W
                if (leaf->result == GameResult::DRAW) {
                    outcomes[1] = 1.0;
                } else if ((leaf->result == GameResult::WHITE_WON && !initial_history_.IsBlackToMove()) ||
                           (leaf->result == GameResult::BLACK_WON && initial_history_.IsBlackToMove())) {
                    outcomes[2] = 1.0;
                } else {
                    outcomes[0] = 1.0;
                }
                leaf->Backpropagate(outcomes,0);
            } else {
                if (!leaf->is_expanded) {
                    leaf->Expand(initial_history_.Last().GetBoard().GenerateLegalMoves());
                }
                leaf->AddVirtualLoss();
                pending_batch_.push_back(leaf);

                // Encode the current state and flatten the resulting matrix into the output vector
                auto encoded_matrix = EncodeGameStateForNN(initial_history_);
                for (const auto &row: encoded_matrix) {
                    flat_encoded_states.insert(flat_encoded_states.end(), row.begin(), row.end());
                }
            }

            // Pop all moves to revert initial_history_ for the next selection
            for (size_t i = 0; i < path.size(); ++i) {
                determinized_game_.Pop();
            }
        }
        return flat_encoded_states;
    }

    void MCTS::ApplyEvaluations(std::span<const float> policy_data, std::span<const float> value_data) {
        if (value_data.size() != pending_batch_.size() * 3) {
            throw std::runtime_error("NN evaluation results have incorrect value dimensions.");
        }

        // policy_size is a constant defined in policy_map.h
        if (policy_size * pending_batch_.size() != policy_data.size()) {
            throw std::runtime_error("NN evaluation results have incorrect policy dimensions.");
        }

        for (size_t i = 0; i < pending_batch_.size(); ++i) {
            Node *node = pending_batch_[i];

            auto policy_slice = policy_data.subspan(i * policy_size, policy_size);
            node->FillPolicy(policy_slice);

            auto value_slice = value_data.subspan(i * 3, 3);
            double outcomes[3] = {(double) value_slice[0], (double) value_slice[1], (double) value_slice[2]};
            node->Backpropagate(outcomes);
        }
    }

    Move MCTS::GetBestMove() const {
        int max_visits = -1;
        Node *best_child = nullptr;

        for (const auto *child = root_->first_child.get(); child != nullptr; child = child->next_sibling.get()) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_child = const_cast<Node *>(child);
            }
        }
        return best_child ? best_child->move : Move();
    }

    void MCTS::Redeterminize() {
        // Re-run the determinization process to get a new random sample of dark pieces.
        determinized_game_.determinize();

        // Create a new root node for the new search tree.
        Node* raw_root = StaticPool<Node>::New(nullptr, Move(), 1.0f);
        root_ = PoolUniqPtr<Node>(raw_root);
    }

    std::vector<MoveEvaluation> MCTS::GetRootMoveEvaluations() const {
        std::vector<MoveEvaluation> evals;
        if (!root_) {
            return evals;
        }

        // Iterate through the singly-linked list of the root's children.
        for (const auto* child = root_->first_child.get(); child != nullptr; child = child->next_sibling.get()) {
            MoveEvaluation eval;
            eval.move = child->move;
            eval.visit_count = child->visit_count;
            eval.policy_prior = child->policy_value;

            if (child->visit_count > 0) {
                // The child's outcome values are from its perspective, which is what we want
                // for the player making the move.
                eval.win_prob = child->outcome_values[2] / static_cast<double>(child->visit_count);
                eval.draw_prob = child->outcome_values[1] / static_cast<double>(child->visit_count);
                eval.loss_prob = child->outcome_values[0] / static_cast<double>(child->visit_count);
            }

            evals.push_back(eval);
        }

        return evals;
    }

    const DeterminizedGame &MCTS::GetDeterminizedGame() const {
        return determinized_game_;
    }
} // namespace lczero