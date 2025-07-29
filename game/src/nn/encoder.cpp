//
// Created by zball on 25-7-21.
//

#include "encoder.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "chess/board.h"
#include "chess/bitboard.h"

#include <vector>
#include <string>
#include <map>
#include <stdexcept>

namespace lczero {

    namespace { // Anonymous namespace for helper functions

/**
 * @brief Determines the 0-15 index for a piece on a given square.
 *
 * The 16-dimensional one-hot vector is encoded as follows:
 * - Indices 0-6: Our pieces (R, A, C, P, N, B, K)
 * - Indices 7-13: Their pieces
 * - Index 14: Our Dark Piece
 * - Index 15: Their Dark Piece
 *
 * @param board The board state, viewed from the current player's perspective.
 * @param sq The square to inspect.
 * @return The index (0-15) for the one-hot vector, or -1 if the square is
 * empty or unknown and should not be encoded.
 */
        int GetPieceIndex(const ChessBoard& board, const BoardSquare& sq) {
            const ChessBoard::PieceType pt = board.at(sq);

            // Per the request, do not encode Unknown or Empty squares. The at()
            // method returns PIECE_TYPE_NB for empty squares.
            if (pt == ChessBoard::UNKNOWN || pt == ChessBoard::PIECE_TYPE_NB) {
                return -1;
            }

            if (pt == ChessBoard::DARK) {
                // Assumption: From the current player's perspective, dark pieces
                // on the lower half (rows 0-4) are ours.
                return sq.row() < 5 ? 14 : 15; // Our Dark : Their Dark
            }

            // Check for standard revealed piece types.
            if (pt >= ChessBoard::ROOK && pt < ChessBoard::PIECE_TYPE_NB) {
                const bool is_ours = board.ours().get(sq);
                int piece_offset = -1;

                if (pt == ChessBoard::ROOK) piece_offset = 0;
                else if (pt == ChessBoard::ADVISOR) piece_offset = 1;
                else if (pt == ChessBoard::CANNON) piece_offset = 2;
                else if (pt == ChessBoard::PAWN) piece_offset = 3;
                else if (pt == ChessBoard::KNIGHT) piece_offset = 4;
                else if (pt == ChessBoard::BISHOP) piece_offset = 5;
                else if (pt == ChessBoard::KING) piece_offset = 6;

                return piece_offset + (is_ours ? 0 : 7);
            }

            // Should not be reached given the checks above.
            return -1;
        }

/**
 * @brief Generates a 15-dimensional feature vector for remaining off-board dark pieces.
 */
        void GetDarkPieceFeatures(const DarkPieces* dp, std::vector<float>& features) {
            if (!dp) return;

            std::map<char, int> counts;
            for (int i = 0; i < 16 && dp->pieces[i] != '\0'; ++i) {
                counts[dp->pieces[i]]++;
            }

            // Unary encoding of piece counts. Total dimension: 5+2+2+2+2+2 = 15.
            for (int i = 0; i < 5; ++i) features.push_back(i < counts['p'] ? 1.0f : 0.0f);
            for (int i = 0; i < 2; ++i) features.push_back(i < counts['r'] ? 1.0f : 0.0f);
            for (int i = 0; i < 2; ++i) features.push_back(i < counts['n'] ? 1.0f : 0.0f);
            for (int i = 0; i < 2; ++i) features.push_back(i < counts['b'] ? 1.0f : 0.0f);
            for (int i = 0; i < 2; ++i) features.push_back(i < counts['c'] ? 1.0f : 0.0f);
            for (int i = 0; i < 2; ++i) features.push_back(i < counts['a'] ? 1.0f : 0.0f);
        }

    } // namespace

/**
 * @brief Encodes the game state into a feature matrix for the neural network.
 */
    std::vector<std::vector<float>> EncodeGameStateForNN(const PositionHistory& game_state) {
        const auto& positions = game_state.GetPositions();
        if (positions.empty()) {
            throw std::runtime_error("Cannot encode an empty game state.");
        }

        constexpr int kNumSquares = 90;
        constexpr int kHistoryPlies = 8;
        constexpr int kPieceFeatureDim = 16;

        // 1. Pre-compute global features (same for all tokens).
        std::vector<float> global_features;
        const auto& last_pos = positions.back();

        GetDarkPieceFeatures(last_pos.our_dark(), global_features);
        GetDarkPieceFeatures(last_pos.their_dark(), global_features);
        global_features.push_back(static_cast<float>(last_pos.GetRule50Ply()) / 120.0f);

        for (int i = 0; i < kHistoryPlies; ++i) {
            int history_idx = positions.size() - 1 - i;
            const auto& pos = (history_idx >= 0) ? positions[history_idx] : positions[0];
            global_features.push_back(pos.GetRepetitions() > 0 ? 1.0f : 0.0f);
        }

        // 2. Generate features for each of the 90 tokens (squares).
        std::vector<std::vector<float>> result(kNumSquares);
        for (int s = 0; s < kNumSquares; ++s) {
            BoardSquare sq(s);
            auto& token_features = result[s];

            // Feature: Piece at the square for the last 8 plies (8 * 16 = 128 floats)
            for (int i = 0; i < kHistoryPlies; ++i) {
                int history_idx = positions.size() - 1 - i;
                const auto& pos = (history_idx >= 0) ? positions[history_idx] : positions[0];
                const ChessBoard& board = pos.GetBoard();

                std::vector<float> piece_vec(kPieceFeatureDim, 0.0f);
                const int piece_idx = GetPieceIndex(board, sq);

                // Only set a bit if the piece type is one we encode (idx is valid).
                // Otherwise, the vector for this ply remains all zeros.
                if (piece_idx != -1) {
                    piece_vec[piece_idx] = 1.0f;
                }
                token_features.insert(token_features.end(), piece_vec.begin(), piece_vec.end());
            }

            // 3. Append the pre-computed global features to each token.
            token_features.insert(token_features.end(), global_features.begin(), global_features.end());
        }

        return result;
    }

} // namespace lczero