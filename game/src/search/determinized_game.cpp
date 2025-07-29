//
// Created by zball on 25-7-21.
//

#include "determinized_game.h"

namespace lczero {
    namespace {
        ChessBoard::PieceType char_to_piece_type(char c) {
            switch (c) {
                case 'r':
                    return ChessBoard::PieceType::ROOK;
                case 'a':
                    return ChessBoard::PieceType::ADVISOR;
                case 'c':
                    return ChessBoard::PieceType::CANNON;
                case 'p':
                    return ChessBoard::PieceType::PAWN;
                case 'n':
                    return ChessBoard::PieceType::KNIGHT;
                case 'b':
                    return ChessBoard::PieceType::BISHOP;
                default:
                    return ChessBoard::PieceType::PIECE_TYPE_NB;
            }
        }
    }

    void DeterminizedGame::determinize() {
        dark_board = ChessBoard(); // fill empty

        const auto &position = ph_.Last();
        const auto our_dark = position.our_dark();
        const auto their_dark = position.their_dark();
        const auto &board = position.GetBoard();
        std::vector<ChessBoard::PieceType> our_dark_pool, their_dark_pool;
        for (int i = 0; i < 15; ++i) {
            if (our_dark->pieces[i] != '.') our_dark_pool.push_back(char_to_piece_type(our_dark->pieces[i]));
            if (their_dark->pieces[i] != '.') their_dark_pool.push_back(char_to_piece_type(their_dark->pieces[i]));
        }
        std::random_device rd;
        std::mt19937 g(rd());
        int od = 0, td = 0;
        std::shuffle(our_dark_pool.begin(), our_dark_pool.end(), g);
        std::shuffle(their_dark_pool.begin(), their_dark_pool.end(), g);
        for (const auto square: board.darks()) {
            if (board.ours().get(square)) {
                dark_board.SetPiece(square, true, our_dark_pool[od++]);
            } else {
                dark_board.SetPiece(square, false, their_dark_pool[td++]);
            }
        }
    }
}