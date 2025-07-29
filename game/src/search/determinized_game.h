//
// Created by zball on 25-7-21.
//

#ifndef VERSION_DETERMINIZED_GAME_H
#define VERSION_DETERMINIZED_GAME_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>

#include "chess/position.h"

namespace lczero {
    class DeterminizedGame {
    private:
        PositionHistory& ph_;
        ChessBoard dark_board;
    public:
        DeterminizedGame() = default;
        DeterminizedGame(const DeterminizedGame &other) = default;
        DeterminizedGame(DeterminizedGame &&other) = default;
        DeterminizedGame &operator=(const DeterminizedGame &other) = default;
        DeterminizedGame &operator=(DeterminizedGame &&other) = default;

        DeterminizedGame(PositionHistory &t) : ph_(t), dark_board() {}

        void determinize();

        inline void Append(Move m){
            const auto &board = ph_.Last().GetBoard();
            ChessBoard::PieceType reveal = lczero::ChessBoard::UNKNOWN;
            ChessBoard::PieceType capture = lczero::ChessBoard::UNKNOWN;
            if(board.IsCapture(m)) capture = dark_board.at(m.to());
            if(board.IsReveal(m)) reveal = dark_board.at(m.from());
            ph_.Append(m, reveal, capture);
            dark_board.Mirror();
        }

        inline void Pop(){
            ph_.Pop();
            dark_board.Mirror();
        }

        inline const PositionHistory& GetPositionHistory()const{ return ph_; }
    };
}

#endif //VERSION_DETERMINIZED_GAME_H
