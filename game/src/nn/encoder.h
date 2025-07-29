//
// Created by zball on 25-7-21.
//

#pragma once

#ifndef VERSION_ENCODER_H
#define VERSION_ENCODER_H

#include <vector>
#include "chess/position.h"

// Forward declaration to reduce compilation dependencies.
// The full definition is in chess/gamestate.h.
namespace lczero {
    struct GameState;
}

namespace lczero {

/**
 * @brief Encodes the game state into a feature matrix for the neural network.
 *
 * This function generates an embedding for each of the 90 squares on the board.
 * Each embedding (token) is a concatenated vector of features based on the
 * game history and current position. The specific features are derived from
 * the last 8 plies of the game.
 *
 * @param game_state The current state of the game, including its history.
 * @return A vector of 90 vectors of floats. Each inner vector represents one
 * token and has a fixed length.
 */
    std::vector<std::vector<float>> EncodeGameStateForNN(const lczero::PositionHistory& game_state);

    const int nn_enc_channel_per_token = 167;
    const int nn_enc_token_count = 90;
} // namespace lczero
#endif //VERSION_ENCODER_H
