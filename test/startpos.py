import jieqi_game as game
import numpy as np

def GenerateStartingPosition(
    max_length = 100,
    reveal_award = 1.15,
    capture_award = 1.2,
    capture_dark_award = 1.1,
    mate_award = 30,
    n_sidepiece_lb = 4,
    n_totalpiece_lb = 12,
    verbose = False
):
    defaultGame = game.ChessBoard(game.ChessBoard.hStartposFen)
    ph = game.PositionHistory()
    ph.Reset(defaultGame, 0, 0)
    ph.SetViewpoint(game.ViewPoint.VP_JUDGE)
    det_game = game.DeterminizedGame(ph)
    det_game.determinize()
    if verbose:
        print(game.GetExtFen(ph.Last()))
    for i in range(max_length):
        if ph.ComputeGameResult() != game.GameResult.UNDECIDED:
            break
        board = ph.Last().GetBoard()
        our_pieces = board.ours().count()
        their_pieces = board.theirs().count()
        if our_pieces <= n_sidepiece_lb or their_pieces <= n_sidepiece_lb or our_pieces + their_pieces <= n_totalpiece_lb:
            break
        moves = board.GenerateLegalMoves()
        trait = board.GetMoveTraits(moves)
        trait_score = np.ones((len(trait)), dtype = np.float32)
        traits = np.array(trait, dtype = np.uint8)
        trait_score[traits & 1 == 1] *= reveal_award
        trait_score[traits & 2 == 2] *= capture_award
        trait_score[traits & 4 == 4] *= capture_dark_award
        trait_score[traits & 8 == 8] *= mate_award
        choice = np.random.choice(len(trait), p=trait_score/np.sum(trait_score))
        det_game.append(moves[choice])
        if verbose:
            print(f"{game.GetExtFen(ph.Last())} : With move {moves[choice]}")
    return ph

if __name__ == "__main__":
    GenerateStartingPosition(verbose = True)