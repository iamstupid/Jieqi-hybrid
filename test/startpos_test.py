import jieqi_game as game
import startpos

for i in range(100):
    ph = game.PositionHistory(startpos.GenerateStartingPosition())
    det = game.DeterminizedGame(ph)
    print(game.GetExtFen(ph.Last()))
    det.determinize()

    