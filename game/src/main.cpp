#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>

#include "chess/position.h"
#include "chess/board.h"
#include "chess/bitboard.h"
#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "PIMCTS.h"
#include "determinized_game.h"
#include "nn/policy_map.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace lczero;

PYBIND11_MAKE_OPAQUE(std::vector<Move>)
PYBIND11_MAKE_OPAQUE(std::vector<MoveEvaluation>)


PYBIND11_MODULE(jieqi_game, m) {
    // Initialize magic bitboards on module load
    InitializeMagicBitboards();

    m.doc() = R"pbdoc(
        Leela Chess Zero Game Logic
        ---------------------------

        Python bindings for LCZero chess engine components including
        board representation, move generation, and position handling.

        .. currentmodule:: _core

        .. autosummary::
           :toctree: _generate

           BoardSquare
           BitBoard
           Move
           ChessBoard
           Position
           PositionHistory
           GameState
           GoParams
           MCTS
           DeterminizedGame
           MoveEvaluation
    )pbdoc";
    // m.def("add", [](int a,int b){return a+b;});

    // BoardSquare class
    py::class_<BoardSquare>(m, "BoardSquare")
            .def(py::init<>())
            .def(py::init<std::uint8_t>())
            .def(py::init<int, int>())
            .def(py::init<const std::string&, bool>())
            .def("as_int", &BoardSquare::as_int)
            .def("as_board", &BoardSquare::as_board)
            .def("row", &BoardSquare::row)
            .def("col", &BoardSquare::col)
            .def("Mirror", &BoardSquare::Mirror)
            .def("IsValid", static_cast<bool (BoardSquare::*)() const>(&BoardSquare::IsValid))
            .def_static("IsValidCoordRow", &BoardSquare::IsValidCoordRow)
            .def_static("IsValidCoordCol", &BoardSquare::IsValidCoordCol)
            .def("as_string", &BoardSquare::as_string)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("__repr__", [](const BoardSquare& sq) {
                return "<BoardSquare " + sq.as_string() + ">";
            });

    // BitBoard class
    py::class_<BitBoard>(m, "BitBoard")
            .def(py::init<>())
            .def(py::init<__uint128_t>())
            .def("as_int", &BitBoard::as_int)
            .def("clear", &BitBoard::clear)
            .def("count", &BitBoard::count)
            .def("count_few", &BitBoard::count_few)
            .def("set_if", static_cast<void (BitBoard::*)(BoardSquare, bool)>(&BitBoard::set_if))
            .def("set_if", static_cast<void (BitBoard::*)(std::uint8_t, bool)>(&BitBoard::set_if))
            .def("set_if", static_cast<void (BitBoard::*)(int, int, bool)>(&BitBoard::set_if))
            .def("set", static_cast<void (BitBoard::*)(BoardSquare)>(&BitBoard::set))
            .def("set", static_cast<void (BitBoard::*)(std::uint8_t)>(&BitBoard::set))
            .def("set", static_cast<void (BitBoard::*)(int, int)>(&BitBoard::set))
            .def("reset", static_cast<void (BitBoard::*)(BoardSquare)>(&BitBoard::reset))
            .def("reset", static_cast<void (BitBoard::*)(std::uint8_t)>(&BitBoard::reset))
            .def("reset", static_cast<void (BitBoard::*)(int, int)>(&BitBoard::reset))
            .def("get", static_cast<bool (BitBoard::*)(BoardSquare) const>(&BitBoard::get))
            .def("get", static_cast<bool (BitBoard::*)(std::uint8_t) const>(&BitBoard::get))
            .def("get", static_cast<bool (BitBoard::*)(int, int) const>(&BitBoard::get))
            .def("empty", &BitBoard::empty)
            .def("intersects", &BitBoard::intersects)
            .def("Mirror", &BitBoard::Mirror)
            .def("DebugString", &BitBoard::DebugString)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self &= py::self)
            .def(py::self -= py::self)
            .def(py::self |= py::self)
            .def(py::self | py::self)
            .def(py::self & py::self)
            .def(py::self - py::self)
            .def("__repr__", [](const BitBoard& bb) {
                return "<BitBoard count=" + std::to_string(bb.count()) + ">";
            });

    // Move class
    py::class_<Move>(m, "Move")
            .def(py::init<>())
            .def(py::init<BoardSquare, BoardSquare>())
            .def(py::init<const std::string&, bool>())
            .def("to", &Move::to)
            .def("from", &Move::from)
            .def("SetTo", &Move::SetTo)
            .def("SetFrom", &Move::SetFrom)
            .def("as_packed_int", &Move::as_packed_int)
            .def("as_nn_index", &Move::as_nn_index)
            .def("Mirror", &Move::Mirror)
            .def("as_string", &Move::as_string)
            .def("__bool__", &Move::operator bool)
            .def(py::self == py::self)
            .def("__repr__", [](const Move& move) {
                return "<Move " + move.as_string() + ">";
            });

    // MoveList typedef
    py::bind_vector<MoveList>(m, "MoveList");

    // ChessBoard enums
    py::enum_<ChessBoard::PieceType>(m, "PieceType")
            .value("ROOK", ChessBoard::ROOK)
            .value("ADVISOR", ChessBoard::ADVISOR)
            .value("CANNON", ChessBoard::CANNON)
            .value("PAWN", ChessBoard::PAWN)
            .value("KNIGHT", ChessBoard::KNIGHT)
            .value("BISHOP", ChessBoard::BISHOP)
            .value("KING", ChessBoard::KING)
            .value("KNIGHT_TO", ChessBoard::KNIGHT_TO)
            .value("PAWN_TO_OURS", ChessBoard::PAWN_TO_OURS)
            .value("PAWN_TO_THEIRS", ChessBoard::PAWN_TO_THEIRS)
            .value("DARK", ChessBoard::DARK)
            .value("DARK_TO", ChessBoard::DARK_TO)
            .value("UNKNOWN", ChessBoard::UNKNOWN)
            .value("PIECE_TYPE_NB", ChessBoard::PIECE_TYPE_NB);

    // ChessBoard class
    py::class_<ChessBoard>(m, "ChessBoard")
            .def(py::init<>())
            .def(py::init<const std::string&>())
            .def_readonly_static("kStartposFen", &ChessBoard::kStartposFen)
            .def_readonly_static("hStartposFen", &ChessBoard::hStartposFen)
            .def_readonly_static("kStartposBoard", &ChessBoard::kStartposBoard)
            .def_readonly_static("hStartposBoard", &ChessBoard::hStartposBoard)
            .def("SetFromFen", &ChessBoard::SetFromFen)
            .def("Clear", &ChessBoard::Clear)
            .def("Mirror", &ChessBoard::Mirror)
            .def("GeneratePseudolegalMoves", &ChessBoard::GeneratePseudolegalMoves)
            .def("ApplyMove", &ChessBoard::ApplyMove)
            .def("IsUnderCheck", &ChessBoard::IsUnderCheck)
            .def("HasMatingMaterial", &ChessBoard::HasMatingMaterial)
            .def("GenerateLegalMoves", &ChessBoard::GenerateLegalMoves)
            .def("IsSameMove", &ChessBoard::IsSameMove)
            .def("MakeChase", &ChessBoard::MakeChase)
            .def("UsChased", &ChessBoard::UsChased)
            .def("ThemChased", &ChessBoard::ThemChased)
            .def("Hash", &ChessBoard::Hash)
            .def("DebugString", &ChessBoard::DebugString)
            .def("ours", &ChessBoard::ours)
            .def("theirs", &ChessBoard::theirs)
            .def("rooks", &ChessBoard::rooks)
            .def("advisors", &ChessBoard::advisors)
            .def("cannons", &ChessBoard::cannons)
            .def("pawns", &ChessBoard::pawns)
            .def("knights", &ChessBoard::knights)
            .def("bishops", &ChessBoard::bishops)
            .def("kings", &ChessBoard::kings)
            .def("flipped", &ChessBoard::flipped)
            .def("revealed", &ChessBoard::revealed)
            .def("captured", &ChessBoard::captured)
            .def("at", &ChessBoard::at)
            .def("SetReveal", &ChessBoard::SetReveal)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("__repr__", [](const ChessBoard& board) {
                return "<ChessBoard flipped=" + std::to_string(board.flipped()) + ">";
            });

    // DarkPieces struct
    py::class_<DarkPieces>(m, "DarkPieces")
            .def(py::init<>())
            .def(py::init<const ChessBoard&, int>())
            .def("Remove", &DarkPieces::Remove)
            .def_readwrite("nleft", &DarkPieces::nleft)
            .def_property_readonly("pieces", [](const DarkPieces &p) {
                // Return the array of valid pieces as a string
                std::string pieces_str;
                for (int i = 0; i < 16; ++i) {
                    if (p.pieces[i] != '.') {
                        pieces_str += p.pieces[i];
                    }
                }
                return pieces_str;
            });

    // GameResult enum
    py::enum_<GameResult>(m, "GameResult")
            .value("UNDECIDED", GameResult::UNDECIDED)
            .value("BLACK_WON", GameResult::BLACK_WON)
            .value("DRAW", GameResult::DRAW)
            .value("WHITE_WON", GameResult::WHITE_WON);

    // Position class
    py::class_<Position>(m, "Position")
            .def(py::init<>())
            .def(py::init<const Position&, Move, ChessBoard::PieceType, ChessBoard::PieceType>())
            .def(py::init<const ChessBoard&, int, int>())
            .def_static("FromFen", &Position::FromFen)
            .def("Hash", &Position::Hash)
            .def("IsBlackToMove", &Position::IsBlackToMove)
            .def("GetGamePly", &Position::GetGamePly)
            .def("GetRepetitions", &Position::GetRepetitions)
            .def("GetPliesSincePrevRepetition", &Position::GetPliesSincePrevRepetition)
            .def("SetRepetitions", &Position::SetRepetitions)
            .def("GetRule50Ply", &Position::GetRule50Ply)
            .def("GetBoard", &Position::GetBoard, py::return_value_policy::reference_internal)
            .def("DebugString", &Position::DebugString)
            .def("our_dark", &Position::our_dark, py::return_value_policy::reference_internal)
            .def("their_dark", &Position::their_dark, py::return_value_policy::reference_internal)
            .def("__repr__", [](const Position& pos) {
                return "<Position ply=" + std::to_string(pos.GetGamePly()) +
                       " black_to_move=" + std::to_string(pos.IsBlackToMove()) + ">";
            });

    // PositionHistory class
    py::class_<PositionHistory>(m, "PositionHistory")
            .def(py::init<>())
            .def("Starting", &PositionHistory::Starting, py::return_value_policy::reference_internal)
            .def("Last", &PositionHistory::Last, py::return_value_policy::reference_internal)
            .def("GetPositionAt", &PositionHistory::GetPositionAt, py::return_value_policy::reference_internal)
            .def("Trim", &PositionHistory::Trim)
            .def("Reserve", &PositionHistory::Reserve)
            .def("GetLength", &PositionHistory::GetLength)
            .def("Reset", &PositionHistory::Reset)
            .def("Append", &PositionHistory::Append)
            .def("Pop", &PositionHistory::Pop)
            .def("RuleJudge", &PositionHistory::RuleJudge)
            .def("ComputeGameResult", &PositionHistory::ComputeGameResult)
            .def("IsBlackToMove", &PositionHistory::IsBlackToMove)
            .def("HashLast", &PositionHistory::HashLast)
            .def("DidRepeatSinceLastZeroingMove", &PositionHistory::DidRepeatSinceLastZeroingMove)
            .def("__len__", &PositionHistory::GetLength)
            .def("__repr__", [](const PositionHistory& hist) {
                return "<PositionHistory length=" + std::to_string(hist.GetLength()) + ">";
            });

    // GameState struct
    py::class_<GameState>(m, "GameState")
            .def(py::init<>())
            .def_readwrite("startpos", &GameState::startpos)
            .def_readwrite("moves", &GameState::moves)
            .def("CurrentPosition", &GameState::CurrentPosition)
            .def("GetPositions", &GameState::GetPositions)
            .def("__repr__", [](const GameState& gs) {
                return "<GameState moves=" + std::to_string(gs.moves.size()) + ">";
            });

    // MoveEvaluation struct
    py::class_<MoveEvaluation>(m, "MoveEvaluation")
            .def(py::init<>())
            .def_readwrite("move", &MoveEvaluation::move)
            .def_readwrite("visit_count", &MoveEvaluation::visit_count)
            .def_readwrite("policy_prior", &MoveEvaluation::policy_prior)
            .def_readwrite("win_prob", &MoveEvaluation::win_prob)
            .def_readwrite("draw_prob", &MoveEvaluation::draw_prob)
            .def_readwrite("loss_prob", &MoveEvaluation::loss_prob)
            .def("__repr__", [](const MoveEvaluation& me) {
                return "<MoveEvaluation move=" + me.move.as_string() +
                       " visits=" + std::to_string(me.visit_count) + ">";
            });

    // Bind vector for MoveEvaluation
    py::bind_vector<std::vector<MoveEvaluation>>(m, "MoveEvaluationList");

    // DeterminizedGame class
    py::class_<DeterminizedGame>(m, "DeterminizedGame")
            // ADD THIS CONSTRUCTOR BINDING:
            .def(py::init<PositionHistory &>(),
                 py::arg("history"),
                    // This keep_alive policy is CRUCIAL for memory safety.
                 py::keep_alive<1, 2>(),
                 "Constructs a determinized game from a PositionHistory.")

                    // Existing method bindings...
            .def("determinize", &DeterminizedGame::determinize)
            .def("append", &DeterminizedGame::Append, "Appends a move to the determinized history.")
            .def("pop", &DeterminizedGame::Pop, "Pops a move from the determinized history.")
            .def("get_position_history", &DeterminizedGame::GetPositionHistory,
                 "Gets a reference to the underlying position history.",
                 py::return_value_policy::reference_internal)
            .def("__repr__", [](const DeterminizedGame& dg) {
                const auto& history = dg.GetPositionHistory();
                return "<DeterminizedGame history_len=" +
                       std::to_string(history.GetLength()) + ">";
            });

    // PIMCTS class
    py::class_<MCTS>(m, "MCTS")
            .def(py::init<const PositionHistory&, int, float>(),
                 py::arg("history"), py::arg("batch_size"), py::arg("cpuct"),
                 "Initializes the MCTS search tree.")

            .def("run_search_batch", [](MCTS &mcts) {
                return py::array(py::cast(mcts.RunSearchBatch()));
            }, "Selects a batch of leaf nodes and returns their game states encoded for the NN.")
            .def("run_search", [](MCTS &mcts) {
                // Note: this is genuinely only useful for root search, where evaluation can be cached
                // and we need the encoding of root node as training data
                return py::array(py::cast(mcts.RunSearch()));
            }, "Selects a leaf node and returns its game states encoded for the NN.")

            .def("apply_evaluations", [](MCTS &mcts,
                                         py::array_t<float, py::array::c_style | py::array::forcecast> policy_array,
                                         py::array_t<float, py::array::c_style | py::array::forcecast> value_array) {
                std::span<const float> policy_data(policy_array.data(), policy_array.size());
                std::span<const float> value_data(value_array.data(), value_array.size());
                mcts.ApplyEvaluations(policy_data, value_data);
            }, py::arg("policy_data"), py::arg("value_data"), "Applies NN evaluation results (policy and value) to the batch of nodes.")

            .def("get_best_move", &MCTS::GetBestMove,
                 "Returns the best move found so far based on visit counts.")

            .def("redeterminize", &MCTS::Redeterminize,
                 "Resets the search tree and re-runs determinization with a new random sample.")

            .def("get_root_move_evaluations", &MCTS::GetRootMoveEvaluations,
                 "Gets the aggregated search statistics for each legal move from the root.")

            .def("get_determinized_game", &MCTS::GetDeterminizedGame,
                 "Gets a reference to the current determinized game state.",
                 py::return_value_policy::reference_internal);


    // GoParams struct for UCI
    py::class_<GoParams>(m, "GoParams")
            .def(py::init<>())
            .def_readwrite("wtime", &GoParams::wtime)
            .def_readwrite("btime", &GoParams::btime)
            .def_readwrite("winc", &GoParams::winc)
            .def_readwrite("binc", &GoParams::binc)
            .def_readwrite("movestogo", &GoParams::movestogo)
            .def_readwrite("depth", &GoParams::depth)
            .def_readwrite("nodes", &GoParams::nodes)
            .def_readwrite("movetime", &GoParams::movetime)
            .def_readwrite("infinite", &GoParams::infinite)
            .def_readwrite("searchmoves", &GoParams::searchmoves)
            .def_readwrite("ponder", &GoParams::ponder);

    // Utility functions
    m.def("GetFen", &GetFen, "Get FEN notation for position");
    m.def("MoveFromNNIndex", &MoveFromNNIndex, "Get move from neural network index");
    m.def("InitializeMagicBitboards", &InitializeMagicBitboards, "Initialize magic bitboard structures");

    m.attr("POLICY_SIZE") = policy_size;
    m.attr("nn_feature_dim") = 167;
    m.attr("nn_token_count") = 90;

    // 2. Expose the C-style array as a read-only NumPy array.
    // This creates a view of the C++ memory without copying it.
    m.attr("K_ATTN_POLICY_MAP") = py::array_t<short>(
            {8100},          // Shape of the array
            {sizeof(short)},        // Strides (size of one element)
            kAttnPolicyMap,         // Pointer to the data
            py::none()              // A dummy "base" object; tells numpy not to manage this memory
    );
    // Module version
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}