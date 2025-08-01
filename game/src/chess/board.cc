/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "chess/board.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <set>

#include "utils/exception.h"

#ifndef NO_PEXT
// Include header for pext instruction.
#include <immintrin.h>
#ifdef _MSC_VER
#define pext(b, m, s) ((_pext_u64(b._Word[1], m._Word[1]) << s) | _pext_u64(b._Word[0], m._Word[0]))
#else
#define pext(b, m, s) ((_pext_u64(b >> 64, m >> 64) << s) | _pext_u64(b, m))
#endif
#endif

namespace lczero {

    const char* ChessBoard::kStartposFen =
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    static const char initBoard[]="rnbakabnr          c     c p p p p p                  P P P P P C     C          RNBAKABNR";

    const char* ChessBoard::hStartposFen =
            "hhhhkhhhh/9/1h5h1/h1h1h1h1h/9/9/H1H1H1H1H/1H5H1/9/HHHHKHHHH w - - 0 1";

    const ChessBoard ChessBoard::kStartposBoard(ChessBoard::kStartposFen);
    const ChessBoard ChessBoard::hStartposBoard(ChessBoard::hStartposFen);

    void ChessBoard::Clear() {
        *this = ChessBoard();
    }

    void ChessBoard::Mirror() {
        our_pieces_.Mirror();
        their_pieces_.Mirror();
        std::swap(our_pieces_, their_pieces_);
        rooks_.Mirror();
        advisors_.Mirror();
        cannons_.Mirror();
        pawns_.Mirror();
        knights_.Mirror();
        bishops_.Mirror();
        darks_.Mirror();
        unknowns_.Mirror();
        our_king_.Mirror();
        their_king_.Mirror();
        std::swap(our_king_, their_king_);
        flipped_ = !flipped_;
    }

    namespace {
        using Direction = std::pair<int, int>;

        Direction NORTH = {1, 0};
        Direction EAST  = {0, 1};
        Direction SOUTH = {-1, 0};
        Direction WEST  = {0, -1};
        Direction NORTH_WEST = {1, -1};
        Direction NORTH_EAST = {1, 1};
        Direction SOUTH_WEST = {-1, -1};
        Direction SOUTH_EAST = {-1, 1};

        static const std::set<Direction> kBishopDirections = {
                {2, 2}, {-2, 2}, {2, -2}, {-2, -2}};

        static const std::set<Direction> kKnightDirections = {
                {-2, -1}, {-2, 1}, {2, -1}, {2, 1}, {1, -2}, {1, 2}, {-1, -2}, {-1, 2}};

        constexpr __uint128_t Palace = __uint128_t(0x70381CULL) << 64 | __uint128_t(0xE07038ULL);

        constexpr __uint128_t FileABB = __uint128_t(0x20100ULL) << 64 | __uint128_t(0x8040201008040201ULL);
        constexpr __uint128_t FileCBB = FileABB << 2;
        constexpr __uint128_t FileEBB = FileABB << 4;
        constexpr __uint128_t FileGBB = FileABB << 6;
        constexpr __uint128_t FileIBB = FileABB << 8;

        constexpr __uint128_t Rank0BB = 0x1FF;
        constexpr __uint128_t Rank1BB = Rank0BB << (ChessBoard::FILE_NB * 1);
        constexpr __uint128_t Rank2BB = Rank0BB << (ChessBoard::FILE_NB * 2);
        constexpr __uint128_t Rank3BB = Rank0BB << (ChessBoard::FILE_NB * 3);
        constexpr __uint128_t Rank4BB = Rank0BB << (ChessBoard::FILE_NB * 4);
        constexpr __uint128_t Rank5BB = Rank0BB << (ChessBoard::FILE_NB * 5);
        constexpr __uint128_t Rank6BB = Rank0BB << (ChessBoard::FILE_NB * 6);
        constexpr __uint128_t Rank7BB = Rank0BB << (ChessBoard::FILE_NB * 7);
        constexpr __uint128_t Rank8BB = Rank0BB << (ChessBoard::FILE_NB * 8);
        constexpr __uint128_t Rank9BB = Rank0BB << (ChessBoard::FILE_NB * 9);

        const BitBoard BishopBB = ((FileABB | FileEBB | FileIBB) & (Rank2BB | Rank7BB)) |
                                  ((FileCBB | FileGBB) & (Rank0BB | Rank4BB | Rank5BB | Rank9BB));
        constexpr BitBoard PawnFileBB = FileABB | FileCBB | FileEBB | FileGBB | FileIBB;
        constexpr BitBoard HalfBB[2] = { Rank0BB | Rank1BB | Rank2BB | Rank3BB | Rank4BB,
        Rank5BB | Rank6BB | Rank7BB | Rank8BB | Rank9BB };
    constexpr BitBoard PawnBB[2] = { HalfBB[1].as_int() | ((Rank3BB | Rank4BB) & PawnFileBB.as_int()),
    HalfBB[0].as_int() | ((Rank6BB | Rank5BB) & PawnFileBB.as_int()) };

BitBoard PseudoAttacks[ChessBoard::PIECE_TYPE_NB+1][90];

// Magic bitboard routines and structures.
// We use so-called "fancy" magic bitboards.

// Structure holding all relevant magic parameters per square.
struct MagicParams {
    // Relevant occupancy mask.
    __uint128_t mask_;
    // Pointer to lookup table.
    BitBoard* attacks_table_;
#if defined(NO_PEXT)
    // Magic number.
  __uint128_t magic_number_;
#endif
    // Number of bits to shift.
    uint8_t shift_bits_;

    // Compute the attack's index using the 'magic bitboards' approach
    unsigned index(BitBoard occupied) const {
#if defined(NO_PEXT)
        return unsigned(((occupied.as_int() & mask_) * magic_number_) >> shift_bits_);
#else
        return unsigned(pext(occupied.as_int(), mask_, shift_bits_));
#endif
    }
};

#if defined(NO_PEXT)
// Magic numbers determined via trial and error with random number generator
// such that the number of relevant occupancy bits suffice to index the attacks
// tables with only constructive collisions.
#define B(h, l) (__uint128_t(h) << 64) ^ __uint128_t(l)
constexpr __uint128_t kRookMagicNumbers[] = {
    B(0x4040000414000A40, 0x8A08C0010C100400), B(0x0520004802000020, 0x2000030408010008),
    B(0x7040010400065040, 0x0018400000034001), B(0x4300008808100040, 0x40084200E4040004),
    B(0x0400200200400100, 0x40080001000000A8), B(0x4040010001200049, 0x0019808808840100),
    B(0x064002A0C000410B, 0x000500000A200000), B(0x0200000900040084, 0x0800810000064000),
    B(0x0080010400860A02, 0x0000088000400121), B(0x50002000085A0000, 0x20483041002001DA),
    B(0x028A500012000060, 0x8010000120101204), B(0x4000400110002040, 0x80200022A0040210),
    B(0x0440400101000080, 0x0900010040080000), B(0x00BA800080818008, 0x4200500401000200),
    B(0x0000800200084010, 0x0401000800000080), B(0x184080009024A104, 0x0008050004000002),
    B(0x0100400440900010, 0x818280000022A200), B(0x0C10800001200900, 0x460480A042200120),
    B(0x002060100004C392, 0x80005840C0080300), B(0x8220008400204000, 0x2000102800008000),
    B(0x0A18002400004000, 0x0800900002004000), B(0x2008201000302020, 0x0010000010A50000),
    B(0x008C001000240020, 0x0204000012008000), B(0x0000020008008010, 0x0440100426200020),
    B(0x1080010004108002, 0x1002100000008000), B(0x6000008000020214, 0x0914040403015061),
    B(0x2008004000048A18, 0x4042004008010228), B(0xA830100008000480, 0x5C30400000000020),
    B(0x000802080C000180, 0x001000000000C440), B(0x0291680010002120, 0x8010800920000040),
    B(0x2011020008002040, 0x0002400A40020401), B(0x10000400B0004000, 0x0901801002010000),
    B(0x0800900100040008, 0x2020020200010020), B(0x0000600008000800, 0x010200004A020081),
    B(0x2002086040001220, 0x1044048420402100), B(0x2211008020002101, 0x0900080840008104),
    B(0x1101242108040004, 0x40400000000000A4), B(0x2400040084220000, 0x4000040080400000),
    B(0x0280200080840040, 0x0020000000041430), B(0x0018000050080010, 0x4008080020908008),
    B(0x000CA10000240004, 0x1000400840186000), B(0x0880100400800200, 0x08020800A4820106),
    B(0x0060400481080021, 0x0400118200101410), B(0x0000002004200004, 0x0800440048300881),
    B(0x2A10040440300009, 0x0100200500281502), B(0x0E02A42800150200, 0x1080000000090A20),
    B(0x8088040004004B00, 0x0420000082128104), B(0x2008200142000200, 0x0810004000800C00),
    B(0x4201200200042200, 0x0448004200020402), B(0x0100800301A01800, 0x1820080200070042),
    B(0x018C800081016800, 0x0122140050100A01), B(0x0101200040000200, 0x0200004001841220),
    B(0x0000A00020000200, 0x010000100D104000), B(0x0040400002C28400, 0x0040408008400070),
    B(0x0000200800008021, 0x0000100842001000), B(0x8020003100028004, 0x4000900100000C0A),
    B(0x0008000080004000, 0x40001000400000C0), B(0x004A020008008001, 0x0002020004000A02),
    B(0x0200800200401004, 0x0008100020004001), B(0x0002800200204004, 0x0004006044000080),
    B(0x2103200040000401, 0x0001000400008440), B(0x09200810A0408408, 0x0002240400061000),
    B(0x510000124AA00304, 0x000080200A180001), B(0x0122080208004100, 0x0080004090048080),
    B(0x0010080001000301, 0x10C0020084001001), B(0x0100040084040084, 0x0200040010800200),
    B(0x0010400A000500A0, 0x3500022D10085010), B(0x1006810000140020, 0x0080010408102004),
    B(0x6000200040002000, 0x2280024883004800), B(0x0220400020041000, 0x8100020000001000),
    B(0x000006004C410409, 0x0400058000400040), B(0x00000110041804A2, 0x2600010000002884),
    B(0x81001A0204100A02, 0x0121000000006000), B(0x000408000C0000A0, 0x6002600310000000),
    B(0x0000780020840010, 0x6000400000440008), B(0x0230812012001020, 0x200080008A100080),
    B(0x6822020000102000, 0x0480400048000100), B(0x100C800041002000, 0x10010010C4000081),
    B(0x0000400240012013, 0x0A04800040002001), B(0x48080204A0022011, 0x0102000000000800),
    B(0x0108000810041000, 0x2021000000880002), B(0x1001842002002200, 0x0008A04000200202),
    B(0x0840100008042060, 0x4002082000040000), B(0x0430C22040410001, 0x0010002000004B80),
    B(0x4410088281000201, 0x8208002000000208), B(0x4000800034408080, 0x04280080002004B0),
    B(0x0000E00010400004, 0x0400002000048022), B(0x0000400040000170, 0x0204414000000408),
    B(0x0000004500402202, 0x1043010000048108), B(0x211A200001280000, 0x8608902000084008)
};

constexpr __uint128_t kBishopMagicNumbers[] = {
    B(0x00376C0000480001, 0x0880010041200001), B(0x0017E20100000000, 0x0001000000000000),
    B(0x0419840C00040020, 0x0000480288204041), B(0x0013100000100202, 0x40000004000000A0),
    B(0x2109900100320000, 0x2808002002A00120), B(0x0203240002068000, 0x0001D10108800100),
    B(0x0001964090001018, 0x7002040148001205), B(0x0225F90800201080, 0x000304120101C208),
    B(0x0800AF8102030000, 0x0680000002000388), B(0x000017B328894000, 0x2104028200050000),
    B(0x20800F9804080100, 0x1004A80010030002), B(0x0460332208000040, 0x040C04410001200A),
    B(0x8000492C20003008, 0x0004004044000800), B(0x100004C540000000, 0x00002042000014C0),
    B(0x5102466642401000, 0xA000100020240264), B(0x0200012980701000, 0x08C0004004000020),
    B(0x004400BE80004818, 0x8203008000801404), B(0x4420006BC5086000, 0x6010000000202085),
    B(0x00214004A9000100, 0x1400004008002E80), B(0x0010600325200200, 0x0120802002144002),
    B(0x94080C0120000480, 0x2044412202002000), B(0x6448800110400100, 0x0000000000106400),
    B(0x000100008A041200, 0x11101401100C1090), B(0x4280900041000000, 0x60002410002000C0),
    B(0x08C0488120024214, 0x0008414880202291), B(0x2080B40050544300, 0x0424A04000000002),
    B(0x8001930019090445, 0x040061000C000104), B(0x00103040126E2283, 0x000002008420C3B0),
    B(0x0200070100368420, 0x0000040208040002), B(0x0000030808040800, 0x0A00128000001000),
    B(0x0121028100114080, 0x5010280481100082), B(0x0121028100114080, 0x5010280481100082),
    B(0x0121028100114080, 0x5010280481100082), B(0x4024004101028240, 0x0000A20000500060),
    B(0x410802C8710C8812, 0x1001810000130814), B(0x42100025C1058000, 0x0020440120000018),
    B(0x0418008ED8000822, 0x40420400820000F0), B(0x0000030778300000, 0x2100801404028000),
    B(0x0024800990801240, 0x00040200080C0040), B(0x1080010B0C800008, 0x1800401040602208),
    B(0xC202408A6460000C, 0x0809031100000800), B(0x00901910C9040000, 0x0000000240A00001),
    B(0xC202408A6460000C, 0x0809031100000800), B(0x010020416EC20018, 0xC001400040040410),
    B(0x081000003F200160, 0x6010C01040000008), B(0x0120409000C081E7, 0x0000804200001812),
    B(0x20020222000000B7, 0xB011000400201000), B(0x0008061084112165, 0x8880040002418026),
    B(0x0008061084112165, 0x8880040002418026), B(0x0100600000882031, 0x8000004200810001),
    B(0x0008061084112165, 0x8880040002418026), B(0x0006000205000216, 0x60100044010001A8),
    B(0x000001804000080D, 0xD80008400012001C), B(0x2480000804001007, 0xE8108020001000F0),
    B(0x440C400000015800, 0xA9CC00150A080410), B(0x440C400000015800, 0xA9CC00150A080410),
    B(0x1540008800400400, 0x6200082800020120), B(0x0800000000300482, 0x9203008100100013),
    B(0x0800000000300482, 0x9203008100100013), B(0x0800000000300482, 0x9203008100100013),
    B(0x0800000000300482, 0x9203008100100013), B(0x8410020001102F08, 0x0422000208C08000),
    B(0x0100000010090058, 0x0388010061000102), B(0x001001440101010C, 0x0034444800000000),
    B(0x1000480021008050, 0x8829480490100020), B(0x0800704010804022, 0x4810A00000020000),
    B(0xC00A810014000512, 0x0208402410204220), B(0x2082000900000108, 0x0024500444001400),
    B(0x8040002000409004, 0x8002110011010809), B(0x000C200005410002, 0x1004404404480000),
    B(0x3020084020120848, 0x0801782844520000), B(0xB080420104000502, 0x2901310038803052),
    B(0x008A408108080000, 0xF692812040001287), B(0x0813000800008008, 0x7D40000485880010),
    B(0x0008020031500100, 0xE588D01000044000), B(0x0010028910042039, 0x3320018000410404),
    B(0x0008020031500100, 0xE588D01000044000), B(0x0010028910042039, 0x3320018000410404),
    B(0x2000000400000000, 0x894C002004240100), B(0x8100400021200040, 0x157E000202900082),
    B(0x40028212A0028210, 0x03F2000810100800), B(0x0000010000020010, 0x00EEC00000020220),
    B(0x800400A23C070820, 0x112FB88050021000), B(0x0108084604040181, 0x4032402000500400),
    B(0x020000040018A404, 0x2126413000200014), B(0x0000010018000004, 0x400C58201000A800),
    B(0x0000102200440200, 0x00C65940000C4000), B(0x9018511020008110, 0x2103130220180000),
    B(0x0224200000201000, 0x2403CE4013013004), B(0x1000002048040400, 0x00015F8040A04004)
};

constexpr __uint128_t kKnightMagicNumbers[] = {
    B(0x61CE000000010400, 0x4201008902036000), B(0x1C22500000100020, 0x0004800000810008),
    B(0x1308200880800080, 0x00104080A0092024), B(0x1308200880800080, 0x00104080A0092024),
    B(0xA462006100000008, 0x50140140000000D2), B(0x8231008011002000, 0x40CA0820C00A8010),
    B(0x012A800008080108, 0x1020810002000202), B(0x0520B30201043000, 0x0000000021281060),
    B(0x00B9458428840314, 0x1000142420881020), B(0x88142A8014000005, 0x1808000050040008),
    B(0x520B1482010A0600, 0x00C1101000800080), B(0x0009840100000040, 0x8018809400414400),
    B(0x0009840100000040, 0x8018809400414400), B(0x1202054810000110, 0x80060010001C0000),
    B(0x0101128860010001, 0x2400306000002880), B(0xC800814620442401, 0x2040008400000004),
    B(0x0020C24700003021, 0x010000112012410C), B(0x8000492C20003008, 0x0004004044000800),
    B(0x40405B8421040220, 0x0B1008000000A100), B(0xEA09088430C00000, 0x2040028604200040),
    B(0x00189410A0484020, 0x0005808002000100), B(0x0034558200120011, 0x80000808D4000C08),
    B(0x000224C302000001, 0x0000081384002048), B(0x1881221020000090, 0x2080060220085018),
    B(0x0000462040040000, 0x00204800000010A0), B(0xA1003042625408C4, 0x0040000001804200),
    B(0x08A015A411800421, 0x206808A000B58080), B(0x12202062A4000000, 0x1060841010010000),
    B(0x0700302912810200, 0x9000050641080000), B(0x840A818211020000, 0x4A12040800060000),
    B(0x08200412402A0400, 0x208400A010000001), B(0x0640130864800400, 0x0000048024400010),
    B(0x000C010220920198, 0x0000085000A08000), B(0x0040851022104100, 0x8002088008020000),
    B(0x0004081940090000, 0x004008000000000A), B(0x400000102003C404, 0x4001201280800002),
    B(0x0039915222114200, 0x2001020088D00044), B(0x0810088800A54100, 0x4C00800000000000),
    B(0x01480020C1028000, 0x1041412400048A00), B(0x000200190A241002, 0x1000600400001881),
    B(0x000200190A241002, 0x1000600400001881), B(0x0002000010221020, 0x0000200040000220),
    B(0x00000A0063423444, 0x0041002C15811008), B(0x8408022CD1220272, 0x0080000000008802),
    B(0x4A00004010169A10, 0x0100000800040008), B(0x0000014402494220, 0x1000284880004200),
    B(0x602060C0C1013082, 0x4010204109000200), B(0x8220000000462041, 0x2200010900210882),
    B(0x0002100001124860, 0x0000482060002000), B(0x24000000008010C8, 0x0000240000800080),
    B(0x0088400012084408, 0x200C004809000101), B(0x200000408422442E, 0x2000000208808000),
    B(0x0800300000112A99, 0x0084400100011A80), B(0x2100080081080428, 0x9400800000161402),
    B(0x8140044080142052, 0x84C0402026140104), B(0x108481218460E780, 0x4500008000091C98),
    B(0x0000000400202040, 0x21A00493280C2008), B(0x41460100E2202013, 0x2010104000014000),
    B(0x0080240000021009, 0x10002B0000413480), B(0x040C800010002004, 0x12240404A0010080),
    B(0x4000A01800051004, 0x9240000002008428), B(0x5280001000200813, 0x1514000202081248),
    B(0x4006840000003114, 0x00B4000020010000), B(0x04040801000A2486, 0x4196800008040000),
    B(0x4001020000804101, 0x2A13142028000000), B(0x0880012000000081, 0x454808C800020000),
    B(0x1210000010400008, 0x9089040854800880), B(0xC020310000641048, 0x1108100040010940),
    B(0x0201824042004009, 0x04109089000000C0), B(0x0600046000000281, 0x0201246020420400),
    B(0x04A0000018302002, 0x015308011A001062), B(0x6100000011000502, 0x0150224801012048),
    B(0x0000010000009000, 0x9D409000004B0800), B(0x2008000108044000, 0x8819C00110000000),
    B(0x0800000005043080, 0x1049000040200000), B(0x0100000201200300, 0x11088C283040C000),
    B(0xC0000A0410400090, 0x0844850000801001), B(0x0E28500304880000, 0x042A440012400200),
    B(0x0E28500304880000, 0x042A440012400200), B(0x2002000010082220, 0x05D2090084008001),
    B(0x5000000420004020, 0x02B3044000020900), B(0x1100000A08003808, 0x2056A21800008065),
    B(0x0500520018002900, 0x02A8554221024000), B(0x0200000040981008, 0x0010624080400880),
    B(0x4C21100000503845, 0x48505620010020A8), B(0x0000080128000212, 0x0318930800201481),
    B(0x24800404A0014000, 0x0902510600000022), B(0x0000400010004021, 0x940216C804002002),
    B(0x000011A601010400, 0x200B1CA100000002), B(0x0020108001000020, 0x851A866140000000)
};

constexpr __uint128_t kKnightToMagicNumbers[] = {
    B(0x00376C0000480001, 0x0880010041200001), B(0x0031800000419802, 0x1045004484220000),
    B(0x0419840C00040020, 0x0000480288204041), B(0x0013100000100202, 0x40000004000000A0),
    B(0x2109900100320000, 0x2808002002A00120), B(0x0203240002068000, 0x0001D10108800100),
    B(0x0001964090001018, 0x7002040148001205), B(0x0540CC0040000001, 0x0208902A02886205),
    B(0x0800AF8102030000, 0x0680000002000388), B(0x2C00107044117186, 0x0472208000024020),
    B(0xB000196440104004, 0x5000001080300028), B(0x1090420810000060, 0x8800043010004000),
    B(0x0840010200480040, 0x0801020002608000), B(0x04100088C0802000, 0x0382004108292000),
    B(0x0A11124202400C01, 0x0006948004100020), B(0x0101128860010001, 0x2400306000002880),
    B(0x5102466642401000, 0xA000100020240264), B(0x00A1085280050288, 0x40002810D0000004),
    B(0x00214004A9000100, 0x1400004008002E80), B(0x0009000420008840, 0x4881300000000210),
    B(0x94080C0120000480, 0x2044412202002000), B(0x6448800110400100, 0x0000000000106400),
    B(0x000100008A041200, 0x11101401100C1090), B(0x4280900041000000, 0x60002410002000C0),
    B(0x08C0488120024214, 0x0008414880202291), B(0x0400420008100290, 0x1002041368140101),
    B(0x8001930019090445, 0x040061000C000104), B(0x00103040126E2283, 0x000002008420C3B0),
    B(0x094811601A014100, 0x0800200020504000), B(0x2081022101040000, 0x00203082D8080080),
    B(0x0001C22080422008, 0x0020000400000215), B(0x2002810C01200004, 0xC020800D00800000),
    B(0x0121028100114080, 0x5010280481100082), B(0x2410204448084080, 0x000400A801B04AC0),
    B(0x9242002000049000, 0x110830840040A100), B(0x048010505009C000, 0xA001000012034000),
    B(0x82A4001B1C412000, 0x30110082060A2002), B(0xC008800204391000, 0x0C40000490000320),
    B(0x4802000D14C18021, 0x0000080007860100), B(0x0090000440102220, 0x9304000004200180),
    B(0x2040012381001282, 0x04804080104A4000), B(0x8804008142E90810, 0x060202A081400000),
    B(0x4000005288008460, 0x400A0C4040000000), B(0x0801021809100408, 0x1600046000284400),
    B(0x9000004026400608, 0xC800000422248286), B(0x000008000242012D, 0x1240080242000548),
    B(0x0040006002210044, 0x0600008000408000), B(0x0040006002210044, 0x0600008000408000),
    B(0x0000200008408189, 0x0002000022000020), B(0x0000200008408189, 0x0002000022000020),
    B(0x1208010040122808, 0x4080424482000080), B(0x0C00290018902002, 0x4204100000000000),
    B(0x2002000080082001, 0x1154005000013100), B(0x050682C125125402, 0x6018410002308020),
    B(0x440C400000015800, 0xA9CC00150A080410), B(0x0000440400044C00, 0x2000080040100500),
    B(0x0000440400044C00, 0x2000080040100500), B(0x0070000221504231, 0x8804501401000108),
    B(0x8028000400002000, 0x24C2000001000000), B(0x0802008004009005, 0x0242031008001000),
    B(0x0800000000300482, 0x9203008100100013), B(0x2002001888002442, 0x0084820010000000),
    B(0x0A005400041404C8, 0x0684000202310040), B(0x001001440101010C, 0x0034444800000000),
    B(0x0000030000020041, 0x00211000002C0800), B(0x0800704010804022, 0x4810A00000020000),
    B(0xC00A810014000512, 0x0208402410204220), B(0x2082000900000108, 0x0024500444001400),
    B(0x8040002000409004, 0x8002110011010809), B(0x000C200005410002, 0x1004404404480000),
    B(0x0268502400100021, 0x080A201840802080), B(0xB080420104000502, 0x2901310038803052),
    B(0x0080800000000000, 0x6602A44001811000), B(0xA00004085400000A, 0x41804A020060C540),
    B(0x00852620805C000A, 0xC40A682004014006), B(0x0000840204000026, 0x0800044801090460),
    B(0x0000840204000026, 0x0800044801090460), B(0x0008400140000018, 0x0220891004810800),
    B(0x0008400140000018, 0x0220891004810800), B(0x0000180202001008, 0x0263410200040040),
    B(0x0008044001000000, 0x017600A208008084), B(0x0000010000020010, 0x00EEC00000020220),
    B(0x0182400000081100, 0x0061801124000088), B(0x0108084604040181, 0x4032402000500400),
    B(0x020000040018A404, 0x2126413000200014), B(0x0000010018000004, 0x400C58201000A800),
    B(0x0000102200440200, 0x00C65940000C4000), B(0x9018511020008110, 0x2103130220180000),
    B(0x0000102200440200, 0x00C65940000C4000), B(0x1000002048040400, 0x00015F8040A04004)
};

#undef B
#endif

// Magic parameters
static MagicParams rook_magic_params[90];
static MagicParams cannon_magic_params[90];
static MagicParams bishop_magic_params[90];
static MagicParams knight_magic_params[90];
static MagicParams knight_to_magic_params[90];
// knight to is used to check if getting attacked (knight is asymmetric)
static MagicParams dark_magic_params[90];
static MagicParams dark_to_magic_params[90];
// dark_to is also used to check getting attacked since dark knights are asymmetric

// Precomputed attacks bitboard tables.
static BitBoard rook_attacks_table[0x108000];
static BitBoard cannon_attacks_table[0x108000];
static BitBoard bishop_attacks_table[0x500];
static BitBoard knight_attacks_table[0x380];
static BitBoard knight_to_attacks_table[0x3E0];

static BoardSquare BetweenSQ[90][90];

// RankBB() and FileBB() return a bitboard representing all the squares on the given file or rank.
constexpr BitBoard RankBB(int r) {
    return Rank0BB << (ChessBoard::FILE_NB * r);
}

constexpr BitBoard FileBB(int f) {
    return FileABB << f;
}

static inline int Distance(BoardSquare x, BoardSquare y) {
    return std::max(std::abs(x.row() - y.row()), std::abs(x.col() - y.col()));
}

// safe_destination() returns the bitboard of target square for the given step
// from the given square. If the step is off the board, returns empty bitboard.
inline BitBoard SafeDestination(BoardSquare s, Direction step) {
    BoardSquare to = s + step;
    return to.IsValid() && Distance(s, to) <= 2 ? to.as_board() : BitBoard(0);
}

template <ChessBoard::PieceType pt>
static BitBoard SlidingAttack(BoardSquare sq, BitBoard occupied) {
    assert(pt == ChessBoard::ROOK || pt == ChessBoard::CANNON);
    BitBoard attack = BitBoard(0);

    for (auto const& d : { NORTH, SOUTH, WEST, EAST })
    {
        bool hurdle = false;
        for (BoardSquare s = sq + d; s.IsValid() && Distance(s - d, s) == 1; s += d)
        {
            if (pt == ChessBoard::ROOK || hurdle)
                attack.set(s);

            if (occupied.get(s))
            {
                if (pt == ChessBoard::CANNON && !hurdle)
                    hurdle = true;
                else
                    break;
            }
        }
    }

    return attack;
}

template <ChessBoard::PieceType pt>
BitBoard LameLeaperPath(Direction d, BoardSquare s) {
    BitBoard b = BitBoard(0);
    BoardSquare to = s + d;
    if (!to.IsValid() || Distance(s, to) >= 4)
        return b;

    // If piece type is by knight attacks, swap the source and destination square
    if (pt == ChessBoard::KNIGHT_TO) {
        std::swap(s, to);
        d.first = -d.first;
        d.second = -d.second;
    }

    Direction dr = {d.first > 0 ? 1 : -1, 0};
    Direction df = {0, d.second > 0 ? 1 : -1};

    int diff = std::abs(to.col() - s.col()) - std::abs(to.row() - s.row());
    if (diff > 0)
        s += df;
    else if (diff < 0)
        s += dr;
    else
        s += df, s += dr;

    b.set(s);
    return b;
}

template <ChessBoard::PieceType pt>
BitBoard LameLeaperPath(BoardSquare s) {
    BitBoard b = BitBoard(0);
    for (const auto& d : pt == ChessBoard::BISHOP ? kBishopDirections : kKnightDirections)
        b |= LameLeaperPath<pt>(d, s);
    return b;
}

template <ChessBoard::PieceType pt>
BitBoard LameLeaperAttack(BoardSquare s, BitBoard occupied) {
    BitBoard b = BitBoard(0);
    for (const auto& d : pt == ChessBoard::BISHOP  ? kBishopDirections : kKnightDirections)
    {
        BoardSquare to = s + d;
        if (to.IsValid() && Distance(s, to) < 4 && !((LameLeaperPath<pt>(d, s) & occupied).as_int()))
            b.set(to);
    }
    return b;
}

constexpr BitBoard Shift(Direction D, BitBoard b) {
return  D == NORTH ? (b & ~Rank9BB).as_int() << 9 : D == SOUTH ?  b.as_int()             >> 9
: D == EAST  ? (b & ~FileIBB).as_int() << 1 : D == WEST  ? (b & ~FileABB).as_int() >> 1
: BitBoard(0);
}

// PawnAttacksBB() returns the squares attacked by pawns from the squares in the given bitboard.
const BitBoard PawnAttacksBB(BoardSquare s) {
    BitBoard b = s.as_board();
    BitBoard attack = Shift(NORTH, b);
    if (s.row() > ChessBoard::RANK_4)
        attack |= Shift(WEST, b) | Shift(EAST, b);
    return attack;
}

// PawnAttacksToBB() returns the squares that if there is a pawn, it can attack the square s
template <ChessBoard::PieceType Pt>
const BitBoard PawnAttacksToBB(BoardSquare s) {
    bool ours = Pt == ChessBoard::PAWN_TO_OURS;
    BitBoard b = s.as_board();
    BitBoard attack = Shift(ours ? NORTH : SOUTH, b);
    if ((ours && s.row() < ChessBoard::RANK_5) || (!ours && s.row() > ChessBoard::RANK_4))
        attack |= Shift(WEST, b) | Shift(EAST, b);
    return attack;
}

static inline MagicParams GetMagicParams(const BoardSquare square, const ChessBoard::PieceType Pt) {
    // this utility is used only for generating attack tables of dark pieces
    assert(square.IsValid());

    int s = square.as_int();
    MagicParams ret;

    switch (Pt)
    {
        case ChessBoard::ROOK     : return rook_magic_params[s];
        case ChessBoard::CANNON   : return cannon_magic_params[s];
        case ChessBoard::BISHOP   : return bishop_magic_params[s];
        case ChessBoard::KNIGHT   : return knight_magic_params[s];
        case ChessBoard::KNIGHT_TO: return knight_to_magic_params[s];
        default                   :
            ret.mask_ = 0;
            ret.shift_bits_ = 0;
            ret.attacks_table_ = PseudoAttacks[ChessBoard::kStartposBoard.at(square)]+s;
            return ret;
    }
}

// Builds attacks table.
template <ChessBoard::PieceType pt>
static void BuildAttacksTable(MagicParams* magic_params,
                              BitBoard* attacks_table) {
    // Offset into lookup table.
    uint32_t table_offset = 0;

    // Initialize for all board squares.
    for (unsigned square = 0; square < 90; square++) {
        const BoardSquare b_sq(square);

        if(pt == ChessBoard::DARK || pt == ChessBoard::DARK_TO){
            ChessBoard::PieceType rpt = ChessBoard::kStartposBoard.at(b_sq);
            if(pt == ChessBoard::DARK_TO && rpt == ChessBoard::KNIGHT) rpt = ChessBoard::KNIGHT_TO;
            magic_params[square] = GetMagicParams(square, rpt);
        }else {
            // Board edges are not considered in the relevant occupancies
            BitBoard edges = ((Rank0BB | Rank9BB) - RankBB(b_sq.row())) | ((FileABB | FileIBB) - FileBB(b_sq.col()));

            // Calculate relevant occupancy masks.
            BitBoard mask = pt == ChessBoard::ROOK ? SlidingAttack<pt>(b_sq, BitBoard(0)) :
                            pt == ChessBoard::CANNON ? rook_magic_params[square].mask_ :
                            LameLeaperPath<pt>(square);
            if (pt != ChessBoard::KNIGHT_TO)
                mask -= edges;

            MagicParams &m = magic_params[square];

            // Set mask.
            m.mask_ = mask.as_int();

#if defined(NO_PEXT)
            // Set number of shifted bits. The magic numbers have been chosen such that
        // the number of relevant occupancy bits suffice to index the attacks table.
        m.shift_bits_ = 128 - mask.count();
#else
            // Set number of shifted bits. PEXT shift is the bit count of low 64 bits
            m.shift_bits_ = BitBoard(uint64_t(mask.as_int())).count();
#endif

            // Set pointer to lookup table.
            m.attacks_table_ = &attacks_table[table_offset];

            // Clear attacks table (used for sanity check later on).
            for (int i = 0; i < (1 << mask.count()); i++) {
                m.attacks_table_[i] = BitBoard(0);
            }

            // Build square attacks table for every possible relevant occupancy
            // bitboard.
            __uint128_t b = 0;
            do {
                // Calculate magic index.
                uint64_t index = m.index(b);
                // Calculate attack.
                BitBoard attacks = pt == ChessBoard::ROOK || pt == ChessBoard::CANNON ?
                                   SlidingAttack<pt>(b_sq, b) :
                                   LameLeaperAttack<pt>(b_sq, b);
#if defined(NO_PEXT)
                // Sanity check. The magic numbers have been chosen such that
          // the number of relevant occupancy bits suffice to index the attacks
          // table. If the table already contains an attacks bitboard, possible
          // collisions should be constructive.
          if (m.attacks_table_[index] != 0 && m.attacks_table_[index] != attacks) {
            throw Exception("Invalid magic number!");
          }
#endif
                // Update table.
                m.attacks_table_[index] = attacks;
                b = (b - m.mask_) & m.mask_;
            } while (b);

            // Update table offset.
            table_offset += (1 << mask.count());
        }
    }
}

// Returns the attacks bitboard for the given board square and the given occupied piece bitboard.
template<ChessBoard::PieceType Pt>
static inline BitBoard GetAttacks(const BoardSquare square,
                                  const BitBoard pieces = BitBoard(0)) {
    assert(square.IsValid());

    int s = square.as_int();

    switch (Pt)
    {
        case ChessBoard::ROOK     : return rook_magic_params[s].attacks_table_[rook_magic_params[s].index(pieces)];
        case ChessBoard::CANNON   : return cannon_magic_params[s].attacks_table_[cannon_magic_params[s].index(pieces)];
        case ChessBoard::BISHOP   : return bishop_magic_params[s].attacks_table_[bishop_magic_params[s].index(pieces)];
        case ChessBoard::KNIGHT   : return knight_magic_params[s].attacks_table_[knight_magic_params[s].index(pieces)];
        case ChessBoard::KNIGHT_TO: return knight_to_magic_params[s].attacks_table_[knight_to_magic_params[s].index(pieces)];
        case ChessBoard::DARK     : return dark_magic_params[s].attacks_table_[dark_magic_params[s].index(pieces)];
        case ChessBoard::DARK_TO  : return dark_to_magic_params[s].attacks_table_[dark_to_magic_params[s].index(pieces)];
        default                   : return PseudoAttacks[Pt][s];
    }
}

// Returns the attacks bitboard for the given board square and the given occupied piece bitboard.
static inline BitBoard GetAttacks(ChessBoard::PieceType Pt,
                                  const BoardSquare square,
                                  const BitBoard pieces = BitBoard(0)) {
    assert(square.IsValid());

    int s = square.as_int();

    switch (Pt)
    {
        case ChessBoard::ROOK     : return GetAttacks<ChessBoard::ROOK     >(square, pieces);
        case ChessBoard::CANNON   : return GetAttacks<ChessBoard::CANNON   >(square, pieces);
        case ChessBoard::BISHOP   : return GetAttacks<ChessBoard::BISHOP   >(square, pieces);
        case ChessBoard::KNIGHT   : return GetAttacks<ChessBoard::KNIGHT   >(square, pieces);
        case ChessBoard::KNIGHT_TO: return GetAttacks<ChessBoard::KNIGHT_TO>(square, pieces);
        case ChessBoard::DARK     : return GetAttacks<ChessBoard::DARK     >(square, pieces);
        case ChessBoard::DARK_TO  : return GetAttacks<ChessBoard::DARK_TO  >(square, pieces);
        default                   : return PseudoAttacks[Pt][s];
    }
}

}  // namespace

void InitializeMagicBitboards() {
#if defined(NO_PEXT)
    // Set magic numbers for all board squares.
  for (unsigned square = 0; square < 90; square++) {
    rook_magic_params[square].magic_number_ = kRookMagicNumbers[square];
    cannon_magic_params[square].magic_number_= kRookMagicNumbers[square];
    bishop_magic_params[square].magic_number_ = kBishopMagicNumbers[square];
    knight_magic_params[square].magic_number_ = kKnightMagicNumbers[square];
    knight_to_magic_params[square].magic_number_ = kKnightToMagicNumbers[square];
  }
#endif

    // Build attacks tables.
    BuildAttacksTable<ChessBoard::ROOK>(rook_magic_params, rook_attacks_table);
    BuildAttacksTable<ChessBoard::CANNON>(cannon_magic_params, cannon_attacks_table);
    BuildAttacksTable<ChessBoard::BISHOP>(bishop_magic_params, bishop_attacks_table);
    BuildAttacksTable<ChessBoard::KNIGHT>(knight_magic_params, knight_attacks_table);
    BuildAttacksTable<ChessBoard::KNIGHT_TO>(knight_to_magic_params, knight_to_attacks_table);

    for (unsigned square = 0; square < 90; square++) {
        const BoardSquare b_sq(square);
        PseudoAttacks[ChessBoard::PAWN][square] = PawnAttacksBB(b_sq);
        PseudoAttacks[ChessBoard::PAWN_TO_OURS][square] = PawnAttacksToBB<ChessBoard::PAWN_TO_OURS>(b_sq);
        PseudoAttacks[ChessBoard::PAWN_TO_THEIRS][square] = PawnAttacksToBB<ChessBoard::PAWN_TO_THEIRS>(b_sq);

        // Only generate pseudo attacks in the palace squares for king
        if (Palace & b_sq.as_board()) {
            for (const auto& d : { NORTH, SOUTH, WEST, EAST } )
                PseudoAttacks[ChessBoard::KING][square] |= SafeDestination(b_sq, d);
            PseudoAttacks[ChessBoard::KING][square] &= Palace;
        }

        for (const auto& d : { NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST } )
            PseudoAttacks[ChessBoard::ADVISOR][square] |= SafeDestination(b_sq, d);

        PseudoAttacks[ChessBoard::KNIGHT][square] = LameLeaperAttack<ChessBoard::KNIGHT>(b_sq, BitBoard(0));

        for (unsigned square2 = 0; square2 < 90; square2++)
        {
            const BoardSquare b_sq2(square2);

            if (PseudoAttacks[ChessBoard::KNIGHT][square].intersects(b_sq2.as_board()))
                BetweenSQ[square][square2] = *LameLeaperPath<ChessBoard::KNIGHT_TO>(
                        Direction{b_sq2.row() - b_sq.row(), b_sq2.col() - b_sq.col()}, square).begin();
        }
    }

    BuildAttacksTable<ChessBoard::DARK>(dark_magic_params, nullptr);
    BuildAttacksTable<ChessBoard::DARK_TO>(dark_to_magic_params, nullptr);
    // dark pieces doesn't own their own attack tables
}

MoveList ChessBoard::GeneratePseudolegalMoves() const {
    MoveList result;
    result.reserve(60);
    for (auto source : our_pieces_) {
        // Rook
        if (rooks_.get(source)) {
            for (const auto& destination : GetAttacks<ROOK>(source, our_pieces_ | their_pieces_) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Advisor
        if (advisors_.get(source)) {
            for (const auto& destination : GetAttacks<ADVISOR>(source) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Cannon
        if (cannons_.get(source)) {
            // Non-Capture
            BitBoard attacks = GetAttacks<ROOK>(source, our_pieces_ | their_pieces_) - (our_pieces_ | their_pieces_);
            // Capture
            attacks |= GetAttacks<CANNON>(source, our_pieces_ | their_pieces_) & their_pieces_;
            for (const auto& destination : attacks) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Pawns.
        if (pawns_.get(source)) {
            for (const auto& destination : GetAttacks<PAWN>(source) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Knight
        if (knights_.get(source)) {
            for (const auto& destination : GetAttacks<KNIGHT>(source, our_pieces_ | their_pieces_) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Bishop
        if (bishops_.get(source)) {
            for (const auto& destination : GetAttacks<BISHOP>(source, our_pieces_ | their_pieces_) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // Dark
        if (darks_.get(source)) {
            for (const auto& destination : GetAttacks<DARK>(source, our_pieces_ | their_pieces_) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
        // King
        if (source == our_king_) {
            for (const auto& destination : GetAttacks<KING>(source) - our_pieces_) {
                result.emplace_back(source, destination);
            }
            continue;
        }
    }
    return result;
}  // namespace lczero

template<typename... T>
void ResetSquare(const BoardSquare& s, T&... args) {
    (..., args.reset(s));
}

template<typename... T>
void SetIfSquare(const BoardSquare& from, const BoardSquare& to, T&... args) {
    (..., args.set_if(to, args.get(from)));
}

bool ChessBoard::ApplyMove(Move move) {
    auto from = move.from();
    auto to = move.to();

    // Move in our pieces.
    our_pieces_.reset(from);
    our_pieces_.set(to);

    // reset captured and revealed
    captured_ = 255u;
    revealed_ = 255u;

    // Remove captured piece.
    bool reset_50_moves = their_pieces_.get(to);
    if (reset_50_moves) {
        ResetSquare(to, their_pieces_, rooks_, advisors_, cannons_, pawns_, knights_, bishops_, darks_);
        captured_ = to.as_int();
    }

    // King
    if (from == our_king_) {
        our_king_ = to;
        return reset_50_moves;
    }

    if (darks_.get(from)){
        revealed_ = to.as_int(); // the piece at to after all has to be revealed
        unknowns_.set(from);
        ResetSquare(from, rooks_, advisors_, cannons_, pawns_, knights_, bishops_, darks_);
        // after the move, this dark piece has move type of unknown
    }

    // Ordinary move.
    SetIfSquare(from, to, rooks_, advisors_, cannons_, pawns_, knights_, bishops_, darks_, unknowns_);
    ResetSquare(from, rooks_, advisors_, cannons_, pawns_, knights_, bishops_, darks_, unknowns_);

    // Move id_board
    if (flipped_)
        from.Mirror(), to.Mirror();
    id_board_[to.as_int()] = id_board_[from.as_int()];
    id_board_[from.as_int()] = 0;

    return reset_50_moves;
}

template<bool our>
BitBoard ChessBoard::CheckersTo(const BoardSquare& ksq, const BitBoard &occupied) const {
    BitBoard checkers = BitBoard(0);
    // Rooks.
    checkers |= GetAttacks<ROOK>(ksq, occupied) & rooks_;
    // Cannons.
    checkers |= GetAttacks<CANNON>(ksq, occupied) & cannons_;
    // Pawns.
    checkers |= GetAttacks<our ? PAWN_TO_OURS : PAWN_TO_THEIRS>(ksq) & pawns_;
    // Knights.
    checkers |= GetAttacks<KNIGHT_TO>(ksq, occupied) & knights_;
    // Bishops -- in Jieqi, bishops and advisors are able to capture opponent king
    checkers |= GetAttacks<BISHOP>(ksq, occupied) & bishops_;
    // Advisors
    checkers |= GetAttacks<ADVISOR>(ksq) & advisors_;
    return checkers & (our ? their_pieces_ : our_pieces_);
}

// Explicit template instantiation
template BitBoard ChessBoard::CheckersTo<true>(const BoardSquare& ksq, const BitBoard& occupied) const;
template BitBoard ChessBoard::CheckersTo<false>(const BoardSquare& ksq, const BitBoard& occupied) const;

BitBoard ChessBoard::RecapturesTo(const BoardSquare &sq) const {
    BitBoard attackers = BitBoard(0);
    BitBoard occupied = our_pieces_ | their_pieces_;
    // Rooks.
    attackers |= GetAttacks<ROOK>(sq, occupied) & rooks_;
    // Advisors.
    attackers |= GetAttacks<ADVISOR>(sq) & advisors_;
    // Cannons.
    attackers |= GetAttacks<CANNON>(sq, occupied) & cannons_;
    // Pawns.
    attackers |= GetAttacks<PAWN_TO_OURS>(sq) & pawns_;
    // Knights.
    attackers |= GetAttacks<KNIGHT_TO>(sq, occupied) & knights_;
    // Bishop
    attackers |= GetAttacks<BISHOP>(sq, occupied) & bishops_;
    // King
    attackers |= GetAttacks<KING>(sq, occupied) & their_king_.as_board();
    return attackers & their_pieces_;
}

bool ChessBoard::IsSameMove(Move move1, Move move2) const {
    return move1 == move2;
}

template<bool our>
bool ChessBoard::IsLegalMove(Move move) const {
    // Occupied
    BitBoard occupied = our_pieces_ | their_pieces_;
    occupied.reset(move.from());
    occupied.set(move.to());

    BoardSquare our_king = our_king_;
    BoardSquare their_king = their_king_;
    if (!our)
        std::swap(our_king, their_king);

    // Flying general
    BoardSquare ksq = our_king == move.from() ? move.to() : our_king;
    if (GetAttacks<ROOK>(ksq, occupied).get(their_king))
        return false;

    // If the moving piece is a king, check whether the destination square
    // is not under attack after the move.
    if (ksq != our_king)
        return !CheckersTo<our>(ksq, occupied).as_int();

    // A non-king move is legal if the king is not under attack after the move.
    BitBoard checkers = CheckersTo<our>(ksq, occupied);
    checkers.reset(move.to());
    return !checkers.as_int();
}

int ChessBoard::MakeChase(BoardSquare to) const {
    if (flipped_)
        to.Mirror();
    return 1 << id_board_[to.as_int()];
}

uint16_t ChessBoard::UsChased() const {
    uint16_t chase = 0;

    // Add chase information for a type of attacker
    auto addChase = [&] (PieceType attackerType, const BitBoard& attacker) {
        for (const auto& from : attacker & our_pieces_) {
            BitBoard attacks = GetAttacks(attackerType, from, our_pieces_ | their_pieces_) & their_pieces_;

            // Exclude attacks on unpromoted pawns and checks
            attacks -= kings() | (pawns_ & HalfBB[1]);

            // Attacks against stronger pieces
            BitBoard candidates = BitBoard(0);
            if (attackerType == KNIGHT || attackerType == CANNON)
                candidates = attacks & rooks_;
            if (attackerType == ADVISOR || attackerType == BISHOP)
                candidates = attacks & (rooks_ | knights_ | cannons_);
            attacks -= candidates;
            for (const auto & to : candidates) {
                if (IsLegalMove(Move(from, to)))
                    chase |= MakeChase(to);
            }

            // Attacks against potentially unprotected pieces
            for (const auto & to : attacks) {
                Move m = Move(from, to);

                if (IsLegalMove(m))
                {
                    bool trueChase = true;
                    ChessBoard after(*this);
                    after.ApplyMove(m);
                    BitBoard recaptures = after.RecapturesTo(to);
                    for (const auto& s : recaptures) {
                        if (after.IsLegalMove<false>(Move(s, to))) {
                            trueChase = false;
                            break;
                        }
                    }

                    if (trueChase) {
                        // Exclude mutual/symmetric attacks except pins
                        if (attacker.get(to)) {
                            if (   (attackerType == KNIGHT && !(GetAttacks<KNIGHT>(to, our_pieces_ | their_pieces_).get(from)))
                                   || !IsLegalMove<false>(Move(to, from)))
                                chase |= MakeChase(to);
                        }
                        else
                            chase |= MakeChase(to);
                    }
                }
            }
        }
    };

    // King and pawn can legally perpetual chase
    addChase(ROOK, rooks_);
    addChase(ADVISOR, advisors_);
    addChase(CANNON, cannons_);
    addChase(KNIGHT, knights_);
    addChase(BISHOP, bishops_);

    return chase;
}

uint16_t ChessBoard::ThemChased() const {
    auto board = *this;
    board.Mirror();
    return board.UsChased();
}

MoveList ChessBoard::GenerateLegalMoves() const {
    MoveList result = GeneratePseudolegalMoves();
    result.erase(
            std::remove_if(result.begin(), result.end(),
                           [&](Move m) { return !IsLegalMove(m); }),
            result.end());
    return result;
}

std::vector<uint8_t> ChessBoard::GetMoveTraits(const lczero::MoveList &ml) const {
    std::vector<uint8_t> result;
    for(Move t: ml){
        uint8_t traits = 0;
        if(darks_.get(t.from())) traits |= 1; // is a reveal
        if(their_pieces_.get(t.to())) traits |= 2; // is a capture
        if(darks_.get(t.to())) traits |= 4; // captures a dark piece
        if(their_king_ == t.to()) traits |= 8; // if the move is a mate
        result.push_back(traits);
    }
    return result;
}

void ChessBoard::SetFromFen(std::string fen, int* rule50_ply, int* moves) {
    Clear();
    int row = 9;
    int col = 0;

    // Remove any trailing whitespaces to detect eof after the last field.
    fen.erase(std::find_if(fen.rbegin(), fen.rend(),
                           [](char c) { return !std::isspace(c); })
                      .base(),
              fen.end());

    std::istringstream fen_str(fen);
    std::string board;
    fen_str >> board;
    std::string who_to_move = "w";
    if (!fen_str.eof()) fen_str >> who_to_move;
    std::string trash;
    if (!fen_str.eof()) fen_str >> trash;
    if (!fen_str.eof()) fen_str >> trash;
    int rule50_halfmoves = 0;
    if (!fen_str.eof()) fen_str >> rule50_halfmoves;
    int total_moves = 1;
    if (!fen_str.eof()) fen_str >> total_moves;
    if (!fen_str) throw Exception("Bad fen string: " + fen);

    for (char c : board) {
        if (c == '/') {
            --row;
            if (row < 0) throw Exception("Bad fen string (too many rows): " + fen);
            col = 0;
            continue;
        }
        if (std::isdigit(c)) {
            col += c - '0';
            continue;
        }
        if (col >= 9) throw Exception("Bad fen string (too many columns): " + fen);

        if (std::isupper(c)) {
            // White piece.
            our_pieces_.set(row, col);
        } else {
            // Black piece.
            their_pieces_.set(row, col);
        }

        if (c == 'H' || c == 'h') {
            darks_.set(row, col);
            c = initBoard[BoardSquare(row, col).as_int()];
            // dark piece = dark + piece type
        }

        if (c == 'K') {
            our_king_.set(row, col);
            if (BitBoard(our_king_.as_board() & Palace).count_few() == 0)
                throw Exception("Bad fen string: (king not in palace) " + fen);
        } else if (c == 'k') {
            their_king_.set(row, col);
            if (BitBoard(their_king_.as_board() & Palace).count_few() == 0)
                throw Exception("Bad fen string: (king not in palace) " + fen);
        } else if (c == 'R' || c == 'r') {
            rooks_.set(row, col);
        } else if (c == 'A' || c == 'a') {
            advisors_.set(row, col);
        } else if (c == 'C' || c == 'c') {
            cannons_.set(row, col);
        } else if (c == 'P' || c == 'p') {
            pawns_.set(row, col);
        } else if (c == 'N' || c == 'n') {
            knights_.set(row, col);
        } else if (c == 'B' || c == 'b') {
            bishops_.set(row, col);
        } else if (c == 'U' || c == 'u') {
            unknowns_.set(row, col);
        } else {
            throw Exception("Bad fen string: " + fen);
        }
        ++col;
    }

    // Setup id_board
    uint8_t our = 0;
    uint8_t their = 0;
    for (const auto& sq : our_pieces_ | their_pieces_) {
        id_board_[sq.as_int()] = our_pieces_.get(sq) ? our++ : their++;
    }

    if (who_to_move == "b" || who_to_move == "B") {
        Mirror();
    } else if (who_to_move != "w" && who_to_move != "W") {
        throw Exception("Bad fen string (side to move): " + fen);
    }
    if (rule50_ply) *rule50_ply = rule50_halfmoves;
    if (moves) *moves = total_moves * 2 - (flipped_ ? 1 : 2);
}

bool ChessBoard::HasMatingMaterial() const {
    if (pawns_.count() == 0 && rooks_.count_few() == 0 && knights_.count_few() == 0) {

        enum DrawLevel : int {
            NO_DRAW,      // There is no drawing situation exists
            DIRECT_DRAW,  // A draw can be directly yielded without any checks
            MATE_DRAW     // We need to check for mate before yielding a draw
        };

        DrawLevel level = [&] {
            // No cannons left on the board
            if (cannons_.count_few() == 0) {
                return DIRECT_DRAW;
            }

            // One cannon left on the board
            if (cannons_.count_few() == 1) {
                // See which side is holding this cannon, and this side must not possess
                // any advisors
                BitBoard cannon_side_occ = our_pieces_;
                BitBoard non_cannon_side_occ = their_pieces_;
                if ((cannon_side_occ & cannons_).count_few() == 0) {
                    std::swap(cannon_side_occ, non_cannon_side_occ);
                }
                if ((advisors_ & cannon_side_occ).count_few() == 0) {
                    // No advisors left on the board
                    if ((advisors_ & non_cannon_side_occ).count_few() == 0) {
                        return DIRECT_DRAW;
                    }

                    // One advisor left on the board
                    if ((advisors_ & non_cannon_side_occ).count_few() == 1) {
                        return (bishops_ & cannon_side_occ).count_few() == 0 ? DIRECT_DRAW
                                                                             : MATE_DRAW;
                    }

                    // Two advisors left on the board
                    if ((bishops_ & cannon_side_occ).count_few() == 0) {
                        return MATE_DRAW;
                    }
                }
            }

            // Two cannons left on the board, one for each side, and no advisors left
            // on the board
            if ((cannons_ & our_pieces_).count_few() == 1 &&
                (cannons_ & their_pieces_).count_few() == 1 &&
                advisors_.count_few() == 0) {
                return bishops_.count_few() == 0 ? DIRECT_DRAW : MATE_DRAW;
            }

            return NO_DRAW;
        }();

        if (level != NO_DRAW) {
            if (level == MATE_DRAW) {
                for (const auto& move : GenerateLegalMoves()) {
                    ChessBoard after(*this);
                    after.ApplyMove(move);
                    after.Mirror();
                    if (after.GenerateLegalMoves().size() == 0)
                        return true;
                }
            }
            return false;
        }
    }

    return true;
}

std::string ChessBoard::DebugString() const {
    std::string result;
    for (int i = 9; i >= 0; --i) {
        for (int j = 0; j < 9; ++j) {
            if (!our_pieces_.get(i, j) && !their_pieces_.get(i, j)) {
                result += '.';
                continue;
            }
            if (our_king_ == i * 9 + j) {
                result += 'K';
                continue;
            }
            if (their_king_ == i * 9 + j) {
                result += 'k';
                continue;
            }
            char c = '?';
            if (darks_.get(i, j)) {
                c = 'h';
            }else if (rooks_.get(i, j)) {
                c = 'r';
            } else if (advisors_.get(i, j)) {
                c = 'a';
            } else if (cannons_.get(i, j)) {
                c = 'c';
            } else if (pawns_.get(i, j)) {
                c = 'p';
            } else if (knights_.get(i, j)) {
                c = 'n';
            } else if (bishops_.get(i, j)) {
                c = 'b';
            }
            if (our_pieces_.get(i, j)) c = std::toupper(c);
            result += c;
        }
        if (i == 0) {
            result += flipped_ ? " (from black's eyes)" : " (from white's eyes)";
            result += " Hash: " + std::to_string(Hash());
        }
        result += '\n';
    }
    return result;
}

}  // namespace lczero
