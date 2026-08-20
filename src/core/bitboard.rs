use std::iter::FusedIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// A 64-bit bitboard.
///
/// Squares are little-endian rank-and-file indexed, i.e. bit `n` corresponds
/// to square `n` where `square = rank * 8 + file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(u64::MAX);

    pub const fn new(bits: u64) -> Self {
        Bitboard(bits)
    }

    /// Returns the bitboard containing only the given square.
    pub const fn from_square(square: usize) -> Self {
        Bitboard(1u64 << square)
    }

    /// Returns the raw `u64` value backing this bitboard.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Returns whether the bitboard is empty (contains no set bits).
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of set bits (population count).
    pub const fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Returns whether the bitboard contains the given square.
    pub const fn contains(self, square: usize) -> bool {
        (self.0 & (1u64 << square)) != 0
    }

    /// Iterates over the index of each set bit, ascending.
    pub fn iter_indices(self) -> BitIndexIter {
        BitIndexIter { bits: self.0 }
    }

    /// Iterates over each set bit as a single-bit bitboard, ascending.
    pub fn iter_bits(self) -> BitIter {
        BitIter { bits: self.0 }
    }
}

impl From<u64> for Bitboard {
    fn from(bits: u64) -> Self {
        Bitboard(bits)
    }
}

impl From<Bitboard> for u64 {
    fn from(bb: Bitboard) -> Self {
        bb.0
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;

    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;

    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;

    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;

    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

/// Stockfish-style display of a bitboard.
///
/// Set squares are shown as `X` and empty squares as spaces.
impl std::fmt::Display for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitboard: {:064b}\n", self.0)?;
        for rank in (0..8).rev() {
            writeln!(f, " +---+---+---+---+---+---+---+---+")?;
            for file in 0..8 {
                let square = rank * 8 + file;
                let marker = if self.contains(square) { 'X' } else { ' ' };
                write!(f, " | {}", marker)?;
            }
            writeln!(f, " |")?;
        }
        write!(f, " +---+---+---+---+---+---+---+---+\n")
    }
}

/// Iterator over the indices of each set bit in a bitboard, ascending.
#[derive(Debug, Clone)]
pub struct BitIndexIter {
    bits: u64,
}

impl Iterator for BitIndexIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let index = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1;
        Some(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bits.count_ones() as usize;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for BitIndexIter {
    fn next_back(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let index = 63 - self.bits.leading_zeros() as usize;
        self.bits &= !(1u64 << index);
        Some(index)
    }
}

impl ExactSizeIterator for BitIndexIter {}
impl FusedIterator for BitIndexIter {}

/// Iterator over each set bit as a single-bit bitboard, ascending.
#[derive(Debug, Clone)]
pub struct BitIter {
    bits: u64,
}

impl Iterator for BitIter {
    type Item = Bitboard;

    fn next(&mut self) -> Option<Bitboard> {
        if self.bits == 0 {
            return None;
        }
        let lsb = self.bits & self.bits.wrapping_neg();
        self.bits &= self.bits - 1;
        Some(Bitboard(lsb))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bits.count_ones() as usize;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for BitIter {
    fn next_back(&mut self) -> Option<Bitboard> {
        if self.bits == 0 {
            return None;
        }
        let msb = 1u64 << (63 - self.bits.leading_zeros());
        self.bits &= !msb;
        Some(Bitboard(msb))
    }
}

impl ExactSizeIterator for BitIter {}
impl FusedIterator for BitIter {}

pub struct BitboardTables {
    pub north_ray: [Bitboard; 64],
    pub south_ray: [Bitboard; 64],
    pub east_ray: [Bitboard; 64],
    pub west_ray: [Bitboard; 64],
    pub northeast_ray: [Bitboard; 64],
    pub northwest_ray: [Bitboard; 64],
    pub southeast_ray: [Bitboard; 64],
    pub southwest_ray: [Bitboard; 64],
    pub knight_attacks: [Bitboard; 64],
    pub king_attacks: [Bitboard; 64],
    pub white_pawn_attacks: [Bitboard; 64],
    pub black_pawn_attacks: [Bitboard; 64],
    pub white_pawn_moves: [Bitboard; 64],
    pub black_pawn_moves: [Bitboard; 64],
}

const fn generate_bitboard_tables() -> BitboardTables {
    let mut north_ray = [Bitboard::EMPTY; 64];
    let mut south_ray = [Bitboard::EMPTY; 64];
    let mut east_ray = [Bitboard::EMPTY; 64];
    let mut west_ray = [Bitboard::EMPTY; 64];
    let mut northeast_ray = [Bitboard::EMPTY; 64];
    let mut northwest_ray = [Bitboard::EMPTY; 64];
    let mut southeast_ray = [Bitboard::EMPTY; 64];
    let mut southwest_ray = [Bitboard::EMPTY; 64];
    let mut knight_attacks = [Bitboard::EMPTY; 64];
    let mut king_attacks = [Bitboard::EMPTY; 64];
    let mut white_pawn_attacks = [Bitboard::EMPTY; 64];
    let mut black_pawn_attacks = [Bitboard::EMPTY; 64];
    let mut white_pawn_moves = [Bitboard::EMPTY; 64];
    let mut black_pawn_moves = [Bitboard::EMPTY; 64];

    let mut square = 0;
    while square < 64 {
        let rank = square / 8;
        let file = square % 8;

        // Sliding rays.  The source square is deliberately not included.
        let mut distance = 1;
        while rank + distance < 8 {
            north_ray[square] =
                Bitboard(north_ray[square].0 | Bitboard::from_square(square + distance * 8).0);
            distance += 1;
        }

        distance = 1;
        while distance <= rank {
            south_ray[square] =
                Bitboard(south_ray[square].0 | Bitboard::from_square(square - distance * 8).0);
            distance += 1;
        }

        distance = 1;
        while file + distance < 8 {
            east_ray[square] =
                Bitboard(east_ray[square].0 | Bitboard::from_square(square + distance).0);
            distance += 1;
        }

        distance = 1;
        while distance <= file {
            west_ray[square] =
                Bitboard(west_ray[square].0 | Bitboard::from_square(square - distance).0);
            distance += 1;
        }

        distance = 1;
        while rank + distance < 8 && file + distance < 8 {
            northeast_ray[square] =
                Bitboard(northeast_ray[square].0 | Bitboard::from_square(square + distance * 9).0);
            distance += 1;
        }

        distance = 1;
        while rank + distance < 8 && distance <= file {
            northwest_ray[square] =
                Bitboard(northwest_ray[square].0 | Bitboard::from_square(square + distance * 7).0);
            distance += 1;
        }

        distance = 1;
        while distance <= rank && file + distance < 8 {
            southeast_ray[square] =
                Bitboard(southeast_ray[square].0 | Bitboard::from_square(square - distance * 7).0);
            distance += 1;
        }

        distance = 1;
        while distance <= rank && distance <= file {
            southwest_ray[square] =
                Bitboard(southwest_ray[square].0 | Bitboard::from_square(square - distance * 9).0);
            distance += 1;
        }

        // Non-sliding attacks.
        let knight_offsets = [
            (1isize, 2isize),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        let mut offset = 0;
        while offset < knight_offsets.len() {
            let target_file = file as isize + knight_offsets[offset].0;
            let target_rank = rank as isize + knight_offsets[offset].1;
            if target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8 {
                knight_attacks[square] = Bitboard(
                    knight_attacks[square].0
                        | Bitboard::from_square((target_rank * 8 + target_file) as usize).0,
                );
            }
            offset += 1;
        }

        let mut target_rank = rank.saturating_sub(1);
        while target_rank <= rank + 1 && target_rank < 8 {
            let mut target_file = file.saturating_sub(1);
            while target_file <= file + 1 && target_file < 8 {
                if target_rank != rank || target_file != file {
                    king_attacks[square] = Bitboard(
                        king_attacks[square].0
                            | Bitboard::from_square(target_rank * 8 + target_file).0,
                    );
                }
                target_file += 1;
            }
            target_rank += 1;
        }

        // White pawns move towards increasing ranks.
        if rank < 7 {
            white_pawn_moves[square] =
                Bitboard(white_pawn_moves[square].0 | Bitboard::from_square(square + 8).0);
            if rank == 1 {
                white_pawn_moves[square] =
                    Bitboard(white_pawn_moves[square].0 | Bitboard::from_square(square + 16).0);
            }
            if file > 0 {
                white_pawn_attacks[square] =
                    Bitboard(white_pawn_attacks[square].0 | Bitboard::from_square(square + 7).0);
            }
            if file < 7 {
                white_pawn_attacks[square] =
                    Bitboard(white_pawn_attacks[square].0 | Bitboard::from_square(square + 9).0);
            }
        }

        // Black pawns move towards decreasing ranks.
        if rank > 0 {
            black_pawn_moves[square] =
                Bitboard(black_pawn_moves[square].0 | Bitboard::from_square(square - 8).0);
            if rank == 6 {
                black_pawn_moves[square] =
                    Bitboard(black_pawn_moves[square].0 | Bitboard::from_square(square - 16).0);
            }
            if file > 0 {
                black_pawn_attacks[square] =
                    Bitboard(black_pawn_attacks[square].0 | Bitboard::from_square(square - 9).0);
            }
            if file < 7 {
                black_pawn_attacks[square] =
                    Bitboard(black_pawn_attacks[square].0 | Bitboard::from_square(square - 7).0);
            }
        }

        square += 1;
    }

    BitboardTables {
        north_ray,
        south_ray,
        east_ray,
        west_ray,
        northeast_ray,
        northwest_ray,
        southeast_ray,
        southwest_ray,
        knight_attacks,
        king_attacks,
        white_pawn_attacks,
        black_pawn_attacks,
        white_pawn_moves,
        black_pawn_moves,
    }
}

pub static BITBOARDS: BitboardTables = generate_bitboard_tables();
