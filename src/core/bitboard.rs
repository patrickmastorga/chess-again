use std::fmt::Display;
use std::iter::FusedIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// A 64-bit bitboard.
///
/// Squares are little-endian rank-and-file indexed, i.e. bit `n` corresponds
/// to square `n` where `square = rank * 8 + file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(u64::MAX);

    /// Returns the bitboard containing only the given square.
    pub const fn from_square(square: usize) -> Self {
        Bitboard(1u64 << square)
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

    /// Returns the index of the least significant set bit, or `None` if empty.
    pub const fn lsb_index(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }

    /// Returns the index of the most significant set bit, or `None` if empty.
    pub const fn msb_index(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(63 - self.0.leading_zeros() as usize)
        }
    }

    /// Sets the bit corresponding to the given square.
    pub const fn set(&mut self, square: usize) {
        self.0 |= 1u64 << square;
    }

    /// Clears the bit corresponding to the given square.
    pub const fn clear(&mut self, square: usize) {
        self.0 &= !(1u64 << square);
    }

    pub const fn with_set(&self, square: usize) -> Self {
        Bitboard(self.0 | (1u64 << square))
    }

    pub const fn with_clear(&self, square: usize) -> Self {
        Bitboard(self.0 & !(1u64 << square))
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
impl Display for Bitboard {
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
