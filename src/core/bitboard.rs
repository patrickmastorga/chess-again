use std::iter::FusedIterator;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr,
    ShrAssign,
};

/// A 64-bit bitboard.
///
/// Squares are little-endian rank-and-file indexed, i.e. bit `n` corresponds
/// to square `n` where `square = rank * 8 + file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    /// The empty bitboard.
    pub const EMPTY: Bitboard = Bitboard(0);
    /// The full 64-square bitboard.
    pub const ALL: Bitboard = Bitboard(u64::MAX);

    /// Wraps a raw `u64` value as a bitboard.
    #[inline]
    pub const fn new(bits: u64) -> Self {
        Bitboard(bits)
    }

    /// Returns the bitboard containing only the given square.
    #[inline]
    pub const fn from_square(square: usize) -> Self {
        Bitboard(1u64 << square)
    }

    /// Returns the raw `u64` value backing this bitboard.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Returns whether the bitboard contains no set bits.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether the bitboard contains at least one set bit.
    #[inline]
    pub const fn is_non_empty(self) -> bool {
        !self.is_empty()
    }

    /// The total number of set bits (population count).
    #[inline]
    pub const fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Index of the least-significant set bit. Returns `None` if empty.
    #[inline]
    pub const fn first_index(self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }

    /// Returns the bitboard containing only the least-significant set bit,
    /// or the empty bitboard if `self` is empty.
    #[inline]
    pub const fn first(self) -> Self {
        Bitboard(self.0 & self.0.wrapping_neg())
    }

    /// Removes the least-significant set bit, returning the resulting bitboard.
    #[inline]
    pub const fn pop_lsb(self) -> Self {
        Bitboard(self.0 & self.0.wrapping_sub(1))
    }

    /// Iterates over the index of each set bit, ascending.
    #[inline]
    pub fn iter_indices(self) -> BitIndexIter {
        BitIndexIter { bits: self.0 }
    }

    /// Iterates over each set bit as a single-bit bitboard, ascending.
    #[inline]
    pub fn iter_bits(self) -> BitIter {
        BitIter(self.iter_indices())
    }
}

impl From<u64> for Bitboard {
    #[inline]
    fn from(bits: u64) -> Self {
        Bitboard(bits)
    }
}

impl From<Bitboard> for u64 {
    #[inline]
    fn from(bb: Bitboard) -> Self {
        bb.0
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Shl<usize> for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn shl(self, rhs: usize) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl ShlAssign<usize> for Bitboard {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        self.0 <<= rhs;
    }
}

impl Shr<usize> for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn shr(self, rhs: usize) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl ShrAssign<usize> for Bitboard {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
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
        let n = self.bits.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitIndexIter {}
impl FusedIterator for BitIndexIter {}

/// Iterator over each set bit as a single-bit bitboard, ascending.
#[derive(Debug, Clone)]
pub struct BitIter(BitIndexIter);

impl Iterator for BitIter {
    type Item = Bitboard;

    fn next(&mut self) -> Option<Bitboard> {
        self.0.next().map(Bitboard::from_square)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for BitIter {}
impl FusedIterator for BitIter {}
