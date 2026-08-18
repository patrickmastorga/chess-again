use std::iter::FusedIterator;

/// Convenience helpers for the `u64` bitboards used throughout the engine.
///
/// Squares are little-endian rank-and-file indexed, i.e. bit `n` corresponds
/// to square `n` where `square = rank * 8 + file`.
pub trait BitboardExt: Copy + Eq {
    /// The empty bitboard.
    const EMPTY: u64 = 0;
    /// The full 64-square bitboard.
    const ALL: u64 = u64::MAX;

    /// Returns whether the bitboard contains no set bits.
    fn is_empty(self) -> bool;

    /// Returns whether the bitboard contains at least one set bit.
    fn is_non_empty(self) -> bool {
        !self.is_empty()
    }

    /// The total number of set bits (population count).
    fn count(self) -> usize;

    /// Index of the least-significant set bit. Returns `None` if empty.
    fn first_index(self) -> Option<usize>;

    /// Returns the bitboard containing only the least-significant set bit,
    /// or the empty bitboard if `self` is empty.
    fn first(self) -> Self;

    /// Removes the least-significant set bit, returning the resulting bitboard.
    fn pop_lsb(self) -> Self;

    /// Iterates over the index of each set bit, ascending.
    fn iter_indices(self) -> BitIndexIter;

    /// Iterates over each set bit as a single-bit bitboard, ascending.
    fn iter_bits(self) -> BitIter;
}

impl BitboardExt for u64 {
    fn is_empty(self) -> bool {
        self == 0
    }

    fn count(self) -> usize {
        self.count_ones() as usize
    }

    fn first_index(self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.trailing_zeros() as usize)
        }
    }

    fn first(self) -> Self {
        self & self.wrapping_neg()
    }

    fn pop_lsb(self) -> Self {
        self & self.wrapping_sub(1)
    }

    fn iter_indices(self) -> BitIndexIter {
        BitIndexIter { bits: self }
    }

    fn iter_bits(self) -> BitIter {
        BitIter(self.iter_indices())
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
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next().map(|index| 1u64 << index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for BitIter {}
impl FusedIterator for BitIter {}
