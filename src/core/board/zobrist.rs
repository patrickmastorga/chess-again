use super::{Color, PieceType};

pub struct Zobrist {
    pieces: [u64; 12 * 64],
    en_passant: [u64; 8],
    castling: [u64; 4],
    color: u64,
}

impl Zobrist {
    #[inline]
    pub fn piece_square(&self, piece_type: PieceType, color: Color, square: usize) -> u64 {
        let piece = (color as usize) * 6 + (piece_type as usize);
        self.pieces[(piece << 6) | square]
    }

    #[inline]
    pub fn color(&self) -> u64 {
        self.color
    }

    #[inline]
    pub fn kingside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize]
    }

    #[inline]
    pub fn queenside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize + 2]
    }

    #[inline]
    pub fn en_passant(&self, file: usize) -> u64 {
        self.en_passant[file]
    }
}

// define static zobrist struct containing constants generated at build time
include!(concat!(env!("OUT_DIR"), "/zobrist.rs"));
