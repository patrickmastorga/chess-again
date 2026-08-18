use super::{PieceType, Color};

pub struct Zobrist {
    pieces: [u64; 12 * 64],
    en_passant: [u64; 8],
    castling: [u64; 4],
    color: u64,
}

impl Zobrist {
    pub fn piece_square(&self, piece_type: PieceType, color: Color, square: usize) -> u64 {
        let piece = (color as usize * 6) + piece_type as usize;
        self.pieces[(piece << 6) | square]
    }

    pub fn color(&self) -> u64 {
        self.color
    }

    pub fn kingside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize]
    }

    pub fn queenside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize + 2]
    }

    pub fn en_passant(&self, file: usize) -> u64 {
        self.en_passant[file]
    }
}

// read the zobrist hashing constants generated at build time
include!(concat!(env!("OUT_DIR"), "/zobrist.rs"));