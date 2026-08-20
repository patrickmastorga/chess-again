use super::{Color, PieceType};

pub struct Zobrist {
    pieces: [u64; 12 * 64],
    en_passant: [u64; 64],
    castling: [u64; 4],
    color: u64,
}

impl Zobrist {
    const fn generate(seed: u64) -> Self {
        let mut state = seed;
        let mut pieces = [0u64; 12 * 64];
        let mut en_passant = [0u64; 64];
        let mut castling = [0u64; 4];
        let color = splitmix64_next(&mut state);

        let mut i: usize = 0;
        while i < 12 * 64 {
            pieces[i] = splitmix64_next(&mut state);
            i += 1;
        }
        i = 0;
        while i < 64 {
            en_passant[i] = splitmix64_next(&mut state);
            i += 1;
        }
        i = 0;
        while i < 4 {
            castling[i] = splitmix64_next(&mut state);
            i += 1;
        }
        Self {
            pieces,
            en_passant,
            castling,
            color,
        }
    }

    pub const fn piece_square(&self, piece_type: PieceType, color: Color, square: usize) -> u64 {
        let piece = (color as usize) * 6 + (piece_type as usize);
        self.pieces[(piece << 6) | square]
    }

    pub const fn color(&self) -> u64 {
        self.color
    }

    pub const fn kingside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize]
    }

    pub const fn queenside_castling(&self, color: Color) -> u64 {
        self.castling[color as usize + 2]
    }

    pub const fn en_passant(&self, square: usize) -> u64 {
        self.en_passant[square]
    }
}

const fn splitmix64_next(state: &mut u64) -> u64 {
    // increment the state
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

    // apply MurmurHash3 mixer to state
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    z ^ (z >> 31)
}

pub static ZOBRIST: Zobrist = Zobrist::generate(0x9E3779B97F4A7C15u64);
