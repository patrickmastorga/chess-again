mod movegen;
mod zobrist;

use crate::core::bitboard::Bitboard;
use crate::core::utils;
pub use movegen::Move;
use zobrist::ZOBRIST;

/// The two chess colors, with values matching array indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White = 0,
    Black = 1,
}

/// The six piece types, with values matching array indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    colors: [Bitboard; 2], // indexed by Color
    pieces: [Bitboard; 6], // indexed by PieceType
    zobrist: u64,
    active_color: Color,
    en_passant_file: Option<u8>,
    castling_rights: u8,
}

impl Board {
    pub fn new() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut board = Self {
            colors: [Bitboard::EMPTY; 2],
            pieces: [Bitboard::EMPTY; 6],
            zobrist: 0,
            active_color: Color::White,
            en_passant_file: None,
            castling_rights: 0,
        };

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err("Invalid FEN: must have 6 space-separated fields".to_string());
        }

        // piece placement
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("Invalid FEN: piece placement data must have 8 ranks".to_string());
        }
        for (i, rank) in ranks.iter().enumerate() {
            let mut file_count = 0;
            for c in rank.chars() {
                if c.is_digit(10) {
                    file_count += c.to_digit(10).unwrap();
                } else if "prnbqkPRNBQK".contains(c) {
                    let color = if c.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let piece_type = match c {
                        'p' | 'P' => PieceType::Pawn,
                        'n' | 'N' => PieceType::Knight,
                        'b' | 'B' => PieceType::Bishop,
                        'r' | 'R' => PieceType::Rook,
                        'q' | 'Q' => PieceType::Queen,
                        'k' | 'K' => PieceType::King,
                        _ => unreachable!(),
                    };
                    let square = (7 - i) * 8 + (file_count as usize);
                    board.put_piece(piece_type, color, square);
                    file_count += 1;
                } else {
                    return Err(format!(
                        "Invalid FEN: invalid character '{}' in piece placement data",
                        c
                    ));
                }
            }
            if file_count != 8 {
                return Err(format!(
                    "Invalid FEN: rank {} must have exactly 8 squares",
                    8 - i
                ));
            }
        }

        // active color
        let active_color = parts[1];
        if active_color == "b" {
            board.flip_color();
        } else if active_color != "w" {
            return Err("Invalid FEN: active color must be 'w' or 'b'".to_string());
        }

        // castling rights
        let castling_rights = parts[2];
        for c in castling_rights.chars() {
            match c {
                'K' => board.put_kingside_castling(Color::White),
                'Q' => board.put_queenside_castling(Color::White),
                'k' => board.put_kingside_castling(Color::Black),
                'q' => board.put_queenside_castling(Color::Black),
                '-' => {}
                _ => return Err(format!("Invalid FEN: invalid castling right '{}'", c)),
            }
        }

        // en passant
        let en_passant = parts[3];
        if en_passant != "-" {
            let Ok(square) = utils::parse_algebraic(en_passant) else {
                return Err(format!(
                    "Invalid FEN: invalid en passant square '{}'",
                    en_passant
                ));
            };
            board.put_en_passant(square & 0b111);
        }

        // halfmove and fullmove clocks ignored by board type
        Ok(board)
    }

    pub fn piece_at(&self, square: usize) -> Option<(PieceType, Color)> {
        let bit = Bitboard::from_square(square);
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            if self.pieces[piece_type as usize] & bit != Bitboard::EMPTY {
                let color = if self.colors[Color::Black as usize] & bit != Bitboard::EMPTY {
                    Color::Black
                } else {
                    Color::White
                };
                return Some((piece_type, color));
            }
        }
        None
    }

    pub fn can_kingside_castle(&self, color: Color) -> bool {
        match color {
            Color::White => self.castling_rights & 0b0001 != 0,
            Color::Black => self.castling_rights & 0b0010 != 0,
        }
    }

    pub fn can_queenside_castle(&self, color: Color) -> bool {
        match color {
            Color::White => self.castling_rights & 0b0100 != 0,
            Color::Black => self.castling_rights & 0b1000 != 0,
        }
    }

    fn flip_color(&mut self) {
        self.active_color = match self.active_color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        self.zobrist ^= ZOBRIST.color();
    }

    /// unsafe: assumes from empty state
    fn put_piece(&mut self, piece_type: PieceType, color: Color, square: usize) {
        let bit = Bitboard::from_square(square);
        self.pieces[piece_type as usize] |= bit;
        self.colors[color as usize] |= bit;
        self.zobrist ^= ZOBRIST.piece_square(piece_type, color, square);
    }

    /// unsafe: assumes from empty state
    fn put_kingside_castling(&mut self, color: Color) {
        let bit = match color {
            Color::White => 0b0001,
            Color::Black => 0b0010,
        };
        self.castling_rights |= bit;
        self.zobrist ^= ZOBRIST.kingside_castling(color);
    }

    /// unsafe: assumes from empty state
    fn put_queenside_castling(&mut self, color: Color) {
        let bit = match color {
            Color::White => 0b0100,
            Color::Black => 0b1000,
        };
        self.castling_rights |= bit;
        self.zobrist ^= ZOBRIST.queenside_castling(color);
    }

    /// unsafe: assumes from empty state
    fn put_en_passant(&mut self, file: usize) {
        self.en_passant_file = Some(file as u8);
        self.zobrist ^= ZOBRIST.en_passant(file);
    }
}

/// Stockfish-style board representation
impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            write!(f, " +---+---+---+---+---+---+---+---+\n")?;
            for file in 0..8 {
                let square = rank * 8 + file;
                let piece_char = match self.piece_at(square) {
                    Some((piece_type, color)) => {
                        let c = match piece_type {
                            PieceType::Pawn => 'P',
                            PieceType::Knight => 'N',
                            PieceType::Bishop => 'B',
                            PieceType::Rook => 'R',
                            PieceType::Queen => 'Q',
                            PieceType::King => 'K',
                        };
                        if color == Color::Black {
                            c.to_ascii_lowercase()
                        } else {
                            c
                        }
                    }
                    None => ' ',
                };
                write!(f, " | {}", piece_char)?;
            }
            writeln!(f, " |")?;
        }
        write!(f, " +---+---+---+---+---+---+---+---+\n")
    }
}
