use crate::core::{Board, CastlingRights, Color, PieceType, utils};

struct Fen {
    piece_placement: String,
    active_color: String,
    castling_rights: String,
    en_passant: String,
    halfmove_clock: String,
    fullmove_number: String,
}

impl Fen {
    pub fn from_str(fen_string: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen_string.split_whitespace().collect();
        if parts.len() != 6 {
            return Err("FEN string must have 6 space-separated fields.".to_string());
        }

        let piece_placement = parts[0].to_string();
        let active_color = parts[1].to_string();
        let castling_rights = parts[2].to_string();
        let en_passant = parts[3].to_string();
        let halfmove_clock = parts[4].to_string();
        let fullmove_number = parts[5].to_string();

        // verify format
        let ranks: Vec<&str> = piece_placement.split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN piece placement must contain 8 ranks.".to_string());
        }
        for (rank_index, rank) in ranks.iter().enumerate() {
            let mut file_count = 0;
            for character in rank.chars() {
                match character {
                    '1'..='8' => file_count += character.to_digit(10).unwrap(),
                    'p' | 'n' | 'b' | 'r' | 'q' | 'k' | 'P' | 'N' | 'B' | 'R' | 'Q' | 'K' => {
                        file_count += 1
                    }
                    _ => {
                        return Err(format!(
                            "Invalid FEN piece placement character '{character}' in rank {}.",
                            8 - rank_index
                        ));
                    }
                }
            }
            if file_count != 8 {
                return Err(format!(
                    "FEN rank {} must contain exactly 8 files.",
                    8 - rank_index
                ));
            }
        }

        if active_color != "w" && active_color != "b" {
            return Err("FEN active color must be 'w' or 'b'.".to_string());
        }

        if castling_rights != "-" {
            let mut seen = 0u8;
            for character in castling_rights.chars() {
                let flag = match character {
                    'K' => 0b0001,
                    'Q' => 0b0010,
                    'k' => 0b0100,
                    'q' => 0b1000,
                    _ => return Err(format!("Invalid FEN castling right '{character}'.")),
                };
                if seen & flag != 0 {
                    return Err(format!("Duplicate FEN castling right '{character}'."));
                }
                seen |= flag;
            }
        }

        if en_passant != "-" {
            if utils::parse_algebraic(&en_passant).is_err() {
                return Err(format!("Invalid FEN en passant square '{}'.", en_passant));
            }
        }

        if halfmove_clock.parse::<u32>().is_err() {
            return Err(format!("Invalid FEN halfmove clock '{}'.", halfmove_clock));
        }
        match fullmove_number.parse::<u32>() {
            Ok(0) | Err(_) => {
                return Err(format!(
                    "Invalid FEN fullmove number '{}'.",
                    fullmove_number
                ));
            }
            Ok(_) => {}
        }

        // HERE

        let fen = Fen {
            piece_placement,
            active_color,
            castling_rights,
            en_passant,
            halfmove_clock,
            fullmove_number,
        };
        Ok(fen)
    }

    pub fn piece_placement(&self) -> Vec<(usize, PieceType, Color)> {
        let mut pieces = Vec::new();
        for (rank_index, rank) in self.piece_placement.split('/').enumerate() {
            let mut file_index = 0;
            for character in rank.chars() {
                match character {
                    '1'..='8' => file_index += character.to_digit(10).unwrap() as usize,
                    'p' | 'n' | 'b' | 'r' | 'q' | 'k' | 'P' | 'N' | 'B' | 'R' | 'Q' | 'K' => {
                        let color = if character.is_uppercase() {
                            Color::White
                        } else {
                            Color::Black
                        };
                        let piece_type = match character.to_ascii_lowercase() {
                            'p' => PieceType::Pawn,
                            'n' => PieceType::Knight,
                            'b' => PieceType::Bishop,
                            'r' => PieceType::Rook,
                            'q' => PieceType::Queen,
                            'k' => PieceType::King,
                            _ => unreachable!(),
                        };
                        pieces.push(((7 - rank_index) * 8 + file_index, piece_type, color));
                        file_index += 1;
                    }
                    _ => unreachable!("piece placement was validated above"),
                }
            }
        }
        pieces
    }

    pub fn active_color(&self) -> Color {
        match self.active_color.as_str() {
            "w" => Color::White,
            "b" => Color::Black,
            _ => unreachable!("active color was validated above"),
        }
    }

    pub fn castling_rights(&self) -> CastlingRights {
        let mut rights = CastlingRights::empty();
        for character in self.castling_rights.chars() {
            match character {
                'K' => rights.set_kingside(Color::White, true),
                'Q' => rights.set_queenside(Color::White, true),
                'k' => rights.set_kingside(Color::Black, true),
                'q' => rights.set_queenside(Color::Black, true),
                '-' => {}
                _ => unreachable!("castling rights were validated above"),
            }
        }
        rights
    }

    pub fn en_passant(&self) -> Option<u8> {
        match self.en_passant.as_str() {
            "-" => None,
            square => Some(utils::parse_algebraic(square).unwrap() as u8),
        }
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock.parse().unwrap()
    }

    pub fn fullmove_number(&self) -> u32 {
        self.fullmove_number.parse().unwrap()
    }
}

impl Board {
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let fen = Fen::from_str(fen).map_err(|error| format!("Invalid FEN: {error}"))?;
        let pieces = fen.piece_placement();
        let halfmove_number = 2 * (fen.fullmove_number() - 1) + fen.active_color() as u32;
        Board::new(
            &pieces,
            fen.castling_rights(),
            fen.en_passant(),
            halfmove_number,
            fen.halfmove_clock(),
        )
        .map_err(|error| format!("Invalid FEN position: {error}"))
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                let square = rank * 8 + file;
                match self.piece_at(square) {
                    Some((piece_type, color)) => {
                        if empty_count > 0 {
                            fen.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        let c = match piece_type {
                            PieceType::Pawn => 'p',
                            PieceType::Knight => 'n',
                            PieceType::Bishop => 'b',
                            PieceType::Rook => 'r',
                            PieceType::Queen => 'q',
                            PieceType::King => 'k',
                        };
                        fen.push(if color == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        });
                    }
                    None => {
                        empty_count += 1;
                    }
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push(' ');
        fen.push(if self.active_color() == Color::White {
            'w'
        } else {
            'b'
        });
        fen.push(' ');
        if self.castling_rights() == CastlingRights::empty() {
            fen.push('-');
        } else {
            if self.castling_rights().kingside(Color::White) {
                fen.push('K');
            }
            if self.castling_rights().queenside(Color::White) {
                fen.push('Q');
            }
            if self.castling_rights().kingside(Color::Black) {
                fen.push('k');
            }
            if self.castling_rights().queenside(Color::Black) {
                fen.push('q');
            }
        }
        fen.push(' ');
        match self.en_passant_square() {
            Some(square) => {
                fen.push_str(&utils::square_to_algebraic(square as usize));
            }
            None => {
                fen.push('-');
            }
        }
        fen.push_str(&format!(
            " {} {}",
            self.rule50(),
            self.halfmove_number() / 2 + 1
        )); // halfmove and fullmove clocks
        fen
    }
}
