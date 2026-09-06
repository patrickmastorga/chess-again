use crate::core::{Board, Color, Move, MoveType, PieceType, utils};
use chrono::Local;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

pub struct Game {
    board: Board,
    legal_moves: Vec<Move>,
    result: Option<GameResult>,
    san_moves: Vec<String>,
    event: String,
    site: String,
    date: String,
    round: String,
    white: String,
    black: String,
    tags: Vec<(String, String)>,
}

impl Game {
    pub fn new() -> Self {
        Self::starting_at_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("starting fen is legal.")
    }

    pub fn starting_at_fen(fen: &str) -> Result<Self, String> {
        let mut game = Self {
            board: Board::from_fen(fen)?,
            legal_moves: Vec::new(),
            result: None,
            san_moves: Vec::new(),
            event: "?".to_string(),
            site: "?".to_string(),
            date: Local::now().format("%Y.%m.%d").to_string(),
            round: "?".to_string(),
            white: "?".to_string(),
            black: "?".to_string(),
            tags: Vec::new(),
        };
        if fen != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" {
            game.tags.push(("SetUp".to_string(), "1".to_string()));
            game.tags.push(("FEN".to_string(), fen.to_string()));
        }
        game.update_legal_moves();
        Ok(game)
    }

    pub fn result(&self) -> Option<GameResult> {
        self.result
    }

    pub fn legal_moves(&self) -> &Vec<Move> {
        &self.legal_moves
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn make_move(&mut self, mv: Move) -> Result<Option<crate::core::PieceType>, String> {
        if !self.legal_moves.contains(&mv) {
            return Err(format!("Illegal move: {mv}"));
        }
        self.san_moves
            .push(mv.to_san(&self.board).expect("move is legal."));
        let captured_piece = self.board.make_move_unchecked(mv);
        self.update_legal_moves();
        if self.board.in_check() {
            let last = self
                .san_moves
                .last_mut()
                .expect("There must be a last move.");
            if self.legal_moves.is_empty() {
                *last = format!("{}#", last);
            } else {
                *last = format!("{}+", last);
            }
        }
        Ok(captured_piece)
    }

    pub fn forfeit(&mut self, color: Color) -> Result<(), String> {
        if self.result.is_some() {
            return Err("Cannot forfeit a game that already has a result.".to_string());
        }
        self.result = Some(match color {
            Color::White => GameResult::BlackWin,
            Color::Black => GameResult::WhiteWin,
        });
        self.legal_moves.clear();
        Ok(())
    }

    pub fn draw(&mut self) -> Result<(), String> {
        if self.result.is_some() {
            return Err("Cannot declare a draw for a game that already has a result.".to_string());
        }
        self.result = Some(GameResult::Draw);
        self.legal_moves.clear();
        Ok(())
    }

    pub fn set_event(&mut self, event: impl Into<String>) {
        self.event = event.into();
    }

    pub fn set_site(&mut self, site: impl Into<String>) {
        self.site = site.into();
    }

    pub fn set_date(&mut self, date: impl Into<String>) {
        self.date = date.into();
    }

    pub fn set_round(&mut self, round: impl Into<String>) {
        self.round = round.into();
    }

    pub fn set_white(&mut self, white: impl Into<String>) {
        self.white = white.into();
    }

    pub fn set_black(&mut self, black: impl Into<String>) {
        self.black = black.into();
    }

    pub fn add_tag(
        &mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), String> {
        let name = name.into();
        if matches!(
            name.as_str(),
            "Event" | "Site" | "Date" | "Round" | "White" | "Black" | "Result"
        ) {
            return Err(format!("The tag '{name}' belongs to the Seven Tag Roster."));
        }
        if self.tags.iter().any(|(tag, _)| tag == &name) {
            return Err(format!("Duplicate PGN tag: {name}"));
        }
        self.tags.push((name, value.into()));
        Ok(())
    }

    pub fn to_pgn(&self) -> Result<String, String> {
        let result = self
            .result
            .ok_or_else(|| "Cannot export an unfinished game as PGN.".to_string())?;
        let result_text = match result {
            GameResult::WhiteWin => "1-0",
            GameResult::BlackWin => "0-1",
            GameResult::Draw => "1/2-1/2",
        };

        let mut pgn = String::new();
        for (name, value) in [
            ("Event", self.event.as_str()),
            ("Site", self.site.as_str()),
            ("Date", self.date.as_str()),
            ("Round", self.round.as_str()),
            ("White", self.white.as_str()),
            ("Black", self.black.as_str()),
            ("Result", result_text),
        ] {
            pgn.push_str(&format!("[{name} \"{}\"]\n", escape_pgn_value(value)));
        }
        for (name, value) in &self.tags {
            pgn.push_str(&format!("[{name} \"{}\"]\n", escape_pgn_value(value)));
        }
        pgn.push('\n');

        let starting_halfmove = self
            .board
            .halfmove_number()
            .saturating_sub(self.san_moves.len() as u32);
        let mut move_number = starting_halfmove / 2 + 1;
        let white_to_move = starting_halfmove % 2 == 0;
        if !white_to_move && !self.san_moves.is_empty() {
            pgn.push_str(&format!("{move_number}... "));
        }
        for (index, san) in self.san_moves.iter().enumerate() {
            let white_move = white_to_move == (index % 2 == 0);
            if white_move {
                pgn.push_str(&format!("{move_number}. {san} "));
            } else {
                pgn.push_str(&format!("{san} "));
                move_number += 1;
            }
        }
        pgn.push_str(result_text);
        Ok(pgn)
    }

    fn update_legal_moves(&mut self) {
        if self.board.is_threefold_repetition() || self.board.is_insufficient_material() {
            self.legal_moves.clear();
            self.result = Some(GameResult::Draw);
            return;
        }
        self.legal_moves = self.board.legal_moves();
        if self.legal_moves.is_empty() {
            if self.board.in_check() {
                self.result = Some(match self.board.active_color() {
                    Color::White => GameResult::BlackWin,
                    Color::Black => GameResult::WhiteWin,
                });
            } else {
                self.result = Some(GameResult::Draw);
            }
        } else {
            self.result = None;
        }
    }
}

impl Move {
    pub fn from_san(san: &str, board: &Board) -> Result<Self, String> {
        let mut san = san.trim();
        if san.is_empty() {
            return Err("SAN move cannot be empty.".to_string());
        }
        san = san.trim_end_matches(['+', '#']);

        if san == "O-O" || san == "0-0" {
            return unique_legal_move(
                board,
                |mv| matches!(mv.move_type, MoveType::Castling(true)),
                san,
            );
        }
        if san == "O-O-O" || san == "0-0-0" {
            return unique_legal_move(
                board,
                |mv| matches!(mv.move_type, MoveType::Castling(false)),
                san,
            );
        }

        // parse an optional promotion suffix, e.g. "e8=Q".
        let mut promotion = None;
        if let Some((body, promo)) = san.split_once('=') {
            let mut chars = promo.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => promotion = Some(promotion_piece(c as u8)?),
                _ => return Err(format!("Invalid promotion in SAN move: {san}")),
            }
            san = body;
        }

        // the last two characters must be the destination square.
        if san.len() < 2 {
            return Err(format!("Invalid SAN move: {san}"));
        }
        let (head, destination) = san.split_at(san.len() - 2);
        let to = utils::parse_algebraic(destination)? as u8;

        // the remaining head consists of an optional piece letter,
        // optional disambiguation (file and/or rank of the origin square),
        // and an optional 'x' capture marker.
        let bytes = head.as_bytes();
        let mut index = 0;

        let piece_type = match bytes.first() {
            Some(b'K') => {
                index += 1;
                PieceType::King
            }
            Some(b'Q') => {
                index += 1;
                PieceType::Queen
            }
            Some(b'R') => {
                index += 1;
                PieceType::Rook
            }
            Some(b'B') => {
                index += 1;
                PieceType::Bishop
            }
            Some(b'N') => {
                index += 1;
                PieceType::Knight
            }
            _ => PieceType::Pawn,
        };

        let mut from_file = None;
        let mut from_rank = None;
        for &b in &bytes[index..] {
            match b {
                b'x' => {} // capture marker; legality is checked against legal moves
                b'a'..=b'h' if from_file.is_none() => from_file = Some(b - b'a'),
                b'1'..=b'8' if from_rank.is_none() => from_rank = Some(b - b'1'),
                _ => return Err(format!("Invalid SAN move: {san}")),
            }
        }

        unique_legal_move(
            board,
            |mv| {
                mv.to == to
                    && match (mv.move_type, promotion) {
                        (MoveType::Promotion(p), Some(expected)) => p == expected,
                        (MoveType::Promotion(_), None) => false,
                        (_, Some(_)) => false,
                        _ => true,
                    }
                    && from_file.is_none_or(|f| mv.from % 8 == f)
                    && from_rank.is_none_or(|r| mv.from / 8 == r)
                    && board
                        .piece_at(mv.from as usize)
                        .is_some_and(|(pt, _)| pt == piece_type)
            },
            san,
        )
    }

    // ignores check annotations
    pub fn to_san(&self, board: &Board) -> Result<String, String> {
        let legal_moves = board.legal_moves();
        if !legal_moves.contains(self) {
            return Err(format!("Illegal move: {self}"));
        }

        let (piece_type, color) = board
            .piece_at(self.from as usize)
            .expect("There must be a piece here.");

        match self.move_type {
            MoveType::Castling(true) => Ok("O-O".to_string()),
            MoveType::Castling(false) => Ok("O-O-O".to_string()),
            _ => {
                let capture = match self.move_type {
                    MoveType::EnPassant(_) => true,
                    _ => board.piece_at(self.to as usize).is_some(),
                };
                let destination = utils::square_to_algebraic(self.to as usize);
                let mut notation = String::new();

                if piece_type != PieceType::Pawn {
                    notation.push(piece_letter(piece_type));

                    let alternatives: Vec<Move> = legal_moves
                        .into_iter()
                        .filter(|mv| {
                            mv.to == self.to
                                && mv.from != self.from
                                && board
                                    .piece_at(mv.from as usize)
                                    .expect("There must be a piece here.")
                                    == (piece_type, color)
                        })
                        .collect();
                    if !alternatives.is_empty() {
                        let same_file = alternatives.iter().any(|mv| mv.from % 8 == self.from % 8);
                        let same_rank = alternatives.iter().any(|mv| mv.from / 8 == self.from / 8);
                        if !same_file {
                            notation.push((b'a' + self.from % 8) as char);
                        } else if !same_rank {
                            notation.push((b'1' + self.from / 8) as char);
                        } else {
                            notation.push_str(&utils::square_to_algebraic(self.from as usize));
                        }
                    }
                } else if capture {
                    notation.push((b'a' + self.from % 8) as char);
                }

                if capture {
                    notation.push('x');
                }
                notation.push_str(&destination);
                if let MoveType::Promotion(promoted_to) = self.move_type {
                    notation.push('=');
                    notation.push(piece_letter(promoted_to));
                }
                Ok(notation)
            }
        }
    }
}

const fn piece_letter(piece_type: PieceType) -> char {
    match piece_type {
        PieceType::King => 'K',
        PieceType::Queen => 'Q',
        PieceType::Rook => 'R',
        PieceType::Bishop => 'B',
        PieceType::Knight => 'N',
        PieceType::Pawn => 'P',
    }
}

fn promotion_piece(character: u8) -> Result<PieceType, String> {
    match character {
        b'Q' => Ok(PieceType::Queen),
        b'R' => Ok(PieceType::Rook),
        b'B' => Ok(PieceType::Bishop),
        b'N' => Ok(PieceType::Knight),
        _ => Err("Invalid promotion piece in SAN move.".to_string()),
    }
}

fn escape_pgn_value(value: &str) -> String {
    // PGN tag values are quoted strings. Escape the two characters that can
    // otherwise change the meaning of the value or terminate the string.
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unique_legal_move<F>(board: &Board, mut predicate: F, san: &str) -> Result<Move, String>
where
    F: FnMut(&Move) -> bool,
{
    let mut matching_move = None;
    for mv in board.legal_moves().iter().filter(|mv| predicate(mv)) {
        if matching_move.is_some() {
            return Err(format!("Ambiguous SAN move: {san}"));
        }
        matching_move = Some(*mv);
    }
    matching_move.ok_or_else(|| format!("Invalid or illegal SAN move: {san}"))
}
