use std::fmt::{Display, Formatter};

use crate::board::{Board, Move, MoveType, PieceType, Square};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UciMove {
    source: Square,
    dest: Square,
    promotion: Option<PieceType>,
}

impl UciMove {
    pub const NULL: Self = Self {
        source: Square::from_index_unchecked(0),
        dest: Square::from_index_unchecked(0),
        promotion: None,
    };

    pub const fn source(&self) -> Square {
        self.source
    }

    pub const fn dest(&self) -> Square {
        self.dest
    }

    pub const fn promotion(&self) -> Option<PieceType> {
        self.promotion
    }

    pub const fn new(
        from: Square,
        to: Square,
        promotion: Option<PieceType>,
    ) -> Result<Self, &'static str> {
        if matches!(promotion, Some(PieceType::Pawn) | Some(PieceType::King)) {
            return Err("Invalid UCI move: promotion must be to a knight, bishop, rook, or queen");
        }
        Ok(Self {
            source: from,
            dest: to,
            promotion,
        })
    }

    pub fn from_string(s: &str) -> Result<Self, String> {
        let input = s.trim().to_ascii_lowercase();
        if input == "0000" {
            return Ok(Self::NULL);
        }
        let input_bytes = input.as_bytes();
        if input_bytes.len() != 4 && input_bytes.len() != 5 {
            return Err(format!("Invalid UCI move: '{input}'"));
        }

        let from = Square::from_algebraic(&input[0..2])
            .map_err(|_| format!("Invalid UCI move: '{input}'"))?;
        let to = Square::from_algebraic(&input[2..4])
            .map_err(|_| format!("Invalid UCI move: '{input}'"))?;
        let promotion = match input_bytes.get(4) {
            None => None,
            Some(b'n') => Some(PieceType::Knight),
            Some(b'b') => Some(PieceType::Bishop),
            Some(b'r') => Some(PieceType::Rook),
            Some(b'q') => Some(PieceType::Queen),
            Some(_) => return Err(format!("Invalid UCI promotion: '{input}'")),
        };
        Self::new(from, to, promotion).map_err(str::to_string)
    }

    pub fn mirrored(&self) -> Self {
        Self {
            source: self.source.mirrored(),
            dest: self.dest.mirrored(),
            promotion: self.promotion,
        }
    }
}

impl Display for UciMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if *self == Self::NULL {
            return write!(f, "0000");
        }

        write!(
            f,
            "{}{}",
            self.source().algebraic(),
            self.dest().algebraic()
        )?;
        if let Some(piece_type) = self.promotion() {
            let promotion = match piece_type {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                PieceType::Pawn | PieceType::King => return Err(std::fmt::Error),
            };
            write!(f, "{promotion}")?;
        }
        Ok(())
    }
}

impl From<Move> for UciMove {
    fn from(mv: Move) -> Self {
        let promotion = match mv.move_type() {
            MoveType::Promotion(piece) => Some(piece),
            _ => None,
        };
        UciMove::new(mv.source(), mv.dest(), promotion)
            .expect("a generated Move must be a valid UCI move")
    }
}

impl Move {
    pub fn from_uci(uci_move: UciMove, board: &Board) -> Result<Self, String> {
        board
            .legal_moves()
            .into_iter()
            .find(|mv| {
                mv.source() == uci_move.source()
                    && mv.dest() == uci_move.dest()
                    && match (mv.move_type(), uci_move.promotion()) {
                        (MoveType::Promotion(piece), Some(promotion)) => piece == promotion,
                        (MoveType::Promotion(_), None) => false,
                        (_, Some(_)) => false,
                        (_, None) => true,
                    }
            })
            .ok_or_else(|| "UCI move must be legal on the given board.".to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
    Cp(i64),   // centipawns, from the engine's point of view
    Mate(i64), // mate in N; negative means the engine is getting mated
}

/// What an engine returns for one completed search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchResult {
    pub best_move: UciMove,
    pub eval: Option<Score>,
}

/// The minimal surface a playable engine needs.
pub trait Engine {
    /// Starts a search on `fen`. `time` is milliseconds, `None` searches until
    /// stopped. Any previous search is discarded.
    fn start_search_async(&mut self, fen: &str, time: Option<u64>) -> Result<(), String>;

    /// Returns the result if a search has finished since the last call.
    fn poll(&mut self) -> Result<Option<SearchResult>, String>;

    /// Ends the running search, blocking until it is fully over, and returns
    /// its result. `Ok(None)` means no search was running.
    fn stop(&mut self) -> Result<Option<SearchResult>, String>;
}
