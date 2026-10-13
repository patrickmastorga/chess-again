use std::fmt::{Display, Formatter};

use crate::bitboard::Bitboard;

/// The two chess colors, with values matching array indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
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

/// A colored chess piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

impl Piece {
    pub const fn new(piece_type: PieceType, color: Color) -> Self {
        Self { piece_type, color }
    }
}

/// A board square using `rank * 8 + file`, with `a1 = 0` and `h8 = 63`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(u8);

impl Square {
    /// Creates a square from zero-based rank and file coordinates.
    pub fn new(rank: u8, file: u8) -> Result<Self, String> {
        if rank >= 8 || file >= 8 {
            return Err("square rank and file must be in 0..8".to_string());
        }
        Ok(Self(rank * 8 + file))
    }

    /// Creates a square from a `rank * 8 + file` index.
    pub fn from_index(index: u8) -> Result<Self, String> {
        if index >= 64 {
            return Err("square index must be in 0..64".to_string());
        }
        Ok(Self(index))
    }

    /// Creates a square from a valid `rank * 8 + file` index without checking.
    pub(crate) const fn from_index_unchecked(index: usize) -> Self {
        Self(index as u8)
    }

    /// Creates a square from valid zero-based rank and file coordinates without checking.
    pub(crate) const fn new_unchecked(rank: u8, file: u8) -> Self {
        Self(rank * 8 + file)
    }

    /// Parses an algebraic square name such as `"e4"`.
    pub fn from_algebraic(notation: &str) -> Result<Self, String> {
        let bytes = notation.as_bytes();
        if bytes.len() != 2 {
            return Err(format!("invalid square: {notation}"));
        }
        let file = bytes[0]
            .checked_sub(b'a')
            .filter(|file| *file < 8)
            .ok_or_else(|| format!("invalid square: {notation}"))?;
        let rank = bytes[1]
            .checked_sub(b'1')
            .filter(|rank| *rank < 8)
            .ok_or_else(|| format!("invalid square: {notation}"))?;
        Self::new(rank, file)
    }

    /// Zero-based rank, with rank 1 represented by zero.
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Zero-based file, with file a represented by zero.
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// The `rank * 8 + file` index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The algebraic name, e.g. `"e4"`.
    pub fn algebraic(self) -> String {
        format!(
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }

    /// Whether this is a light square (`a1` is dark).
    pub const fn is_light(self) -> bool {
        (self.rank() + self.file()) % 2 == 1
    }

    pub const fn mirrored(self) -> Self {
        Self(self.0 ^ 0b111000)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveType {
    Regular,
    Castling(bool),
    EnPassant(u8),
    Promotion(PieceType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    source: Square,
    dest: Square,
    move_type: MoveType,
}

impl Move {
    pub const fn source(&self) -> Square {
        self.source
    }

    pub const fn dest(&self) -> Square {
        self.dest
    }

    pub const fn move_type(&self) -> MoveType {
        self.move_type
    }

    pub(crate) const fn new(source: Square, dest: Square, move_type: MoveType) -> Self {
        Self {
            source,
            dest,
            move_type,
        }
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            self.source().algebraic(),
            self.dest().algebraic()
        )?;

        if let MoveType::Promotion(piece_type) = self.move_type() {
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

/// Represents the castling rights for both sides in a chess game.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub fn empty() -> Self {
        Self(0u8)
    }

    pub fn bits(&self) -> u8 {
        self.0
    }

    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub fn set_kingside(&mut self, color: Color, can_castle: bool) {
        match color {
            Color::White => {
                if can_castle {
                    self.0 |= 0b0001;
                } else {
                    self.0 &= !0b0001;
                }
            }
            Color::Black => {
                if can_castle {
                    self.0 |= 0b0100;
                } else {
                    self.0 &= !0b0100;
                }
            }
        }
    }

    pub fn set_queenside(&mut self, color: Color, can_castle: bool) {
        match color {
            Color::White => {
                if can_castle {
                    self.0 |= 0b0010;
                } else {
                    self.0 &= !0b0010;
                }
            }
            Color::Black => {
                if can_castle {
                    self.0 |= 0b1000;
                } else {
                    self.0 &= !0b1000;
                }
            }
        }
    }

    pub fn kingside(&self, color: Color) -> bool {
        match color {
            Color::White => self.0 & 0b0001 != 0,
            Color::Black => self.0 & 0b0100 != 0,
        }
    }

    pub fn queenside(&self, color: Color) -> bool {
        match color {
            Color::White => self.0 & 0b0010 != 0,
            Color::Black => self.0 & 0b1000 != 0,
        }
    }

    pub fn flipped(&self) -> Self {
        let black = self.0 >> 2;
        let white = self.0 & 0b0011;
        Self((black) | (white << 2))
    }
}

/// Board metadata not deduceable when undoing moves
#[derive(Debug, Clone, Copy)]
struct BoardMetadata {
    castling_rights: CastlingRights,
    ep_square: Option<Square>,
    rule50: u8,
    zobrist: u64,
}

/// Primary structure for representing and manipulating a chess board state
#[derive(Debug, Clone)]
pub struct Board {
    colors: [Bitboard; 2], // indexed by Color
    pieces: [Bitboard; 6], // indexed by PieceType
    halfmove_number: u32,
    active_color: Color,
    metadata: Vec<BoardMetadata>,
    moves: Vec<(Move, Option<PieceType>)>,
}
impl Board {
    pub fn new(
        pieces: &[(Square, Piece)],
        castling_rights: CastlingRights,
        ep_square: Option<Square>,
        halfmove_number: u32,
        rule50: u8,
    ) -> Result<Self, String> {
        let rule50 = rule50.min(100);
        let active_color = if halfmove_number % 2 == 0 {
            Color::White
        } else {
            Color::Black
        };
        let mut board = Board {
            colors: [Bitboard::EMPTY; 2],
            pieces: [Bitboard::EMPTY; 6],
            halfmove_number,
            active_color,
            metadata: vec![BoardMetadata {
                castling_rights,
                ep_square,
                rule50: rule50,
                zobrist: 0,
            }],
            moves: Vec::new(),
        };

        for &(square, piece) in pieces {
            board.put_piece(piece.piece_type, piece.color, square.index());
        }

        if active_color == Color::Black {
            *board.zobrist_mut() ^= zobrist::color();
        }

        if castling_rights.kingside(Color::White) {
            *board.zobrist_mut() ^= zobrist::kingside_castling(Color::White);
        }
        if castling_rights.queenside(Color::White) {
            *board.zobrist_mut() ^= zobrist::queenside_castling(Color::White);
        }
        if castling_rights.kingside(Color::Black) {
            *board.zobrist_mut() ^= zobrist::kingside_castling(Color::Black);
        }
        if castling_rights.queenside(Color::Black) {
            *board.zobrist_mut() ^= zobrist::queenside_castling(Color::Black);
        }

        if let Some(ep_square) = ep_square {
            *board.zobrist_mut() ^= zobrist::en_passant(ep_square.index());
        }

        // validate board

        // one king of each color
        if board.kings(Color::White).count() != 1 {
            return Err("There must be exactly one white king.".to_string());
        }
        if board.kings(Color::Black).count() != 1 {
            return Err("There must be exactly one black king.".to_string());
        }
        let white_king_index = board
            .kings(Color::White)
            .lsb_index()
            .expect("validated above.");
        let black_king_index = board
            .kings(Color::Black)
            .lsb_index()
            .expect("validated above.");

        // castling rights are valid
        if castling_rights.kingside(Color::White)
            && (white_king_index != 4 || !board.rooks(Color::White).contains(7))
        {
            return Err(
                "Invalid white kingside castling right. The king and rook do not reside on thier original squares"
                .to_string()
            );
        }
        if castling_rights.queenside(Color::White)
            && (white_king_index != 4 || !board.rooks(Color::White).contains(0))
        {
            return Err(
                "Invalid white queenside castling right. The king and rook do not reside on thier original squares"
                .to_string()
            );
        }
        if castling_rights.kingside(Color::Black)
            && (black_king_index != 60 || !board.rooks(Color::Black).contains(63))
        {
            return Err(
                "Invalid black kingside castling right. The king and rook do not reside on thier original squares"
                .to_string()
            );
        }
        if castling_rights.queenside(Color::Black)
            && (black_king_index != 60 || !board.rooks(Color::Black).contains(56))
        {
            return Err(
                "Invalid black queenside castling right. The king and rook do not reside on thier original squares"
                .to_string()
            );
        }

        // en passant square is valid
        if let Some(ep_square) = ep_square {
            let ep_square = ep_square.index();
            *board.zobrist_mut() ^= zobrist::en_passant(ep_square);
            let (source, destination, color) = match ep_square >> 3 {
                2 => {
                    let source = 1 * 8 + (ep_square & 0b111);
                    let destination = 3 * 8 + (ep_square & 0b111);
                    (source, destination, Color::Black)
                }
                5 => {
                    let source = 6 * 8 + (ep_square & 0b111);
                    let destination = 4 * 8 + (ep_square & 0b111);
                    (source, destination, Color::White)
                }
                _ => {
                    return Err(
                        "En passant square must be on either the 3rd or 6th ranks.".to_string()
                    );
                }
            };
            if active_color != color
                || board.all_pieces().contains(source)
                || board.all_pieces().contains(ep_square)
                || !board.pawns(color.opposite()).contains(destination)
            {
                return Err(
                    "Invalid en passant square. A pawn double jump could not have occured on the last move"
                    .to_string()
                );
            }
        }

        if (rule50 as u32) > halfmove_number {
            return Err("rule50 cannot be larger than the halfmove number.".to_string());
        }

        Ok(board)
    }

    // public getters

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.piece_at_index(square.index())
    }

    /// All pieces on the board with their squares, in no particular order.
    pub fn pieces(&self) -> Vec<(Square, Piece)> {
        let mut pieces = Vec::with_capacity(32);
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            for color in [Color::Black, Color::White] {
                for square in
                    (self.pieces[piece_type as usize] & self.colors[color as usize]).iter_indices()
                {
                    if self.pieces_of_color(color).contains(square) {
                        pieces.push((
                            Square::from_index_unchecked(square),
                            Piece::new(piece_type, color),
                        ));
                    }
                }
            }
        }
        pieces
    }

    pub(crate) fn piece_at_index(&self, square: usize) -> Option<Piece> {
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            if self.pieces[piece_type as usize].contains(square) {
                let color = if self.pieces_of_color(Color::Black).contains(square) {
                    Color::Black
                } else {
                    Color::White
                };
                return Some(Piece::new(piece_type, color));
            }
        }
        None
    }

    pub fn castling_rights(&self) -> CastlingRights {
        self.metadata
            .last()
            .expect("Metadata should always be available.")
            .castling_rights
    }

    pub fn ep_square(&self) -> Option<Square> {
        self.metadata
            .last()
            .expect("Metadata should always be available.")
            .ep_square
    }

    pub fn halfmove_number(&self) -> u32 {
        self.halfmove_number
    }

    pub fn active_color(&self) -> Color {
        self.active_color
    }

    pub fn rule50(&self) -> u8 {
        self.metadata
            .last()
            .expect("Metadata should always be available.")
            .rule50
    }

    pub fn zobrist(&self) -> u64 {
        self.metadata
            .last()
            .expect("Metadata should always be available.")
            .zobrist
    }

    pub fn is_threefold_repetition(&self) -> bool {
        self.repetition_count() >= 3
    }

    pub fn repetition_count(&self) -> u32 {
        let current_zobrist = self.zobrist();
        let history_length = (self.rule50() as usize + 1).min(self.metadata.len());
        self.metadata
            .iter()
            .rev()
            .take(history_length)
            .filter(|metadata| metadata.zobrist == current_zobrist)
            .count() as u32
    }

    pub fn is_insufficient_material(&self) -> bool {
        // Any pawn, rook, or queen means that a checkmate is possible.
        if !(self.pawns(Color::White) | self.pawns(Color::Black)).is_empty()
            || !(self.rooks(Color::White) | self.rooks(Color::Black)).is_empty()
            || !(self.queens(Color::White) | self.queens(Color::Black)).is_empty()
        {
            return false;
        }

        let white_knights = self.knights(Color::White).count();
        let black_knights = self.knights(Color::Black).count();
        let total_knights = white_knights + black_knights;
        let total_bishops = self.bishops(Color::White).count() + self.bishops(Color::Black).count();

        // Two knights can provide mating material, as can a knight against
        // any other minor piece (including another knight).
        if total_knights >= 2 || (total_knights > 0 && total_bishops > 0) {
            return false;
        }

        if total_bishops == 0 {
            // King versus king, or king and one knight versus king.
            return true;
        }

        // Bishops can mate only when bishops of both square colours are
        // present. Square 0 is a dark square with this board's indexing.
        let bishops = self.bishops(Color::White) | self.bishops(Color::Black);
        let has_dark_square_bishop = bishops
            .iter_indices()
            .any(|square| (square / 8 + square % 8) % 2 == 0);
        let has_light_square_bishop = bishops
            .iter_indices()
            .any(|square| (square / 8 + square % 8) % 2 != 0);

        !(has_dark_square_bishop && has_light_square_bishop)
    }

    pub fn all_pieces(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    pub fn pieces_of_color(&self, color: Color) -> Bitboard {
        self.colors[color as usize]
    }

    pub fn pawns(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::Pawn as usize] & self.pieces_of_color(color)
    }

    pub fn knights(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::Knight as usize] & self.pieces_of_color(color)
    }

    pub fn bishops(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::Bishop as usize] & self.pieces_of_color(color)
    }

    pub fn rooks(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::Rook as usize] & self.pieces_of_color(color)
    }

    pub fn queens(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::Queen as usize] & self.pieces_of_color(color)
    }

    pub fn kings(&self, color: Color) -> Bitboard {
        self.pieces[PieceType::King as usize] & self.pieces_of_color(color)
    }

    // public mutable methods

    pub fn make_move_unchecked(&mut self, mv: Move) -> Option<PieceType> {
        let source = mv.source().index();
        let destination = mv.dest().index();
        let friendly_color = self.active_color();
        let opponent_color = friendly_color.opposite();
        self.halfmove_number += 1;
        self.metadata.push(
            *self
                .metadata
                .last()
                .expect("Metadata should always be available."),
        );
        let rule50 = self.rule50().saturating_add(1).min(100);
        *self.rule50_mut() = rule50;
        *self.en_passant_square_mut() = None;
        self.flip_color();
        let captured_piece = match mv.move_type() {
            MoveType::Regular => {
                let moving_piece = self
                    .piece_at_index(source)
                    .map(|piece| piece.piece_type)
                    .expect("Source square should have a piece.");
                self.revoke_castling_rights_for_piece(moving_piece, source);
                let captured_piece = self
                    .piece_at_index(destination)
                    .map(|piece| piece.piece_type);
                if moving_piece == PieceType::Pawn {
                    *self.rule50_mut() = 0;
                    if source.abs_diff(destination) == 16 {
                        self.put_en_passant((source + destination) / 2);
                    }
                }
                if let Some(captured_piece) = captured_piece {
                    *self.rule50_mut() = 0;
                    self.remove_piece(captured_piece, opponent_color, destination);
                    self.revoke_castling_rights_for_piece(captured_piece, destination);
                }
                self.remove_piece(moving_piece, friendly_color, source);
                self.put_piece(moving_piece, friendly_color, destination);
                captured_piece
            }
            MoveType::Promotion(promoted_to) => {
                *self.rule50_mut() = 0;
                let captured_piece = self
                    .piece_at_index(destination)
                    .map(|piece| piece.piece_type);
                if let Some(captured_piece) = captured_piece {
                    self.remove_piece(captured_piece, opponent_color, destination);
                    self.revoke_castling_rights_for_piece(captured_piece, destination);
                }
                self.remove_piece(PieceType::Pawn, friendly_color, source);
                self.put_piece(promoted_to, friendly_color, destination);
                captured_piece
            }
            MoveType::EnPassant(target) => {
                *self.rule50_mut() = 0;
                self.remove_piece(PieceType::Pawn, opponent_color, target as usize);
                self.remove_piece(PieceType::Pawn, friendly_color, source);
                self.put_piece(PieceType::Pawn, friendly_color, destination);
                Some(PieceType::Pawn)
            }
            MoveType::Castling(kingside) => {
                self.set_kingside_castling(friendly_color, false);
                self.set_queenside_castling(friendly_color, false);
                let (rook_source, rook_destination) = if kingside {
                    (source + 3, source + 1)
                } else {
                    (source - 4, source - 1)
                };
                self.remove_piece(PieceType::Rook, friendly_color, rook_source);
                self.put_piece(PieceType::Rook, friendly_color, rook_destination);
                self.remove_piece(PieceType::King, friendly_color, source);
                self.put_piece(PieceType::King, friendly_color, destination);
                None
            }
        };
        self.moves.push((mv, captured_piece));
        captured_piece
    }

    pub fn unmake_move(&mut self) -> Move {
        let (mv, captured_piece) = self.moves.pop().expect("Move stack should not be empty.");
        let source = mv.source().index();
        let destination = mv.dest().index();
        let opponent_color = self.active_color();
        let friendly_color = opponent_color.opposite();
        self.metadata.pop();
        self.halfmove_number -= 1;
        self.flip_color();
        match mv.move_type() {
            MoveType::Regular => {
                let moving_piece = self
                    .piece_at_index(destination)
                    .map(|piece| piece.piece_type)
                    .expect("Destination square should have a piece.");
                self.remove_piece_no_zobrist(moving_piece, friendly_color, destination);
                self.put_piece_no_zobrist(moving_piece, friendly_color, source);
                if let Some(captured_piece) = captured_piece {
                    self.put_piece_no_zobrist(captured_piece, opponent_color, destination);
                }
            }
            MoveType::Promotion(promoted_to) => {
                self.remove_piece_no_zobrist(promoted_to, friendly_color, destination);
                self.put_piece_no_zobrist(PieceType::Pawn, friendly_color, source);
                if let Some(captured_piece) = captured_piece {
                    self.put_piece_no_zobrist(captured_piece, opponent_color, destination);
                }
            }
            MoveType::EnPassant(target) => {
                self.remove_piece_no_zobrist(PieceType::Pawn, friendly_color, destination);
                self.put_piece_no_zobrist(PieceType::Pawn, friendly_color, source);
                self.put_piece_no_zobrist(PieceType::Pawn, opponent_color, target as usize);
            }
            MoveType::Castling(kingside) => {
                let (rook_source, rook_destination) = if kingside {
                    (source + 3, source + 1)
                } else {
                    (source - 4, source - 1)
                };
                self.remove_piece_no_zobrist(PieceType::Rook, friendly_color, rook_destination);
                self.put_piece_no_zobrist(PieceType::Rook, friendly_color, rook_source);
                self.remove_piece_no_zobrist(PieceType::King, friendly_color, destination);
                self.put_piece_no_zobrist(PieceType::King, friendly_color, source);
            }
        }
        mv
    }

    // private mutable getters

    fn rule50_mut(&mut self) -> &mut u8 {
        &mut self
            .metadata
            .last_mut()
            .expect("Metadata should always be available.")
            .rule50
    }

    fn en_passant_square_mut(&mut self) -> &mut Option<Square> {
        &mut self
            .metadata
            .last_mut()
            .expect("Metadata should always be available.")
            .ep_square
    }

    fn zobrist_mut(&mut self) -> &mut u64 {
        &mut self
            .metadata
            .last_mut()
            .expect("Metadata should always be available.")
            .zobrist
    }

    fn castling_rights_mut(&mut self) -> &mut CastlingRights {
        &mut self
            .metadata
            .last_mut()
            .expect("Metadata should always be available.")
            .castling_rights
    }

    // private mutable methods

    fn flip_color(&mut self) {
        self.active_color = self.active_color.opposite();
        *self.zobrist_mut() ^= zobrist::color();
    }

    fn set_kingside_castling(&mut self, color: Color, value: bool) {
        if self.castling_rights().kingside(color) != value {
            self.castling_rights_mut().set_kingside(color, value);
            *self.zobrist_mut() ^= zobrist::kingside_castling(color);
        }
    }

    fn set_queenside_castling(&mut self, color: Color, value: bool) {
        if self.castling_rights().queenside(color) != value {
            self.castling_rights_mut().set_queenside(color, value);
            *self.zobrist_mut() ^= zobrist::queenside_castling(color);
        }
    }

    fn revoke_castling_rights_for_piece(&mut self, piece_type: PieceType, square: usize) {
        match (piece_type, square) {
            (PieceType::King, 4) => {
                self.set_kingside_castling(Color::White, false);
                self.set_queenside_castling(Color::White, false);
            }
            (PieceType::King, 60) => {
                self.set_kingside_castling(Color::Black, false);
                self.set_queenside_castling(Color::Black, false);
            }
            (PieceType::Rook, 0) => self.set_queenside_castling(Color::White, false),
            (PieceType::Rook, 7) => self.set_kingside_castling(Color::White, false),
            (PieceType::Rook, 56) => self.set_queenside_castling(Color::Black, false),
            (PieceType::Rook, 63) => self.set_kingside_castling(Color::Black, false),
            _ => {}
        }
    }

    /// unsafe: does not check for existing pieces on the square
    fn put_piece(&mut self, piece_type: PieceType, color: Color, square: usize) {
        self.pieces[piece_type as usize].set(square);
        self.colors[color as usize].set(square);
        *self.zobrist_mut() ^= zobrist::piece_square(piece_type, color, square);
    }

    /// unsafe: does not check for existing pieces on the square
    fn put_piece_no_zobrist(&mut self, piece_type: PieceType, color: Color, square: usize) {
        self.pieces[piece_type as usize].set(square);
        self.colors[color as usize].set(square);
    }

    /// unsafe: assumes piece is present on the square
    fn remove_piece(&mut self, piece_type: PieceType, color: Color, square: usize) {
        self.pieces[piece_type as usize].clear(square);
        self.colors[color as usize].clear(square);
        *self.zobrist_mut() ^= zobrist::piece_square(piece_type, color, square);
    }

    /// unsafe: assumes piece is present on the square
    fn remove_piece_no_zobrist(&mut self, piece_type: PieceType, color: Color, square: usize) {
        self.pieces[piece_type as usize].clear(square);
        self.colors[color as usize].clear(square);
    }

    /// unsafe: assumes en passant square is unset
    fn put_en_passant(&mut self, square: usize) {
        *self.en_passant_square_mut() = Some(Square::from_index_unchecked(square));
        *self.zobrist_mut() ^= zobrist::en_passant(square);
    }
}

impl Display for Board {
    /// Stockfish-style board representation
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Board: {}\n", self.to_fen())?;
        for rank in (0..8).rev() {
            write!(f, " +---+---+---+---+---+---+---+---+\n")?;
            for file in 0..8 {
                let square = rank * 8 + file;
                let piece_char = match self.piece_at_index(square) {
                    Some(piece) => {
                        let c = match piece.piece_type {
                            PieceType::Pawn => 'P',
                            PieceType::Knight => 'N',
                            PieceType::Bishop => 'B',
                            PieceType::Rook => 'R',
                            PieceType::Queen => 'Q',
                            PieceType::King => 'K',
                        };
                        if piece.color == Color::Black {
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

mod zobrist {
    use crate::{Color, PieceType};

    const fn splitmix64_next(state: &mut u64) -> u64 {
        // increment the state
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        // apply MurmurHash3 mixer to state
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    struct ZobristTables {
        pieces: [u64; 12 * 64],
        en_passant: [u64; 64],
        castling: [u64; 4],
        color: u64,
    }
    const fn generate_zobrist_tables(seed: u64) -> ZobristTables {
        let mut state = seed;
        let mut pieces = [0u64; 12 * 64];
        let mut en_passant = [0u64; 64];
        let mut castling = [0u64; 4];

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
        let color = splitmix64_next(&mut state);

        ZobristTables {
            pieces,
            en_passant,
            castling,
            color,
        }
    }

    const ZOBRIST_TABLES: ZobristTables = generate_zobrist_tables(0x9E3779B97F4A7C15u64);

    pub const fn piece_square(piece_type: PieceType, color: Color, square: usize) -> u64 {
        let piece = (color as usize) * 6 + (piece_type as usize);
        ZOBRIST_TABLES.pieces[(piece << 6) | square]
    }

    pub const fn color() -> u64 {
        ZOBRIST_TABLES.color
    }

    pub const fn kingside_castling(color: Color) -> u64 {
        ZOBRIST_TABLES.castling[color as usize]
    }

    pub const fn queenside_castling(color: Color) -> u64 {
        ZOBRIST_TABLES.castling[color as usize + 2]
    }

    pub const fn en_passant(square: usize) -> u64 {
        ZOBRIST_TABLES.en_passant[square]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_indexing_and_colors() {
        let a1 = Square::new(0, 0).expect("a1 is valid");
        let h8 = Square::new(7, 7).expect("h8 is valid");
        let h1 = Square::new(0, 7).expect("h1 is valid");
        let e4 = Square::new(3, 4).expect("e4 is valid");
        assert_eq!(a1.index(), 0);
        assert_eq!(h8.index(), 63);
        assert!(!a1.is_light());
        assert!(h1.is_light());
        assert!(e4.is_light());
        assert_eq!(Square::from_algebraic("e4"), Ok(e4));
        assert_eq!(e4.algebraic(), "e4");
        assert!(Square::new(8, 0).is_err());
        assert!(Square::from_algebraic("i9").is_err());
    }

    #[test]
    fn pieces_lists_starting_position() {
        let mut board = Board::starting();
        let pieces = board.pieces();
        assert_eq!(pieces.len(), 32);
        let e2 = Square::from_algebraic("e2").expect("e2 is valid");
        let e4 = Square::from_algebraic("e4").expect("e4 is valid");
        let e7 = Square::from_algebraic("e7").expect("e7 is valid");
        assert!(pieces.contains(&(e2, Piece::new(PieceType::Pawn, Color::White))));
        assert!(pieces.contains(&(e7, Piece::new(PieceType::Pawn, Color::Black))));

        // The list tracks moves.
        let mv = board
            .legal_moves()
            .iter()
            .find(|mv| mv.source() == e2 && mv.dest() == e4)
            .copied()
            .expect("e2e4 should be legal");
        board.make_move_unchecked(mv);
        let pieces = board.pieces();
        assert_eq!(pieces.len(), 32);
        assert!(!pieces.iter().any(|(square, _)| *square == e2));
        assert!(pieces.contains(&(e4, Piece::new(PieceType::Pawn, Color::White))));
    }
}
