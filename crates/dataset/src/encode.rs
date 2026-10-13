use chess_again_core::{
    Board, CastlingRights, Color, GameResult, Piece, PieceType, Square, UciMove,
};

pub const RECORD_RSM_SIZE: usize = 36;
pub const RECORD_ERSM_SIZE: usize = 37;

/// Number of from/to policy indices: one per `source * 64 + dest` pair.
pub const N_NON_PROMOTION_MOVE_INDICES: usize = 64 * 64;
/// Number of promotion policy indices: one per promotion cell and piece.
pub const N_PROMOTION_MOVE_INDICES: usize = 8 * 8 * 4;
/// Size of the flattened policy vector the model predicts: every from-to pair
pub const N_MOVE_INDICES: usize = N_NON_PROMOTION_MOVE_INDICES + N_PROMOTION_MOVE_INDICES;

/// A type that can be decoded from a fixed-size byte array.
pub trait DecodeableFixedSize: Sized {
    /// Number of bytes the type occupies when encoded.
    const SIZE: usize;

    /// Decodes a value from the first [`Self::SIZE`] bytes of `bytes`.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Fully decodes the record and checks the invariants a valid record must
    /// hold, returning a message describing the first violation.
    ///
    /// This is the single entry point dataset validation uses to decide
    /// whether a record is sound, so every check a record type needs belongs
    /// here rather than in the validator.
    fn validate(&self) -> Result<(), String>;
}

/// Training data record format, without the evaluation:
/// 32 bytes position data + en passant
/// 1 byte (1b side to move + 7b rule50)
/// 1 byte (4b castling rights + 2b repetition flags + 2b result)
/// 2 bytes next move, storing the flattened policy index (13 bits used)
/// Total: 36 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordRSM {
    pub bytes: [u8; RECORD_RSM_SIZE],
}

impl RecordRSM {
    // encodes the position, forcing the side to move to be at the bottom
    pub fn encode(
        pieces: &[(Square, Piece)],
        side_to_move: Color,
        castling_rights: CastlingRights,
        ep_square: Option<Square>,
        rule50: u8,
        repetition_count: u32,
        next_move: UciMove,
        result: GameResult,
    ) -> Self {
        let mut bytes = [0u8; RECORD_RSM_SIZE];
        let repetition = repetition_count.clamp(2, 3) as u8;

        if side_to_move == Color::Black {
            for &(square, piece) in pieces {
                let index = square.mirrored().index();
                let value = piece_value(piece, true);
                if index % 2 == 0 {
                    bytes[index / 2] |= value;
                } else {
                    bytes[index / 2] |= value << 4;
                }
            }
            if let Some(ep_square) = ep_square {
                let ep_square = ep_square.mirrored().index();
                if ep_square % 2 == 1 {
                    bytes[ep_square / 2] |= 13 << 4;
                } else {
                    bytes[ep_square / 2] |= 13;
                }
            }
            // Results are always stored relative to the side to move
            // (0 = win, 1 = draw, 2 = loss) so records stay color symmetric.
            let result = match result {
                GameResult::BlackWin => 0u8,
                GameResult::Draw => 1u8,
                GameResult::WhiteWin => 2u8,
            };
            let castling_rights = castling_rights.flipped().bits();
            bytes[32] = (side_to_move as u8) << 7 | (rule50 as u8);
            bytes[33] = castling_rights << 4 | repetition << 2 | result;
            bytes[34..36]
                .copy_from_slice(&(encode_move(next_move.mirrored()) as u16).to_le_bytes());
            Self { bytes }
        } else {
            for &(square, piece) in pieces {
                let index = square.index();
                let value = piece_value(piece, false);
                if index % 2 == 0 {
                    bytes[index / 2] |= value;
                } else {
                    bytes[index / 2] |= value << 4;
                }
            }
            if let Some(ep_square) = ep_square {
                let ep_square = ep_square.index();
                if ep_square % 2 == 1 {
                    bytes[ep_square / 2] |= 13 << 4;
                } else {
                    bytes[ep_square / 2] |= 13;
                }
            }
            let result = match result {
                GameResult::WhiteWin => 0u8,
                GameResult::Draw => 1u8,
                GameResult::BlackWin => 2u8,
            };
            let castling_rights = castling_rights.bits();
            bytes[32] = (side_to_move as u8) << 7 | (rule50 as u8);
            bytes[33] = castling_rights << 4 | repetition << 2 | result;
            bytes[34..36].copy_from_slice(&(encode_move(next_move) as u16).to_le_bytes());
            Self { bytes }
        }
    }

    /// Decodes the record unnormalized: the side to move is restored.
    pub fn decode(&self) -> Result<RecordRSMDecodeResult, String> {
        self._decode(true)
    }

    pub fn from_board(board: &Board, next_move: UciMove, result: GameResult) -> Self {
        let pieces = board.pieces();
        Self::encode(
            &pieces,
            board.active_color(),
            board.castling_rights(),
            board.ep_square(),
            board.rule50(),
            board.repetition_count(),
            next_move,
            result,
        )
    }

    pub fn to_board(&self) -> Result<(Board, UciMove, GameResult), String> {
        let decoded = self.decode()?;
        let halfmove_number =
            decoded.rule50 as u32 + decoded.rule50 as u32 % 2 + decoded.side_to_move as u32;
        Ok((
            Board::new(
                &decoded.pieces,
                decoded.castling_rights,
                decoded.ep_square,
                halfmove_number,
                decoded.rule50,
            )?,
            decoded.next_move,
            decoded.result,
        ))
    }

    // decodes the position, while keeping it normalized (white is to move)
    pub fn to_board_force_white(&self) -> Result<(Board, Color, UciMove, GameResult), String> {
        let decoded = self._decode(false)?;
        let halfmove_number = decoded.rule50 as u32 + decoded.rule50 as u32 % 2;
        Ok((
            Board::new(
                &decoded.pieces,
                decoded.castling_rights,
                decoded.ep_square,
                halfmove_number,
                decoded.rule50,
            )?,
            decoded.side_to_move,
            decoded.next_move,
            decoded.result,
        ))
    }

    fn _decode(&self, unnormalize: bool) -> Result<RecordRSMDecodeResult, String> {
        let bytes = self.bytes;
        let side_to_move = if bytes[32] >> 7 == 1 {
            Color::Black
        } else {
            Color::White
        };

        if unnormalize && side_to_move == Color::Black {
            let mut pieces = Vec::new();
            let mut ep_square = None;
            for i in 0..32 {
                for (offset, value) in [(0, bytes[i] & 0x0F), (1, bytes[i] >> 4)] {
                    let square = Square::from_index((i * 2 + offset) as u8)
                        .expect("guaranteed valid square")
                        .mirrored();
                    if value == 13 {
                        ep_square = Some(square);
                    } else if value != 0 {
                        pieces.push((square, from_piece_value(value, true)?));
                    }
                }
            }
            let rule50 = (bytes[32] & 0x7F) as u32;
            let castling_rights = CastlingRights::from_bits(bytes[33] >> 4).flipped();
            let repetition_count = (bytes[33] >> 2) & 0b11;
            let result = match bytes[33] & 0b11 {
                0 => GameResult::BlackWin,
                1 => GameResult::Draw,
                2 => GameResult::WhiteWin,
                result => return Err(format!("invalid result value: {result}")),
            };
            let next_move =
                decode_move(u16::from_le_bytes([bytes[34], bytes[35]]) as usize)?.mirrored();
            Ok(RecordRSMDecodeResult {
                pieces,
                side_to_move,
                castling_rights,
                ep_square,
                rule50: rule50 as u8,
                repetition_count: repetition_count as u32,
                next_move,
                result,
            })
        } else {
            let mut pieces = Vec::new();
            let mut ep_square = None;
            for i in 0..32 {
                for (offset, value) in [(0, bytes[i] & 0x0F), (1, bytes[i] >> 4)] {
                    let square = Square::from_index((i * 2 + offset) as u8)
                        .expect("guaranteed valid square");
                    if value == 13 {
                        ep_square = Some(square);
                    } else if value != 0 {
                        pieces.push((square, from_piece_value(value, false)?));
                    }
                }
            }
            let rule50 = (bytes[32] & 0x7F) as u32;
            let castling_rights = CastlingRights::from_bits(bytes[33] >> 4);
            let repetition_count = (bytes[33] >> 2) & 0b11;
            let result = match bytes[33] & 0b11 {
                0 => GameResult::WhiteWin,
                1 => GameResult::Draw,
                2 => GameResult::BlackWin,
                result => return Err(format!("invalid result value: {result}")),
            };
            let next_move = decode_move(u16::from_le_bytes([bytes[34], bytes[35]]) as usize)?;
            Ok(RecordRSMDecodeResult {
                pieces,
                side_to_move,
                castling_rights,
                ep_square,
                rule50: rule50 as u8,
                repetition_count: repetition_count as u32,
                next_move,
                result,
            })
        }
    }
}

impl DecodeableFixedSize for RecordRSM {
    const SIZE: usize = RECORD_RSM_SIZE;

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut record = [0u8; RECORD_RSM_SIZE];
        record.copy_from_slice(&bytes[..RECORD_RSM_SIZE]);
        Self { bytes: record }
    }

    fn validate(&self) -> Result<(), String> {
        // Full decode: every field must survive the round trip.
        self.decode()?;
        // And the record must decode to a position with White to move, which
        // is the normalization the encoder promises.
        let (board, ..) = self.to_board_force_white()?;
        if board.active_color() != Color::White {
            return Err("decoded board does not have White to move".to_string());
        }
        Ok(())
    }
}

pub struct RecordRSMDecodeResult {
    pub pieces: Vec<(Square, Piece)>,
    pub side_to_move: Color,
    pub castling_rights: CastlingRights,
    pub ep_square: Option<Square>,
    pub rule50: u8,
    pub repetition_count: u32,
    pub next_move: UciMove,
    pub result: GameResult,
}

/// Training data record format: a [`RecordRSM`] followed by the evaluation.
/// 36 bytes of position data and next move
/// 1 byte stockfish eval
/// Total: 37 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordERSM {
    pub bytes: [u8; RECORD_ERSM_SIZE],
}

impl RecordERSM {
    // encodes the position, forcing the side to move to be at the bottom
    pub fn encode(
        pieces: &[(Square, Piece)],
        side_to_move: Color,
        castling_rights: CastlingRights,
        ep_square: Option<Square>,
        rule50: u8,
        repetition_count: u32,
        next_move: UciMove,
        eval: f32,
        result: GameResult,
    ) -> Self {
        let rsm = RecordRSM::encode(
            pieces,
            side_to_move,
            castling_rights,
            ep_square,
            rule50,
            repetition_count,
            next_move,
            result,
        );
        let eval = if side_to_move == Color::Black {
            encode_eval(-eval)
        } else {
            encode_eval(eval)
        };

        let mut bytes = [0u8; RECORD_ERSM_SIZE];
        bytes[..RECORD_RSM_SIZE].copy_from_slice(&rsm.bytes);
        bytes[RECORD_RSM_SIZE] = eval;
        Self { bytes }
    }

    /// The record without its evaluation.
    pub fn rsm(&self) -> RecordRSM {
        let mut bytes = [0u8; RECORD_RSM_SIZE];
        bytes.copy_from_slice(&self.bytes[..RECORD_RSM_SIZE]);
        RecordRSM { bytes }
    }

    /// Decodes the record unnormalized: the side to move is restored.
    pub fn decode(&self) -> Result<RecordERSMDecodeResult, String> {
        self._decode(true)
    }

    pub fn from_board(board: &Board, next_move: UciMove, eval: f32, result: GameResult) -> Self {
        let pieces = board.pieces();
        Self::encode(
            &pieces,
            board.active_color(),
            board.castling_rights(),
            board.ep_square(),
            board.rule50(),
            board.repetition_count(),
            next_move,
            eval,
            result,
        )
    }

    pub fn to_board(&self) -> Result<(Board, UciMove, f32, GameResult), String> {
        let decoded = self.decode()?;
        let halfmove_number =
            decoded.rule50 as u32 + decoded.rule50 as u32 % 2 + decoded.side_to_move as u32;
        Ok((
            Board::new(
                &decoded.pieces,
                decoded.castling_rights,
                decoded.ep_square,
                halfmove_number,
                decoded.rule50,
            )?,
            decoded.next_move,
            decoded.eval,
            decoded.result,
        ))
    }

    // decodes the position, while keeping it normalized (white is to move)
    pub fn to_board_force_white(&self) -> Result<(Board, Color, UciMove, f32, GameResult), String> {
        let decoded = self._decode(false)?;
        let halfmove_number = decoded.rule50 as u32 + decoded.rule50 as u32 % 2;
        Ok((
            Board::new(
                &decoded.pieces,
                decoded.castling_rights,
                decoded.ep_square,
                halfmove_number,
                decoded.rule50,
            )?,
            decoded.side_to_move,
            decoded.next_move,
            decoded.eval,
            decoded.result,
        ))
    }

    fn _decode(&self, unnormalize: bool) -> Result<RecordERSMDecodeResult, String> {
        let decoded = self.rsm()._decode(unnormalize)?;
        let eval = decode_eval(self.bytes[RECORD_RSM_SIZE]);
        Ok(RecordERSMDecodeResult {
            pieces: decoded.pieces,
            side_to_move: decoded.side_to_move,
            castling_rights: decoded.castling_rights,
            ep_square: decoded.ep_square,
            rule50: decoded.rule50,
            repetition_count: decoded.repetition_count,
            next_move: decoded.next_move,
            eval: match decoded.side_to_move {
                Color::Black if unnormalize => -eval,
                _ => eval,
            },
            result: decoded.result,
        })
    }
}

impl DecodeableFixedSize for RecordERSM {
    const SIZE: usize = RECORD_ERSM_SIZE;

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut record = [0u8; RECORD_ERSM_SIZE];
        record.copy_from_slice(&bytes[..RECORD_ERSM_SIZE]);
        Self { bytes: record }
    }

    fn validate(&self) -> Result<(), String> {
        self.decode()?;
        let (board, ..) = self.to_board_force_white()?;
        if board.active_color() != Color::White {
            return Err("decoded board does not have White to move".to_string());
        }
        Ok(())
    }
}

pub struct RecordERSMDecodeResult {
    pub pieces: Vec<(Square, Piece)>,
    pub side_to_move: Color,
    pub castling_rights: CastlingRights,
    pub ep_square: Option<Square>,
    pub rule50: u8,
    pub repetition_count: u32,
    pub next_move: UciMove,
    pub eval: f32,
    pub result: GameResult,
}

const fn piece_value(piece: Piece, flip: bool) -> u8 {
    let friendly = match (piece.color, flip) {
        (Color::White, false) | (Color::Black, true) => true,
        (Color::Black, false) | (Color::White, true) => false,
    };
    friendly as u8 * 6u8 + piece.piece_type as u8 + 1
}

fn from_piece_value(value: u8, flip: bool) -> Result<Piece, String> {
    if !(1..=12).contains(&value) {
        return Err(format!("invalid piece value: {}", value));
    }
    let value = value - 1;
    let friendly = (value / 6) != 0;
    let piece_type = match value % 6 {
        0 => PieceType::Pawn,
        1 => PieceType::Knight,
        2 => PieceType::Bishop,
        3 => PieceType::Rook,
        4 => PieceType::Queen,
        5 => PieceType::King,
        _ => unreachable!(),
    };
    let color = match (friendly, flip) {
        (true, false) | (false, true) => Color::White,
        (false, false) | (true, true) => Color::Black,
    };
    Ok(Piece::new(piece_type, color))
}

fn encode_eval(cp: f32) -> u8 {
    (1.0 / (1.0 + (-0.00368208 * cp).exp()) * 255.0).round() as u8
}

fn decode_eval(value: u8) -> f32 {
    -(1.0 / (value as f32 / 255.0).clamp(1e-6, 1.0 - 1e-6) - 1.0).ln() / 0.00368208
}

/// Flattens a move into its index in the model's policy vector.
pub fn encode_move(mv: UciMove) -> usize {
    match mv.promotion() {
        None => mv.source().index() * 64 + mv.dest().index(),
        Some(piece_type) => {
            let piece = match piece_type {
                PieceType::Knight => 0,
                PieceType::Bishop => 1,
                PieceType::Rook => 2,
                PieceType::Queen => 3,
                // Not reachable: UciMove rejects pawn/king promotions.
                PieceType::Pawn | PieceType::King => 4,
            };
            N_NON_PROMOTION_MOVE_INDICES
                + (mv.source().file() as usize * 8 + mv.dest().file() as usize) * 4
                + piece
        }
    }
}

/// Rebuilds a move from its index in the model's policy vector.
pub fn decode_move(index: usize) -> Result<UciMove, String> {
    if index >= N_MOVE_INDICES {
        return Err(format!("move index out of range: {index}"));
    }

    if index < N_NON_PROMOTION_MOVE_INDICES {
        let source = Square::from_index((index / 64) as u8)?;
        let dest = Square::from_index((index % 64) as u8)?;
        return UciMove::new(source, dest, None).map_err(str::to_string);
    }

    let promotion = index - N_NON_PROMOTION_MOVE_INDICES;
    let piece = match promotion % 4 {
        0 => PieceType::Knight,
        1 => PieceType::Bishop,
        2 => PieceType::Rook,
        _ => PieceType::Queen,
    };
    let source = Square::new(6, (promotion / 4 / 8) as u8)?;
    let dest = Square::new(7, (promotion / 4 % 8) as u8)?;
    UciMove::new(source, dest, Some(piece)).map_err(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_again_core::Move;

    #[test]
    fn round_trips_black_to_move_record() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordERSM::from_board(&board, mv.into(), 42.0, GameResult::WhiteWin);

        let (decoded_board, decoded_move, eval, result) =
            record.to_board().expect("record should decode");

        assert_eq!(decoded_board.to_fen(), board.to_fen());
        assert_eq!(decoded_move, UciMove::from(mv));
        assert_eq!(result, GameResult::WhiteWin);
        assert!((eval - 42.0).abs() < 5.0);
    }

    #[test]
    fn decodes_black_to_move_record_from_white_perspective() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordERSM::from_board(&board, mv.into(), 42.0, GameResult::WhiteWin);

        let (decoded_board, side_to_move, decoded_move, white_view_eval, white_view_result) =
            record
                .to_board_force_white()
                .expect("record should decode normalized");

        assert_eq!(side_to_move, Color::Black);
        assert_eq!(
            decoded_board.to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        assert_eq!(decoded_move.mirrored(), UciMove::from(mv));
        assert!((white_view_eval + 42.0).abs() < 5.0);
        assert_eq!(white_view_result, GameResult::BlackWin);
    }

    #[test]
    fn repetition_count_does_not_overflow_into_castling_rights() {
        let mut board = Board::starting();
        for san in ["e4", "e5", "Ke2", "Ke7", "Ke1", "Ke8", "Nf3"] {
            let mv = Move::from_san(san, &board).expect("setup move should be legal");
            board.make_move_unchecked(mv);
        }
        for san in ["Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8", "Nf3"] {
            let mv = Move::from_san(san, &board).expect("repetition move should be legal");
            board.make_move_unchecked(mv);
        }

        assert_eq!(board.active_color(), Color::Black);
        assert_eq!(board.castling_rights().bits(), 0);
        assert_eq!(board.repetition_count(), 3);

        // Complete one more loop to reach four occurrences of the same
        // black-to-move position; the record only has room for two bits.
        for san in ["Nf6", "Ng1", "Ng8", "Nf3"] {
            let mv = Move::from_san(san, &board).expect("repetition move should be legal");
            board.make_move_unchecked(mv);
        }
        assert_eq!(board.active_color(), Color::Black);
        assert_eq!(board.castling_rights().bits(), 0);
        assert_eq!(board.repetition_count(), 4);

        let target = Move::from_san("Nf6", &board).expect("target move should be legal");
        let record = RecordERSM::from_board(&board, target.into(), 0.0, GameResult::Draw);

        assert_eq!(record.bytes[33] >> 4, 0);
        assert_eq!((record.bytes[33] >> 2) & 0b11, 0b11);
        record.decode().expect("decoder should accept record");
        record
            .to_board_force_white()
            .expect("decoder should accept record");
    }

    #[test]
    fn every_legal_non_promotion_and_promotion_move_round_trips() {
        // Sweep real positions, including pawns about to promote, and check
        // that every legal move survives encode -> decode (in the normalized
        // frame) and that no two legal moves in a position share an index.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/7P/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/7p/4K3 b - - 0 1",
            "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
            "rnbqkbnr/ppppPppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).expect("test position should be valid");
            let flip = board.active_color() == Color::Black;
            let mut seen = Vec::new();
            for mv in board.legal_moves() {
                // Sharing one index is only valid inside the normalized frame,
                // so normalize before encoding - exactly what record encoding
                // does - and expect decode to hand back that same move.
                let mv = UciMove::from(mv);
                let mv = if flip { mv.mirrored() } else { mv };
                let index = encode_move(mv);
                assert!(
                    index < N_MOVE_INDICES,
                    "{fen}: {} index {index} in range",
                    mv.to_string()
                );
                assert!(
                    !seen.contains(&index),
                    "{fen}: policy index {index} is not unique"
                );
                seen.push(index);
                assert_eq!(
                    decode_move(index).expect("index should decode"),
                    mv,
                    "{fen}: {} should round trip",
                    mv.to_string()
                );
            }
            assert!(!seen.is_empty(), "{fen} should have legal moves");
        }
    }

    #[test]
    fn rejects_move_indices_out_of_range() {
        assert_eq!(N_NON_PROMOTION_MOVE_INDICES, 4096);
        assert_eq!(N_PROMOTION_MOVE_INDICES, 256);
        assert_eq!(N_MOVE_INDICES, 4352);
        assert!(decode_move(N_MOVE_INDICES).is_err());
        assert!(decode_move(usize::MAX).is_err());
    }

    #[test]
    fn from_bytes_reads_the_record_layout() {
        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        let record = RecordERSM::from_board(&board, mv.into(), 25.0, GameResult::Draw);

        let decoded = RecordERSM::from_bytes(&record.bytes);
        assert_eq!(decoded, record);
        assert_eq!(RecordERSM::SIZE, RECORD_ERSM_SIZE);
        decoded.decode().expect("record should decode");
    }

    #[test]
    fn rsm_is_the_ersm_prefix_without_eval() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let uci = UciMove::from(mv);

        let rsm = RecordRSM::from_board(&board, uci, GameResult::WhiteWin);
        let ersm = RecordERSM::from_board(&board, uci, 42.0, GameResult::WhiteWin);

        // The eval is appended, so the first 36 bytes are identical.
        assert_eq!(&ersm.bytes[..RECORD_RSM_SIZE], &rsm.bytes[..]);
        assert_eq!(ersm.rsm(), rsm);

        // Both decode to the same position, move and result.
        let (rsm_board, rsm_move, rsm_result) = rsm.to_board().expect("rsm should decode");
        let (ersm_board, ersm_move, _, ersm_result) = ersm.to_board().expect("ersm should decode");
        assert_eq!(rsm_board.to_fen(), ersm_board.to_fen());
        assert_eq!(rsm_move, ersm_move);
        assert_eq!(rsm_result, ersm_result);

        let (rsm_norm, _, _, _) = rsm.to_board_force_white().expect("rsm should decode");
        let (ersm_norm, _, _, _, _) = ersm.to_board_force_white().expect("ersm should decode");
        assert_eq!(rsm_norm.to_fen(), ersm_norm.to_fen());
    }

    #[test]
    fn rsm_sizes_match_their_layouts() {
        assert_eq!(RecordRSM::SIZE, RECORD_RSM_SIZE);
        assert_eq!(RecordERSM::SIZE, RECORD_ERSM_SIZE);
        assert_eq!(RECORD_RSM_SIZE + 1, RECORD_ERSM_SIZE);
    }

    #[test]
    fn records_store_the_flattened_move_index() {
        let board = Board::from_fen("4k3/7P/8/8/8/8/8/4K3 w - - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("h8=Q", &board).expect("h8=Q should be legal");
        let uci = UciMove::from(mv);
        let record = RecordERSM::from_board(&board, uci, 0.0, GameResult::Draw);

        // The move lives at the end of the RSM prefix, not the end of the
        // full record, because eval was moved to the last byte.
        let stored = u16::from_le_bytes([record.bytes[34], record.bytes[35]]) as usize;
        assert_eq!(stored, encode_move(uci));
        assert_eq!(record.to_board().expect("record should decode").1, uci);
    }
}
