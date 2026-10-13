use chess_again_core::{Color, Piece, PieceType};

/// An SVG asset for one piece.
///
/// `uri` is a stable `bytes://` id that egui uses as the texture cache key, so
/// each piece is rasterized once per display size and reused across frames.
pub struct PieceAsset {
    pub uri: &'static str,
    pub bytes: &'static [u8],
}

const KING_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_klt45.svg",
    bytes: include_bytes!("assets/Chess_klt45.svg"),
};
const KING_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_kdt45.svg",
    bytes: include_bytes!("assets/Chess_kdt45.svg"),
};
const QUEEN_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_qlt45.svg",
    bytes: include_bytes!("assets/Chess_qlt45.svg"),
};
const QUEEN_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_qdt45.svg",
    bytes: include_bytes!("assets/Chess_qdt45.svg"),
};
const ROOK_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_rlt45.svg",
    bytes: include_bytes!("assets/Chess_rlt45.svg"),
};
const ROOK_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_rdt45.svg",
    bytes: include_bytes!("assets/Chess_rdt45.svg"),
};
const BISHOP_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_blt45.svg",
    bytes: include_bytes!("assets/Chess_blt45.svg"),
};
const BISHOP_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_bdt45.svg",
    bytes: include_bytes!("assets/Chess_bdt45.svg"),
};
const KNIGHT_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_nlt45.svg",
    bytes: include_bytes!("assets/Chess_nlt45.svg"),
};
const KNIGHT_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_ndt45.svg",
    bytes: include_bytes!("assets/Chess_ndt45.svg"),
};
const PAWN_WHITE: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_plt45.svg",
    bytes: include_bytes!("assets/Chess_plt45.svg"),
};
const PAWN_BLACK: PieceAsset = PieceAsset {
    uri: "bytes://chess-again/gui/assets/Chess_pdt45.svg",
    bytes: include_bytes!("assets/Chess_pdt45.svg"),
};

pub const fn asset(piece: Piece) -> PieceAsset {
    match (piece.piece_type, piece.color) {
        (PieceType::King, Color::White) => KING_WHITE,
        (PieceType::King, Color::Black) => KING_BLACK,
        (PieceType::Queen, Color::White) => QUEEN_WHITE,
        (PieceType::Queen, Color::Black) => QUEEN_BLACK,
        (PieceType::Rook, Color::White) => ROOK_WHITE,
        (PieceType::Rook, Color::Black) => ROOK_BLACK,
        (PieceType::Bishop, Color::White) => BISHOP_WHITE,
        (PieceType::Bishop, Color::Black) => BISHOP_BLACK,
        (PieceType::Knight, Color::White) => KNIGHT_WHITE,
        (PieceType::Knight, Color::Black) => KNIGHT_BLACK,
        (PieceType::Pawn, Color::White) => PAWN_WHITE,
        (PieceType::Pawn, Color::Black) => PAWN_BLACK,
    }
}
