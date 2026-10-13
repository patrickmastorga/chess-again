use egui::{Color32, Pos2, Rect, Sense, Ui, Vec2, vec2};

use crate::pieces;
use chess_again_core::{Color, Piece, PieceType, Square};

/// State for displaying a chess board. Hand it to [`GuiBoard::show`] each frame.
pub struct GuiBoard {
    squares: [Option<Piece>; 64],
    highlights: [Option<Color32>; 64],
    flipped: bool,

    /// Fixed board side length in egui points. `None` sizes the board to fill
    /// the available space (it is always square).
    pub size: Option<f32>,
    /// Fill color of the light squares.
    pub light_color: Color32,
    /// Fill color of the dark squares.
    pub dark_color: Color32,
    /// Piece images are drawn at this fraction of the square size (0..=1).
    pub piece_scale: f32,
    /// A piece drawn floating over the board, centered at a screen position —
    /// e.g. the piece under a dragging cursor. Drawn on top of everything,
    /// pixel-exactly at that position.
    pub hovering_piece: Option<(Piece, Pos2)>,
    /// Opacity (0..=1) at which [`GuiBoard::hovering_piece`] is drawn.
    pub hovering_opacity: f32,
}

impl Default for GuiBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl GuiBoard {
    /// An empty board, white perspective.
    pub fn new() -> Self {
        Self {
            squares: [None; 64],
            highlights: [None; 64],
            flipped: false,
            size: None,
            light_color: Color32::from_rgb(240, 217, 181),
            dark_color: Color32::from_rgb(181, 136, 99),
            piece_scale: 1.0,
            hovering_piece: None,
            hovering_opacity: 0.7,
        }
    }

    /// An empty board set up with the standard chess starting position.
    pub fn starting_position() -> Self {
        let mut board = Self::new();
        let back_rank = [
            PieceType::Rook,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Queen,
            PieceType::King,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Rook,
        ];
        for file in 0u8..8 {
            let back = back_rank[file as usize];
            board.set_piece(
                Square::new(0, file).expect("guarenteed valid square"),
                Some(Piece::new(back, Color::White)),
            );
            board.set_piece(
                Square::new(1, file).expect("guarenteed valid square"),
                Some(Piece::new(PieceType::Pawn, Color::White)),
            );
            board.set_piece(
                Square::new(6, file).expect("guarenteed valid square"),
                Some(Piece::new(PieceType::Pawn, Color::Black)),
            );
            board.set_piece(
                Square::new(7, file).expect("guarenteed valid square"),
                Some(Piece::new(back, Color::Black)),
            );
        }
        board
    }

    pub fn set_pieces(&mut self, pieces: &[(Square, Piece)]) {
        self.clear_pieces();
        for (square, piece) in pieces {
            self.set_piece(*square, Some(*piece));
        }
    }

    /// Places a piece on a square; `None` empties the square.
    pub fn set_piece(&mut self, square: Square, piece: Option<Piece>) {
        self.squares[square.index() as usize] = piece;
    }

    pub fn clear_pieces(&mut self) {
        self.squares = [None; 64];
    }

    /// The piece on a square, if any.
    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.squares[square.index() as usize]
    }

    /// All pieces on the board with their squares, in no particular order.
    pub fn pieces(&self) -> Vec<(Square, Piece)> {
        self.squares
            .iter()
            .enumerate()
            .filter_map(|(index, piece)| {
                piece.map(|piece| {
                    (
                        Square::from_index(index as u8).expect("guarenteed valid square"),
                        piece,
                    )
                })
            })
            .collect()
    }

    /// Empties the board (pieces and highlights).
    pub fn clear(&mut self) {
        self.clear_pieces();
        self.clear_highlights();
    }

    /// Colors a square (a translucent `Color32` reads best). Highlighting the
    /// same square twice replaces the earlier highlight.
    pub fn set_highlight(&mut self, square: Square, color: Color32) {
        self.highlights[square.index()] = Some(color);
    }

    /// Removes all square highlights.
    pub fn clear_highlights(&mut self) {
        self.highlights = [None; 64];
    }

    /// The current perspective.
    pub fn flipped(&self) -> bool {
        self.flipped
    }

    /// Sets the perspective: `false` shows white at the bottom, `true` black.
    pub fn set_flipped(&mut self, flipped: bool) {
        self.flipped = flipped;
    }

    /// Flips the perspective.
    pub fn flip(&mut self) {
        self.flipped = !self.flipped;
    }

    /// The square under a screen position (e.g. the pointer), or `None` if the
    /// position is outside the board rect.
    pub fn square_at(&self, board_rect: Rect, pos: Pos2) -> Option<Square> {
        if !board_rect.contains(pos) {
            return None;
        }
        let square_side = board_rect.width() / 8.0;
        let visual_file = ((pos.x - board_rect.left()) / square_side)
            .floor()
            .clamp(0.0, 7.0) as u8;
        let visual_rank = ((pos.y - board_rect.top()) / square_side)
            .floor()
            .clamp(0.0, 7.0) as u8;
        let (rank, file) = if self.flipped {
            (visual_rank, 7 - visual_file)
        } else {
            (7 - visual_rank, visual_file)
        };
        Some(Square::new(rank, file).expect("guarenteed valid square"))
    }

    /// Draws the board and returns the plain [`egui::Response`] of the board
    /// widget.
    pub fn show(&mut self, ui: &mut Ui) -> egui::Response {
        // Pieces are drawn with `egui::Image` on `bytes://` URIs, which needs the
        // egui_extras SVG loader. Installing is idempotent and cheap.
        egui_extras::install_image_loaders(ui.ctx());

        let side = self.size.unwrap_or_else(|| ui.available_size().min_elem());
        // Center the square board in the remaining space (vertical slack on
        // wide windows, horizontal slack on tall ones). The response reports
        // the board rect itself, not the full allocation, so callers that
        // hit-test against `response.rect` (e.g. InteractiveGuiBoard) stay
        // correct.
        let (board_rect, response) = ui
            .centered_and_justified(|ui| {
                ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag())
            })
            .inner;
        let response = response.with_new_rect(board_rect);

        let square_side = side / 8.0;
        let painter = ui.painter_at(board_rect);

        for visual_rank in 0u8..8 {
            for visual_file in 0u8..8 {
                let (rank, file) = if self.flipped {
                    (visual_rank, 7 - visual_file)
                } else {
                    (7 - visual_rank, visual_file)
                };
                let square = Square::new(rank, file).expect("guarenteed valid square");

                let square_rect = Rect::from_min_size(
                    board_rect.left_top()
                        + vec2(
                            visual_file as f32 * square_side,
                            visual_rank as f32 * square_side,
                        ),
                    Vec2::splat(square_side),
                );

                let square_color = if square.is_light() {
                    self.light_color
                } else {
                    self.dark_color
                };
                painter.rect_filled(square_rect, 0, square_color);

                if let Some(color) = self.highlights[square.index() as usize] {
                    painter.rect_filled(square_rect, 0, color);
                }

                if let Some(piece) = self.squares[square.index() as usize] {
                    let piece_rect =
                        square_rect.shrink(0.5 * (1.0 - self.piece_scale) * square_side);
                    let asset = pieces::asset(piece);
                    // `paint_at` rasterizes the SVG at the exact pixel size of
                    // `piece_rect`, so pieces stay crisp at any board size.
                    egui::Image::from_bytes(asset.uri, asset.bytes).paint_at(ui, piece_rect);
                }
            }
        }

        if let Some((piece, pos)) = self.hovering_piece {
            let rect = Rect::from_center_size(pos, Vec2::splat(square_side));
            let piece_rect = rect.shrink(0.5 * (1.0 - self.piece_scale) * square_side);
            let asset = pieces::asset(piece);
            let mut painter = ui.painter_at(ui.clip_rect());
            painter.set_opacity(self.hovering_opacity.clamp(0.0, 1.0));
            egui::Image::from_bytes(asset.uri, asset.bytes).paint_at(ui, piece_rect);
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    #[test]
    fn starting_position_has_correct_pieces() {
        let board = GuiBoard::starting_position();
        assert_eq!(
            board.piece_at(Square::new(0, 4).expect("e1 is valid")),
            Some(Piece::new(PieceType::King, Color::White))
        );
        assert_eq!(
            board.piece_at(Square::new(7, 3).expect("d8 is valid")),
            Some(Piece::new(PieceType::Queen, Color::Black))
        );
        assert_eq!(
            board.piece_at(Square::new(3, 4).expect("e4 is valid")),
            None
        );
    }

    #[test]
    fn square_at_respects_perspective() {
        let mut board = GuiBoard::new();
        let board_rect = Rect::from_min_size(pos2(0.0, 0.0), Vec2::splat(800.0));

        // White perspective: a8 at the top-left, a1 at the bottom-left.
        assert_eq!(
            board.square_at(board_rect, pos2(10.0, 10.0)),
            Some(Square::new(7, 0).expect("a8 is valid"))
        );
        assert_eq!(
            board.square_at(board_rect, pos2(10.0, 790.0)),
            Some(Square::new(0, 0).expect("a1 is valid"))
        );
        assert_eq!(
            board.square_at(board_rect, pos2(790.0, 790.0)),
            Some(Square::new(0, 7).expect("h1 is valid"))
        );
        assert_eq!(board.square_at(board_rect, pos2(900.0, 10.0)), None);

        // Black perspective: the board is rotated 180 degrees, a1 ends up in
        // the top-right corner.
        board.set_flipped(true);
        assert_eq!(
            board.square_at(board_rect, pos2(10.0, 10.0)),
            Some(Square::new(0, 7).expect("h1 is valid"))
        );
        assert_eq!(
            board.square_at(board_rect, pos2(790.0, 10.0)),
            Some(Square::new(0, 0).expect("a1 is valid"))
        );
        assert_eq!(
            board.square_at(board_rect, pos2(790.0, 790.0)),
            Some(Square::new(7, 0).expect("a8 is valid"))
        );
    }

    #[test]
    fn pieces_getter_lists_the_starting_position() {
        let board = GuiBoard::starting_position();
        let pieces = board.pieces();
        assert_eq!(pieces.len(), 32);
        let e2 = Square::new(1, 4).expect("e2 is valid");
        assert!(pieces.contains(&(e2, Piece::new(PieceType::Pawn, Color::White))));
    }
}
