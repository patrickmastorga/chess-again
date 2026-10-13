use egui::{Color32, Pos2};

use chess_again_core::{Game, Move, MoveType, Piece, PieceType, Square};

use crate::board::GuiBoard;

const SELECTION_COLOR: Color32 = Color32::from_rgba_premultiplied(10, 46, 16, 140);
const TARGET_COLOR: Color32 = Color32::from_rgba_premultiplied(7, 30, 10, 90);

/// An interactive chess board that plays by the rules of chess
pub struct InteractiveGuiBoard {
    /// The game being played and displayed, public for direct manipulation by the app.
    pub game: Game,
    /// Whether pointer interaction is processed. `true` by default.
    pub enabled: bool,

    /// The display board.
    board: GuiBoard,
    /// The selected square (persisted highlight, toggled by grabbing and
    /// returning a piece to its home square).
    selected: Option<Square>,
    /// The piece currently held under the pointer, and its origin square.
    dragging: Option<(Piece, Square)>,
    /// Whether the pointer button was down on the board last frame, for
    /// press/release edge detection.
    pointer_was_down: bool,
    /// Zobrist hash of the position last frame, so programmatic changes made
    /// by the app (engine replies, new games) are detected and stale
    /// interaction state is dropped.
    last_zobrist: u64,
}

/// What [`InteractiveGuiBoard::show`] reports each frame.
pub struct InteractionResult {
    /// The plain [`egui::Response`] of the underlying board widget.
    pub response: egui::Response,
    /// The move the interaction committed to the game this frame, if any —
    /// the hook for "player moved, engine should reply".
    pub moved: Option<Move>,
}

impl Default for InteractiveGuiBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl InteractiveGuiBoard {
    /// A new interactive board with the standard chess starting position.
    pub fn new() -> Self {
        let game = Game::new();
        let last_zobrist = game.board().zobrist();
        Self {
            game,
            board: GuiBoard::starting_position(),
            enabled: true,
            selected: None,
            dragging: None,
            pointer_was_down: false,
            last_zobrist,
        }
    }

    /// The current perspective.
    pub fn flipped(&self) -> bool {
        self.board.flipped()
    }

    /// Sets the perspective: `false` shows white at the bottom, `true` black.
    pub fn set_flipped(&mut self, flipped: bool) {
        self.board.set_flipped(flipped);
    }

    /// Flips the perspective.
    pub fn flip(&mut self) {
        self.board.flip();
    }

    /// The selected square, if any.
    pub fn selected(&self) -> Option<Square> {
        self.selected
    }

    /// The piece currently held under the pointer, and its origin square.
    pub fn dragging(&self) -> Option<(Piece, Square)> {
        self.dragging
    }

    /// Draws the board, processes pointer input, and updates the game.
    pub fn show(&mut self, ui: &mut egui::Ui) -> InteractionResult {
        // The app may have changed the position between frames (engine reply,
        // new game): drop stale interaction state so highlights and held
        // pieces can't outlive the position they were made against.
        let zobrist = self.game.board().zobrist();
        if zobrist != self.last_zobrist {
            self.selected = None;
            self.dragging = None;
        }
        self.last_zobrist = zobrist;

        // Mirror the game into the display board. While a piece is held, its
        // origin square renders empty and the piece floats under the pointer.
        // (Highlights are re-applied below each frame.)
        self.board.clear_highlights();
        let holding = self.dragging.map(|(_, from)| from);
        self.board.set_pieces(
            &self
                .game
                .board()
                .pieces()
                .into_iter()
                .filter(|(square, _)| Some(*square) != holding)
                .collect::<Vec<_>>(),
        );

        // Highlight the selected (or held) piece and its legal destinations.
        let focus = self
            .selected
            .or_else(|| self.dragging.map(|(_, from)| from));
        if let Some(from) = focus {
            self.board.set_highlight(from, SELECTION_COLOR);
            for mv in self.moves_from(from) {
                self.board.set_highlight(mv.dest(), TARGET_COLOR);
            }
        }

        // A held piece follows the pointer pixel-exactly, lichess-style.
        self.board.hovering_piece = match (self.dragging, self.pointer_pos(ui)) {
            (Some((piece, _)), Some(pos)) => Some((piece, pos)),
            _ => None,
        };

        let response = self.board.show(ui);
        let moved = self.handle_pointer(&response);
        InteractionResult { response, moved }
    }

    /// The pointer position in screen coordinates, if any.
    fn pointer_pos(&self, ui: &egui::Ui) -> Option<Pos2> {
        ui.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()))
    }

    /// Advances the interaction state machine for this frame.
    ///
    /// Press/release edges come from `is_pointer_button_down_on`, which is
    /// true from the press frame onward (no click-vs-drag decision window),
    /// and `interact_pointer_pos` tracks the pointer even outside the board
    /// rect while held.
    fn handle_pointer(&mut self, response: &egui::Response) -> Option<Move> {
        let down = response.is_pointer_button_down_on();
        let pos = response.interact_pointer_pos().or(response.hover_pos());
        let square = pos.and_then(|pos| self.board.square_at(response.rect, pos));
        let mut moved = None;

        if !down && self.pointer_was_down {
            // Mouse-up: a held piece is either dropped or toggled.
            if let Some((_, from)) = self.dragging.take() {
                match square {
                    // Released on the home square: toggle the selection (a
                    // fresh grab selects, re-grabbing the selected piece
                    // deselects).
                    Some(to) if to == from => {
                        self.selected = if self.selected == Some(from) {
                            None
                        } else {
                            Some(from)
                        };
                    }
                    // Released elsewhere: move if legal; a failed drop snaps
                    // back and clears the selection so nothing stays
                    // highlighted. Off-board keeps the selection, like
                    // lichess.
                    Some(to) => {
                        moved = self.try_move(from, to);
                        if to != from {
                            self.selected = None;
                        }
                    }
                    None => {}
                }
            }
        } else if down && !self.pointer_was_down && self.enabled && self.game.result().is_none() {
            // Mouse-down: on a legal destination with a piece selected, move
            // immediately (no mouse-up needed). Otherwise an own piece is
            // picked up and selected; anything else deselects.
            if let Some(square) = square {
                let mut handled = false;
                if let Some(from) = self.selected
                    && from != square
                {
                    moved = self.try_move(from, square);
                    handled = moved.is_some();
                    if handled {
                        self.selected = None;
                    }
                }
                if !handled {
                    let turn = self.game.board().active_color();
                    match self.game.board().piece_at(square) {
                        Some(piece) if piece.color == turn => {
                            self.dragging = Some((piece, square));
                        }
                        _ => self.selected = None,
                    }
                }
            }
        }

        self.pointer_was_down = down;
        moved
    }

    /// The legal moves of the side to move that start on `from`.
    fn moves_from(&self, from: Square) -> Vec<Move> {
        self.game
            .legal_moves()
            .iter()
            .filter(|mv| mv.source() == from)
            .copied()
            .collect()
    }

    /// Finds the legal move matching a from/to pair, auto-queening promotions.
    fn find_move(&self, from: Square, to: Square) -> Option<Move> {
        self.game
            .legal_moves()
            .iter()
            .filter(|mv| mv.source() == from && mv.dest() == to)
            .find(|mv| {
                !matches!(
                    mv.move_type(),
                    MoveType::Promotion(piece) if piece != PieceType::Queen
                )
            })
            .copied()
    }

    /// Plays a legal move if one matches `from`/`to`, returning it.
    fn try_move(&mut self, from: Square, to: Square) -> Option<Move> {
        match self.find_move(from, to) {
            Some(mv) => {
                let _ = self.game.make_move(mv);
                Some(mv)
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_again_core::Color;
    use egui::{Order, Rect, Vec2};

    /// A synthetic response for driving `handle_pointer` without a live ui:
    /// an 800x800 board at the origin, pointer at a square center.
    fn response(down: bool, pos: Option<Pos2>) -> egui::Response {
        egui::Response {
            ctx: egui::Context::default(),
            layer_id: egui::LayerId::new(Order::Background, egui::Id::new("test")),
            id: egui::Id::new("test"),
            rect: Rect::from_min_size(egui::Pos2::ZERO, Vec2::splat(800.0)),
            interact_rect: Rect::from_min_size(egui::Pos2::ZERO, Vec2::splat(800.0)),
            sense: egui::Sense::click_and_drag(),
            interact_pointer_pos_or_nan: pos.unwrap_or(egui::Pos2::NAN),
            intrinsic_size_or_nan: Vec2::ZERO,
            flags: if down {
                egui::response::Flags::ENABLED | egui::response::Flags::IS_POINTER_BUTTON_DOWN_ON
            } else {
                egui::response::Flags::ENABLED
            },
        }
    }

    fn center(square: Square) -> Pos2 {
        let visual_rank = 7 - square.rank();
        egui::pos2(
            (square.file() as f32 + 0.5) * 100.0,
            (visual_rank as f32 + 0.5) * 100.0,
        )
    }

    fn e2() -> Square {
        Square::from_algebraic("e2").expect("e2 is valid")
    }
    fn e4() -> Square {
        Square::from_algebraic("e4").expect("e4 is valid")
    }

    #[test]
    fn drag_commits_legal_move_on_release() {
        let mut igt = InteractiveGuiBoard::new();

        // Press own pawn, release on e4: the move is played.
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(
            igt.dragging(),
            Some((Piece::new(PieceType::Pawn, Color::White), e2()))
        );
        let moved = igt.handle_pointer(&response(false, Some(center(e4()))));
        assert!(moved.is_some());
        assert_eq!(igt.dragging(), None);
        assert_eq!(
            igt.game.board().piece_at(e4()).map(|p| p.piece_type),
            Some(PieceType::Pawn)
        );
        assert_eq!(igt.game.board().piece_at(e2()), None);
        assert_eq!(igt.selected(), None);
    }

    #[test]
    fn release_on_home_toggles_selection() {
        let mut igt = InteractiveGuiBoard::new();

        // Grab and release on the same square: becomes selected.
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(
            igt.handle_pointer(&response(false, Some(center(e2())))),
            None
        );
        assert_eq!(igt.selected(), Some(e2()));

        // Grab and release again: toggles off.
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(
            igt.handle_pointer(&response(false, Some(center(e2())))),
            None
        );
        assert_eq!(igt.selected(), None);
    }

    #[test]
    fn illegal_drop_snaps_back_and_deselects() {
        let mut igt = InteractiveGuiBoard::new();

        // A white rook on a1 cannot move to a3 (blocked by the a2 pawn).
        let a1 = Square::from_algebraic("a1").expect("a1 is valid");
        let a3 = Square::from_algebraic("a3").expect("a3 is valid");
        assert_eq!(igt.handle_pointer(&response(true, Some(center(a1)))), None);
        assert_eq!(igt.handle_pointer(&response(false, Some(center(a3)))), None);
        assert!(igt.game.board().piece_at(a1).is_some());
        assert_eq!(
            igt.selected(),
            None,
            "failed drop must not leave highlights"
        );
    }

    #[test]
    fn disabled_board_ignores_input() {
        let mut igt = InteractiveGuiBoard::new();
        igt.enabled = false;
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(igt.dragging(), None);
        assert_eq!(
            igt.handle_pointer(&response(false, Some(center(e4())))),
            None
        );
        assert_eq!(
            igt.game.board().piece_at(e2()).map(|p| p.piece_type),
            Some(PieceType::Pawn)
        );
    }

    #[test]
    fn mousedown_on_legal_target_moves_immediately() {
        let mut igt = InteractiveGuiBoard::new();

        // Select the e2 pawn (grab + release on home).
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(
            igt.handle_pointer(&response(false, Some(center(e2())))),
            None
        );
        assert_eq!(igt.selected(), Some(e2()));

        // Mousedown on e4 alone must commit the move.
        let moved = igt.handle_pointer(&response(true, Some(center(e4()))));
        assert!(moved.is_some());
        assert_eq!(
            igt.game.board().piece_at(e4()).map(|p| p.piece_type),
            Some(PieceType::Pawn)
        );
        assert_eq!(igt.selected(), None);
    }

    #[test]
    fn programmatic_position_change_clears_stale_state() {
        let mut igt = InteractiveGuiBoard::new();
        assert_eq!(
            igt.handle_pointer(&response(true, Some(center(e2())))),
            None
        );
        assert_eq!(
            igt.handle_pointer(&response(false, Some(center(e2())))),
            None
        );
        assert_eq!(igt.selected(), Some(e2()));

        // The app swaps in a different game; the selection must not survive.
        igt.game =
            Game::starting_at_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
                .expect("legal fen");
        egui::__run_test_ui(|ui| {
            let _ = igt.show(ui);
        });
        assert_eq!(igt.selected(), None);
    }
}
