use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use clap::Parser;
use eframe::egui;
use egui::Color32;

use chess_again_core::{Color, Engine, Game, GameResult, Move, Score, UciEngine};
use chess_again_gui::InteractiveGuiBoard;

const ENGINE_MOVETIME_MS: u64 = 1_000;
/// How often egui wakes up to poll the engine while it is searching.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "stockfish")]
    engine: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([720.0, 750.0]),
        ..Default::default()
    };

    let engine_path = args.engine.display().to_string();
    let mut engine =
        UciEngine::new(&args.engine).with_context(|| format!("spawning {engine_path}"))?;
    let engine_name = engine.name.clone();
    let mut board = InteractiveGuiBoard::new();
    let mut status = String::new();
    let mut eval: Option<String> = None;
    let mut thinking = false;

    // A fresh game can leave the engine to move first.
    if is_engine_turn(&board) {
        match engine.start_search_async(&board.game.board().to_fen(), Some(ENGINE_MOVETIME_MS)) {
            Ok(()) => thinking = true,
            Err(error) => status = error,
        }
    }

    let result = eframe::run_ui_native("engine demo", native_options, move |ui, _frame| {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut restart = ui.button("New game").clicked();
                if ui.button("Flip board").clicked() {
                    board.flip();
                    restart = true;
                }
                if restart {
                    // End any search before the board moves on; `stop` blocks
                    // until the engine is idle, so no reply can arrive later.
                    let stopped = engine.stop();
                    thinking = false;
                    eval = None;
                    board.game = Game::new();
                    if is_engine_turn(&board) {
                        match engine.start_search_async(
                            &board.game.board().to_fen(),
                            Some(ENGINE_MOVETIME_MS),
                        ) {
                            Ok(()) => thinking = true,
                            Err(error) => status = error,
                        }
                    }
                    match (stopped, status.is_empty()) {
                        (Err(error), _) => status = error,
                        (Ok(_), true) => {
                            status = if board.flipped() {
                                "New game — you play Black.".to_owned()
                            } else {
                                "New game — you play White.".to_owned()
                            }
                        }
                        (Ok(_), false) => {}
                    }
                }

                if let Some(result) = board.game.result() {
                    ui.colored_label(
                        match (result, engine_color(&board)) {
                            (GameResult::WhiteWin, Color::White) => Color32::RED,
                            (GameResult::WhiteWin, Color::Black) => Color32::GREEN,
                            (GameResult::BlackWin, Color::White) => Color32::GREEN,
                            (GameResult::BlackWin, Color::Black) => Color32::RED,
                            (GameResult::Draw, _) => Color32::YELLOW,
                        },
                        match result {
                            GameResult::WhiteWin => "White wins",
                            GameResult::BlackWin => "Black wins",
                            GameResult::Draw => "Draw",
                        },
                    );
                } else {
                    ui.label(format!(
                        "{} to move",
                        match board.game.board().active_color() {
                            Color::White => "White",
                            Color::Black => "Black",
                        }
                    ));
                }
                ui.separator();
                if thinking {
                    ui.weak(format!("{engine_name} is thinking…"));
                } else if let Some(eval) = &eval {
                    ui.weak(eval);
                } else {
                    ui.weak(&status);
                }
            });

            if thinking {
                ui.ctx().request_repaint_after(POLL_INTERVAL);
            }

            match engine.poll() {
                Ok(Some(result)) => {
                    thinking = false;
                    eval = result
                        .eval
                        .map(|score| format_eval(score, engine_color(&board)));
                    match Move::from_uci(result.best_move, board.game.board())
                        .map_err(|error| error.to_string())
                        .and_then(|mv| board.game.make_move(mv))
                    {
                        Ok(_) => {}
                        Err(error) => status = error,
                    }
                }
                Ok(None) => {}
                Err(error) => status = error,
            }

            board.enabled = !is_engine_turn(&board) && !thinking;
            let interaction = board.show(ui);

            if interaction.moved.is_some()
                && board.game.result().is_none()
                && !board.game.legal_moves().is_empty()
            {
                match engine
                    .start_search_async(&board.game.board().to_fen(), Some(ENGINE_MOVETIME_MS))
                {
                    Ok(()) => thinking = true,
                    Err(error) => status = error,
                }
            }
        });
    });
    result.map_err(|error| anyhow::anyhow!("eframe: {error}"))
}

/// The side the engine plays: the one at the top of the board.
fn engine_color(board: &InteractiveGuiBoard) -> Color {
    if board.flipped() {
        Color::White
    } else {
        Color::Black
    }
}

fn is_engine_turn(board: &InteractiveGuiBoard) -> bool {
    board.game.result().is_none() && board.game.board().active_color() == engine_color(board)
}

/// Renders a score from White's point of view, e.g. "+0.32" or "mate in 3".
fn format_eval(score: Score, engine_color: Color) -> String {
    // Scores are from the engine's point of view.
    let white_pov = |value: i64| {
        if engine_color == Color::Black {
            -value
        } else {
            value
        }
    };
    match score {
        Score::Cp(cp) => format!("{:+.2}", white_pov(cp) as f64 / 100.0),
        Score::Mate(n) => match white_pov(n) {
            n if n > 0 => format!("mate in {n} for White"),
            n if n < 0 => format!("mate in {} for Black", -n),
            _ => "mate".to_owned(),
        },
    }
}
