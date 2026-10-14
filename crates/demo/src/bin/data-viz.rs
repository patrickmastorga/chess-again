use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::data::dataloader::DataLoader;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::Device;
use clap::Parser;
use eframe::egui::{self, Color32};

use chess_again_core::{Board, GameResult, UciMove};
use chess_again_dataset::{RecordRSM, StreamingLoader};
use chess_again_gui::GuiBoard;

const SOURCE_HIGHLIGHT_COLOR: Color32 = Color32::from_rgba_premultiplied(16, 75, 35, 150);
const DEST_HIGHLIGHT_COLOR: Color32 = Color32::from_rgba_premultiplied(35, 140, 60, 150);

#[derive(Parser)]
struct Args {
    #[arg(default_value = "data/validate")]
    dataset_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Debug, Clone)]
struct VizBatcher;

impl Batcher<RecordRSM, Vec<(Board, UciMove, GameResult)>> for VizBatcher {
    fn batch(
        &self,
        records: Vec<RecordRSM>,
        _device: &Device,
    ) -> Vec<(Board, UciMove, GameResult)> {
        records
            .into_iter()
            .map(|record| {
                record
                    .to_board()
                    .expect("training records must decode successfully")
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Built without a device, then bound to one: `to_device` is how a loader
    // learns which device to assemble batches on.
    let loader = StreamingLoader::<RecordRSM, VizBatcher, Vec<(Board, UciMove, GameResult)>>::new(
        &args.dataset_dir,
        VizBatcher,
        1,
        1,
        1,
        args.seed,
    )?
    .to_device(&Device::flex());
    // The UI closure is `'static`, so the loader it iterates has to be owned by
    // the closure too, and the iterator is built inside it.
    let mut current = loader
        .iter()
        .next()
        .transpose()?
        .and_then(|mut batch| batch.pop())
        .with_context(|| format!("dataset {} has no records", args.dataset_dir.display()))?;
    let mut board = GuiBoard::new();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([760.0, 800.0]),
        ..Default::default()
    };

    eframe::run_ui_native("Dataset visualizer", native_options, move |ui, _frame| {
        let mut examples = loader.iter();
        egui::CentralPanel::default().show(ui, |ui| {
            ui.label(format!("FEN: {}", current.0.to_fen()));
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Result: {}",
                    match current.2 {
                        GameResult::WhiteWin => "White win",
                        GameResult::BlackWin => "Black win",
                        GameResult::Draw => "Draw",
                    }
                ));
                ui.separator();
                ui.label(format!("Move: {}", current.1));
            });

            board.set_pieces(&current.0.pieces());
            board.clear_highlights();
            board.set_highlight(current.1.source(), SOURCE_HIGHLIGHT_COLOR);
            board.set_highlight(current.1.dest(), DEST_HIGHLIGHT_COLOR);
            if board.show(ui).clicked() {
                if let Some(next) = examples
                    .next()
                    .transpose()
                    .ok()
                    .flatten()
                    .and_then(|mut batch| batch.pop())
                {
                    current = next;
                    ui.ctx().request_repaint();
                }
            }
        });
    })
    .map_err(|error| anyhow::anyhow!("eframe: {error}"))
}
