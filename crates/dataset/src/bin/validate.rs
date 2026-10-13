//! Manual dataset validator.
//!
//! There is deliberately no command line parsing: edit the constants below and
//! run
//!
//! ```text
//! cargo run --release -p chess-again-dataset --bin validate
//! ```
//!
//! The record type is chosen by which `validate_dataset` call in `main` is
//! uncommented, so the same binary checks RSM and ERSM datasets.

use std::path::PathBuf;

use anyhow::Result;
use chess_again_dataset::RecordRSM;
use chess_again_dataset::validate_dataset;

/// Dataset directory holding `manifest.csv` and its zstd shards.
const DATASET_DIR: &str = "data/train";

/// Validation workers. `0` means one per available core.
const WORKERS: usize = 0;

fn main() -> Result<()> {
    let workers = if WORKERS == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        WORKERS
    };
    let dataset_dir = PathBuf::from(DATASET_DIR);
    let report = validate_dataset::<RecordRSM>(&dataset_dir, workers)?;

    println!("Dataset: {}", dataset_dir.display());
    println!("Workers: {workers}");
    print!("{report}");

    if report.is_valid() {
        println!("\nDataset validation passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "dataset validation failed: {} decode errors, {} count errors",
            report.decode_errors,
            report.count_errors
        )
    }
}
