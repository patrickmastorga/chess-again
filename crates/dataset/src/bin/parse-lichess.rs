use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use chess_again_dataset::parse::{Config, run};

#[derive(Parser)]
#[command(
    name = "parse-lichess",
    about = "lichess .pgn.zst → packed training shards"
)]
struct Args {
    /// One or more input .pgn.zst files.
    #[arg(required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    #[arg(short, long)]
    out_dir: PathBuf,
    /// Write RecordERSMs carrying the Stockfish evaluation. When false, plain
    /// RecordRSMs are written and the movetext does not need eval comments.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    eval: bool,
    #[arg(long, default_value_t = 4_194_304)]
    records_per_shard: u64,
    #[arg(long, default_value_t = 2000)]
    min_elo: u32,
    #[arg(long, default_value_t = 300)]
    min_initial_time_seconds: u32,
    #[arg(long, default_value_t = 3)]
    compression_level: i32,
    /// Number of worker threads to use. Defaults to available CPU parallelism, minimum 1.
    #[arg(long)]
    workers: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let workers = args.workers.unwrap_or_else(default_workers);
    if workers == 0 {
        anyhow::bail!("workers must be greater than zero");
    }
    if args.records_per_shard == 0 {
        anyhow::bail!("records_per_shard must be greater than zero");
    }

    let config = Config {
        inputs: args.input,
        out_dir: args.out_dir,
        records_per_shard: args.records_per_shard,
        min_elo: args.min_elo,
        min_initial_time_seconds: args.min_initial_time_seconds,
        compression_level: args.compression_level,
        workers,
        eval: args.eval,
    };
    run(config)
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(1)
}
