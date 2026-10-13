use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;

use chess_again_core::{Board, GameResult, Move};

use crate::encode::{RECORD_ERSM_SIZE, RECORD_RSM_SIZE, RecordERSM, RecordRSM};
use crate::read::read_chunk;

const CHUNK_SIZE: usize = 4_194_304;
const RECORD_BUFFER_CAPACITY: usize = CHUNK_SIZE >> 2;

pub struct Config {
    pub inputs: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub records_per_shard: u64,
    pub min_elo: u32,
    pub min_initial_time_seconds: u32,
    pub compression_level: i32,
    pub workers: usize,
    /// When set, records carry the Stockfish evaluation and are written as
    /// [`RecordERSM`]s; otherwise plain [`RecordRSM`]s are written.
    pub eval: bool,
}

impl Config {
    /// Number of bytes in one record for this configuration.
    pub const fn record_size(&self) -> usize {
        if self.eval {
            RECORD_ERSM_SIZE
        } else {
            RECORD_RSM_SIZE
        }
    }
}

impl std::fmt::Display for Config {
    /// Dumps the whole configuration, one `key = value` per line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "inputs = {}", self.inputs.len())?;
        for input in &self.inputs {
            writeln!(f, "input = {}", input.display())?;
        }
        writeln!(f, "out_dir = {}", self.out_dir.display())?;
        writeln!(f, "records_per_shard = {}", self.records_per_shard)?;
        writeln!(f, "min_elo = {}", self.min_elo)?;
        writeln!(
            f,
            "min_initial_time_seconds = {}",
            self.min_initial_time_seconds
        )?;
        writeln!(f, "compression_level = {}", self.compression_level)?;
        writeln!(f, "workers = {}", self.workers)?;
        writeln!(f, "eval = {}", self.eval)?;
        write!(f, "record_size = {}", self.record_size())
    }
}

struct Shard {
    encoder: Encoder<'static, File>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    pub size: u64,
}

impl Shard {
    fn new(path: PathBuf, compression_level: i32) -> Result<Self> {
        let tmp_path = path.with_extension("zstd.tmp");
        let file = File::create(&tmp_path).with_context(|| {
            format!(
                "failed to create temporary shard file: {}",
                tmp_path.display()
            )
        })?;
        let encoder = Encoder::new(file, compression_level)
            .with_context(|| format!("failed to create zstd encoder for {}", tmp_path.display()))?;
        Ok(Self {
            encoder,
            tmp_path,
            final_path: path,
            size: 0,
        })
    }

    fn write(&mut self, data: &[u8]) -> Result<()> {
        self.encoder.write_all(data).with_context(|| {
            format!(
                "failed to write records to shard {}",
                self.tmp_path.display()
            )
        })?;
        self.size += data.len() as u64;
        Ok(())
    }

    fn close(self) -> Result<(PathBuf, u64)> {
        self.encoder.finish().with_context(|| {
            format!("failed to finish writing shard {}", self.tmp_path.display())
        })?;
        std::fs::rename(&self.tmp_path, &self.final_path).with_context(|| {
            format!(
                "failed to rename shard {} to {}",
                self.tmp_path.display(),
                self.final_path.display()
            )
        })?;
        eprintln!("wrote {} bytes to {}", self.size, self.final_path.display());
        Ok((self.final_path, self.size))
    }
}

pub fn run(cfg: Config) -> Result<()> {
    std::fs::create_dir_all(&cfg.out_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", cfg.out_dir))?;

    let (chunk_tx, chunk_rx) = crossbeam_channel::bounded::<String>(2 * cfg.workers);
    let (buf_tx, buf_rx) = crossbeam_channel::bounded::<Vec<u8>>(2 * cfg.workers);

    // shared scope for immutable cfg reference across threads
    std::thread::scope(|scope| -> Result<()> {
        let decompressor = scope.spawn({
            let inputs = &cfg.inputs;
            move || run_decompressor(inputs, chunk_tx)
        });

        let mut workers = Vec::with_capacity(cfg.workers);
        for _ in 0..cfg.workers {
            let chunk_rx = chunk_rx.clone();
            let buf_tx = buf_tx.clone();
            let cfg = &cfg;
            workers.push(scope.spawn(move || run_worker(cfg, chunk_rx, buf_tx)));
        }
        // drop the coordinator's copy
        drop(chunk_rx);
        drop(buf_tx);

        let writer = scope.spawn({
            let cfg = &cfg;
            move || run_writer(cfg, buf_rx)
        });

        decompressor
            .join()
            .map_err(|_| anyhow::anyhow!("decompressor thread panicked"))??;

        let mut games = 0u64;
        for worker in workers {
            games += worker
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread panicked"))??;
        }

        let records = writer
            .join()
            .map_err(|_| anyhow::anyhow!("writer thread panicked"))??;

        eprintln!("processed {games} games; wrote {records} records");
        Ok(())
    })
}

fn run_decompressor(inputs: &[PathBuf], chunk_tx: Sender<String>) -> Result<()> {
    for input in inputs {
        let name = input.display().to_string();
        let file = File::open(input).with_context(|| format!("failed to open {name}"))?;
        let mut decoder = Decoder::new(file)
            .with_context(|| format!("failed to create zstd decoder for {name}"))?;
        let mut buf = vec![0u8; CHUNK_SIZE];
        // Bytes of `buf` holding the tail of a game split across chunks
        let mut carried = 0usize;

        loop {
            let filled = read_chunk(&mut decoder, &mut buf[carried..])
                .with_context(|| format!("failed to read chunk from {name}"))?;
            let total = carried + filled;

            // A short read means the stream ended mid-chunk.
            if filled < CHUNK_SIZE - carried {
                // if the buffer contains a game
                if buf[..total].iter().any(|b| !b.is_ascii_whitespace()) {
                    // Trailing capacity is dropped
                    buf.truncate(total);
                    send_chunk(&chunk_tx, buf)?;
                }
                break;
            }

            let cut = buf[..total]
                .windows(3)
                .rposition(|w| w == b"\n\n[")
                .filter(|&i| i > 0)
                .map(|i| i + 2)
                .with_context(|| {
                    format!(
                        "failed to find game boundary in CHUNK SIZE ({CHUNK_SIZE}) bytes of {name}"
                    )
                })?;

            carried = total - cut;
            let mut next = vec![0u8; CHUNK_SIZE];
            next[..carried].copy_from_slice(&buf[cut..total]);
            buf.truncate(cut);
            send_chunk(&chunk_tx, buf)?;
            buf = next;
        }
    }
    Ok(())
}

fn send_chunk(tx: &Sender<String>, bytes: Vec<u8>) -> Result<()> {
    let text = String::from_utf8(bytes).context("input is not valid UTF-8")?;
    tx.send(text)
        .map_err(|_| anyhow::anyhow!("workers disconnected before input was exhausted"))
}

fn run_worker(cfg: &Config, chunk_rx: Receiver<String>, buf_tx: Sender<Vec<u8>>) -> Result<u64> {
    let mut games = 0u64;

    for chunk in chunk_rx {
        let mut records = Vec::with_capacity(RECORD_BUFFER_CAPACITY);

        for game in chunk.split("\n\n[").filter(|game| !game.trim().is_empty()) {
            let Some((headers, movetext)) = game.split_once("\n\n") else {
                continue;
            };
            if !filter(headers, movetext, cfg) {
                continue;
            }
            games += 1;
            let game_start = records.len();
            let parsed = if cfg.eval {
                parse_movetext_eval(movetext, &mut records)
            } else {
                parse_movetext(movetext, &mut records)
            };
            if let Err(error) = parsed {
                records.truncate(game_start);
                games -= 1;
                eprintln!("Failed to parse movetext: {error}\n{movetext}");
                continue;
            }
        }

        if !records.is_empty() {
            buf_tx
                .send(records)
                .map_err(|_| anyhow::anyhow!("writer disconnected before workers finished"))?;
        }
    }
    Ok(games)
}

fn filter(headers: &str, movetext: &str, cfg: &Config) -> bool {
    // Only the eval parser needs evaluation annotations; the plain parser
    // tolerates their absence (and their presence).
    if cfg.eval && !movetext.contains("[%eval ") {
        return false;
    }

    let mut white_elo = None;
    let mut black_elo = None;
    let mut initial_time_seconds = None;

    for line in headers.lines() {
        let line = line.trim().strip_prefix('[').unwrap_or(line.trim());
        let Some((name, value)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let Some(value) = value
            .trim()
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix("\"]"))
        else {
            continue;
        };
        match name {
            "WhiteElo" => white_elo = value.parse::<u32>().ok(),
            "BlackElo" => black_elo = value.parse::<u32>().ok(),
            "TimeControl" => {
                initial_time_seconds = value
                    .split_once('+')
                    .and_then(|(base, _)| base.parse::<u32>().ok());
            }
            _ => {}
        }
    }

    matches!((white_elo, black_elo, initial_time_seconds),
        (Some(white), Some(black), Some(initial_time))
            if white >= cfg.min_elo
                && black >= cfg.min_elo
                && initial_time >= cfg.min_initial_time_seconds
    )
}

/// Parses any lichess movetext, extracting every SAN move and the result.
///
/// Unlike [`parse_movetext_eval`] this does not require evaluation
/// annotations, so it also accepts games that only carry clock comments.
fn parse_movetext(movetext: &str, records: &mut Vec<u8>) -> Result<(), String> {
    let mut board = Board::starting();

    let result = match movetext.split_whitespace().last() {
        Some("1-0") => GameResult::WhiteWin,
        Some("0-1") => GameResult::BlackWin,
        Some("1/2-1/2") => GameResult::Draw,
        Some("*") => return Ok(()),
        Some(result) => return Err(format!("Invalid or unfinished game result: {result}")),
        None => return Err("Missing game result.".to_string()),
    };

    for token in strip_comments(movetext).split_whitespace() {
        // Comments are gone, so what remains is SAN, move numbers, NAGs and
        // the result. Move numbers look like `12.` / `12...`; castling uses
        // 0-0 / 0-0-0, which must be kept.
        if token.starts_with(|c: char| c.is_ascii_digit() && c != '0') || token.starts_with('$') {
            continue;
        }
        if matches!(token, "1-0" | "0-1" | "1/2-1/2" | "*") {
            break;
        }

        let san = token.trim_end_matches(['!', '?']);
        if san.is_empty() {
            continue;
        }
        let mv = Move::from_san(san, &board)?;
        let record = RecordRSM::from_board(&board, mv.into(), result);
        records.extend_from_slice(&record.bytes);
        board.make_move_unchecked(mv);
    }
    Ok(())
}

/// Removes every `{ ... }` comment group from a movetext.
fn strip_comments(movetext: &str) -> String {
    let mut out = String::with_capacity(movetext.len());
    let mut depth = 0usize;
    for c in movetext.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn parse_movetext_eval(movetext: &str, records: &mut Vec<u8>) -> Result<(), String> {
    let mut board = Board::starting();

    let result = match movetext.split_whitespace().last() {
        Some("1-0") => GameResult::WhiteWin,
        Some("0-1") => GameResult::BlackWin,
        Some("1/2-1/2") => GameResult::Draw,
        Some("*") => return Ok(()),
        Some(result) => return Err(format!("Invalid or unfinished game result: {result}")),
        None => return Err("Missing game result.".to_string()),
    };

    for move_text in movetext.split(". ").skip(1) {
        // extract san and eval
        let Some((san, after_san)) = move_text.split_once(char::is_whitespace) else {
            return Err("Expected a comment after SAN move.".to_string());
        };
        let Some((_, eval)) = after_san.split_once("[%eval ") else {
            break;
        };
        let Some((eval, _)) = eval.split_once(']') else {
            return Err(format!("Invalid evaluation after SAN move: {san}"));
        };

        let san = san.trim_end_matches(['!', '?']);

        // validate san
        let mv = Move::from_san(san, &board)?;
        // validate eval
        let eval = if let Some(mate) = eval.strip_prefix('#') {
            let mate = mate
                .parse::<i32>()
                .map_err(|_| format!("Invalid evaluation after SAN move: {san}"))?;
            if mate.is_negative() {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            let eval = eval
                .parse::<f32>()
                .map_err(|_| format!("Invalid evaluation after SAN move: {san}"))?;
            if !eval.is_finite() {
                return Err(format!("Invalid evaluation after SAN move: {san}"));
            }
            eval * 100.0
        };

        let record = RecordERSM::from_board(&board, mv.into(), eval, result);
        records.extend_from_slice(&record.bytes);
        board.make_move_unchecked(mv);
    }
    Ok(())
}

fn run_writer(cfg: &Config, buf_rx: Receiver<Vec<u8>>) -> Result<u64> {
    let mut shard: Option<Shard> = None;
    let mut shard_index = 0u64;
    let mut records_written = 0u64;
    let mut manifest = Vec::new();

    for records in buf_rx {
        let mut offset = 0;

        while offset < records.len() {
            if shard.is_none() {
                let path = cfg.out_dir.join(format!("shard-{shard_index:06}.zst"));
                shard = Some(Shard::new(path, cfg.compression_level)?);
            }

            let record_size = cfg.record_size();
            let current = shard.as_mut().expect("shard was created above");
            let current_records = current.size / record_size as u64;
            let remaining_records = (cfg.records_per_shard - current_records) as usize;
            let available_records = (records.len() - offset) / record_size;
            let count = remaining_records.min(available_records);
            let byte_count = count * record_size;
            current.write(&records[offset..offset + byte_count])?;
            offset += byte_count;
            records_written += count as u64;

            if count == remaining_records {
                let completed = shard.take().expect("shard was created above");
                manifest.push(completed.close()?);
                shard_index += 1;
            }
        }
    }

    if let Some(shard) = shard {
        manifest.push(shard.close()?);
    }
    write_manifest(cfg, &manifest)?;
    write_meta(cfg)?;
    Ok(records_written)
}

/// Writes the manifest file listing all shard files and their record counts.
fn write_manifest(cfg: &Config, entries: &[(PathBuf, u64)]) -> Result<()> {
    let out_dir = &cfg.out_dir;
    let final_path = out_dir.join("manifest.csv");
    let tmp_path = out_dir.join("manifest.csv.tmp");
    let mut file = File::create(&tmp_path)
        .with_context(|| format!("failed to create manifest {}", tmp_path.display()))?;

    writeln!(file, "filename,records")?;
    for (path, size) in entries {
        let filename = path
            .file_name()
            .expect("shard path must include a filename")
            .to_string_lossy();
        let records = size / cfg.record_size() as u64;
        writeln!(file, "{filename},{records}")?;
    }
    file.flush()
        .with_context(|| format!("failed to write manifest {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "failed to rename manifest {} to {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    eprintln!("wrote manifest to {}", final_path.display());
    Ok(())
}

/// Dumps the run configuration next to the manifest so a dataset can be
/// reproduced from its own metadata.
fn write_meta(cfg: &Config) -> Result<()> {
    let final_path = cfg.out_dir.join("meta");
    let tmp_path = cfg.out_dir.join("meta.tmp");
    let mut file = File::create(&tmp_path)
        .with_context(|| format!("failed to create meta {}", tmp_path.display()))?;
    writeln!(file, "{cfg}")?;
    file.flush()
        .with_context(|| format!("failed to write meta {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "failed to rename meta {} to {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    eprintln!("wrote meta to {}", final_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecodeableFixedSize;
    use crate::encode::RECORD_RSM_SIZE;

    const MOVETEXT_WITH_CLOCKS: &str = "1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:10:00] } 2. Nf3 { [%clk 0:09:59] } 2... Nc6 { [%clk 0:09:59] } 3. Bb5 { [%clk 0:09:58] } 3... a6 1/2-1/2";

    const MOVETEXT_WITH_EVALS: &str = "1. e4 { [%eval 0.18] [%clk 0:01:00] } 1... e5 { [%eval 0.22] [%clk 0:01:00] } 2. Nf3?! { [%eval 0.69] [%clk 0:01:00] } 1-0";

    /// One complete PGN game, shaped so that concatenating many of them
    /// produces the `\n\n[` separator `run_decompressor` splits on.
    fn game(id: usize, padding: usize) -> String {
        let mut game = format!(
            "[Event \"g{id}\"]\n[WhiteElo \"2000\"]\n\n1. e4 {{ [%clk 0:10:00] }} e5 1-0\n\n"
        );
        if padding > 0 {
            // Grow the movetext comment so a game can be sized deliberately.
            let insert = game.len() - 1;
            let mut filler = String::with_capacity(padding);
            filler.push_str("{ ");
            filler.extend(std::iter::repeat_n('x', padding.saturating_sub(3)));
            filler.push_str(" }");
            game.insert_str(insert, &filler);
        }
        game
    }

    /// Writes `raw` as a zstd file and returns the chunks `run_decompressor`
    /// produces for it.
    fn decompress_to_chunks(raw: &str) -> Vec<String> {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "chess-again-decompressor-{unique_id}-{}.pgn.zst",
            raw.len()
        ));
        let compressed =
            zstd::stream::encode_all(raw.as_bytes(), 1).expect("input should compress");
        std::fs::write(&path, compressed).expect("compressed input should be written");

        let (tx, rx) = crossbeam_channel::unbounded::<String>();
        run_decompressor(std::slice::from_ref(&path), tx).expect("decompression should succeed");
        let chunks: Vec<String> = rx.into_iter().collect();
        std::fs::remove_file(&path).expect("temporary input should be removed");
        chunks
    }

    #[test]
    fn decompressor_preserves_bytes_across_chunk_boundaries() {
        // More than two chunks of games, so the split path runs repeatedly and
        // a carried tail crosses a boundary at least twice.
        let mut raw = String::new();
        let mut id = 0;
        while raw.len() < CHUNK_SIZE * 2 + 4096 {
            raw.push_str(&game(id, 0));
            id += 1;
        }

        let chunks = decompress_to_chunks(&raw);

        assert!(
            chunks.len() >= 3,
            "expected several chunks, got {}",
            chunks.len()
        );
        // No byte may be lost, duplicated or reordered by the carry logic.
        assert_eq!(chunks.concat(), raw);
        // Every chunk must end on a game boundary, so nothing is cut in half.
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(
                chunk.ends_with("\n\n"),
                "chunk should end at a game boundary, ends with {:?}",
                &chunk[chunk.len().saturating_sub(8)..]
            );
            assert!(chunk.len() <= CHUNK_SIZE);
        }
    }

    #[test]
    fn decompressor_drops_a_whitespace_only_input() {
        assert!(decompress_to_chunks("\n\n\n   \n").is_empty());
    }

    #[test]
    fn decompressor_reports_a_game_larger_than_a_chunk() {
        // One game alone fills the buffer, leaving no boundary to split on.
        let mut raw = game(0, 0);
        raw.push_str(&game(1, CHUNK_SIZE + 4096));
        raw.push_str(&game(2, 0));

        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("chess-again-oversized-{unique_id}.pgn.zst"));
        let compressed =
            zstd::stream::encode_all(raw.as_bytes(), 1).expect("input should compress");
        std::fs::write(&path, compressed).expect("compressed input should be written");

        let (tx, _rx) = crossbeam_channel::unbounded::<String>();
        let error = run_decompressor(std::slice::from_ref(&path), tx)
            .expect_err("an oversized game should be reported, not panic");
        std::fs::remove_file(&path).expect("temporary input should be removed");

        let message = format!("{error:#}");
        assert!(
            message.contains("failed to find game boundary"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn strips_clock_and_eval_comments() {
        let stripped = strip_comments(MOVETEXT_WITH_CLOCKS);
        assert!(!stripped.contains('{'), "opening braces should be gone");
        assert!(!stripped.contains('}'), "closing braces should be gone");
        assert!(!stripped.contains("%clk"));
        assert!(stripped.contains("1. e4"));
        assert!(stripped.contains("1... e5"));
    }

    #[test]
    fn parses_movetext_without_evals() {
        let mut records = Vec::new();
        parse_movetext(MOVETEXT_WITH_CLOCKS, &mut records).expect("movetext should parse");
        // Six plies, each one 36-byte record.
        assert_eq!(records.len(), 6 * RECORD_RSM_SIZE);
    }

    #[test]
    fn parses_movetext_with_evals_too() {
        let mut records = Vec::new();
        parse_movetext(MOVETEXT_WITH_EVALS, &mut records).expect("movetext should parse");
        assert_eq!(records.len(), 3 * RECORD_RSM_SIZE);
    }

    #[test]
    fn keeps_castling_moves() {
        // Castling is written 0-0 / 0-0-0 and must not be mistaken for a move
        // number, which is why 0 is excluded from the digit filter.
        let movetext =
            "1. e4 { [%clk 0:10:00] } 1... e5 2. Nf3 2... Nc6 3. Bc4 3... Bc5 4. 0-0 1-0";
        let mut records = Vec::new();
        parse_movetext(movetext, &mut records).expect("movetext should parse");
        assert_eq!(records.len(), 7 * RECORD_RSM_SIZE);
    }

    #[test]
    fn skips_unfinished_games() {
        let mut records = Vec::new();
        parse_movetext("1. e4 e5 *", &mut records).expect("unfinished game should be skipped");
        assert!(records.is_empty());
    }

    #[test]
    fn rejects_a_missing_result() {
        let mut records = Vec::new();
        assert!(parse_movetext("1. e4 e5", &mut records).is_err());
    }

    #[test]
    fn rejects_an_illegal_move() {
        let mut records = Vec::new();
        // Qh5 cannot follow e4/e5/Ke2 style nonsense; use a clearly illegal SAN.
        assert!(parse_movetext("1. e4 e5 2. Qz9 1-0", &mut records).is_err());
    }

    #[test]
    fn eval_parser_still_reads_evals() {
        let mut records = Vec::new();
        parse_movetext_eval(MOVETEXT_WITH_EVALS, &mut records).expect("movetext should parse");
        assert_eq!(records.len(), 3 * RECORD_ERSM_SIZE);
        let record = RecordERSM::from_bytes(&records[..RECORD_ERSM_SIZE]);
        record.decode().expect("record should decode");
    }

    #[test]
    fn eval_parser_requires_evals() {
        // The eval parser stops as soon as it runs out of eval comments, which
        // is why the clock-only movetext yields no records.
        let mut records = Vec::new();
        parse_movetext_eval(MOVETEXT_WITH_CLOCKS, &mut records).expect("movetext should parse");
        assert!(records.is_empty());
    }

    #[test]
    fn record_size_follows_the_eval_flag() {
        let mut cfg = Config {
            inputs: Vec::new(),
            out_dir: PathBuf::new(),
            records_per_shard: 1,
            min_elo: 0,
            min_initial_time_seconds: 0,
            compression_level: 1,
            workers: 1,
            eval: true,
        };
        assert_eq!(cfg.record_size(), RECORD_ERSM_SIZE);
        cfg.eval = false;
        assert_eq!(cfg.record_size(), RECORD_RSM_SIZE);
    }
}
