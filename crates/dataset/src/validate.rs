//! Dataset validation.
//!
//! There are exactly two things that can be wrong with a shard:
//!
//! 1. **Count errors** - the record count in the manifest does not match the
//!    number of whole records actually in the shard.
//! 2. **Decode errors** - a record fails [`DecodeableFixedSize::validate`].
//!
//! Shards are decompressed in a stream of record-aligned chunks. Each chunk is
//! owned by the worker that receives it, so no shard is held in memory whole
//! and nothing needs to be shared through an `Arc`.

use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, bounded};

use crate::encode::DecodeableFixedSize;
use crate::dataloader::read_manifest;
use crate::read::read_chunk;

/// Records decoded per chunk. Each chunk becomes one worker job.
const RECORDS_PER_CHUNK: usize = 2048;
/// Errors kept per thread for the report. Counts stay exact regardless: a
/// sample is only pushed while the collecting `Vec` is below this length.
const MAX_SAMPLES: usize = 20;

/// One record that failed to decode.
#[derive(Debug, Clone)]
pub struct DecodeError {
    pub shard: PathBuf,
    /// Index of the record within its shard.
    pub record_index: usize,
    pub message: String,
}

/// One shard whose record count disagrees with the manifest.
#[derive(Debug, Clone)]
pub struct CountError {
    pub shard: PathBuf,
    pub expected_records: u64,
    pub actual_records: usize,
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Whole records found across every shard.
    pub records_checked: usize,
    pub shards_checked: usize,
    pub decode_errors: u64,
    pub count_errors: u64,
    /// A bounded sample of the decode errors, not all of them.
    pub errors: Vec<DecodeError>,
    pub count_mismatches: Vec<CountError>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.decode_errors == 0 && self.count_errors == 0
    }
}

impl Display for ValidationReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Shards checked: {}", self.shards_checked)?;
        writeln!(f, "Records checked: {}", self.records_checked)?;
        writeln!(f, "Decode errors: {}", self.decode_errors)?;
        writeln!(f, "Count errors: {}", self.count_errors)?;

        for error in &self.count_mismatches {
            writeln!(
                f,
                "{}: manifest lists {} records, shard contains {}",
                error.shard.display(),
                error.expected_records,
                error.actual_records
            )?;
        }
        for error in &self.errors {
            writeln!(
                f,
                "{} record {}: {}",
                error.shard.display(),
                error.record_index,
                error.message
            )?;
        }
        if self.errors.len() == MAX_SAMPLES {
            writeln!(f, "(showing at most {MAX_SAMPLES} errors)")?;
        }

        Ok(())
    }
}

/// A record-aligned slice of a decompressed shard, owned by its worker.
struct Chunk<R> {
    shard: PathBuf,
    /// Index of the first record in this chunk, within the shard.
    first_record: usize,
    records: Vec<R>,
}

/// Validates every record in a sharded dataset.
///
/// `R` is the record type and must match the on-disk format: the shard is
/// divided by `R::SIZE`, so validating an RSM dataset as ERSM (or the reverse)
/// reports count errors.
pub fn validate_dataset<R>(
    dataset_dir: impl AsRef<Path>,
    worker_count: usize,
) -> Result<ValidationReport>
where
    R: DecodeableFixedSize + Send + 'static,
{
    if worker_count == 0 {
        bail!("worker count must be greater than zero");
    }

    let dataset_dir = dataset_dir.as_ref();
    let shards = read_manifest(dataset_dir)?;

    let (sender, receiver) = bounded::<Chunk<R>>(worker_count * 2);
    let workers: Vec<_> = (0..worker_count)
        .map(|worker_index| {
            let receiver = receiver.clone();
            thread::Builder::new()
                .name(format!("dataset-validator-{worker_index}"))
                .spawn(move || validate_worker::<R>(receiver))
                .expect("failed to spawn dataset validation worker")
        })
        .collect();
    drop(receiver);

    // This thread is the only one that builds a report; the workers and the
    // producer just return counts plus a bounded sample of what went wrong.
    let produced = produce_chunks::<R>(&shards, &sender);
    drop(sender);

    let mut report = ValidationReport::default();
    for worker in workers {
        let (decode_errors, samples) = worker
            .join()
            .map_err(|_| anyhow::anyhow!("validation worker thread panicked"))?;
        report.decode_errors += decode_errors;
        report.errors.extend(samples);
    }

    let (records_checked, shards_checked, count_errors, count_mismatches) = produced?;
    report.records_checked = records_checked;
    report.shards_checked = shards_checked;
    report.count_errors = count_errors;
    report.count_mismatches = count_mismatches;

    report.errors.sort_by(|a, b| {
        (&a.shard, a.record_index, &a.message).cmp(&(&b.shard, b.record_index, &b.message))
    });
    report.errors.truncate(MAX_SAMPLES);

    Ok(report)
}

/// Streams each shard in record-aligned chunks, checking counts and handing
/// the chunks to the workers.
fn produce_chunks<R>(
    shards: &[(PathBuf, u64)],
    sender: &crossbeam_channel::Sender<Chunk<R>>,
) -> Result<(usize, usize, u64, Vec<CountError>)>
where
    R: DecodeableFixedSize,
{
    // One buffer per shard, reused across chunks; the workers take ownership
    // of what they are sent, so the buffer is only the staging area.
    let chunk_bytes = RECORDS_PER_CHUNK * R::SIZE;

    let mut records_checked = 0usize;
    let mut shards_checked = 0usize;
    let mut count_errors = 0u64;
    let mut count_error_samples = Vec::new();

    for (shard, manifest_records) in shards {
        eprintln!("Validating {}...", shard.display());
        shards_checked += 1;

        let file =
            File::open(shard).with_context(|| format!("failed to open {}", shard.display()))?;
        let mut decoder = zstd::stream::read::Decoder::with_buffer(BufReader::new(file))
            .with_context(|| format!("failed to create zstd decoder for {}", shard.display()))?;

        let mut buffer = vec![0u8; chunk_bytes];
        let mut records_seen = 0usize;
        let mut first_record = 0usize;

        loop {
            let filled = read_chunk(&mut decoder, &mut buffer)
                .with_context(|| format!("failed to decompress {}", shard.display()))?;
            if filled == 0 {
                break;
            }

            // A trailing fragment shorter than one record is a truncated
            // shard; it is dropped rather than decoded.
            let complete = filled / R::SIZE * R::SIZE;
            if complete == 0 {
                break;
            }

            let records: Vec<R> = buffer[..complete]
                .chunks_exact(R::SIZE)
                .map(R::from_bytes)
                .collect();
            records_seen += records.len();

            sender
                .send(Chunk {
                    shard: shard.clone(),
                    first_record,
                    records,
                })
                .map_err(|_| anyhow::anyhow!("validation workers stopped unexpectedly"))?;
            first_record += complete / R::SIZE;

            if filled < buffer.len() {
                break;
            }
        }

        records_checked += records_seen;
        if records_seen as u64 != *manifest_records {
            count_errors += 1;
            if count_error_samples.len() < MAX_SAMPLES {
                count_error_samples.push(CountError {
                    shard: shard.clone(),
                    expected_records: *manifest_records,
                    actual_records: records_seen,
                });
            }
        }
    }

    Ok((
        records_checked,
        shards_checked,
        count_errors,
        count_error_samples,
    ))
}

/// Returns the number of decode errors seen, plus a bounded sample of them.
fn validate_worker<R>(receiver: Receiver<Chunk<R>>) -> (u64, Vec<DecodeError>)
where
    R: DecodeableFixedSize,
{
    let mut decode_errors = 0u64;
    let mut decode_error_samples = Vec::new();

    while let Ok(chunk) = receiver.recv() {
        for (offset, record) in chunk.records.iter().enumerate() {
            if let Err(message) = record.validate() {
                decode_errors += 1;
                if decode_error_samples.len() < MAX_SAMPLES {
                    decode_error_samples.push(DecodeError {
                        shard: chunk.shard.clone(),
                        record_index: chunk.first_record + offset,
                        message,
                    });
                }
            }
        }
    }

    (decode_errors, decode_error_samples)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::RECORD_RSM_SIZE;
    use crate::{RecordERSM, RecordRSM};
    use chess_again_core::{Board, GameResult, Move};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_dir(label: &str) -> PathBuf {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after the Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("chess-again-{label}-{unique_id}"));
        std::fs::create_dir_all(&directory).expect("temporary dataset directory should be created");
        directory
    }

    /// Writes `raw` as a shard with a manifest claiming `manifest_records`.
    fn write_dataset(directory: &Path, raw: &[u8], manifest_records: usize) {
        let compressed = zstd::stream::encode_all(raw, 1).expect("test shard should compress");
        std::fs::write(directory.join("shard-000.zst"), compressed)
            .expect("test shard should be written");
        std::fs::write(
            directory.join("manifest.csv"),
            format!("filename,records\nshard-000.zst,{manifest_records}\n"),
        )
        .expect("test manifest should be written");
    }

    fn one_record() -> [u8; RECORD_RSM_SIZE] {
        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        RecordRSM::from_board(&board, mv.into(), GameResult::Draw).bytes
    }

    #[test]
    fn accepts_a_sound_shard() {
        let directory = temporary_dir("validate-ok");
        let record = one_record();
        write_dataset(&directory, &record.repeat(5), 5);

        let report = validate_dataset::<RecordRSM>(&directory, 2).expect("validation runs");

        assert!(report.is_valid(), "{report}");
        assert_eq!(report.records_checked, 5);
        assert_eq!(report.decode_errors, 0);
        assert_eq!(report.count_errors, 0);
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }

    #[test]
    fn reports_a_decode_error_with_the_right_index() {
        let directory = temporary_dir("validate-decode");
        let mut raw = one_record().repeat(4);
        // Corrupt the third record only.
        raw[2 * RECORD_RSM_SIZE] = 14; // invalid piece value
        write_dataset(&directory, &raw, 4);

        let report = validate_dataset::<RecordRSM>(&directory, 2).expect("validation runs");

        assert!(!report.is_valid());
        assert_eq!(report.decode_errors, 1);
        assert_eq!(report.count_errors, 0);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].record_index, 2);
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }

    #[test]
    fn reports_a_count_mismatch_without_a_decode_error() {
        let directory = temporary_dir("validate-count");
        // Two whole records on disk, but the manifest claims five.
        write_dataset(&directory, &one_record().repeat(2), 5);

        let report = validate_dataset::<RecordRSM>(&directory, 2).expect("validation runs");

        assert!(!report.is_valid());
        assert_eq!(report.count_errors, 1);
        assert_eq!(report.decode_errors, 0);
        assert_eq!(report.count_mismatches.len(), 1);
        assert_eq!(report.count_mismatches[0].expected_records, 5);
        assert_eq!(report.count_mismatches[0].actual_records, 2);
        // The records themselves are fine, so the total is still reported.
        assert_eq!(report.records_checked, 2);
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }

    #[test]
    fn spans_multiple_chunks_with_stable_indices() {
        let directory = temporary_dir("validate-chunks");
        // More than one chunk, with a bad record in the second chunk.
        let total = RECORDS_PER_CHUNK + 8;
        let mut raw = one_record().repeat(total);
        raw[RECORDS_PER_CHUNK * RECORD_RSM_SIZE] = 14;
        write_dataset(&directory, &raw, total);

        let report = validate_dataset::<RecordRSM>(&directory, 3).expect("validation runs");

        assert_eq!(report.records_checked, total);
        assert_eq!(report.decode_errors, 1);
        assert_eq!(report.errors[0].record_index, RECORDS_PER_CHUNK);
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }

    #[test]
    fn a_record_size_mismatch_shows_up_as_a_count_error() {
        let directory = temporary_dir("validate-ersm");
        let record = one_record();
        // Records written as RSM (36 bytes) but validated as ERSM (37 bytes),
        // so the shard does not divide evenly and the counts disagree.
        write_dataset(&directory, &record.repeat(4), 4);

        let report = validate_dataset::<RecordERSM>(&directory, 2).expect("validation runs");

        assert_eq!(report.count_errors, 1);
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }

    #[test]
    fn rejects_a_zero_worker_count() {
        let directory = temporary_dir("validate-workers");
        assert!(validate_dataset::<RecordRSM>(&directory, 0).is_err());
        std::fs::remove_dir_all(directory).expect("temporary dataset should be removed");
    }
}
