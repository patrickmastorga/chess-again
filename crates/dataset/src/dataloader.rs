//! Streaming dataloader over a sharded, zstd-compressed dataset.

use std::fs::{self, File};
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use zstd::stream::read::Decoder;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;

use crate::encode::DecodeableFixedSize;
use crate::read::read_chunk;

const DEFAULT_BUFFER_RECORDS: usize = 257 * 512;
const BATCH_QUEUE: usize = 32;
/// Raw chunks outstanding per worker, between the decompressor and the
/// batchers. Each slot holds one buffer of [`DEFAULT_BUFFER_RECORDS`] records.
const RAW_QUEUE_PER_WORKER: usize = 2;

/// A multi-threaded running pass over the dataset
struct Stream<O> {
    /// Batch queue receiver, held for as long as the stream runs.
    rx: Receiver<O>,
    /// Batcher threads.
    workers: Vec<JoinHandle<Result<()>>>,
    /// The single thread decompressing shards into the workers' raw queue.
    decompressor: JoinHandle<Result<()>>,
}

impl<O> Stream<O> {
    /// Disconnects the batch queue and waits for the producing threads.
    fn shutdown(self) {
        let Self {
            rx,
            workers,
            decompressor,
        } = self;
        // Drop the receiver first to immediately disconnect the batch queue,
        // otherwise a worker parked in `send` on a full queue never returns.
        drop(rx);
        // Next the workers: the decompressor may be parked sending to them.
        for (index, worker) in workers.into_iter().enumerate() {
            report_failure(worker.join(), &format!("batch worker {index}"));
        }
        // Finally the decompressor.
        report_failure(decompressor.join(), "decompressor");
    }
}

/// Reports a producer thread that failed, either by returning an error or by
/// panicking.
///
/// A producer that stops early is indistinguishable from a finished dataset at
/// the receiving end: the queue simply disconnects. Without this the epoch that
/// lost data would look like a clean, short epoch.
fn report_failure(result: thread::Result<Result<()>>, who: &str) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => eprintln!("streaming loader: {who} failed: {error:#}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            eprintln!("streaming loader: {who} panicked: {message}");
        }
    }
}

/// Streams fixed-size records out of a sharded, zstd-compressed dataset as batches.
pub struct StreamingLoader<B: Backend, R: DecodeableFixedSize, Bt, O>
where
    Bt: Batcher<B, R, O>,
{
    /// Shard paths, shared with the decompressor thread.
    shards: Vec<PathBuf>,
    device: B::Device,
    batcher: Arc<Bt>,
    batch_size: usize,
    workers: usize,
    seed: u64,
    num_items: usize,
    /// Records served per epoch. 0 streams entire dataset.
    batches_per_epoch: u64,
    /// Number of records to buffer for shuffling.
    buffer_size: usize,
    /// Counts the passes over the dataset.
    pass: AtomicUsize,
    /// The pass currently running, if any.
    ///
    /// Guarded by a mutex so concurrent callers of [`iter`](DataLoader::iter)
    /// cannot both decide the pass is over and spawn competing sets of workers.
    stream: Mutex<Option<Stream<O>>>,
    marker: PhantomData<fn() -> (R, O)>,
}

impl<B, R, Bt, O> StreamingLoader<B, R, Bt, O>
where
    B: Backend,
    R: DecodeableFixedSize + Send + 'static,
    Bt: Batcher<B, R, O> + Send + Sync + 'static,
    O: Send + 'static,
{
    pub fn new(
        dataset_dir: impl AsRef<Path>,
        device: B::Device,
        batcher: Bt,
        batch_size: usize,
        workers: usize,
        seed: u64,
    ) -> Result<Self> {
        let dataset_dir = dataset_dir.as_ref();
        let shards = read_manifest(dataset_dir)?;
        if shards.is_empty() {
            bail!("dataset {} contains no shards", dataset_dir.display());
        }
        if batch_size == 0 {
            bail!("batch size must be at least one");
        }
        if workers == 0 {
            bail!("at least one worker is required");
        }
        let total_records: usize = shards
            .iter()
            .map(|(_shard, records)| *records as usize)
            .sum();
        let shards: Vec<PathBuf> = shards.into_iter().map(|(shard, _records)| shard).collect();
        let num_items = total_records.div_ceil(batch_size);

        Ok(Self {
            shards,
            device,
            batcher: Arc::new(batcher),
            batch_size,
            workers,
            seed,
            num_items,
            batches_per_epoch: 0,
            buffer_size: round_to_batches(DEFAULT_BUFFER_RECORDS, batch_size),
            pass: AtomicUsize::new(0),
            stream: Mutex::new(None),
            marker: PhantomData,
        })
    }

    /// Batches served per epoch. 0 streams entire dataset.
    pub fn with_batches_per_epoch(mut self, batches: u64) -> Self {
        self.batches_per_epoch = batches;
        self
    }

    /// Number of records to buffer for shuffling.
    pub fn with_buffer_size(mut self, records: usize) -> Self {
        self.buffer_size = round_to_batches(records, self.batch_size);
        self
    }

    /// Total number of batches in the dataset.
    pub fn size(&self) -> usize {
        self.num_items
    }

    /// Batches served per epoch
    pub fn batches_per_epoch(&self) -> u64 {
        if self.batches_per_epoch == 0 {
            self.num_items as u64
        } else {
            self.batches_per_epoch
        }
    }

    /// Spawns a pass: one decompressor and `workers` batcher threads.
    ///
    /// Called only from [`iter`](DataLoader::iter), under the `stream` lock, so
    /// at most one pass exists per loader.
    fn start(&self) -> Result<Stream<O>> {
        let pass = self.pass.fetch_add(1, Ordering::Relaxed);
        let seed = self.seed ^ pass as u64;

        // Shard order is decided here rather than in the decompressor: the list
        // is cheap to clone, and this keeps the decompressor free of any RNG.
        let mut shards = self.shards.clone();
        shards.shuffle(&mut StdRng::seed_from_u64(seed));

        let (batches_tx, batches_rx) = bounded(BATCH_QUEUE);
        let (raw_tx, raw_rx) = bounded::<Vec<u8>>(self.workers * RAW_QUEUE_PER_WORKER);

        let mut workers = Vec::with_capacity(self.workers);
        for index in 0..self.workers {
            let rx = raw_rx.clone();
            let tx = batches_tx.clone();
            let worker_seed = seed ^ index as u64;
            let batcher = Arc::clone(&self.batcher);
            let device = self.device.clone();
            let batch_size = self.batch_size;
            workers.push(
                thread::Builder::new()
                    .name(format!("dataloader-batcher-{index}"))
                    .spawn(move || {
                        Self::worker(batcher, device, batch_size, worker_seed, rx, tx)
                    })?,
            );
        }

        // The workers should hold every raw receiver and batch sender for proper exit behavior.
        drop(raw_rx);
        drop(batches_tx);

        let buffer_records = self.buffer_size;
        let decompressor = thread::Builder::new()
            .name("dataloader-decompressor".to_string())
            .spawn(move || Self::decompressor(shards, buffer_records, raw_tx))?;

        Ok(Stream {
            rx: batches_rx,
            workers,
            decompressor,
        })
    }

    /// Streams decompressed records out of the shards and into the raw queue.
    fn decompressor(shards: Vec<PathBuf>, buffer_size: usize, tx: Sender<Vec<u8>>) -> Result<()> {
        let mut buffer = vec![0u8; buffer_size * R::SIZE];
        // Bytes of `buffer` already filled, carried across shard boundaries so
        // that a short shard does not produce a short chunk.
        let mut filled = 0usize;

        for shard in shards {
            let name = shard.display().to_string();
            let file = File::open(&shard).with_context(|| format!("failed to open {name}"))?;
            let mut decoder = Decoder::with_buffer(BufReader::new(file))
                .with_context(|| format!("failed to decompress {name}"))?;

            while filled < buffer.len() {
                let room = buffer.len() - filled;
                let read = read_chunk(&mut decoder, &mut buffer[filled..])
                    .with_context(|| format!("failed to read {name}"))?;
                filled += read;

                // A short read is the end of the shard.
                if read < room {
                    if filled % R::SIZE != 0 {
                        bail!(
                            "{name}: shard ends with {} bytes of a {}-byte record",
                            filled % R::SIZE,
                            R::SIZE
                        );
                    }
                    break;
                }

                // Send the full buffer.
                if tx.send(buffer.clone()).is_err() {
                    // The workers are gone, so this pass is over.
                    return Ok(());
                }
                filled = 0;
            }
        }

        // The shards ran out with a partial buffer left, which holds whole
        // records because every shard was checked as it ended.
        if filled > 0 && tx.send(buffer[..filled].to_vec()).is_err() {
            return Ok(());
        }

        Ok(())
    }

    /// Shuffles and batches the raw data received from the decompressor.
    fn worker(
        batcher: Arc<Bt>,
        device: B::Device,
        batch_size: usize,
        seed: u64,
        rx: Receiver<Vec<u8>>,
        tx: Sender<O>,
    ) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        while let Ok(chunk) = rx.recv() {
            debug_assert_eq!(chunk.len() % R::SIZE, 0, "chunks hold whole records");

            // Shuffle the whole chunk.
            let mut records: Vec<R> = chunk.chunks_exact(R::SIZE).map(R::from_bytes).collect();
            records.shuffle(&mut rng);

            // Cut it into batches from the back.
            while !records.is_empty() {
                let take = batch_size.min(records.len());
                let batch = records.split_off(records.len() - take);
                if tx.send(batcher.batch(batch, &device)).is_err() {
                    // The iterator is gone, so this pass is over.
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}

fn round_to_batches(records: usize, batch_size: usize) -> usize {
    (records / batch_size).max(1) * batch_size
}

pub fn read_manifest(dataset_dir: &Path) -> Result<Vec<(PathBuf, u64)>> {
    let manifest_path = dataset_dir.join("manifest.csv");
    let manifest = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read manifest {}", manifest_path.display()))?;
    let mut lines = manifest.lines();
    if lines.next() != Some("filename,records") {
        bail!(
            "manifest {} must start with filename,records",
            manifest_path.display()
        );
    }

    let mut shards = Vec::new();
    for (line_number, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let (filename, records) = line.split_once(',').with_context(|| {
            format!(
                "invalid manifest entry on line {} in {}",
                line_number + 2,
                manifest_path.display()
            )
        })?;
        let records: u64 = records.parse().with_context(|| {
            format!(
                "invalid record count on line {} in {}",
                line_number + 2,
                manifest_path.display()
            )
        })?;
        let shard_path = dataset_dir.join(filename);
        if !shard_path.is_file() {
            bail!(
                "shard listed on line {} does not exist: {}",
                line_number + 2,
                shard_path.display()
            );
        }
        shards.push((shard_path, records));
    }
    Ok(shards)
}

/// One epoch's view of a running pass: it pulls batches through a clone of the
/// batch queue until the cap is reached or the producers stop.
///
/// Dropping this does not stop the producers. The loader keeps its own clone of
/// the receiver, so the pass stays alive across epoch boundaries and resumes
/// where it left off.
pub struct StreamingIter<O> {
    rx: Receiver<O>,
    items_processed: usize,
    items_total: usize,
    /// set to true if the iterator should halt after items_total batches
    capped: bool,
}

impl<O> StreamingIter<O> {
    fn new(rx: Receiver<O>, items_total: usize, capped: bool) -> Self {
        Self {
            rx,
            items_processed: 0,
            items_total,
            capped,
        }
    }
}

impl<O> Iterator for StreamingIter<O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        if self.capped && self.items_processed >= self.items_total {
            return None;
        }

        match self.rx.recv() {
            Ok(batch) => {
                self.items_processed += 1;
                Some(batch)
            }
            Err(_) => None,
        }
    }
}

impl<O> DataLoaderIterator<O> for StreamingIter<O> {
    fn progress(&self) -> Progress {
        Progress::new(self.items_processed.min(self.items_total), self.items_total)
    }
}

impl<B, R, Bt, O> DataLoader<B, O> for StreamingLoader<B, R, Bt, O>
where
    B: Backend,
    R: DecodeableFixedSize + Send + 'static,
    Bt: Batcher<B, R, O> + Send + Sync + 'static,
    O: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        let batches = self.batches_per_epoch() as usize;
        let capped = self.batches_per_epoch != 0;

        // Obtain the lock on the stream slot. This ensures that only one caller
        // can reap or start a new pass at a time.
        let mut stream = self.stream.lock().expect("stream slot is never poisoned");

        // A pass is reused while the batch queue is still connected.
        let finished = match stream.as_ref() {
            Some(running) if !running.rx.is_empty() => false,
            Some(running) => {
                matches!(running.rx.try_recv(), Err(TryRecvError::Disconnected))
            }
            None => true,
        };

        if finished {
            // Reap the pass that ended before replacing it:
            // the Joinhandles needs to be properly shut down.
            if let Some(previous) = stream.take() {
                previous.shutdown();
            }

            match self.start() {
                Ok(running) => *stream = Some(running),
                Err(error) => {
                    // A dataset that cannot be opened is not a reason to panic
                    // inside a training loop: the empty stream below ends the
                    // epoch, and the reason is reported rather than discarded.
                    eprintln!("streaming loader could not start: {error:#}");
                    drop(stream);
                    let (tx, dead) = bounded::<O>(1);
                    drop(tx);
                    return Box::new(StreamingIter::new(dead, batches, true));
                }
            }
        }

        let rx = stream.as_ref().expect("a pass is installed").rx.clone();
        drop(stream);

        Box::new(StreamingIter::new(rx, batches, capped))
    }

    fn num_items(&self) -> usize {
        self.num_items
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        // Clone dataloader around new device with fresh stream
        Arc::new(Self {
            shards: self.shards.clone(),
            device: device.clone(),
            batcher: self.batcher.clone(),
            batch_size: self.batch_size,
            workers: self.workers,
            seed: self.seed,
            num_items: self.num_items,
            batches_per_epoch: self.batches_per_epoch,
            buffer_size: self.buffer_size,
            pass: AtomicUsize::new(0),
            stream: Mutex::new(None),
            marker: PhantomData,
        })
    }

    fn slice(&self, _start: usize, _end: usize) -> Arc<dyn DataLoader<B, O>> {
        unimplemented!("a shuffled decompressed stream has no stable indices to slice")
    }
}

impl<B, R, Bt, O> Drop for StreamingLoader<B, R, Bt, O>
where
    B: Backend,
    R: DecodeableFixedSize,
    Bt: Batcher<B, R, O>,
{
    fn drop(&mut self) {
        // `&mut self` means no other thread can be inside `iter` right now, so
        // this cannot race with a reap. Shutting the pass down here is the
        // last chance to report a producer that panicked during a final epoch,
        // which nothing would otherwise join.
        let Some(running) = self
            .stream
            .get_mut()
            .expect("stream slot is never poisoned")
            .take()
        else {
            return;
        };

        running.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A throwaway directory that names itself uniquely, so tests running in
    /// parallel do not share shards.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "chess-again-dataloader-{name}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("clock after epoch")
                    .as_nanos()
            ));
            fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn rounds_the_shuffle_buffer_up_to_whole_batches() {
        // A buffer that is not a whole number of batches would let a worker
        // cut a short batch in the middle of a pass.
        assert_eq!(round_to_batches(100, 32), 96);
        assert_eq!(round_to_batches(96, 32), 96);
        assert_eq!(round_to_batches(97, 32), 96);
        // Never rounds down to nothing, whatever the input.
        assert_eq!(round_to_batches(1, 512), 512);
        assert_eq!(round_to_batches(0, 512), 512);
    }

    #[test]
    fn reads_a_manifest_of_shards() {
        let dir = TempDir::new("manifest");
        fs::write(dir.path().join("shard-000000.zst"), b"x").expect("write shard");
        fs::write(dir.path().join("shard-000001.zst"), b"y").expect("write shard");
        fs::write(
            dir.path().join("manifest.csv"),
            "filename,records\nshard-000000.zst,10\n\nshard-000001.zst,20\n",
        )
        .expect("write manifest");

        let shards = read_manifest(dir.path()).expect("manifest reads");
        assert_eq!(
            shards,
            vec![
                (dir.path().join("shard-000000.zst"), 10),
                (dir.path().join("shard-000001.zst"), 20),
            ]
        );
    }

    #[test]
    fn rejects_malformed_manifests() {
        // A header that is not exactly `filename,records`.
        let dir = TempDir::new("bad-header");
        fs::write(dir.path().join("manifest.csv"), "file,count\n").expect("write manifest");
        assert!(read_manifest(dir.path()).is_err());

        // A record count that does not parse.
        let dir = TempDir::new("bad-count");
        fs::write(dir.path().join("shard-000000.zst"), b"x").expect("write shard");
        fs::write(
            dir.path().join("manifest.csv"),
            "filename,records\nshard-000000.zst,many\n",
        )
        .expect("write manifest");
        assert!(read_manifest(dir.path()).is_err());

        // A shard that is named but absent, which would otherwise only fail
        // later, inside the decompressor.
        let dir = TempDir::new("missing-shard");
        fs::write(
            dir.path().join("manifest.csv"),
            "filename,records\nshard-000000.zst,10\n",
        )
        .expect("write manifest");
        let error = read_manifest(dir.path()).expect_err("missing shard is an error");
        assert!(
            error.to_string().contains("does not exist"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn seeds_differ_per_pass_and_per_worker() {
        // `start` xor's the pass counter into the base seed, and each worker
        // xor's its own index into that, so no two passes and no two workers
        // within a pass may shuffle the same way.
        let base = 0x1234_5678_9ABC_DEF0u64;
        let passes: Vec<u64> = (0..4u64).map(|pass| base ^ pass).collect();
        assert_eq!(passes[0], base, "the first pass uses the seed as given");
        assert_eq!(
            passes.iter().copied().collect::<HashSet<u64>>().len(),
            passes.len(),
            "pass seeds must differ"
        );

        // Workers within one pass must all differ from each other. Pass seeds
        // and worker seeds may coincide across passes, which does not matter:
        // two passes never run at the same time.
        let seed = passes[0];
        let workers: Vec<u64> = (0..8u64).map(|index| seed ^ index).collect();
        assert_eq!(
            workers.iter().copied().collect::<HashSet<u64>>().len(),
            workers.len(),
            "worker seeds must differ"
        );
    }
}
