//! Streaming dataloader over a sharded, zstd-compressed dataset.

use std::fs::{self, File};
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use zstd::stream::read::Decoder;

use burn::data::dataloader::batcher::Batcher;

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::data::dataset::DatasetError;
use burn::tensor::Device;

use crate::encode::Decodeable;
use crate::read::read_chunk;

const DEFAULT_WINDOW_BATCHES: usize = 256;
const BATCH_QUEUE: usize = 64;

/// Delegator for shard order for worker threads.
struct ShardOrderer {
    shards: Vec<PathBuf>,
    order: Vec<usize>,
    rng: StdRng,
}

impl ShardOrderer {
    fn new(shards: Vec<PathBuf>, seed: u64) -> Self {
        assert!(!shards.is_empty(), "no shards provided");
        Self {
            shards,
            order: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn next_shard(&mut self) -> PathBuf {
        if self.order.is_empty() {
            self.order = (0..self.shards.len()).collect();
            self.order.shuffle(&mut self.rng);
        }
        self.order
            .pop()
            .map(|index| self.shards[index].clone())
            .expect("order is not empty.")
    }
}

/// A multi-threaded stream over the dataset
struct Stream<T> {
    /// Receiver for the stream of items, always held so that the stream remains connected.
    rx: Receiver<T>,
    workers: Vec<JoinHandle<Result<()>>>,
}

impl<T> Stream<T> {
    /// Disconnects the item queue and waits for the producing threads,
    /// returning the first failure among them.
    fn shutdown(self) -> Result<()> {
        let Self { rx, workers } = self;
        // Drop the receiver first to immediately disconnect the item queue,
        // otherwise a worker parked in `send` on a full queue never returns.
        drop(rx);
        // Next join all the workers
        let mut failure = None;
        for (index, worker) in workers.into_iter().enumerate() {
            if let Err(error) = report_failure(worker.join(), &format!("worker {index}")) {
                failure.get_or_insert(error);
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Reports a producer thread that failed, either by returning an error or by
/// panicking, both to the caller and to the log.
fn report_failure(result: thread::Result<Result<()>>, who: &str) -> Result<()> {
    let error = match result {
        Ok(Ok(())) => return Ok(()),
        Ok(Err(error)) => anyhow!("{who} failed: {error:#}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            anyhow!("{who} panicked: {message}")
        }
    };
    eprintln!("streaming loader: {error:#}");
    Err(error)
}

/// Streams records out of a sharded, zstd-compressed dataset as batches.
pub struct StreamingLoader<R: Decodeable, Bt, O>
where
    Bt: Batcher<R, O>,
{
    shards: Vec<PathBuf>,
    batcher: Arc<Bt>,
    batch_size: usize,
    stream_workers: usize,
    batch_workers: usize,
    seed: u64,
    /// Number of batches served by the dataloader per epoch.
    num_items: usize,
    /// Number of batches to keep in the window for shuffling.
    window_size: usize,
    device: Option<Device>,
    /// The pass currently running, if any. Shared among all copies of the loader.
    stream: Arc<Mutex<Option<Stream<Vec<R>>>>>,
    marker: PhantomData<fn() -> (R, O)>,
}

impl<R, Bt, O> StreamingLoader<R, Bt, O>
where
    R: Decodeable + Send + 'static,
    Bt: Batcher<R, O> + Send + Sync + 'static,
    O: Send + 'static,
{
    pub fn new(
        dataset_dir: impl AsRef<Path>,
        batcher: Bt,
        batch_size: usize,
        stream_workers: usize,
        batch_workers: usize,
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
        if stream_workers == 0 {
            bail!("at least one stream worker is required");
        }
        if stream_workers > shards.len() {
            bail!("number of stream workers cannot exceed number of shards");
        }
        if batch_workers == 0 {
            bail!("at least one batch worker is required");
        }
        let dataset_size: usize = shards.iter().map(|(_, records)| *records as usize).sum();
        let shards: Vec<PathBuf> = shards.into_iter().map(|(shard, _records)| shard).collect();
        let num_items = dataset_size.div_ceil(batch_size);

        Ok(Self {
            shards,
            batcher: Arc::new(batcher),
            batch_size,
            stream_workers,
            batch_workers,
            seed,
            num_items,
            window_size: DEFAULT_WINDOW_BATCHES * batch_size,
            device: None,
            stream: Arc::new(Mutex::new(None)),
            marker: PhantomData,
        })
    }

    /// Batches served per epoch; defaults to the manifest's dataset size.
    pub fn with_batches_per_epoch(mut self, batches: usize) -> Self {
        assert!(batches > 0, "batches per epoch must be at least one");
        self.num_items = batches;
        self
    }

    /// Number of batches to buffer for shuffling.
    pub fn with_buffer_size(mut self, batches: usize) -> Self {
        self.window_size = batches * self.batch_size;
        self
    }

    /// Stops the running stream, joining its worker threads.
    ///
    /// A stream never ends on its own, workers
    /// stay alive for the whole run unless this is called or the loader is
    /// dropped. Use this to release them early, or to make the next epoch
    /// start a fresh pass with a new shuffle order.
    ///
    /// Does nothing when no stream is running. The next call to
    /// [`iter`](DataLoader::iter) starts a new one.
    ///
    /// # Blocking
    ///
    /// This waits for the worker threads to unwind, and a worker only
    /// notices it should stop once the batch queue disconnects. A live
    /// batcher associated with a [`StreamingIter`] holds its own handle on
    /// that queue, so calling this while an iterator from this loader is still
    /// alive blocks until that iterator is dropped. Drop the iterator first.
    pub fn shutdown_threads(&self) -> Result<()> {
        let mut stream = self.stream.lock().expect("stream slot is never poisoned");

        if let Some(running) = stream.take() {
            running.shutdown()?;
        }
        Ok(())
    }

    /// Spawns a new stream: `stream_workers` batcher threads.
    ///
    /// Called only from [`iter`](DataLoader::iter), under the `stream` lock, so
    /// at most one pass exists per loader.
    fn start(&self) -> Result<Stream<Vec<R>>> {
        let shards = Arc::new(Mutex::new(ShardOrderer::new(
            self.shards.clone(),
            self.seed,
        )));

        let (tx, rx) = bounded::<Vec<R>>(BATCH_QUEUE);

        let mut workers = Vec::with_capacity(self.stream_workers);
        for index in 0..self.stream_workers {
            let shards = Arc::clone(&shards);
            let window_size = self.window_size;
            let batch_size = self.batch_size;
            let worker_seed = self.seed ^ index as u64;
            let tx = tx.clone();
            workers.push(
                thread::Builder::new()
                    .name(format!("dataloader-stream-worker-{index}"))
                    .spawn(move || {
                        Self::stream_worker(shards, window_size, batch_size, worker_seed, tx)
                    })?,
            );
        }
        // The workers should hold sender for proper exit behavior.
        drop(tx);

        Ok(Stream { rx, workers })
    }

    /// Decodes records from the shards and sends them as unbatched groups of
    /// `batch_size`, shuffled within a window of at least `window_size` records.
    fn stream_worker(
        shards: Arc<Mutex<ShardOrderer>>,
        window_size: usize,
        batch_size: usize,
        seed: u64,
        tx: Sender<Vec<R>>,
    ) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut buffer = vec![0u8; window_size * R::MAX_SIZE];
        let mut filled = 0usize;
        let mut carry = Vec::<R>::new();

        loop {
            let shard = shards
                .lock()
                .expect("shards is never poisoned!")
                .next_shard();
            let name = shard.display().to_string();
            let file = File::open(&shard).with_context(|| format!("failed to open {name}"))?;
            let mut decoder = Decoder::with_buffer(BufReader::new(file))
                .with_context(|| format!("failed to decompress {name}"))?;

            loop {
                let room = buffer.len() - filled;
                let read = read_chunk(&mut decoder, &mut buffer[filled..])
                    .with_context(|| format!("failed to read {name}"))?;
                filled += read;

                if read < room {
                    break;
                }

                let (mut decoded, used) = R::read_from_bytes(&buffer[..filled], usize::MAX)
                    .map_err(|error| anyhow!("{error}"))
                    .with_context(|| format!("failed to decode {name}"))?;
                if used == 0 && read != 0 {
                    bail!("decode failure in {name}");
                }
                buffer.copy_within(used..filled, 0);
                filled -= used;

                decoded.extend(carry);
                decoded.shuffle(&mut rng);
                while decoded.len() >= batch_size {
                    let batch = decoded.split_off(decoded.len() - batch_size);
                    if tx.send(batch).is_err() {
                        return Ok(());
                    }
                }
                carry = decoded;
            }
        }
    }

    /// Spawns the threads that turn unbatched groups into batches on this
    /// loader's device, until `num_items` batches have been claimed.
    fn batch_worker(
        rx: Receiver<Vec<R>>,
        tx: Sender<O>,
        batcher: Arc<Bt>,
        device: Device,
        claimed: Arc<AtomicUsize>,
        quota: usize,
    ) -> Result<()> {
        while claimed.fetch_add(1, Ordering::Relaxed) < quota {
            let Ok(records) = rx.recv() else {
                return Ok(());
            };
            if tx.send(batcher.batch(records, &device)).is_err() {
                return Ok(());
            }
        }
        Ok(())
    }

    /// A handle onto the same stream with its own quota and device.
    fn handle(&self, num_items: usize, device: Option<Device>) -> Self {
        Self {
            shards: self.shards.clone(),
            batcher: Arc::clone(&self.batcher),
            batch_size: self.batch_size,
            stream_workers: self.stream_workers,
            batch_workers: self.batch_workers,
            seed: self.seed,
            num_items,
            window_size: self.window_size,
            device,
            stream: Arc::clone(&self.stream),
            marker: PhantomData,
        }
    }
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

pub struct StreamingIter<O> {
    /// The batch queue. `None` once the epoch is over.
    rx: Option<Receiver<O>>,
    batchers: Vec<JoinHandle<Result<()>>>,
    items_processed: usize,
    items_total: usize,
}

/// An epoch that fails before it starts: a closed channel whose only item is
/// the generic short-epoch error.
fn dead_iter<O: 'static>() -> Box<dyn DataLoaderIterator<O>> {
    let (tx, rx) = bounded::<O>(0);
    drop(tx);
    Box::new(StreamingIter {
        rx: Some(rx),
        batchers: Vec::new(),
        items_processed: 0,
        // Any positive total makes the closed channel read as a short epoch
        // rather than an empty one.
        items_total: 1,
    })
}

impl<O> Iterator for StreamingIter<O> {
    type Item = Result<O, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.as_ref()?.recv() {
            Ok(batch) => {
                self.items_processed += 1;
                Some(Ok(batch))
            }
            Err(_) => {
                drop(self.rx.take());
                for (index, batcher) in self.batchers.drain(..).enumerate() {
                    let _ = report_failure(batcher.join(), &format!("batcher {index}"));
                }
                if self.items_processed >= self.items_total {
                    return None;
                }
                Some(Err(DatasetError::new(std::io::Error::other(
                    "streaming loader stopped before the epoch was complete",
                ))))
            }
        }
    }
}

impl<O> Drop for StreamingIter<O> {
    fn drop(&mut self) {
        drop(self.rx.take());
        for (index, batcher) in self.batchers.drain(..).enumerate() {
            let _ = report_failure(batcher.join(), &format!("batcher {index}"));
        }
    }
}

impl<O> DataLoaderIterator<O> for StreamingIter<O> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.items_processed,
            self.items_total,
            Some("batches".to_string()),
        )
    }
}

impl<R, Bt, O> DataLoader<O> for StreamingLoader<R, Bt, O>
where
    R: Decodeable + Send + 'static,
    Bt: Batcher<R, O> + Send + Sync + 'static,
    O: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // Obtain the lock on the stream slot. This ensures that only one caller
        // can reap or start a new pass at a time.
        let mut stream = self.stream.lock().expect("stream slot is never poisoned");

        // A pass is reused while the item queue is still connected.
        let finished = match stream.as_ref() {
            Some(running) => matches!(running.rx.try_recv(), Err(TryRecvError::Disconnected)),
            None => true,
        };

        if finished {
            // Reap the pass that ended before replacing it: the handles need to
            // be joined. A dead pass is an error, not a reason to start over.
            if let Some(previous) = stream.take() {
                let _ = previous.shutdown();
            }

            match self.start() {
                Ok(running) => *stream = Some(running),
                Err(error) => {
                    eprintln!("streaming loader: {error:#}");
                    return dead_iter();
                }
            }
        }

        let device = self
            .device
            .clone()
            .expect("streaming loader has no device: call to_device before iterating");

        let running = stream.as_ref().expect("a pass was started above");
        let (tx, rx) = bounded::<O>(2 * self.batch_workers);
        let claimed = Arc::new(AtomicUsize::new(0));
        let mut batchers = Vec::with_capacity(self.batch_workers);
        for index in 0..self.batch_workers {
            let (stream_rx, tx) = (running.rx.clone(), tx.clone());
            let (batcher, device) = (Arc::clone(&self.batcher), device.clone());
            let (claimed, quota) = (Arc::clone(&claimed), self.num_items);
            match thread::Builder::new()
                .name(format!("dataloader-batcher-{index}"))
                .spawn(move || Self::batch_worker(stream_rx, tx, batcher, device, claimed, quota))
            {
                Ok(handle) => batchers.push(handle),
                Err(error) => {
                    eprintln!("streaming loader: {error}");
                    drop(batchers);
                    return dead_iter();
                }
            }
        }
        drop(tx);

        Box::new(StreamingIter {
            rx: Some(rx),
            batchers,
            items_processed: 0,
            items_total: self.num_items,
        })
    }

    fn num_items(&self) -> usize {
        self.num_items
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<O>> {
        Arc::new(self.handle(self.num_items, Some(device.clone())))
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<O>> {
        Arc::new(self.handle(end - start, self.device.clone()))
    }
}

impl<R, Bt, O> Drop for StreamingLoader<R, Bt, O>
where
    R: Decodeable,
    Bt: Batcher<R, O>,
{
    fn drop(&mut self) {
        // Shutting the pass down here is the last chance to report a producer
        // that panicked during a final epoch, which nothing would otherwise
        // join.
        if Arc::strong_count(&self.stream) > 1 {
            return;
        }
        let Some(running) = self
            .stream
            .lock()
            .expect("stream slot is never poisoned")
            .take()
        else {
            return;
        };
        let _ = running.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{DecodeableFixedSize, RecordRSM};

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

    /// Writes `records` one-byte records to a fresh directory, so tests can
    /// build loaders over a dataset of a known size.
    fn one_byte_record_dataset(name: &str, records: u8) -> TempDir {
        let dir = TempDir::new(name);
        let mut raw = Vec::new();
        for index in 0..records {
            let mut record = [0u8; <RecordRSM as DecodeableFixedSize>::SIZE];
            // A recognisable payload, so a duplicate is detectable.
            record[0] = index;
            raw.extend_from_slice(&record);
        }
        let compressed = zstd::stream::encode_all(raw.as_slice(), 1).expect("compress");
        fs::write(dir.path().join("shard-000000.zst"), compressed).expect("write shard");
        fs::write(
            dir.path().join("manifest.csv"),
            format!("filename,records\nshard-000000.zst,{records}\n"),
        )
        .expect("write manifest");
        dir
    }

    #[test]
    fn to_device_returns_an_independent_loader() {
        // `to_device` names the device a loader builds batches on, because the
        // device is captured when a pass starts. It must therefore hand back a
        // loader with its own pass rather than one sharing a running stream,
        // which would keep building for whichever device started it.
        let dir = one_byte_record_dataset("to-device", 64);

        // One record per batch, so the 64 records are 64 batches and a cap of
        // 32 is a real cap rather than the whole dataset.
        let loader = StreamingLoader::<RecordRSM, _, Vec<u8>>::new(
            dir.path(),
            PassthroughBatcher,
            1,
            1,
            2,
            0,
        )
        .expect("loader builds")
        .with_batches_per_epoch(32);

        let first = loader.to_device(&Device::flex());
        let second = loader.to_device(&Device::flex());

        // Each is a whole capped epoch, not a share of the other: a pass is
        // never shared, so neither can starve the other of batches.
        let collect = |loader: &Arc<dyn DataLoader<Vec<u8>>>| {
            let mut batches = Vec::new();
            for batch in loader.iter() {
                batches.push(batch.expect("batch reads"));
            }
            batches
        };
        assert_eq!(collect(&first).len(), 32);
        assert_eq!(collect(&second).len(), 32);
    }

    /// One record per batch, so every batch is identifiable by its payload.
    struct PassthroughBatcher;

    impl Batcher<RecordRSM, Vec<u8>> for PassthroughBatcher {
        fn batch(&self, items: Vec<RecordRSM>, _device: &Device) -> Vec<u8> {
            items.into_iter().flat_map(|record| record.bytes).collect()
        }
    }

    #[test]
    fn a_failed_pass_is_reported() {
        // A shard that is not zstd at all kills the stream workers. The epoch
        // has to say so rather than look like a short one.
        let dir = TempDir::new("bad-shard");
        fs::write(dir.path().join("shard-000000.zst"), b"not zstd at all").expect("write shard");
        fs::write(
            dir.path().join("manifest.csv"),
            "filename,records\nshard-000000.zst,64\n",
        )
        .expect("write manifest");

        let loader = StreamingLoader::<RecordRSM, _, Vec<u8>>::new(
            dir.path(),
            PassthroughBatcher,
            1,
            1,
            1,
            0,
        )
        .expect("loader builds")
        .to_device(&Device::flex());

        let epoch: Vec<_> = loader.iter().collect();
        assert_eq!(epoch.len(), 1, "a failed epoch yields exactly one item");
        assert!(epoch[0].is_err(), "a failed epoch must report an error");
    }
}
