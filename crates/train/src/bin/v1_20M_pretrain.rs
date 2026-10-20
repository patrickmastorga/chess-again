//! V1 pretraining using all CUDA devices in the allocation.

use anyhow::{Result, bail};
use burn::tensor::{Device, DeviceType};
use chess_again_network::model::V1_20M_CONFIG;
use chess_again_train::{
    V1PretrainConfig, pretrain_v1, v1_default_optimizer, v1_pretrain_scheduler,
};

const STEPS_PER_EPOCH: u64 = 2_048;
const EPOCHS: usize = 64;
const BATCH_SIZE: usize = 2_048;

const STREAM_WORKERS: usize = 4;
const BATCH_WORKERS: usize = 2;

use chess_again_train::time_loader;

/// Times the loader alone for one epoch of batches, with no model involved.
#[allow(dead_code)]
fn run_loader_profile() -> Result<()> {
    let device = Device::enumerate(DeviceType::Cuda)
        .autodiff()
        .into_vec()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no CUDA devices found"))?;

    let seconds = time_loader(
        device,
        STREAM_WORKERS,
        BATCH_WORKERS,
        BATCH_SIZE,
        STEPS_PER_EPOCH,
        std::path::Path::new("data/train"),
    )?;
    eprintln!(
        "loader: {seconds:.2}s for {STEPS_PER_EPOCH} batches of {BATCH_SIZE} ({:.1} ms/batch)",
        seconds * 1000.0 / STEPS_PER_EPOCH as f64
    );
    Ok(())
}

fn config() -> V1PretrainConfig {
    let total_steps = (STEPS_PER_EPOCH * EPOCHS as u64) as usize;
    let warmup_steps = (total_steps as f64 * 0.02) as u64;
    let scheduler = v1_pretrain_scheduler(5e-4, 1e-5, 1e-6, warmup_steps, total_steps)
        .expect("warmup is a small fraction of the run");

    V1PretrainConfig::new(V1_20M_CONFIG, v1_default_optimizer(), scheduler)
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1_20M_pretrain".to_string())
        .with_epochs(1)
        .with_steps_per_epoch(STEPS_PER_EPOCH)
        .with_batch_size(BATCH_SIZE)
        .with_stream_workers(STREAM_WORKERS)
        .with_batch_workers(BATCH_WORKERS)
}

fn main() -> Result<()> {
    let available = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    eprintln!("{} CUDA device(s) found:", available.len());
    if available.is_empty() {
        bail!("no CUDA devices found; check the allocation and CUDA driver");
    }
    for (index, device) in available.iter().enumerate() {
        eprintln!("  {index}: {device:?}");
    }

    // DDP runs the same number of batches on every device, so train on the
    // most devices that still divide the batch evenly. With 8 devices and a
    // batch of 2048 that is all 8; on a 4-GPU allocation it is 4.
    let device_count = (1..=available.len())
        .rev()
        .find(|count| BATCH_SIZE.is_multiple_of(*count))
        .expect("one device always divides the batch");
    let devices = available.into_iter().take(device_count).collect::<Vec<_>>();

    eprintln!(
        "using {device_count} device(s): batch {BATCH_SIZE} = {} records per device",
        BATCH_SIZE / device_count
    );

    run_loader_profile()?;
    pretrain_v1(config(), devices)
}
