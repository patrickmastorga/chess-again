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

const STREAM_WORKERS: usize = 2;
const BATCH_WORKERS: usize = 1;

fn config() -> V1PretrainConfig {
    let total_steps = (STEPS_PER_EPOCH * EPOCHS as u64) as usize;
    let warmup_steps = (total_steps as f64 * 0.02) as u64;
    let scheduler = v1_pretrain_scheduler(5e-4, 1e-5, 1e-6, warmup_steps, total_steps)
        .expect("warmup is a small fraction of the run");

    V1PretrainConfig::new(V1_20M_CONFIG, v1_default_optimizer(), scheduler)
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1_20M_pretrain".to_string())
        .with_epochs(EPOCHS)
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

    let device_count = (1..=available.len())
        .rev()
        .find(|count| BATCH_SIZE.is_multiple_of(*count))
        .expect("one device always divides the batch");
    let devices = available.into_iter().take(device_count).collect::<Vec<_>>();

    eprintln!(
        "using {device_count} device(s): batch {BATCH_SIZE} = {} records per device",
        BATCH_SIZE / device_count
    );

    pretrain_v1(config(), devices)
}
