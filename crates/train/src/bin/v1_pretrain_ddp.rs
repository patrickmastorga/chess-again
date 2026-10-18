//! V1 pretraining run across several GPUs with distributed data parallel.
//!
//! DDP needs a collective-capable backend, so this binary declares
//! `required-features = ["cuda"]` and is not built without it. Run it with:
//!
//! ```text
//! cargo run --release --features cuda -p chess-again-train --bin v1_pretrain_ddp
//! ```
//!
//! Every CUDA device on the node is used, so the run scales with whatever the
//! allocation granted. The assertions in `main` refuse to start on hardware the
//! recipe was not tuned for.

use anyhow::{Result, bail};
use burn::tensor::{Device, DeviceType};
use chess_again_network::model::V1_0_CONFIG;
use chess_again_train::{
    V1PretrainConfig, pretrain_v1_ddp, v1_default_optimizer, v1_pretrain_scheduler,
};

const STEPS_PER_EPOCH: u64 = 2_048;
const EPOCHS: usize = 64;
const BATCH_SIZE: usize = 2_048;
const EXPECTED_DEVICES: usize = 8;

fn config() -> V1PretrainConfig {
    let total_steps = (STEPS_PER_EPOCH * EPOCHS as u64) as usize;
    // 2% of the run, which is 2621 steps over the 131072 this run performs.
    let warmup_steps = (total_steps as f64 * 0.02) as u64;
    let scheduler = v1_pretrain_scheduler(5e-4, 1e-5, 1e-6, warmup_steps, total_steps)
        .expect("warmup is a small fraction of the run");

    V1PretrainConfig::new(V1_0_CONFIG, v1_default_optimizer(), scheduler)
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1-ddp".to_string())
        .with_epochs(EPOCHS)
        .with_steps_per_epoch(STEPS_PER_EPOCH)
        .with_batch_size(BATCH_SIZE)
        .with_grad_accumulation(1)
        .with_data_workers(4)
        .with_seed(0)
}

fn main() -> Result<()> {
    let available = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    eprintln!("{} CUDA device(s) found:", available.len());
    for (index, device) in available.iter().enumerate() {
        eprintln!("  {index}: {device:?}");
    }

    if available.is_empty() {
        bail!(
            "no CUDA devices: was this built with --features cuda, and is the node's driver visible?"
        );
    }
    if !BATCH_SIZE.is_multiple_of(available.len()) {
        bail!(
            "batch size {BATCH_SIZE} does not divide across {} device(s); \
             every device must run the same number of batches or the all-reduce deadlocks",
            available.len()
        );
    }
    if available.len() != EXPECTED_DEVICES {
        eprintln!(
            "warning: expected {EXPECTED_DEVICES} devices but found {}; the recipe \
             (batch {BATCH_SIZE}, {STEPS_PER_EPOCH} steps/epoch) was tuned for {EXPECTED_DEVICES}",
            available.len()
        );
    }

    let batch_per_device = BATCH_SIZE / available.len();
    eprintln!(
        "using {} device(s): batch {BATCH_SIZE} = {batch_per_device} records per device",
        available.len()
    );

    pretrain_v1_ddp(config(), &available)
}
