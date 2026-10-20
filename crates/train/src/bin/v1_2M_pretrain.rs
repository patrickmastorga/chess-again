//! V1 pretraining run.

use anyhow::Result;
use chess_again_network::{default_device, model::V1_2M_CONFIG};
use chess_again_train::{
    V1PretrainConfig, pretrain_v1, v1_default_optimizer, v1_pretrain_scheduler,
};

const STEPS_PER_EPOCH: u64 = 1024;
const EPOCHS: usize = 32;
const BATCH_SIZE: usize = 512;
const GRAD_ACCUMULATION: usize = 2;

fn config() -> V1PretrainConfig {
    let total_steps = (STEPS_PER_EPOCH * EPOCHS as u64) as usize;
    let warmup_steps = (total_steps as f64 * 0.02) as u64;
    let scheduler = v1_pretrain_scheduler(5e-4, 1e-5, 1e-6, warmup_steps, total_steps)
        .expect("warmup is a small fraction of the run");

    V1PretrainConfig::new(V1_2M_CONFIG, v1_default_optimizer(), scheduler)
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1_2M_pretrain".to_string())
        .with_stream_workers(1)
        .with_batch_workers(1)
        .with_epochs(EPOCHS)
        .with_steps_per_epoch(STEPS_PER_EPOCH)
        .with_batch_size(BATCH_SIZE)
        .with_grad_accumulation(GRAD_ACCUMULATION)
}

fn main() -> Result<()> {
    // Autodiff is runtime state in Burn 0.22: enable it on the device before
    // the model and its inputs are created.
    pretrain_v1(config(), vec![default_device().autodiff()])
}
