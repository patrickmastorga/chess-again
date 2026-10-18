//! V1 pretraining run.

use anyhow::Result;
use chess_again_network::{default_device, model::V1_0_CONFIG};
use chess_again_train::{
    V1PretrainConfig, pretrain_v1, v1_default_optimizer, v1_pretrain_scheduler,
};

/// Optimizer steps for the whole run, so the cosine can span it.
const STEPS: u64 = 2048;
const EPOCHS: usize = 1;

fn config() -> Result<V1PretrainConfig> {
    // The schedule needs the run's total length, so it is built here and handed
    // to the config rather than derived inside the training loop.
    let scheduler =
        v1_pretrain_scheduler(5e-4, 1e-5, 1e-6, 1_000, (STEPS * EPOCHS as u64) as usize)?;
    Ok(
        V1PretrainConfig::new(V1_0_CONFIG, v1_default_optimizer(), scheduler)
            // Data and run shape.
            .with_train_dir("data/train".to_string())
            .with_test_dir("data/validate".to_string())
            .with_artifact_dir("artifacts/v1".to_string())
            .with_epochs(EPOCHS)
            .with_steps_per_epoch(STEPS)
            .with_batch_size(512)
            .with_grad_accumulation(1)
            .with_data_workers(4)
            .with_seed(0),
    )
}

fn main() -> Result<()> {
    // Autodiff is runtime state in Burn 0.22: enable it on the device before
    // the model and its inputs are created.
    pretrain_v1(config()?, default_device().autodiff())
}
