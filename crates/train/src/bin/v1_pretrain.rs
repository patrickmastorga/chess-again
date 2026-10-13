//! V1 pretraining run.

use anyhow::Result;
use chess_again_network::{default_device, model::V1_0_CONFIG};
use chess_again_train::{V1PretrainConfig, pretrain_v1, v1_default_optimizer};

fn config() -> V1PretrainConfig {
    V1PretrainConfig::new(V1_0_CONFIG, v1_default_optimizer())
        // Data and run shape.
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1".to_string())
        .with_epochs(1)
        .with_steps_per_epoch(4_883)
        .with_batch_size(512)
        .with_grad_accumulation(1)
        .with_data_workers(4)
        .with_seed(0)
        // Learning rate schedule.
        .with_learning_rate(5e-4)
        .with_warmup_start_lr(1e-6)
        .with_min_learning_rate(1e-5)
        .with_warmup_steps(1_000)
}

fn main() -> Result<()> {
    // Autodiff is runtime state in Burn 0.22: enable it on the device before
    // the model and its inputs are created.
    pretrain_v1(config(), default_device().autodiff())
}
