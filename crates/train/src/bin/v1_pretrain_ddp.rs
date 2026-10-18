//! V1 pretraining run across several GPUs with distributed data parallel.
//!
//! DDP needs a collective-capable backend, so this binary declares
//! `required-features = ["cuda"]` and is not built without it. Run it with:
//!
//! ```text
//! cargo run --release --features cuda -p chess-again-train --bin v1_pretrain_ddp
//! ```

use anyhow::{Result, bail};
use burn::tensor::{Device, DeviceType};
use chess_again_network::model::V1_0_CONFIG;
use chess_again_train::{V1PretrainConfig, pretrain_v1_ddp, v1_default_optimizer};

/// GPUs this run wants.
const DEVICES: usize = 2;

fn config() -> V1PretrainConfig {
    V1PretrainConfig::new(V1_0_CONFIG, v1_default_optimizer())
        // Data and run shape.
        .with_train_dir("data/train".to_string())
        .with_test_dir("data/validate".to_string())
        .with_artifact_dir("artifacts/v1-ddp".to_string())
        .with_epochs(1)
        .with_steps_per_epoch(4_883)
        // Total across every device, so each takes `batch_size / DEVICES`.
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
    let available = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    eprintln!("{} CUDA device(s) found:", available.len());
    for (index, device) in available.iter().enumerate() {
        eprintln!("  {index}: {device:?}");
    }

    if available.len() < DEVICES {
        bail!(
            "this run wants {DEVICES} devices but only {} are available",
            available.len()
        );
    }
    if available.len() > DEVICES {
        eprintln!(
            "note: {} device(s) available, using the first {DEVICES}",
            available.len()
        );
    }

    let devices: Vec<Device> = available.into_iter().take(DEVICES).collect();
    pretrain_v1_ddp(config(), &devices)
}
