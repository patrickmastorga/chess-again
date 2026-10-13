//! V1 trainer.
//!
//! There is no command line: training runs are configured by editing
//! [`config`] below and running `cargo run -p chess-again-train`.

use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};
use chess_again_dataset::{RecordRSM, StreamingLoader};
use chess_again_network::{DefaultBackend, V1Config, default_device, model::V1_0_CONFIG};
use chess_again_train::{V1Batch, V1Batcher, V1TrainModel};

/// Every setting for a training run, including the ones that used to be command
/// line flags.
///
/// `Config` supplies `save`/`load` and the JSON representation, so this single
/// struct is also what lands in `config.json` next to the checkpoints.
#[derive(Config, Debug)]
struct TrainConfig {
    model: V1Config,
    optimizer: AdamConfig,
    #[config(default = "\"data/train\".to_string()")]
    train_dir: String,
    #[config(default = "\"data/validate\".to_string()")]
    test_dir: String,
    #[config(default = "\"artifacts/v1\".to_string()")]
    artifact_dir: String,
    #[config(default = 1)]
    epochs: usize,
    #[config(default = 2_500_096)]
    records_per_epoch: u64,
    #[config(default = 512)]
    batch_size: usize,
    #[config(default = 4)]
    data_workers: usize,
    #[config(default = 0)]
    seed: u64,
    #[config(default = 5e-4)]
    learning_rate: f64,
    resume_checkpoint: Option<usize>,
}

/// Builds the settings for the next run. Edit the values below, then
/// `cargo run -p chess-again-train`.
///
/// This is a function rather than a `const` because the built-in config types
/// (`AdamConfig` in particular) only expose builders, not public fields. The
/// required and defaulted fields are separated exactly as the derive does:
/// `new` takes the required ones, the defaults come from the attributes above.
fn config() -> TrainConfig {
    TrainConfig::new(
        V1_0_CONFIG,
        AdamConfig::new()
            .with_beta_2(0.98)
            .with_epsilon(1e-7)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0))),
    )
    // Data and run shape. Edit these.
    .with_train_dir("data/train".to_string())
    .with_test_dir("data/validate".to_string())
    .with_artifact_dir("artifacts/v1".to_string())
    .with_epochs(1)
    .with_records_per_epoch(2_500_096)
    .with_batch_size(512)
    .with_data_workers(4)
    .with_seed(0)
    .with_learning_rate(5e-4)
}

pub fn main() -> Result<()> {
    type B = Autodiff<DefaultBackend>;
    type InferenceB = <B as AutodiffBackend>::InnerBackend;

    let config = config();
    fs::create_dir_all(&config.artifact_dir)
        .with_context(|| format!("failed to create {}", config.artifact_dir))?;

    let device = default_device();
    // The loader caps epochs in batches; the setting is in records so that it
    // stays meaningful if the batch size changes.
    let batches_per_epoch = config.records_per_epoch.div_ceil(config.batch_size as u64);
    let train_loader = StreamingLoader::<B, RecordRSM, _, V1Batch<B>>::new(
        &config.train_dir,
        device.clone(),
        V1Batcher,
        config.batch_size,
        config.data_workers,
        config.seed,
    )
    .with_context(|| format!("failed to open train dataset {}", config.train_dir))?
    .with_batches_per_epoch(batches_per_epoch);
    let test_loader = StreamingLoader::<InferenceB, RecordRSM, _, V1Batch<InferenceB>>::new(
        &config.test_dir,
        device.clone(),
        V1Batcher,
        config.batch_size,
        config.data_workers,
        config.seed,
    )
    .with_context(|| format!("failed to open test dataset {}", config.test_dir))?;

    let artifact_dir = Path::new(&config.artifact_dir);
    config
        .save(artifact_dir.join("config.json"))
        .with_context(|| {
            format!(
                "failed to save training config in {}",
                artifact_dir.display()
            )
        })?;
    eprintln!(
        "training on {} batches of up to {} records: {} epochs of up to {} batches each",
        train_loader.size(),
        config.batch_size,
        config.epochs,
        train_loader.batches_per_epoch(),
    );

    let model = V1TrainModel::new(config.model.init::<B>(&device));
    let optimizer = config.optimizer.init::<B, V1TrainModel<B>>();
    let learner = Learner::new(model, optimizer, config.learning_rate);

    let mut training =
        SupervisedTraining::new(artifact_dir, Arc::new(train_loader), Arc::new(test_loader))
            .num_epochs(config.epochs)
            // Report the running training loss to the renderer each iteration. The
            // metric reads synced (non-autodiff) tensors, so it uses the inner backend.
            .metric_train_numeric(LossMetric::<InferenceB>::new())
            .with_file_checkpointer(DefaultFileRecorder::<FullPrecisionSettings>::new());
    if let Some(checkpoint) = config.resume_checkpoint {
        training = training.checkpoint(checkpoint);
    }
    training.launch(learner);

    Ok(())
}
