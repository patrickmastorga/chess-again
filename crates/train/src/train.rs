use anyhow::{Context, Result, anyhow, bail};
use burn::config::Config;
use burn::data::dataloader::DataLoader;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLrSchedulerConfig;
use burn::lr_scheduler::{
    LrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig, linear::LinearLrSchedulerConfig,
    sequential::SequentialLrSchedulerConfig,
};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::tensor::ExecutionError;
#[cfg(feature = "cuda")]
use burn::tensor::distributed::{DistributedConfig, ReduceOperation};
use burn::tensor::{Device, Tensor};
#[cfg(feature = "cuda")]
use burn::train::ExecutionStrategy;
use burn::train::metric::{Adaptor, LearningRateMetric, LossInput, LossMetric};
use burn::train::{InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep};
use chess_again_dataset::{RecordRSM, StreamingLoader};
use chess_again_network::{V1Config, V1Model, V1Output};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::batcher::{V1Batch, V1Batcher};

#[derive(Module, Debug)]
pub struct V1TrainModel {
    model: V1Model,
}

impl V1TrainModel {
    pub fn new(model: V1Model) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &V1Model {
        &self.model
    }
}

/// Weight applied to the WDL term of the combined loss.
const WDL_LOSS_WEIGHT: f32 = 0.1;

/// Computes the masked policy loss, the WDL loss and their weighted sum.
///
/// Shared by training and validation so the two cannot report different losses.
fn v1_losses(model: &V1Model, item: &V1Batch) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
    let device = item.input.device();
    let output = model.forward(item.input.clone());
    // The policy head already predicts the flattened 4096 + 256 vector, so
    // it can be masked and scored against the targets directly.
    let policy_logits = output
        .policy_logits
        .mask_fill(item.policy_illegal_mask.clone(), f32::NEG_INFINITY);
    let cross_entropy = CrossEntropyLossConfig::new().init(&device);
    let policy_loss = cross_entropy.forward(policy_logits, item.policy_targets.clone());
    let wdl_loss = cross_entropy.forward(output.wdl_logits, item.wdl_targets.clone());
    let loss = policy_loss.clone() + wdl_loss.clone().mul_scalar(WDL_LOSS_WEIGHT);
    (loss, policy_loss, wdl_loss)
}

pub struct V1TrainOutput {
    pub output: V1Output,
    /// Total loss, the one reported to the metric system.
    pub loss: Tensor<1>,
    pub policy_loss: Tensor<1>,
    pub wdl_loss: Tensor<1>,
}

pub struct V1InferenceOutput {
    pub output: V1Output,
    /// Validation loss, reported to the metric system for the valid split.
    pub loss: Tensor<1>,
    pub policy_loss: Tensor<1>,
    pub wdl_loss: Tensor<1>,
}

impl ItemLazy for V1InferenceOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        self.loss.device().flush()?;

        Ok(V1InferenceOutput {
            output: self.output,
            loss: self.loss.without_autodiff(),
            policy_loss: self.policy_loss.without_autodiff(),
            wdl_loss: self.wdl_loss.without_autodiff(),
        })
    }
}

impl ItemLazy for V1TrainOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        self.loss.device().flush()?;

        Ok(V1TrainOutput {
            output: self.output,
            loss: self.loss.without_autodiff(),
            policy_loss: self.policy_loss.without_autodiff(),
            wdl_loss: self.wdl_loss.without_autodiff(),
        })
    }
}

impl Adaptor<LossInput> for V1TrainOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<LossInput> for V1InferenceOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl InferenceStep for V1TrainModel {
    type Input = V1Batch;
    type Output = V1InferenceOutput;

    fn step(&self, item: Self::Input) -> Self::Output {
        let output = self.model.forward(item.input.clone());
        // No backward here: the losses are computed only to be reported.
        let (loss, policy_loss, wdl_loss) = v1_losses(&self.model, &item);

        V1InferenceOutput {
            output,
            loss,
            policy_loss,
            wdl_loss,
        }
    }
}

impl TrainStep for V1TrainModel {
    type Input = V1Batch;
    type Output = V1TrainOutput;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.model.forward(item.input.clone());
        let (loss, policy_loss, wdl_loss) = v1_losses(&self.model, &item);
        let grads = loss.clone().backward();

        TrainOutput::new(
            self,
            grads,
            V1TrainOutput {
                output,
                loss,
                policy_loss,
                wdl_loss,
            },
        )
    }
}

/// Settings for a V1 pretraining run; saved beside its checkpoints.
#[derive(Config, Debug)]
pub struct V1PretrainConfig {
    pub model: V1Config,
    pub optimizer: AdamWConfig,
    pub lr_scheduler: LrSchedulerConfig,
    #[config(default = "\"data/train\".to_string()")]
    pub train_dir: String,
    #[config(default = "\"data/validate\".to_string()")]
    pub test_dir: String,
    #[config(default = "\"artifacts/v1\".to_string()")]
    pub artifact_dir: String,
    #[config(default = 1)]
    pub epochs: usize,
    #[config(default = 2048)]
    pub steps_per_epoch: u64,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub grad_accumulation: usize,
    #[config(default = 4)]
    pub stream_workers: usize,
    #[config(default = 2)]
    pub batch_workers: usize,
    #[config(default = 0)]
    pub seed: u64,
    pub resume_checkpoint: Option<usize>,
}

/// AdamW defaults for V1 training.
pub fn v1_default_optimizer() -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_2(0.98)
        .with_epsilon(1e-7)
        .with_weight_decay(1e-6)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)))
}

/// Builds a linear warmup followed by cosine decay.
pub fn v1_pretrain_scheduler(
    peak: f64,
    floor: f64,
    warmup_start: f64,
    warmup_steps: u64,
    total_steps: usize,
) -> Result<LrSchedulerConfig> {
    let warmup_steps = warmup_steps as usize;
    if warmup_steps >= total_steps {
        bail!("warmup of {warmup_steps} steps leaves no room to decay in a {total_steps}-step run");
    }

    Ok(LrSchedulerConfig::Sequential(
        SequentialLrSchedulerConfig::new(
            vec![
                LrSchedulerConfig::Linear(LinearLrSchedulerConfig::new(
                    warmup_start,
                    peak,
                    warmup_steps,
                )),
                LrSchedulerConfig::Cosine(
                    CosineAnnealingLrSchedulerConfig::new(peak, total_steps - warmup_steps)
                        .with_min_lr(floor),
                ),
            ],
            vec![warmup_steps],
        ),
    ))
}

/// Pretrains V1 on one device, or across several with CUDA DDP.
///
/// `batch_size` is the global batch and must divide evenly across devices.
pub fn pretrain_v1(config: V1PretrainConfig, devices: Vec<Device>) -> Result<()> {
    if devices.is_empty() {
        bail!("pretraining needs at least one device");
    }
    if !devices[0].is_autodiff() {
        bail!("the first device must carry autodiff, since the model is initialized there");
    }
    if config.grad_accumulation == 0 {
        bail!("gradient accumulation must be at least one");
    }
    if config.steps_per_epoch == 0 {
        bail!("an epoch must have at least one optimizer step");
    }

    let n_devices = devices.len();
    let batch_size = if n_devices == 1 {
        config.batch_size
    } else {
        if !config.batch_size.is_multiple_of(n_devices) {
            bail!(
                "batch size {} must divide evenly across {} devices",
                config.batch_size,
                n_devices
            );
        }
        #[cfg(not(feature = "cuda"))]
        bail!("multi-device training requires building with the `cuda` feature");
        #[cfg(feature = "cuda")]
        {
            config.batch_size / n_devices
        }
    };

    fs::create_dir_all(&config.artifact_dir)
        .with_context(|| format!("failed to create {}", config.artifact_dir))?;

    // The loader caps epochs in batches, and one optimizer step consumes
    // `grad_accumulation` batches.
    let batches_per_epoch = config.steps_per_epoch * config.grad_accumulation as u64;

    // The loader is built without a device: the training loop calls
    // `to_device` itself, for training and validation alike.
    let train_loader = StreamingLoader::<RecordRSM, _, V1Batch>::new(
        &config.train_dir,
        V1Batcher,
        batch_size,
        config.stream_workers,
        config.batch_workers,
        config.seed,
    )
    .with_context(|| format!("failed to open train dataset {}", config.train_dir))?
    .with_batches_per_epoch(batches_per_epoch as usize);
    // Validation always runs unsplit, at the full batch size.
    let test_loader = StreamingLoader::<RecordRSM, _, V1Batch>::new(
        &config.test_dir,
        V1Batcher,
        config.batch_size,
        1,
        config.batch_workers,
        config.seed,
    )
    .with_context(|| format!("failed to open test dataset {}", config.test_dir))?
    .with_batches_per_epoch(config.steps_per_epoch as usize);

    let artifact_dir = Path::new(&config.artifact_dir);
    config
        .save(artifact_dir.join("config.json"))
        .with_context(|| {
            format!(
                "failed to save training config in {}",
                artifact_dir.display()
            )
        })?;

    let lr_scheduler = ModuleLrSchedulerConfig::new(config.lr_scheduler.clone())
        .init()
        .map_err(|error| anyhow!("invalid learning rate schedule: {error}"))?;
    eprintln!(
        "{} epochs of {} optimizer steps on {} device(s) ({} batches of {} records per device)",
        config.epochs, config.steps_per_epoch, n_devices, batches_per_epoch, batch_size,
    );
    eprintln!(
        "effective batch {} records (batch {} over {} device(s) x accumulation {})",
        config.batch_size as u64 * config.grad_accumulation as u64,
        config.batch_size,
        n_devices,
        config.grad_accumulation,
    );

    let model = V1TrainModel::new(config.model.init(&devices[0]));
    let learner = Learner::new(model, config.optimizer.init(), lr_scheduler);

    // The only difference between the two runs: one device trains in place,
    // several share each batch and all-reduce the gradients.
    #[cfg(feature = "cuda")]
    let strategy = (devices.len() > 1).then(|| {
        ExecutionStrategy::ddp(
            devices,
            DistributedConfig {
                all_reduce_op: ReduceOperation::Mean,
            },
        )
    });

    #[allow(unused_mut)]
    let mut training =
        SupervisedTraining::new(artifact_dir, Arc::new(train_loader), Arc::new(test_loader))
            .num_epochs(config.epochs)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(LearningRateMetric::new())
            .with_default_checkpointers();
    #[cfg(feature = "cuda")]
    let mut training = match strategy {
        Some(strategy) => training.with_training_strategy(strategy.into()),
        None => training,
    };
    if config.grad_accumulation > 1 {
        training = training.grads_accumulation(config.grad_accumulation);
    }
    if let Some(checkpoint) = config.resume_checkpoint {
        training = training.checkpoint(checkpoint);
    }
    let result = training.launch(learner);

    // Burn 0.22 stops instead of panicking when a run fails; report it.
    if let Some(error) = result.error {
        return Err(anyhow::anyhow!("training failed: {error}"));
    }

    Ok(())
}

/// Times iterating `steps` batches on one device, with no model in the way.
pub fn time_loader(
    device: Device,
    data_workers: usize,
    batch_workers: usize,
    batch_size: usize,
    steps: u64,
    data_dir: &Path,
) -> Result<f64> {
    let loader = StreamingLoader::<RecordRSM, _, V1Batch>::new(
        data_dir,
        V1Batcher,
        batch_size,
        data_workers,
        batch_workers,
        0,
    )
    .with_context(|| format!("failed to open dataset {}", data_dir.display()))?
    .with_batches_per_epoch(steps as usize)
    .to_device(&device);

    let start = Instant::now();
    let mut seen = 0u64;
    for batch in loader.iter() {
        let batch = batch.map_err(|error| anyhow!("dataloader failed: {error}"))?;
        std::hint::black_box(batch.input.dims());
        seen += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    if seen < steps {
        bail!("loader produced {seen} of {steps} batches");
    }
    Ok(elapsed)
}

// /// Profiles a short DDP run.
// #[cfg(feature = "cuda")]
// pub fn v1_profile_ddp(
//     devices: &[Device],
//     data_workers: usize,
//     batch_workers: usize,
//     batch_size: usize,
//     steps: u64,
//     data_dir: impl AsRef<Path>,
// ) -> Result<ProfileReport> {
//     if devices.is_empty() {
//         bail!("profiling needs at least one device");
//     }
//     if steps == 0 {
//         bail!("profiling needs at least one step");
//     }
//     if batch_size == 0 {
//         bail!("profiling needs a batch size of at least one");
//     }
//     let data_dir = data_dir.as_ref();

//     let loader_secs = time_loader(
//         devices[0].clone(),
//         data_workers,
//         batch_workers,
//         batch_size,
//         steps,
//         data_dir,
//     )?;
//     let step_secs = time_steps_ddp(
//         devices,
//         data_workers,
//         batch_workers,
//         batch_size,
//         steps,
//         data_dir,
//     )?;

//     let report = ProfileReport {
//         devices: devices.len(),
//         steps,
//         data_workers,
//         batch_workers,
//         batch_size,
//         loader_secs,
//         step_secs,
//     };
//     report.print();
//     Ok(report)
// }

// /// The outcome of one profiling run.
// #[derive(Debug, Clone)]
// pub struct ProfileReport {
//     pub devices: usize,
//     pub steps: u64,
//     pub data_workers: usize,
//     pub batch_workers: usize,
//     pub batch_size: usize,
//     /// Seconds to iterate `steps` batches with no model, on one device.
//     pub loader_secs: f64,
//     /// Seconds for the real training loop to run `steps` optimizer steps.
//     pub step_secs: f64,
// }

// impl ProfileReport {
//     pub fn print(&self) {
//         assert!(self.step_secs > 0.0, "step_secs must be positive");
//         let share = self.loader_secs / self.step_secs;
//         eprintln!("--- profile ---");
//         eprintln!(
//             "devices {} | steps {} | batch {} | data workers {} | batch workers {}",
//             self.devices, self.steps, self.batch_size, self.data_workers, self.batch_workers
//         );
//         eprintln!(
//             "loader alone {:>8.2}s ({:.1} ms/batch)",
//             self.loader_secs,
//             self.loader_secs * 1000.0 / self.steps as f64
//         );
//         eprintln!(
//             "full steps  {:>8.2}s ({:.1} ms/step)",
//             self.step_secs,
//             self.step_secs * 1000.0 / self.steps as f64
//         );
//         eprintln!("loader share of step time: {share:.2}");
//         if share >= 1.0 {
//             eprintln!("verdict: dataloader bound - the loader is slower than training");
//         } else if share >= 0.7 {
//             eprintln!("verdict: likely dataloader bound - more workers may help");
//         } else {
//             eprintln!("verdict: device bound - the loader is not the limit");
//         }
//     }
// }

// /// Times a short DDP run using a constant learning rate.
// fn time_steps_ddp(
//     devices: &[Device],
//     data_workers: usize,
//     batch_workers: usize,
//     batch_size: usize,
//     steps: u64,
//     data_dir: &Path,
// ) -> Result<f64> {
//     let config = V1PretrainConfig::new(
//         V1_2M_CONFIG,
//         v1_default_optimizer(),
//         LrSchedulerConfig::Constant(5e-4),
//     )
//     .with_train_dir(data_dir.display().to_string())
//     .with_test_dir(data_dir.display().to_string())
//     .with_artifact_dir("artifacts/v1-profile".to_string())
//     .with_epochs(1)
//     .with_steps_per_epoch(steps)
//     .with_batch_size(batch_size)
//     .with_grad_accumulation(1)
//     .with_data_workers(data_workers)
//     .with_batch_workers(batch_workers)
//     .with_seed(0);

//     // A profile must not leave checkpoints behind.
//     let dir = Path::new(&config.artifact_dir);
//     if dir.exists() {
//         fs::remove_dir_all(dir).with_context(|| format!("failed to clear {}", dir.display()))?;
//     }

//     let start = Instant::now();
//     pretrain_v1(config.clone(), devices.to_vec())?;
//     Ok(start.elapsed().as_secs_f64())
// }

#[cfg(test)]
mod tests {
    use super::*;

    use chess_again_core::{Board, GameResult, Move};
    use chess_again_dataset::RecordRSM;
    use chess_again_network::{default_device, model::V1_2M_CONFIG};

    use crate::V1Batcher;
    use burn::data::dataloader::batcher::Batcher;

    #[test]
    fn synced_loss_feeds_the_loss_metric() {
        use burn::train::metric::{Adaptor, LossInput, Metric, MetricMetadata};

        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);
        let device = default_device().autodiff();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_2M_CONFIG.init(&device));

        let output = TrainStep::step(&model, batch).item;
        let synced = output.sync().expect("sync should succeed on flex");

        // The adaptor must hand a real batch of losses to the metric.
        let input: LossInput = synced.adapt();
        let mut metric = burn::train::metric::LossMetric::new();
        let metadata = MetricMetadata {
            progress: burn::data::dataloader::Progress::new(1, 1, None),
            iteration: Some(1),
            lr: None,
        };
        let entry = metric
            .update(&input, &metadata)
            .expect("loss metric reads the synced loss");
        assert!(
            !entry.formatted.is_empty(),
            "loss metric should render a value, got {entry:?}"
        );
        // The formatted value must actually contain a number, not a placeholder.
        assert!(
            entry.formatted.chars().any(|c| c.is_ascii_digit()),
            "expected a numeric loss, got {}",
            entry.formatted
        );
        assert_eq!(synced.loss.dims(), [1]);
        assert_eq!(synced.policy_loss.dims(), [1]);
        assert_eq!(synced.wdl_loss.dims(), [1]);
    }

    #[test]
    fn train_step_combines_policy_and_value_losses() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::WhiteWin);
        let device = default_device().autodiff();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_2M_CONFIG.init(&device));

        let output = TrainStep::step(&model, batch).item;

        assert_eq!(output.loss.dims(), [1]);
        assert_eq!(output.policy_loss.dims(), [1]);
        assert_eq!(output.wdl_loss.dims(), [1]);
    }

    /// The validation split needs its own loss and adaptor, or the valid curve
    /// never appears; `Adaptor<()> for T` lets that omission compile.
    #[test]
    fn inference_step_reports_a_loss_for_the_valid_split() {
        use burn::train::metric::{Adaptor, LossInput, Metric, MetricMetadata};

        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::WhiteWin);
        let device = default_device().autodiff();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_2M_CONFIG.init(&device));

        let output = InferenceStep::step(&model, batch);
        let synced = output.sync().expect("sync should succeed on flex");

        assert_eq!(synced.loss.dims(), [1]);
        assert_eq!(synced.policy_loss.dims(), [1]);
        assert_eq!(synced.wdl_loss.dims(), [1]);

        // The valid metric must read a real value, not a placeholder.
        let input: LossInput = synced.adapt();
        let mut metric = burn::train::metric::LossMetric::new();
        let metadata = MetricMetadata {
            progress: burn::data::dataloader::Progress::new(1, 1, None),
            iteration: Some(1),
            lr: None,
        };
        let entry = metric
            .update(&input, &metadata)
            .expect("loss metric reads the synced validation loss");
        assert!(
            entry.formatted.chars().any(|c| c.is_ascii_digit()),
            "expected a numeric validation loss, got {}",
            entry.formatted
        );
    }

    /// Both steps must score the same way, since they share `v1_losses`.
    #[test]
    fn train_and_inference_agree_on_the_loss() {
        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);
        let device = default_device().autodiff();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_2M_CONFIG.init(&device));

        let train_loss = TrainStep::step(&model, batch.clone()).item.loss;
        let valid_loss = InferenceStep::step(&model, batch).loss;

        let train_loss = train_loss.to_data().try_to_vec::<f32>().unwrap()[0];
        let valid_loss = valid_loss.to_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            (train_loss - valid_loss).abs() < 1e-6,
            "train loss {train_loss} and validation loss {valid_loss} should match"
        );
    }
}
