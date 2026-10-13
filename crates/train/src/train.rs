use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLrScheduler;
use burn::lr_scheduler::{
    LrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig, linear::LinearLrSchedulerConfig,
    sequential::SequentialLrSchedulerConfig,
};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::tensor::ExecutionError;
use burn::tensor::{Device, Tensor};
use burn::train::metric::{Adaptor, LearningRateMetric, LossInput, LossMetric};
use burn::train::{InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainOutput, TrainStep};
use chess_again_dataset::{RecordRSM, StreamingLoader};
use chess_again_network::{V1Config, V1Model, V1Output};

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

pub struct V1TrainOutput {
    pub output: V1Output,
    /// Total loss, the one reported to the metric system.
    pub loss: Tensor<1>,
    pub policy_loss: Tensor<1>,
    pub wdl_loss: Tensor<1>,
}

pub struct V1InferenceOutput {
    pub output: V1Output,
}

impl ItemLazy for V1InferenceOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        self.output.policy_logits.device().flush()?;
        Ok(self)
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

impl InferenceStep for V1TrainModel {
    type Input = V1Batch;
    type Output = V1InferenceOutput;

    fn step(&self, item: Self::Input) -> Self::Output {
        V1InferenceOutput {
            output: self.model.forward(item.input),
        }
    }
}

impl TrainStep for V1TrainModel {
    type Input = V1Batch;
    type Output = V1TrainOutput;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let device = item.input.device();
        let output = self.model.forward(item.input);
        // The policy head already predicts the flattened 4096 + 256 vector, so
        // it can be masked and scored against the targets directly.
        let policy_logits = output
            .policy_logits
            .clone()
            .mask_fill(item.policy_illegal_mask, f32::NEG_INFINITY);
        let cross_entropy = CrossEntropyLossConfig::new().init(&device);
        let policy_loss = cross_entropy.forward(policy_logits, item.policy_targets);
        let wdl_loss = cross_entropy.forward(output.wdl_logits.clone(), item.wdl_targets);
        let loss = policy_loss.clone() + wdl_loss.clone().mul_scalar(0.1);
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

/// Every setting for a V1 pretraining run.
///
/// `Config` supplies `save`/`load` and the JSON representation, so a single
/// struct is also what lands in `config.json` next to the checkpoints.
#[derive(Config, Debug)]
pub struct V1PretrainConfig {
    pub model: V1Config,
    /// AdamW, which decouples weight decay from the gradient update.
    pub optimizer: AdamWConfig,
    #[config(default = "\"data/train\".to_string()")]
    pub train_dir: String,
    #[config(default = "\"data/validate\".to_string()")]
    pub test_dir: String,
    #[config(default = "\"artifacts/v1\".to_string()")]
    pub artifact_dir: String,
    #[config(default = 1)]
    pub epochs: usize,
    /// Optimizer steps served per epoch. The loader caps epochs in batches, so
    /// this is converted with `grad_accumulation` when the loaders are built.
    #[config(default = 4_883)]
    pub steps_per_epoch: u64,
    #[config(default = 512)]
    pub batch_size: usize,
    /// Records per optimizer step is `batch_size * grad_accumulation`; the
    /// loader still yields batches of `batch_size`.
    #[config(default = 1)]
    pub grad_accumulation: usize,
    #[config(default = 4)]
    pub data_workers: usize,
    #[config(default = 0)]
    pub seed: u64,
    /// The learning rate the warmup ramps up to and the cosine decays from.
    #[config(default = 5e-4)]
    pub learning_rate: f64,
    /// The learning rate at the first step of the warmup.
    #[config(default = 1e-6)]
    pub warmup_start_lr: f64,
    /// The floor the cosine decays to.
    #[config(default = 1e-5)]
    pub min_learning_rate: f64,
    /// How many optimizer steps the warmup lasts.
    #[config(default = 1_000)]
    pub warmup_steps: u64,
    pub resume_checkpoint: Option<usize>,
}

/// The default AdamW settings for V1 runs: beta_2 = 0.98, epsilon = 1e-7,
/// weight decay 1e-6 and gradient clipping at norm 10.
pub fn v1_default_optimizer() -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_2(0.98)
        .with_epsilon(1e-7)
        .with_weight_decay(1e-6)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)))
}

/// Builds the pretraining schedule: a linear warmup to `learning_rate`,
/// followed by a cosine decay to `min_learning_rate`.
///
/// The cosine spans the run after warmup, so it reaches the floor on the last
/// optimizer step.
fn pretrain_lr_scheduler(config: &V1PretrainConfig, steps: usize) -> Result<ModuleLrScheduler> {
    let warmup_steps = config.warmup_steps as usize;
    if warmup_steps >= steps {
        bail!("warmup of {warmup_steps} steps leaves no room to decay in a {steps}-step run");
    }

    SequentialLrSchedulerConfig::new(
        vec![
            LrSchedulerConfig::Linear(LinearLrSchedulerConfig::new(
                config.warmup_start_lr,
                config.learning_rate,
                warmup_steps,
            )),
            LrSchedulerConfig::Cosine(
                CosineAnnealingLrSchedulerConfig::new(config.learning_rate, steps - warmup_steps)
                    .with_min_lr(config.min_learning_rate),
            ),
        ],
        vec![warmup_steps],
    )
    .init()
    .map_err(|error| anyhow::anyhow!("invalid learning rate schedule: {error}"))
}

/// Runs a V1 pretraining pass on `device`.
///
/// `device` carries the autodiff setting: pass a device from
/// `default_device().autodiff()` to train, since autodiff is runtime state in
/// Burn 0.22 and must be enabled before the model and its inputs are created.
pub fn pretrain_v1(config: V1PretrainConfig, device: Device) -> Result<()> {
    fs::create_dir_all(&config.artifact_dir)
        .with_context(|| format!("failed to create {}", config.artifact_dir))?;

    if config.grad_accumulation == 0 {
        bail!("gradient accumulation must be at least one");
    }
    if config.steps_per_epoch == 0 {
        bail!("an epoch must have at least one optimizer step");
    }
    // The scheduler advances once per optimizer step, so the run has exactly
    // `steps_per_epoch * epochs` of them.
    let total_steps = (config.steps_per_epoch * config.epochs as u64) as usize;
    // The loader caps epochs in batches, and one optimizer step consumes
    // `grad_accumulation` batches.
    let batches_per_epoch = config.steps_per_epoch * config.grad_accumulation as u64;

    let train_loader = StreamingLoader::<RecordRSM, _, V1Batch>::new(
        &config.train_dir,
        device.clone(),
        V1Batcher,
        config.batch_size,
        config.data_workers,
        config.seed,
    )
    .with_context(|| format!("failed to open train dataset {}", config.train_dir))?
    .with_batches_per_epoch(batches_per_epoch);
    let test_loader = StreamingLoader::<RecordRSM, _, V1Batch>::new(
        &config.test_dir,
        device.clone().without_autodiff(),
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

    let lr_scheduler = pretrain_lr_scheduler(&config, total_steps)?;
    eprintln!(
        "{} epochs of {} optimizer steps ({} batches of up to {} records)",
        config.epochs, config.steps_per_epoch, batches_per_epoch, config.batch_size,
    );
    eprintln!(
        "effective batch {} records (batch {} x accumulation {}), {} optimizer steps total",
        config.batch_size as u64 * config.grad_accumulation as u64,
        config.batch_size,
        config.grad_accumulation,
        total_steps,
    );
    eprintln!(
        "lr {} -> {} over {} warmup steps, then {} -> {} over {} decay steps",
        config.warmup_start_lr,
        config.learning_rate,
        config.warmup_steps,
        config.learning_rate,
        config.min_learning_rate,
        total_steps.saturating_sub(config.warmup_steps as usize),
    );

    let model = V1TrainModel::new(config.model.init(&device));
    let learner = Learner::new(model, config.optimizer.init(), lr_scheduler);

    let mut training =
        SupervisedTraining::new(artifact_dir, Arc::new(train_loader), Arc::new(test_loader))
            .num_epochs(config.epochs)
            .metric_train_numeric(LossMetric::new())
            .metric_train_numeric(LearningRateMetric::new())
            .with_default_checkpointers();
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::batcher::Batcher;
    use chess_again_core::{Board, GameResult, Move};
    use chess_again_dataset::RecordRSM;
    use chess_again_network::{default_device, model::V1_0_CONFIG};

    use crate::V1Batcher;

    #[test]
    fn synced_loss_feeds_the_loss_metric() {
        use burn::train::metric::{Adaptor, LossInput, Metric, MetricMetadata};

        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);
        let device = default_device().autodiff();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_0_CONFIG.init(&device));

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
        let model = V1TrainModel::new(V1_0_CONFIG.init(&device));

        let output = TrainStep::step(&model, batch).item;

        assert_eq!(output.loss.dims(), [1]);
        assert_eq!(output.policy_loss.dims(), [1]);
        assert_eq!(output.wdl_loss.dims(), [1]);
    }
}
