use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::Tensor;
use burn::train::metric::{Adaptor, LossInput};
use burn::train::{InferenceStep, ItemLazy, TrainOutput, TrainStep};
use burn::tensor::ExecutionError;
use chess_again_network::{V1Model, V1Output};

use crate::batcher::V1Batch;

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
