use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{
    Tensor, Transaction,
    backend::{AutodiffBackend, Backend},
};
use burn::train::metric::{Adaptor, LossInput};
use burn::train::{InferenceStep, ItemLazy, TrainOutput, TrainStep};
use chess_again_network::{V1Model, V1Output};

use crate::batcher::V1Batch;

#[derive(Module, Debug)]
pub struct V1TrainModel<B: Backend> {
    model: V1Model<B>,
}

impl<B: Backend> V1TrainModel<B> {
    pub fn new(model: V1Model<B>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &V1Model<B> {
        &self.model
    }
}

pub struct V1TrainOutput<B: Backend> {
    pub output: V1Output<B>,
    /// Total loss, the one reported to the metric system.
    pub loss: Tensor<B, 1>,
    pub policy_loss: Tensor<B, 1>,
    pub wdl_loss: Tensor<B, 1>,
}

pub struct V1InferenceOutput<B: Backend> {
    pub output: V1Output<B>,
}

impl<B: Backend> ItemLazy for V1InferenceOutput<B> {
    type ItemSync = ();

    fn sync(self) {}
}

/// Losses moved off the autodiff backend so metrics can read them.
pub struct V1TrainOutputSync<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub policy_loss: Tensor<B, 1>,
    pub wdl_loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> ItemLazy for V1TrainOutput<B> {
    type ItemSync = V1TrainOutputSync<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        // One transfer for all the losses reported to metrics.
        let [loss, policy_loss, wdl_loss] = Transaction::default()
            .register(self.loss)
            .register(self.policy_loss)
            .register(self.wdl_loss)
            .execute()
            .try_into()
            .expect("three registered tensors");

        let device = &Default::default();
        V1TrainOutputSync {
            loss: Tensor::from_data(loss, device),
            policy_loss: Tensor::from_data(policy_loss, device),
            wdl_loss: Tensor::from_data(wdl_loss, device),
        }
    }
}

/// Lets `LossMetric` read the synced total loss.
impl<B: Backend> Adaptor<LossInput<B>> for V1TrainOutputSync<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> InferenceStep for V1TrainModel<B> {
    type Input = V1Batch<B>;
    type Output = V1InferenceOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        V1InferenceOutput {
            output: self.model.forward(item.input),
        }
    }
}

impl<B: AutodiffBackend> TrainStep for V1TrainModel<B> {
    type Input = V1Batch<B>;
    type Output = V1TrainOutput<B>;

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
    use burn::backend::Autodiff;
    use burn::data::dataloader::batcher::Batcher;
    use burn::tensor::Device;
    use chess_again_core::{Board, GameResult, Move};
    use chess_again_dataset::RecordRSM;
    use chess_again_network::{DefaultBackend, model::V1_0_CONFIG};

    use crate::V1Batcher;

    #[test]
    fn synced_loss_feeds_the_loss_metric() {
        use burn::train::metric::{Adaptor, LossInput, Metric, MetricMetadata};

        type B = Autodiff<DefaultBackend>;
        let board = Board::starting();
        let mv = Move::from_san("e4", &board).expect("e4 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);
        let device: Device<B> = Default::default();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_0_CONFIG.init::<B>(&device));

        let output = TrainStep::step(&model, batch).item;
        let synced = output.sync();

        // The adaptor must hand a real batch of losses to the metric.
        let input: LossInput<<B as AutodiffBackend>::InnerBackend> = synced.adapt();
        let mut metric =
            burn::train::metric::LossMetric::<<B as AutodiffBackend>::InnerBackend>::new();
        let metadata = MetricMetadata {
            progress: burn::data::dataloader::Progress::new(1, 1),
            global_progress: burn::data::dataloader::Progress::new(1, 1),
            iteration: Some(1),
            lr: None,
        };
        let entry = metric.update(&input, &metadata);
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
        type B = Autodiff<DefaultBackend>;
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::WhiteWin);
        let device: Device<B> = Default::default();
        let batch = V1Batcher.batch(vec![record], &device);
        let model = V1TrainModel::new(V1_0_CONFIG.init::<B>(&device));

        let output = TrainStep::step(&model, batch).item;

        assert_eq!(output.loss.dims(), [1]);
        assert_eq!(output.policy_loss.dims(), [1]);
        assert_eq!(output.wdl_loss.dims(), [1]);
    }
}
