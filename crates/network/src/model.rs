// V1 MODEL

// INPUT: (64, 20)
// per token:
//  - dims 0 - 11 zero/one-hot piece value
//  - dim 12 en passant target marker
//  - dim 13 side to move
//  - dims 14 - 17 castling rights
//  - dim 18 repetition scalar
//  - dim 19 rule50 scalar
// Token positions are flipped so that the side-to-move is always at the bottom.
// Dims 13-19 are identical on all 64 tokens
// linear projection (64, 20) -> (64, d_model), shared across tokens.

// TRUNK: n_blocks * gab encoder blocks

// OUTPUT: policy head + wdl head

// LOSS WEIGHTS:
// policy 1.0, WDL 0.1, eval 0.1

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use crate::components::{
    GabEncoderBlock, GabEncoderBlockConfig, GabTemplates, PolicyHead, PolicyHeadConfig, ValueHead,
    ValueHeadConfig,
};

pub const N_INPUT_FEATURES: usize = 20;

#[derive(Config, Debug)]
pub struct V1Config {
    pub d_model: usize,
    pub n_blocks: usize,
    pub block: GabEncoderBlockConfig,
    pub policy: PolicyHeadConfig,
    pub value: ValueHeadConfig,
}

// Based on Chessformer 2.5M ablation, total 2,221,955 parameters
pub const V1_2M_CONFIG: V1Config = V1Config {
    d_model: 192,
    n_blocks: 8,
    block: GabEncoderBlockConfig {
        d_model: 192,
        n_heads: 6,
        d_head: 32,
        d_ff: 192,
        d_1: Some(8),
        d_2: 32,
        d_3: 32,
    },
    policy: PolicyHeadConfig { d_model: 192 },
    value: ValueHeadConfig {
        d_model: 192,
        d_hidden: 128,
    },
};

// Medium-sized model, total 20,699,683 parameters
pub const V1_20M_CONFIG: V1Config = V1Config {
    d_model: 512,
    n_blocks: 10,
    block: GabEncoderBlockConfig {
        d_model: 512,
        n_heads: 16,
        d_head: 32,
        d_ff: 768,
        d_1: Some(16),
        d_2: 64,
        d_3: 64,
    },
    policy: PolicyHeadConfig { d_model: 512 },
    value: ValueHeadConfig {
        d_model: 512,
        d_hidden: 128,
    },
};

#[derive(Module, Debug)]
pub struct V1Model {
    input_proj: Linear,
    gab_templates: GabTemplates,
    blocks: Vec<GabEncoderBlock>,
    policy: PolicyHead,
    value: ValueHead,
}

impl V1Config {
    /// Builds a model with random weights using this architecture.
    pub fn init(&self, device: &Device) -> V1Model {
        assert_eq!(self.d_model, self.block.d_model);
        assert_eq!(self.d_model, self.policy.d_model);
        assert_eq!(self.d_model, self.value.d_model);

        V1Model {
            input_proj: LinearConfig::new(N_INPUT_FEATURES, self.d_model).init(device),
            gab_templates: self.block.init_templates(device),
            blocks: (0..self.n_blocks)
                .map(|_| self.block.init(device))
                .collect(),
            policy: self.policy.init(device),
            value: self.value.init(device),
        }
    }
}

impl V1Model {
    /// Runs a forward pass over a batch of encoded boards.
    ///
    /// # Shape
    /// - `input`: `[B, 64, 20]` raw board encoding
    /// - returns: a [`V1Output`] with the flattened policy vector and the
    ///   value head logits
    pub fn forward(&self, input: Tensor<3>) -> V1Output {
        // [B, 64, 20] -> [B, 64, 192] shared-token linear projection.
        let tokens = self.input_proj.forward(input);

        // Trunk; each block generates its own GAB templates and projects them
        // through the shared bias projection.
        let mut x = tokens;
        for block in &self.blocks {
            x = block.forward(x, &self.gab_templates);
        }

        // Heads.
        let policy_logits = self.policy.forward(x.clone());
        let wdl_logits = self.value.forward(x);

        V1Output {
            policy_logits,
            wdl_logits,
        }
    }
}

/// Per-head logits produced by one forward pass of [`V1Model`].
#[derive(Debug, Clone)]
pub struct V1Output {
    /// Flattened policy logits, `[B, 4352]`: the 4096 `source * 64 + dest`
    /// from-to entries followed by the 256 promotion entries indexed
    /// `(from_file * 8 + to_file) * 4 + piece` (piece order: knight, bishop,
    /// rook, queen), matching [`chess_again_dataset::encode_move`].
    pub policy_logits: Tensor<2>,
    /// WDL logits (win / draw / loss), `[B, 3]`.
    pub wdl_logits: Tensor<2>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::components::{N_POLICY_CLASSES, N_SQUARES, N_WDL_CLASSES};
    use crate::default_device;

    #[test]
    fn forward_shapes_batched() {
        let device = default_device();
        let model = V1_2M_CONFIG.init(&device);
        let input = Tensor::<3>::zeros([2, N_SQUARES, N_INPUT_FEATURES], &device);
        let out = model.forward(input);

        assert_eq!(out.policy_logits.dims(), [2, N_POLICY_CLASSES]);
        assert_eq!(out.wdl_logits.dims(), [2, N_WDL_CLASSES]);
    }
}
