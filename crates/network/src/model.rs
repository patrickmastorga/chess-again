// INPUT: (64, 20)
// per token:
//  - dims 0 - 11 zero/one-hot piece value
//  - dim 12 en passant target marker
//  - dim 13 side to move
//  - dims 14 - 17 castling rights
//  - dim 18 repetition scalar
//  - dim 19 rule50 scalar
// Dims 13-19 are identical on all 64 tokens
// input linear projection (64, 20) -> (64, 192), shared across tokens.

// TRUNK: d_model = 192, d_head = 32, n_heads = 6
// 8 x transformer encoder blocks + GAB:
//  - LayerNorm
//  - 3 x linear projection -> (64, 192) -> (64, 192), (64, 192), (64, 192)
//  - reshape each -> (6, 64, 32), 6 x Q, 6 x K, 6 x V
//  - scaled dot product 6 x QK^T/sqrt(32) -> (6, 64, 64)
//  - GAB bias:
//     - mean-pool over tokens (64, 192) -> (192,)
//     - linear projection + GELU + LayerNorm (192,) -> (32,)
//     - linear projection + GELU + LayerNorm + unflatten (32,) -> (192,) -> (6, 32)
//     - linear projection + reshape -> (6, 32) -> (6, 4096) -> (6, 32, 32), shared across model
//  - 6 x softmax(scaled dot product + GAB bias) @ V -> (6, 64, 32)
//  - reshape + linear projection -> (64, 192)
//  - residual connection
//  - LayerNorm
//  - linear projection + GELU + linear projection
//  - residual connection

// OUTPUT:
// policy head:
//  - scaled dot product with d = 192 -> (64, 64) from/to logits
//  - linear projection on last-rank keys -> additive bias, 4 extra logits per penultimate->last-rank traversal
//  - cross entropy loss
// value head:
//  - mean-pool over tokens (64, 192) -> (192,)
//  - LayerNorm
//  - linear projection + RELU (192,) -> (128,)
//  - WDL logits linear projection 128 -> 3; cross entropy loss on game result
//  - eval logits 128 -> 256; cross entropy loss on u8 values

// LOSS WEIGHTS:
// policy 1.0, WDL 0.1, eval 0.1

// AUGMENTATION:
// flip so color-to-move is at the bottom, inputs/outputs relative

// TRAINING:
// batch 512 (grad accumulation), 100-300k steps, lr 5e-4 with warmup, mixed precision, gradient clipping.
// paper reference: 1.4M steps, batch 2048, single A100, ~4 days, lr 5e-4 constant, Nadam (b1=0.9, b2=0.98, eps=1e-7), clip 10, SWA checkpoints.

// BACKEND RULES (Burn 0.22):
// - Models and tensors carry NO backend type parameter. A runtime
//   `Device` selects the backend, so all model/dataset code is written
//   once. Zero backend-specific imports, cfgs, or dtype assumptions
//   outside ONE device-constructor module driven by cargo features:
//     default & tests:  flex   (CPU, pure Rust, eager, deterministic)
//     training (M5):    feature "metal"   (Metal through WGPU, native MSL)
//     later cluster:    feature "cuda"    (A100)
//     later wasm ship:  "webgpu" + flex (inference only)
// - Turn on autodiff on the DEVICE (`device.autodiff()`) before
//   initializing the model or its inputs; there is no `Autodiff<B>`.
// - Tests default to flex. MUST include one cross-backend integration
//   test: identical weights -> flex vs metal forward, assert allclose
//   (fp32 tolerance). This is the project's correctness harness.
// - fp32 first. Half precision = later opt-in, never hardcoded.
//   No deprecated backends. In 0.22 burn-ndarray and burn-tch (LibTorch)
//   are deprecated and burn-candle is removed; use flex/cpu on CPU.
// - Metal notes: #![recursion_limit = "256"] at crate root; fusion +
//   autotune via feature flags; kernel tuning lives in cubecl.toml,
//   never in code. First run on a new shape compiles/tunes kernels
//   lazily; expect recompile + re-autotune after backend or shape changes.
// - Checkpoints via burn-store / burnpack (portable across backends and
//   dtypes). Never legacy recorders (.mpk/.bin).
// - Dataset: implement Burn's Dataset trait over the zstd shards; decode
//   records into (64,20) inputs + targets; persistent dataloader workers;
//   record-level shuffling.
// - Determinism: seed per device; fixed shapes; no data-dependent control
//   flow inside Module::forward.
// - Perf policy: correctness on flex FIRST. Then measure step time on
//   metal (burn-bench / micro-benchmark) BEFORE any micro-optimization.
//   Fusion + autotune are the first and likely only optimizations needed;
//   do nothing else without profiling evidence.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, relu};
use burn::tensor::module::attention;
use burn::tensor::{Device, Tensor};

/// Side length of the board, i.e. squares per rank and file.
pub const BOARD_SIZE: usize = 8;
/// Number of squares on the board.
pub const N_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
/// Number of policy classes per promotion cell: knight, bishop, rook, queen.
pub const N_UNDERPROMOTION: usize = 4;
/// First square of the penultimate rank in the normalized (side-to-move at
/// the bottom) orientation.
pub const PENULTIMATE_RANK_START: usize = 48;
/// First square of the last rank in the normalized orientation.
pub const LAST_RANK_START: usize = 56;
/// Number of promotion policy classes: one per promotion cell and piece.
pub const N_PROMOTION_POLICY_CLASSES: usize = BOARD_SIZE * BOARD_SIZE * N_UNDERPROMOTION;
/// Number of from-to policy classes: one per source/destination pair. The
/// promotion classes are an additional block, not a replacement.
pub const N_FROM_TO_POLICY_CLASSES: usize = N_SQUARES * N_SQUARES;
/// Size of the flattened policy vector the model predicts.
pub const N_POLICY_CLASSES: usize = N_FROM_TO_POLICY_CLASSES + N_PROMOTION_POLICY_CLASSES;
/// Offset of the promotion classes inside the flattened policy vector.
pub const PROMOTION_OFFSET: usize = N_FROM_TO_POLICY_CLASSES;

// ---------------------------------------------------------------------------
// Geometric Attention Bias (GAB) Block
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct GabHeadConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub hidden_width: usize,
    pub template_width: usize,
}

#[derive(Module, Debug)]
pub struct GabHead {
    down: Linear,
    norm_mid: LayerNorm,
    up: Linear,
    norm_out: LayerNorm,
    d_model: usize,
    n_heads: usize,
    template_width: usize,
}

impl GabHeadConfig {
    fn init(&self, device: &Device) -> GabHead {
        GabHead {
            down: LinearConfig::new(self.d_model, self.hidden_width).init(device),
            norm_mid: LayerNormConfig::new(self.hidden_width).init(device),
            up: LinearConfig::new(self.hidden_width, self.n_heads * self.template_width)
                .init(device),
            norm_out: LayerNormConfig::new(self.n_heads * self.template_width).init(device),
            d_model: self.d_model,
            n_heads: self.n_heads,
            template_width: self.template_width,
        }
    }
}

impl GabHead {
    /// Computes the per-head GAB template vectors from the attention input.
    ///
    /// # Shape
    /// - `tokens`: `[B, S, D]` (the normalised attention input)
    /// - returns: `[B, H, D_GAB_TEMPLATES]`
    fn forward(&self, tokens: Tensor<3>) -> Tensor<3> {
        let [batch_size, _seq_len, _] = tokens.dims();

        // Mean-pool over tokens: [B, S, D] -> [B, D]
        let pooled = tokens.mean_dim(1).reshape([batch_size, self.d_model]);

        // [B, D] -> [B, D_GAB_HIDDEN] with GELU + LN -> [B, H*D3] with GELU + LN
        let hidden = self.down.forward(pooled);
        let hidden = gelu(hidden);
        let hidden = self.norm_mid.forward(hidden);
        let hidden = self.up.forward(hidden);
        let hidden = gelu(hidden);
        let hidden = self.norm_out.forward(hidden);

        // Per-head template vectors: [B, H*D3] -> [B, H, D3]
        hidden.reshape([batch_size, self.n_heads, self.template_width])
    }
}

#[derive(Config, Debug)]
pub struct GabBiasProjectionConfig {
    pub template_width: usize,
}

#[derive(Module, Debug)]
pub struct GabBiasProjection {
    /// Shared projection from template width to the full `(S, S)` bias map.
    bias: Linear,
}

impl GabBiasProjectionConfig {
    fn init(&self, device: &Device) -> GabBiasProjection {
        GabBiasProjection {
            bias: LinearConfig::new(self.template_width, N_SQUARES * N_SQUARES).init(device),
        }
    }
}

impl GabBiasProjection {
    /// Projects per-head template vectors into attention bias maps.
    ///
    /// # Shape
    /// - `templates`: `[B, H, D_GAB_TEMPLATES]`
    /// - returns: `[B, H, S, S]`
    fn forward(&self, templates: Tensor<3>) -> Tensor<4> {
        let [batch_size, n_heads, _d_templates] = templates.dims();

        // Per-head bias maps: [B, H, D3] -> [B, H, S*S] -> [B, H, S, S]
        let bias = self.bias.forward(templates);
        bias.reshape([batch_size, n_heads, N_SQUARES, N_SQUARES])
    }
}

// ---------------------------------------------------------------------------
// Encoder block with GAB
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct GabEncoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub gab: GabHeadConfig,
}

#[derive(Module, Debug)]
pub struct GabEncoderBlock {
    norm_attn: LayerNorm,
    gab: GabHead,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    norm_ff: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
}

impl GabEncoderBlockConfig {
    fn init(&self, device: &Device) -> GabEncoderBlock {
        GabEncoderBlock {
            norm_attn: LayerNormConfig::new(self.d_model).init(device),
            gab: self.gab.init(device),
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            norm_ff: LayerNormConfig::new(self.d_model).init(device),
            ff_in: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff_out: LinearConfig::new(self.d_ff, self.d_model).init(device),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_head,
        }
    }
}

impl GabEncoderBlock {
    fn attention(&self, x: Tensor<3>, gab_bias: &GabBiasProjection) -> Tensor<3> {
        let [batch_size, seq_len, _] = x.dims();
        let head_shape = [batch_size, seq_len, self.n_heads, self.d_head];

        // GAB templates are compressed from the attention input itself.
        let gab_bias = gab_bias.forward(self.gab.forward(x.clone()));

        let q = self.q_proj.forward(x.clone()).reshape(head_shape);
        let k = self.k_proj.forward(x.clone()).reshape(head_shape);
        let v = self.v_proj.forward(x).reshape(head_shape);

        // [B, S, H, Dh] -> [B, H, S, Dh]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Built-in scaled dot-product attention with the GAB added to the
        // logits before softmax. Default scale = 1/sqrt(head_dim).
        let context = attention(q, k, v, None, Some(gab_bias), Default::default());

        // [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
        let context = context.swap_dims(1, 2);
        let context = context.reshape([batch_size, seq_len, self.d_model]);
        self.out_proj.forward(context)
    }

    /// Forward pass through the GAB encoder block.
    ///
    /// # Shape
    /// - `x`: `[B, S, D]` input tokens
    /// - `gab_bias`: GAB bias projection
    /// - returns: `[B, S, D]` output tokens
    fn forward(&self, x: Tensor<3>, gab_bias: &GabBiasProjection) -> Tensor<3> {
        let h = self.norm_attn.forward(x.clone());
        let h = self.attention(h, gab_bias);
        let x = x + h;

        let h = self.norm_ff.forward(x.clone());
        let h = self.ff_out.forward(gelu(self.ff_in.forward(h)));
        x + h
    }
}

// ---------------------------------------------------------------------------
// Policy head
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct PolicyHeadConfig {
    pub d_model: usize,
}

impl PolicyHeadConfig {
    fn init(&self, device: &Device) -> PolicyHead {
        PolicyHead {
            sq_from_proj: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            sq_to_proj: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            promotion_bias_proj: LinearConfig::new(self.d_model, N_UNDERPROMOTION)
                .with_bias(false)
                .init(device),
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct PolicyHead {
    sq_from_proj: Linear,
    sq_to_proj: Linear,
    promotion_bias_proj: Linear,
    d_model: usize,
}

impl PolicyHead {
    /// Computes the flattened policy logits.
    ///
    /// # Shape
    /// - `tokens`: `[B, 64, D]` trunk output (board flipped so the side to
    ///   move is at the bottom: tokens 48-55 are the penultimate rank, 56-63
    ///   the last)
    /// - returns: `[B, 4352]` policy logits
    ///
    /// The first 4096 entries are `source * 64 + dest` from-to scores; the
    /// trailing 256 are promotion scores indexed
    /// `(from_file * 8 + to_file) * 4 + piece` (piece order: knight, bishop,
    /// rook, queen), matching [`chess_again_dataset::encode_move`].
    fn forward(&self, tokens: Tensor<3>) -> Tensor<2> {
        let [batch_size, n_squares, _] = tokens.dims();
        debug_assert_eq!(n_squares, N_SQUARES);

        // From/to logits: [B, S, D] @ [B, D, S] -> [B, S, S], scaled by 1/sqrt(d)
        let sq_from = self.sq_from_proj.forward(tokens.clone());
        let sq_to = self.sq_to_proj.forward(tokens);
        let from_to = sq_from
            .matmul(sq_to.clone().transpose())
            .mul_scalar(1.0 / (self.d_model as f32).sqrt());

        // Last-rank to-square vectors: [B, S, D] -> [B, 8, D]
        let last_rank = sq_to.narrow(1, LAST_RANK_START, BOARD_SIZE);
        // Promotion-piece biases: [B, 8, 4]
        let promo_biases = self.promotion_bias_proj.forward(last_rank);

        // Base scores of penultimate->last-rank traversals: [B, 8, 8]
        let base = from_to
            .clone()
            .narrow(1, PENULTIMATE_RANK_START, BOARD_SIZE)
            .narrow(2, LAST_RANK_START, BOARD_SIZE);
        // Add the per-destination-file bias: [B, 8(from), 8(to), 1] + [B, 1, 8(to), 4]
        let promotion_logits = base.reshape([batch_size, BOARD_SIZE, BOARD_SIZE, 1])
            + promo_biases.reshape([batch_size, 1, BOARD_SIZE, N_UNDERPROMOTION]);

        Tensor::cat(
            vec![
                from_to.reshape([batch_size, N_FROM_TO_POLICY_CLASSES]),
                promotion_logits.reshape([batch_size, N_PROMOTION_POLICY_CLASSES]),
            ],
            1,
        )
    }
}

// ---------------------------------------------------------------------------
// Value head
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct ValueHeadConfig {
    pub d_model: usize,
    pub hidden_width: usize,
    pub n_wdl_classes: usize,
}

#[derive(Module, Debug)]
pub struct ValueHead {
    norm: LayerNorm,
    hidden: Linear,
    wdl: Linear,
    d_model: usize,
}

impl ValueHeadConfig {
    fn init(&self, device: &Device) -> ValueHead {
        ValueHead {
            norm: LayerNormConfig::new(self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.hidden_width).init(device),
            wdl: LinearConfig::new(self.hidden_width, self.n_wdl_classes).init(device),
            d_model: self.d_model,
        }
    }
}

impl ValueHead {
    /// Computes WDL logits from the trunk token stream.
    ///
    /// # Shape
    /// - `tokens`: `[B, S, D]`
    /// - returns: `[B, 3]` WDL logits
    fn forward(&self, tokens: Tensor<3>) -> Tensor<2> {
        let [batch_size, _seq_len, _d_model] = tokens.dims();

        // Mean-pool over tokens: [B, S, D] -> [B, D]
        let pooled = tokens.mean_dim(1).reshape([batch_size, self.d_model]);
        let pooled = self.norm.forward(pooled);
        let hidden = relu(self.hidden.forward(pooled));
        self.wdl.forward(hidden)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct V1Config {
    pub n_input_features: usize,
    pub n_blocks: usize,
    pub gab_bias: GabBiasProjectionConfig,
    pub encoder: GabEncoderBlockConfig,
    pub policy: PolicyHeadConfig,
    pub value: ValueHeadConfig,
}

pub const V1_0_CONFIG: V1Config = V1Config {
    n_input_features: 20,
    n_blocks: 8,
    gab_bias: GabBiasProjectionConfig { template_width: 32 },
    encoder: GabEncoderBlockConfig {
        d_model: 192,
        n_heads: 6,
        d_head: 32,
        d_ff: 192,
        gab: GabHeadConfig {
            d_model: 192,
            n_heads: 6,
            hidden_width: 32,
            template_width: 32,
        },
    },
    policy: PolicyHeadConfig { d_model: 192 },
    value: ValueHeadConfig {
        d_model: 192,
        hidden_width: 128,
        n_wdl_classes: 3,
    },
};

#[derive(Module, Debug)]
pub struct V1Model {
    input_proj: Linear,
    gab_bias: GabBiasProjection,
    blocks: Vec<GabEncoderBlock>,
    policy: PolicyHead,
    value: ValueHead,
}

impl V1Config {
    /// Builds a model with random weights using this architecture.
    pub fn init(&self, device: &Device) -> V1Model {
        assert_eq!(
            self.encoder.d_model,
            self.encoder.n_heads * self.encoder.d_head
        );
        assert_eq!(self.encoder.gab.d_model, self.encoder.d_model);
        assert_eq!(self.encoder.gab.n_heads, self.encoder.n_heads);
        assert_eq!(
            self.encoder.gab.template_width,
            self.gab_bias.template_width
        );
        assert_eq!(self.policy.d_model, self.encoder.d_model);
        assert_eq!(self.value.d_model, self.encoder.d_model);

        let blocks = (0..self.n_blocks)
            .map(|_| self.encoder.init(device))
            .collect();
        V1Model {
            input_proj: LinearConfig::new(self.n_input_features, self.encoder.d_model).init(device),
            gab_bias: self.gab_bias.init(device),
            blocks,
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
            x = block.forward(x, &self.gab_bias);
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

    #[test]
    fn forward_shapes_batched() {
        let device = crate::device::default_device();
        let model = V1_0_CONFIG.init(&device);
        let input = Tensor::<3>::zeros([2, N_SQUARES, V1_0_CONFIG.n_input_features], &device);
        let out = model.forward(input);

        assert_eq!(out.policy_logits.dims(), [2, N_POLICY_CLASSES]);
        assert_eq!(out.wdl_logits.dims(), [2, V1_0_CONFIG.value.n_wdl_classes]);
    }

    #[test]
    fn forward_batch_of_one() {
        let device = crate::device::default_device();
        let model = V1_0_CONFIG.init(&device);
        let input = Tensor::<3>::zeros([1, N_SQUARES, V1_0_CONFIG.n_input_features], &device);
        let out = model.forward(input);

        assert_eq!(out.policy_logits.dims(), [1, N_POLICY_CLASSES]);
        assert_eq!(out.wdl_logits.dims(), [1, V1_0_CONFIG.value.n_wdl_classes]);
    }
}
