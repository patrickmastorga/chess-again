use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, relu};
use burn::tensor::module::attention;
use burn::tensor::{Device, Tensor};

pub const BOARD_SIZE: usize = 8;
pub const N_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const N_UNDERPROMOTION: usize = 4;
pub const PENULTIMATE_RANK_START: usize = 48;
pub const LAST_RANK_START: usize = 56;
/// Number of promotion policy classes: one per promotion cell and piece.
pub const N_PROMOTION_POLICY_CLASSES: usize = BOARD_SIZE * BOARD_SIZE * N_UNDERPROMOTION;
/// Number of from-to policy classes: one per source/destination pair
pub const N_FROM_TO_POLICY_CLASSES: usize = N_SQUARES * N_SQUARES;
/// Size of the flattened policy vector the model predicts.
pub const N_POLICY_CLASSES: usize = N_FROM_TO_POLICY_CLASSES + N_PROMOTION_POLICY_CLASSES;
/// Offset of the promotion classes inside the flattened policy vector.
pub const PROMOTION_OFFSET: usize = N_FROM_TO_POLICY_CLASSES;
pub const N_WDL_CLASSES: usize = 3;

// ENCODER BLOCK + GAB:
//  - LayerNorm
//  - 3 x linear projection -> (64, d_model) -> (64, n_head * d_head), (64, n_head * d_head), (64, n_head * d_head)
//  - reshape each -> (n_heads, 64, d_head), n_heads x Q, n_heads x K, n_heads x V
//  - scaled dot product n_heads x QK^T/sqrt(d_head) -> (n_heads, 64, 64)
//  - GAB bias:
//     - linear projection + flatten (64, d_model) -> (64, d_1) -> (64 * d_1,) OR mean-pool over tokens (64, d_model) -> (d_model,)
//     - linear projection + GELU + LayerNorm (64 * d_1,) OR (d_model,) -> (d_2,)
//     - linear projection + GELU + LayerNorm + unflatten (d_2,) -> (n_heads * d_3,) -> (n_heads, d_3)
//     - linear projection + reshape -> (n_heads, d_3) -> (n_heads, 4096) -> (n_heads, 64, 64), shared across model
//  - n_heads x softmax(scaled dot product + GAB bias) @ V -> (n_heads, 64, d_head)
//  - reshape -> (64, n_heads * d_head) + linear projection -> (64, d_model)
//  - residual connection
//  - LayerNorm
//  - linear projection + GELU + linear projection
//  - residual connection

#[derive(Module, Debug)]
pub struct GabTemplates {
    /// Shared projection from d_3 to the full `(S, S)` bias map.
    bias: Linear,
}

impl GabTemplates {
    fn init(d_3: usize, device: &Device) -> Self {
        Self {
            bias: LinearConfig::new(d_3, N_SQUARES * N_SQUARES).init(device),
        }
    }

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

#[derive(Module, Debug)]
pub struct GabHead {
    down_proj: Option<Linear>,
    hidden: Linear,
    norm1: LayerNorm,
    up_proj: Linear,
    norm2: LayerNorm,
    n_heads: usize,
    d_1: Option<usize>,
    d_2: usize,
    d_3: usize,
}

impl GabHead {
    fn init(
        d_model: usize,
        n_heads: usize,
        d_1: Option<usize>,
        d_2: usize,
        d_3: usize,
        device: &Device,
    ) -> GabHead {
        Self {
            down_proj: d_1.map(|d_1| LinearConfig::new(d_model, d_1).init(device)),
            hidden: match d_1 {
                Some(d_1) => LinearConfig::new(64 * d_1, d_2).init(device),
                None => LinearConfig::new(d_model, d_2).init(device),
            },
            norm1: LayerNormConfig::new(d_2).init(device),
            up_proj: LinearConfig::new(d_2, n_heads * d_3).init(device),
            norm2: LayerNormConfig::new(n_heads * d_3).init(device),
            n_heads,
            d_1,
            d_2,
            d_3,
        }
    }

    /// Computes the per-head GAB bias from the attention input.
    ///
    /// # Shape
    /// - `tokens`: `[B, S, D]` (the normalised attention input)
    /// - returns: `[B, H, S, S]` (the per-head attention bias)
    fn forward(&self, tokens: Tensor<3>, templates: &GabTemplates) -> Tensor<4> {
        let [batch_size, _seq_len, d_model] = tokens.dims();
        assert_eq!(
            self.down_proj.is_some(),
            self.d_1.is_some(),
            "this GabHead has down_proj = {} but d_1 = {}",
            self.down_proj.is_some(),
            self.d_1.is_some(),
        );

        let pooled = match &self.down_proj {
            // Project tokens and flatten
            Some(down) => {
                let projected = down.forward(tokens);
                projected.reshape([batch_size, 64 * self.d_1.unwrap()])
            }
            // Mean-pool over tokens: [B, S, D] -> [B, D]
            None => tokens.mean_dim(1).reshape([batch_size, d_model]),
        };

        // [B, 64 * D_1] or [B, D] -> [B, D_2] with GELU + LN -> [B, H * D_3] with GELU + LN -> [B, H, S, S]
        let hidden = self.hidden.forward(pooled);
        let hidden = gelu(hidden);
        let hidden = self.norm1.forward(hidden);
        let hidden = self.up_proj.forward(hidden);
        let hidden = gelu(hidden);
        let hidden = self.norm2.forward(hidden);
        let hidden = hidden.reshape([batch_size, self.n_heads, self.d_3]);
        templates.forward(hidden)
    }
}

#[derive(Config, Debug)]
pub struct GabEncoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    /// dimension of the per-token down-projection in the GAB head. If None,
    /// the down-projection + flattening is replaced with a mean pooling over tokens.
    pub d_1: Option<usize>,
    /// dimension of the hidden layer in the GAB head
    pub d_2: usize,
    /// number of per-head GAB templates
    pub d_3: usize,
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
    d_ff: usize,
}

impl GabEncoderBlockConfig {
    pub fn init_templates(&self, device: &Device) -> GabTemplates {
        GabTemplates::init(self.d_3, device)
    }

    pub fn init(&self, device: &Device) -> GabEncoderBlock {
        GabEncoderBlock {
            norm_attn: LayerNormConfig::new(self.d_model).init(device),
            gab: GabHead::init(
                self.d_model,
                self.n_heads,
                self.d_1,
                self.d_2,
                self.d_3,
                device,
            ),
            q_proj: LinearConfig::new(self.d_model, self.n_heads * self.d_head).init(device),
            k_proj: LinearConfig::new(self.d_model, self.n_heads * self.d_head).init(device),
            v_proj: LinearConfig::new(self.d_model, self.n_heads * self.d_head).init(device),
            out_proj: LinearConfig::new(self.n_heads * self.d_head, self.d_model).init(device),
            norm_ff: LayerNormConfig::new(self.d_model).init(device),
            ff_in: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff_out: LinearConfig::new(self.d_ff, self.d_model).init(device),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_head,
            d_ff: self.d_ff,
        }
    }
}

impl GabEncoderBlock {
    fn gab_attention(&self, x: Tensor<3>, templates: &GabTemplates) -> Tensor<3> {
        let [batch_size, seq_len, _] = x.dims();
        let head_shape = [batch_size, seq_len, self.n_heads, self.d_head];

        // GAB templates are compressed from the attention input itself.
        let gab_bias = self.gab.forward(x.clone(), templates);

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
    pub fn forward(&self, x: Tensor<3>, templates: &GabTemplates) -> Tensor<3> {
        let h = self.norm_attn.forward(x.clone());
        let h = self.gab_attention(h, templates);
        let x = x + h;

        let h = self.norm_ff.forward(x.clone());
        let h = self.ff_out.forward(gelu(self.ff_in.forward(h)));
        x + h
    }
}

// POLICY HEAD:
//  - scaled dot product with d = d_model -> (64, 64) from/to logits
//  - linear projection on last-rank keys -> additive bias, 4 extra logits per penultimate->last-rank traversal
//  - cross entropy loss
//  - Note: moves are always from whites perspective

#[derive(Config, Debug)]
pub struct PolicyHeadConfig {
    pub d_model: usize,
}

impl PolicyHeadConfig {
    pub fn init(&self, device: &Device) -> PolicyHead {
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
    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<2> {
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

// VALUE HEAD:
//  - mean-pool over tokens (64, d_model) -> (d_model,)
//  - LayerNorm
//  - linear projection + RELU (d_model,) -> (d_hidden,)
//  - WDL logits linear projection (d_hidden,) -> (3,); cross entropy loss on game result

#[derive(Config, Debug)]
pub struct ValueHeadConfig {
    pub d_model: usize,
    pub d_hidden: usize,
}

#[derive(Module, Debug)]
pub struct ValueHead {
    norm: LayerNorm,
    hidden: Linear,
    wdl: Linear,
}

impl ValueHeadConfig {
    pub fn init(&self, device: &Device) -> ValueHead {
        ValueHead {
            norm: LayerNormConfig::new(self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            wdl: LinearConfig::new(self.d_hidden, N_WDL_CLASSES).init(device),
        }
    }
}

impl ValueHead {
    /// Computes WDL logits from the trunk token stream.
    ///
    /// # Shape
    /// - `tokens`: `[B, S, D]`
    /// - returns: `[B, 3]` WDL logits
    pub fn forward(&self, tokens: Tensor<3>) -> Tensor<2> {
        let [batch_size, _seq_len, d_model] = tokens.dims();

        // Mean-pool over tokens: [B, S, D] -> [B, D]
        let pooled = tokens.mean_dim(1).reshape([batch_size, d_model]);
        let pooled = self.norm.forward(pooled);
        let hidden = relu(self.hidden.forward(pooled));
        self.wdl.forward(hidden)
    }
}
