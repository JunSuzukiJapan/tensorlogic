//! LLaMA model implementation following Candle's architecture
//!
//! Based on: tmp/candle/candle-transformers/src/models/llama.rs
//!
//! Key design:
//! - Model.forward() handles both prefill and decode
//! - Cache manages K and V tensors for all layers
//! - Position passed as CPU parameter (no GPU shape() calls)

use crate::tensor::Tensor;
use crate::device::MetalDevice;
use crate::error::{TensorError, TensorResult};
use std::collections::HashMap;

/// LLaMA model configuration
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // TinyLlama-1.1B defaults
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            intermediate_size: 5632,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

/// KV Cache for attention layers
///
/// Stores K and V tensors for each layer, handling both prefill and decode:
/// - Prefill: Creates new K/V tensors
/// - Decode: Concatenates new K/V to existing cache
#[derive(Debug, Clone)]
pub struct Cache {
    /// K and V tensors for each layer: Vec<Option<(K, V)>>
    /// None for prefill, Some((K, V)) after first forward pass
    kvs: Vec<Option<(Tensor<half::f16>, Tensor<half::f16>)>>,

    /// Whether to use KV caching (true after prefill)
    pub use_kv_cache: bool,
}

impl Cache {
    /// Create a new empty cache for the given number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            kvs: vec![None; num_layers],
            use_kv_cache: false,
        }
    }

    /// Get K and V for a specific layer
    pub fn get(&self, layer_idx: usize) -> Option<&(Tensor<half::f16>, Tensor<half::f16>)> {
        self.kvs.get(layer_idx).and_then(|kv| kv.as_ref())
    }

    /// Update K and V for a specific layer
    ///
    /// Prefill: Sets initial K/V
    /// Decode: Concatenates new K/V along sequence dimension (dim=0)
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: Tensor<half::f16>,
        v: Tensor<half::f16>,
        _device: &MetalDevice,
    ) -> TensorResult<()> {
        if let Some(Some((ref k_cache, ref v_cache))) = self.kvs.get(layer_idx) {
            // Decode: concat along sequence dimension
            let new_k = Tensor::concat(&[k_cache, &k], 0)?;
            let new_v = Tensor::concat(&[v_cache, &v], 0)?;
            self.kvs[layer_idx] = Some((new_k, new_v));
        } else {
            // Prefill: set initial cache
            self.kvs[layer_idx] = Some((k, v));
        }
        Ok(())
    }
}

/// Single transformer layer
pub struct TransformerLayer {
    // Attention weights
    pub attn_q_weight: Tensor<half::f16>,
    pub attn_k_weight: Tensor<half::f16>,
    pub attn_v_weight: Tensor<half::f16>,
    pub attn_output_weight: Tensor<half::f16>,

    // FFN weights
    pub ffn_gate_weight: Tensor<half::f16>,
    pub ffn_up_weight: Tensor<half::f16>,
    pub ffn_down_weight: Tensor<half::f16>,

    // Norms
    pub attn_norm_weight: Tensor<half::f16>,
    pub ffn_norm_weight: Tensor<half::f16>,
}

/// LLaMA model
///
/// Matches Candle's architecture:
/// ```rust
/// pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>
/// ```
pub struct LlamaModel {
    /// Token embeddings
    pub tok_embeddings: Tensor<half::f16>,

    /// Transformer layers (22 for TinyLlama)
    pub layers: Vec<TransformerLayer>,

    /// Final RMS norm
    pub norm: Tensor<half::f16>,

    /// Output projection (lm_head)
    pub output: Tensor<half::f16>,

    /// Model configuration
    pub config: LlamaConfig,

    /// Metal device
    device: MetalDevice,
}

impl LlamaModel {
    /// Create model from loaded weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor<half::f16>>,
        config: LlamaConfig,
        device: MetalDevice,
    ) -> TensorResult<Self> {
        // Get embeddings
        let tok_embeddings = weights.get("tok_embeddings.weight")
            .ok_or_else(|| TensorError::InvalidOperation("Missing tok_embeddings.weight".to_string()))?
            .clone();

        // Get output projection
        let output = weights.get("output.weight")
            .ok_or_else(|| TensorError::InvalidOperation("Missing output.weight".to_string()))?
            .clone();

        // Get final norm
        let norm = weights.get("norm.weight")
            .ok_or_else(|| TensorError::InvalidOperation("Missing norm.weight".to_string()))?
            .clone();

        // Build layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{}", i);

            let layer = TransformerLayer {
                attn_q_weight: weights.get(&format!("{}.attn_q.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.attn_q.weight", prefix)))?
                    .clone(),
                attn_k_weight: weights.get(&format!("{}.attn_k.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.attn_k.weight", prefix)))?
                    .clone(),
                attn_v_weight: weights.get(&format!("{}.attn_v.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.attn_v.weight", prefix)))?
                    .clone(),
                attn_output_weight: weights.get(&format!("{}.attn_output.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.attn_output.weight", prefix)))?
                    .clone(),

                ffn_gate_weight: weights.get(&format!("{}.ffn_gate.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.ffn_gate.weight", prefix)))?
                    .clone(),
                ffn_up_weight: weights.get(&format!("{}.ffn_up.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.ffn_up.weight", prefix)))?
                    .clone(),
                ffn_down_weight: weights.get(&format!("{}.ffn_down.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.ffn_down.weight", prefix)))?
                    .clone(),

                attn_norm_weight: weights.get(&format!("{}.attn_norm.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.attn_norm.weight", prefix)))?
                    .clone(),
                ffn_norm_weight: weights.get(&format!("{}.ffn_norm.weight", prefix))
                    .ok_or_else(|| TensorError::InvalidOperation(format!("Missing {}.ffn_norm.weight", prefix)))?
                    .clone(),
            };

            layers.push(layer);
        }

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            config,
            device,
        })
    }

    /// Forward pass - handles both prefill and decode
    ///
    /// Candle signature: `pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>`
    ///
    /// Arguments:
    /// - `x`: Input token IDs [seq_len] for prefill or [1] for decode
    /// - `index_pos`: Position index (0 for prefill, current_pos for decode)
    /// - `cache`: KV cache (updated in-place)
    ///
    /// Returns:
    /// - Logits tensor [vocab_size]
    pub fn forward(
        &self,
        _x: &Tensor<half::f16>,
        _index_pos: usize,
        _cache: &mut Cache,
    ) -> TensorResult<Tensor<half::f16>> {
        // TODO: Implement forward pass
        // This will be implemented in subsequent steps
        Err(TensorError::InvalidOperation("Not implemented yet".to_string()))
    }
}
