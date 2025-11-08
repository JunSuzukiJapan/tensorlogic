# Candle LLaMA Implementation - Complete Algorithm Analysis

**Source**: `/tmp/candle/candle-transformers/src/models/llama.rs`
**Date**: 2025-11-08
**Purpose**: Reference implementation for TensorLogic transformer debugging

---

## Table of Contents

1. [Overview](#overview)
2. [Cache Structure](#cache-structure)
3. [RoPE (Rotary Position Embedding)](#rope-rotary-position-embedding)
4. [Attention Mechanism](#attention-mechanism)
5. [MLP/FFN Layer](#mlpffn-layer)
6. [Transformer Block](#transformer-block)
7. [Model Forward Pass](#model-forward-pass)
8. [Critical Shape Transformations](#critical-shape-transformations)
9. [Key Differences from TensorLogic](#key-differences-from-tensorlogic)

---

## Overview

Candle's LLaMA implementation follows the standard transformer architecture with:
- **Grouped Query Attention (GQA)**: Efficient attention with fewer KV heads than Q heads
- **RoPE**: Rotary Position Embeddings applied to Q and K
- **SwiGLU**: Gated FFN activation
- **RMSNorm**: Pre-normalization before attention and FFN
- **KV Caching**: Efficient autoregressive generation

**Model Parameters (TinyLlama 1.1B)**:
- Layers: 22
- Hidden size: 2048
- Q heads: 32
- KV heads: 4
- Head dim: 64
- Vocab size: 32000
- Max position embeddings: 2048

---

## Cache Structure

**Lines 145-216** - Cache initialization and management

```rust
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,      // Causal masks for different sequence lengths
    pub use_kv_cache: bool,             // Enable/disable KV caching
    kvs: Vec<Option<(Tensor, Tensor)>>, // (K, V) pairs for each layer
    cos: Tensor,                        // Precomputed cos for RoPE
    sin: Tensor,                        // Precomputed sin for RoPE
    device: Device,
}
```

### RoPE Frequency Calculation (Lines 155-161)

```rust
fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;  // 64
    (0..head_dim)
        .step_by(2)  // 0, 2, 4, ..., 62
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}
```

**Output**: `[1/10000^(0/64), 1/10000^(2/64), ..., 1/10000^(62/64)]` → 32 frequencies

### Cache Initialization (Lines 164-216)

```rust
pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
    let theta = calculate_default_inv_freq(config);  // [32] frequencies
    let theta = Tensor::new(theta, device)?;

    // Create position indices: [0, 1, 2, ..., max_pos-1]
    let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((config.max_position_embeddings, 1))?  // [2048, 1]
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;  // [2048, 32]

    // Precompute cos and sin for all positions
    let cos = idx_theta.cos()?.to_dtype(dtype)?;  // [2048, 32]
    let sin = idx_theta.sin()?.to_dtype(dtype)?;  // [2048, 32]

    Ok(Self {
        masks: HashMap::new(),
        use_kv_cache,
        kvs: vec![None; config.num_hidden_layers],  // 22 layers
        device: device.clone(),
        cos,
        sin,
    })
}
```

**Key Points**:
- `cos` and `sin` are precomputed for ALL positions [0, max_pos)
- Shape: `[max_position_embeddings=2048, head_dim/2=32]`
- Stored in cache to avoid recomputation

### Causal Mask Generation (Lines 218-229)

```rust
fn mask(&mut self, t: usize) -> Result<Tensor> {
    if let Some(mask) = self.masks.get(&t) {
        Ok(mask.clone())  // Reuse cached mask
    } else {
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }
}
```

**Example** (t=4):
```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

---

## RoPE (Rotary Position Embedding)

**Lines 264-270** - Apply rotary embeddings to Q or K

```rust
fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
    let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;  // [1, num_heads, seq_len, head_dim]

    // Extract cos/sin for current position range
    let cos = cache.cos.narrow(0, index_pos, seq_len)?;  // [seq_len, 32]
    let sin = cache.sin.narrow(0, index_pos, seq_len)?;  // [seq_len, 32]

    candle_nn::rotary_emb::rope(x, &cos, &sin)
}
```

**Parameters**:
- `x`: Query or Key tensor `[batch, num_heads, seq_len, head_dim]`
- `index_pos`: Starting position in sequence (0 for prefill, cumulative for decode)
- `seq_len`: Current sequence length (35 for prefill, 1 for decode)

**Behavior**:
- **Prefill**: `index_pos=0, seq_len=35` → use `cos[0:35]`, `sin[0:35]`
- **Decode**: `index_pos=35, seq_len=1` → use `cos[35:36]`, `sin[35:36]`

---

## Attention Mechanism

**Lines 272-358** - Causal self-attention with GQA and KV caching

### Full Forward Pass

```rust
fn forward(
    &self,
    x: &Tensor,           // [batch=1, seq_len, hidden=2048]
    index_pos: usize,     // Position in sequence
    block_idx: usize,     // Layer index (0-21)
    cache: &mut Cache,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_size) = x.dims3()?;

    // 1. Linear projections
    let q = self.q_proj.forward(x)?;  // [1, seq_len, 2048]
    let k = self.k_proj.forward(x)?;  // [1, seq_len, 256]
    let v = self.v_proj.forward(x)?;  // [1, seq_len, 256]

    // 2. Reshape to multi-head format
    let q = q
        .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?  // [1, seq_len, 32, 64]
        .transpose(1, 2)?         // [1, 32, seq_len, 64]
        .contiguous()?;

    let k = k
        .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?  // [1, seq_len, 4, 64]
        .transpose(1, 2)?         // [1, 4, seq_len, 64]
        .contiguous()?;

    let mut v = v
        .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?  // [1, seq_len, 4, 64]
        .transpose(1, 2)?;        // [1, 4, seq_len, 64]

    // 3. Apply RoPE to Q and K
    let q = self.apply_rotary_emb(&q, index_pos, cache)?;  // [1, 32, seq_len, 64]
    let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;  // [1, 4, seq_len, 64]

    // 4. KV Cache management
    if cache.use_kv_cache {
        if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
            // Append new K, V to cached ones
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;  // [1, 4, cache_len+seq_len, 64]
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;  // [1, 4, cache_len+seq_len, 64]

            // Optional: Truncate if exceeding max position embeddings
            let k_seq_len = k.dims()[2];
            if k_seq_len > self.max_position_embeddings {
                k = k.narrow(D::Minus1, k_seq_len - self.max_position_embeddings, self.max_position_embeddings)?.contiguous()?
            }
            // (Similar for V)
        }
        // Update cache with new K, V
        cache.kvs[block_idx] = Some((k.clone(), v.clone()))
    }

    // 5. Repeat KV for GQA (4 KV heads → 32 Q heads)
    let k = self.repeat_kv(k)?;  // [1, 32, total_len, 64]
    let v = self.repeat_kv(v)?;  // [1, 32, total_len, 64]

    // 6. Scaled dot-product attention
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;  // [1, 32, seq_len, total_len]

    // 7. Apply causal mask (only if seq_len > 1, i.e., prefill)
    let att = if seq_len == 1 {
        att  // No mask for single token (decode phase)
    } else {
        let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
        masked_fill(&att, &mask, f32::NEG_INFINITY)?  // Mask future positions
    };

    // 8. Softmax and weighted sum
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let y = att.matmul(&v.contiguous()?)?;  // [1, 32, seq_len, 64]

    // 9. Reshape back to [batch, seq_len, hidden]
    let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;  // [1, seq_len, 2048]

    // 10. Output projection
    let y = self.o_proj.forward(&y)?;
    Ok(y)
}
```

### Repeat KV for GQA (Lines 360-362)

```rust
fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
    // Repeat each KV head 8 times: 4 heads → 32 heads
    crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
}
```

**Input**: `[1, 4, seq_len, 64]`
**Output**: `[1, 32, seq_len, 64]` (each KV head repeated 8 times)

---

## MLP/FFN Layer

**Lines 406-410** - SwiGLU activation

```rust
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // SwiGLU: silu(W_gate × x) ⊙ (W_up × x)
    let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
    self.c_proj.forward(&x)
}
```

**Shape flow**:
- Input: `[batch, seq_len, hidden=2048]`
- `c_fc1` (W_gate): `[2048, 5632]` → `[batch, seq_len, 5632]`
- `c_fc2` (W_up): `[2048, 5632]` → `[batch, seq_len, 5632]`
- Element-wise: `silu(gate) * up` → `[batch, seq_len, 5632]`
- `c_proj` (W_down): `[5632, 2048]` → `[batch, seq_len, 2048]`

---

## Transformer Block

**Lines 438-455** - Complete transformer layer with residuals

```rust
fn forward(
    &self,
    x: &Tensor,
    index_pos: usize,
    block_idx: usize,
    cache: &mut Cache,
) -> Result<Tensor> {
    // Attention block with pre-norm and residual
    let residual = x;
    let x = self.rms_1.forward(x)?;  // Pre-norm
    let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;

    // FFN block with pre-norm and residual
    let residual = &x;
    let x = self.rms_2.forward(&x)?;  // Pre-norm
    self.mlp.forward(&x)? + residual
}
```

**Pattern**: Pre-LayerNorm architecture
```
x → RMSNorm → Attention → + residual →
    RMSNorm → FFN → + residual → output
```

---

## Model Forward Pass

**Lines 505-515** - 🚨 **CRITICAL**: Token extraction for logits

```rust
pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
    let (_b_sz, seq_len) = x.dims2()?;  // x is token IDs: [batch, seq_len]

    // 1. Embedding
    let mut x = self.wte.forward(x)?;  // [batch, seq_len, hidden]

    // 2. Pass through all transformer blocks
    for (block_idx, block) in self.blocks.iter().enumerate() {
        x = block.forward(&x, index_pos, block_idx, cache)?;
    }
    // x: [batch, seq_len, hidden]

    // 3. Final normalization
    let x = self.ln_f.forward(&x)?;  // [batch, seq_len, hidden]

    // 🚨 CRITICAL: Extract ONLY the last token position
    let x = x.i((.., seq_len - 1, ..))?.contiguous()?;  // [batch, hidden]

    // 4. Project to vocabulary
    let logits = self.lm_head.forward(&x)?;  // [batch, vocab_size]
    logits.to_dtype(DType::F32)
}
```

### 🚨 Why Last Token Extraction is Critical

**Prefill phase** (seq_len = 35):
- Without extraction: `x` is `[1, 35, 2048]` → logits would be `[1, 35, 32000]` ❌
- With extraction: `x` is `[1, 2048]` → logits is `[1, 32000]` ✓

**Decode phase** (seq_len = 1):
- Without extraction: `x` is `[1, 1, 2048]` → extract → `[1, 2048]` ✓
- With extraction: Same result

**Rationale**:
- During prefill, we process entire sequence but only need logits for **next token prediction**
- The last position contains context from all previous positions via causal attention
- All other positions are ignored for generation

---

## Critical Shape Transformations

### Prefill Phase (index_pos=0, seq_len=35)

```
Token IDs:        [1, 35]
    ↓ embedding
Embeddings:       [1, 35, 2048]
    ↓ 22x transformer blocks
Hidden states:    [1, 35, 2048]
    ↓ RMSNorm
Normalized:       [1, 35, 2048]
    ↓ 🚨 EXTRACT LAST TOKEN: x.i((.., seq_len-1, ..))
Last token:       [1, 2048]
    ↓ linear(output.weight)
Logits:           [1, 32000] ✓
```

### Decode Phase (index_pos=35, seq_len=1)

```
Token ID:         [1, 1]
    ↓ embedding
Embedding:        [1, 1, 2048]
    ↓ 22x transformer blocks (with KV cache)
Hidden states:    [1, 1, 2048]
    ↓ RMSNorm
Normalized:       [1, 1, 2048]
    ↓ 🚨 EXTRACT LAST TOKEN: x.i((.., 0, ..))
Single token:     [1, 2048]
    ↓ linear(output.weight)
Logits:           [1, 32000] ✓
```

### Attention Shape Flow (Prefill)

```
Input x:          [1, 35, 2048]
    ↓ q_proj
Q:                [1, 35, 2048] → reshape → [1, 35, 32, 64] → transpose → [1, 32, 35, 64]
    ↓ k_proj
K:                [1, 35, 256]  → reshape → [1, 35, 4, 64]  → transpose → [1, 4, 35, 64]
    ↓ v_proj
V:                [1, 35, 256]  → reshape → [1, 35, 4, 64]  → transpose → [1, 4, 35, 64]
    ↓ apply_rotary_emb
Q_rope:           [1, 32, 35, 64]
K_rope:           [1, 4, 35, 64]
    ↓ repeat_kv (K, V)
K_expanded:       [1, 32, 35, 64]
V_expanded:       [1, 32, 35, 64]
    ↓ Q @ K^T / sqrt(64)
Attention scores: [1, 32, 35, 35]
    ↓ causal mask + softmax
Attention weights:[1, 32, 35, 35]
    ↓ @ V
Attention output: [1, 32, 35, 64]
    ↓ transpose + reshape
Output:           [1, 35, 2048]
    ↓ o_proj
Final:            [1, 35, 2048]
```

---

## Key Differences from TensorLogic

### 1. 🚨 Last Token Extraction (CRITICAL BUG)

**Candle** (Correct):
```rust
let x = x.i((.., seq_len - 1, ..))?.contiguous()?;  // [1, 2048]
let logits = self.lm_head.forward(&x)?;             // [1, 32000]
```

**TensorLogic** (Current - BUGGY):
```tl
let final_norm = rms_norm(h, output_norm)  // h: [35, 2048]
let logits = linear(final_norm, tok_embd)  // logits: [35, 32000] ❌
```

**Fix Required**:
```tl
let final_norm = rms_norm(h, output_norm)      // [35, 2048]
let last_token = slice_last(final_norm, 0)    // [2048] ← Extract last row
let logits = linear(last_token, tok_embd)      // [32000] ✓
```

### 2. RoPE Position Tracking

**Candle**: `index_pos` parameter tracks absolute position
- Prefill: `index_pos=0` (start from position 0)
- Decode iteration 1: `index_pos=35` (prompt length)
- Decode iteration 2: `index_pos=36`
- etc.

**TensorLogic**: Uses `position` parameter
- Should match Candle's `index_pos` behavior

### 3. KV Cache Shape

**Candle**: Caches full K, V tensors with head dimension
- K: `[batch=1, num_kv_heads=4, total_seq_len, head_dim=64]`
- V: `[batch=1, num_kv_heads=4, total_seq_len, head_dim=64]`

**TensorLogic**: Caches flattened K, V
- K: `[total_seq_len, num_kv_heads * head_dim = 256]`
- V: `[total_seq_len, 256]`

Both are valid if handled consistently.

### 4. Causal Mask Behavior

**Candle**: Only applies mask when `seq_len > 1`
```rust
let att = if seq_len == 1 {
    att  // No mask needed for decode
} else {
    masked_fill(&att, &mask, f32::NEG_INFINITY)
};
```

**TensorLogic**: Should follow same pattern (builtin `attention_with_cache` may handle this)

### 5. Weight Tying

**Candle** (Lines 519-523):
```rust
let lm_head = if cfg.tie_word_embeddings {
    Linear::from_weights(wte.embeddings().clone(), None)
} else {
    linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
};
```

**TensorLogic**: Uses `tok_embd` weight for output projection
```tl
let logits = linear(final_norm, tok_embd)  // Weight tying
```

Both are equivalent for TinyLlama which has `tie_word_embeddings=true`.

---

## Summary of Critical Implementation Requirements

1. ✅ **Precompute RoPE cos/sin** for all positions in cache initialization
2. ✅ **Track absolute position** (`index_pos`) across decode iterations
3. 🚨 **Extract last token** before final projection: `x.i((.., seq_len-1, ..))`
4. ✅ **Append to KV cache** during decode, not replace
5. ✅ **Repeat KV heads** for GQA (4 → 32 heads)
6. ✅ **Apply causal mask** only during prefill (seq_len > 1)
7. ✅ **Use weight tying** for output projection

---

## Verification Checklist

- [ ] RoPE frequencies calculated correctly
- [ ] Cos/sin precomputed for max_position_embeddings
- [ ] index_pos increments correctly across decode iterations
- [ ] KV cache appends new tokens, doesn't replace
- [ ] GQA repeat_kv expands 4 KV heads to 32
- [ ] Causal mask only applied during prefill
- [ ] **🚨 Last token extracted before logits projection**
- [ ] Output shape is `[vocab_size]` not `[seq_len, vocab_size]`

---

**End of Document**
