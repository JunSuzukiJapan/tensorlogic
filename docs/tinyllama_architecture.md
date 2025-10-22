# TinyLlama 1.1B Model Architecture

**Model**: TinyLlama-1.1B-Chat-v1.0 (Q4_0 quantized)
**File**: `~/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf`
**Total Parameters**: ~1.1 billion
**Total Tensors**: 201 tensors

## Architecture Overview

TinyLlama is based on the LLaMA 2 architecture with Grouped-Query Attention (GQA).

### Core Specifications

| Parameter | Value |
|-----------|-------|
| Number of Layers | 22 |
| Vocabulary Size | 32,000 tokens |
| Embedding Dimension (d_model) | 2048 |
| FFN Intermediate Dimension (d_ff) | 5632 |
| Number of Attention Heads | 32 (inferred) |
| Head Dimension | 64 (2048 / 32) |
| KV Heads (GQA) | 4 (inferred from 256 / 64) |
| Max Context Length | 2048 tokens |

### Grouped-Query Attention (GQA)

TinyLlama uses Grouped-Query Attention, a LLaMA 2 optimization that reduces KV-cache memory usage:

- **Query**: `[d_model, d_model]` = `[2048, 2048]` - full dimension
- **Key**: `[d_model, kv_dim]` = `[2048, 256]` - reduced dimension (4 KV heads)
- **Value**: `[d_model, kv_dim]` = `[2048, 256]` - reduced dimension (4 KV heads)

Each of the 4 KV heads is shared across 8 query heads (32 total / 4 KV = 8 queries per KV).

### Activation Functions

- **FFN**: SwiGLU (Swish-Gated Linear Unit)
  - `gate = ffn_gate(x)`
  - `up = ffn_up(x)`
  - `hidden = swish(gate) * up`
  - `output = ffn_down(hidden)`
- **Normalization**: RMSNorm (Root Mean Square Layer Normalization)

## Tensor Organization

### Token Embeddings

```
token_embd.weight : [2048, 32000]
```

**Note**: Shape is `[d_model, vocab_size]`, which is transposed from typical PyTorch format.

### Per-Layer Tensors (blk.0 through blk.21)

Each of the 22 transformer layers contains:

#### Attention Weights

```
blk.{i}.attn_norm.weight      : [2048]         # Pre-attention RMSNorm
blk.{i}.attn_q.weight          : [2048, 2048]  # Query projection (full)
blk.{i}.attn_k.weight          : [2048, 256]   # Key projection (grouped)
blk.{i}.attn_v.weight          : [2048, 256]   # Value projection (grouped)
blk.{i}.attn_output.weight     : [2048, 2048]  # Output projection
```

#### Feed-Forward Network (SwiGLU)

```
blk.{i}.ffn_norm.weight        : [2048]         # Pre-FFN RMSNorm
blk.{i}.ffn_gate.weight        : [2048, 5632]  # Gate projection
blk.{i}.ffn_up.weight          : [2048, 5632]  # Up projection
blk.{i}.ffn_down.weight        : [5632, 2048]  # Down projection
```

### Output Projection

```
output_norm.weight : [2048]         # Final RMSNorm
output.weight      : [2048, 32000]  # Logits projection to vocabulary
```

## Forward Pass Algorithm

### 1. Token Embedding

```
input_ids: [batch, seq_len] → int64 token IDs
embeddings = token_embd.weight[:, input_ids]  # [batch, seq_len, d_model]
x = embeddings
```

### 2. Transformer Layers (×22)

For each layer i from 0 to 21:

#### a) Self-Attention

```
# Pre-attention normalization
attn_input = RMSNorm(x, blk.i.attn_norm.weight)

# Query, Key, Value projections
Q = matmul(attn_input, blk.i.attn_q.weight)      # [batch, seq_len, 2048]
K = matmul(attn_input, blk.i.attn_k.weight)      # [batch, seq_len, 256]
V = matmul(attn_input, blk.i.attn_v.weight)      # [batch, seq_len, 256]

# Reshape for multi-head attention
Q = reshape(Q, [batch, seq_len, 32, 64])         # 32 query heads
K = reshape(K, [batch, seq_len, 4, 64])          # 4 KV heads
V = reshape(V, [batch, seq_len, 4, 64])          # 4 KV heads

# Repeat KV for grouped attention (each KV head serves 8 query heads)
K = repeat(K, groups=8)  # [batch, seq_len, 32, 64]
V = repeat(V, groups=8)  # [batch, seq_len, 32, 64]

# Scaled dot-product attention with causal mask
scores = matmul(Q, transpose(K)) / sqrt(64)      # [batch, 32, seq_len, seq_len]
scores = apply_causal_mask(scores)               # Mask future positions
attn_weights = softmax(scores, dim=-1)
attn_output = matmul(attn_weights, V)            # [batch, 32, seq_len, 64]

# Concatenate heads and project
attn_output = reshape(attn_output, [batch, seq_len, 2048])
attn_output = matmul(attn_output, blk.i.attn_output.weight)

# Residual connection
x = x + attn_output
```

#### b) Feed-Forward Network (SwiGLU)

```
# Pre-FFN normalization
ffn_input = RMSNorm(x, blk.i.ffn_norm.weight)

# SwiGLU: gate(x) * swish(up(x))
gate = matmul(ffn_input, blk.i.ffn_gate.weight)    # [batch, seq_len, 5632]
up = matmul(ffn_input, blk.i.ffn_up.weight)        # [batch, seq_len, 5632]
hidden = swish(gate) * up                          # Element-wise multiply
ffn_output = matmul(hidden, blk.i.ffn_down.weight) # [batch, seq_len, 2048]

# Residual connection
x = x + ffn_output
```

### 3. Output Projection

```
# Final normalization
x = RMSNorm(x, output_norm.weight)

# Project to vocabulary logits
logits = matmul(x, output.weight)  # [batch, seq_len, 32000]

# For next token prediction, take last position
next_token_logits = logits[:, -1, :]  # [batch, 32000]
```

## Special Tokens

```
BOS (Beginning of Sequence): <s> (token_id: 1)
EOS (End of Sequence): </s> (token_id: 2)
PAD (Padding): Not typically used in autoregressive generation
```

## ChatML Format

TinyLlama Chat uses the ChatML conversation format:

```
<|system|>
You are a helpful assistant.</s>
<|user|>
What is the capital of Japan?</s>
<|assistant|>
The capital of Japan is Tokyo.</s>
```

## Implementation Notes

### RMSNorm

```rust
fn rms_norm(x: Tensor, weight: Tensor, eps: f32) -> Tensor {
    // RMS = sqrt(mean(x^2) + eps)
    let squared = x.square();
    let mean_squared = squared.mean(dim=-1, keepdim=true);
    let rms = (mean_squared + eps).sqrt();

    // Normalize and scale
    let normalized = x / rms;
    normalized * weight
}
```

Typical `eps = 1e-6`

### SwiGLU

```rust
fn swigu(x: Tensor, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor) -> Tensor {
    let gate = x.matmul(gate_weight);
    let up = x.matmul(up_weight);

    // Swish(gate) * up
    let hidden = swish(gate) * up;

    hidden.matmul(down_weight)
}

fn swish(x: Tensor) -> Tensor {
    x * sigmoid(x)
}
```

### Grouped-Query Attention

Key difference from standard Multi-Head Attention:
- Query has 32 heads
- Key/Value have 4 heads
- Each KV head is shared across 8 query heads
- Reduces KV-cache size by 8x compared to Multi-Query Attention (MQA)

```rust
// Repeat KV heads to match query heads
fn repeat_kv(kv: Tensor, num_groups: usize) -> Tensor {
    // kv shape: [batch, seq_len, kv_heads, head_dim]
    // output shape: [batch, seq_len, kv_heads * num_groups, head_dim]
    kv.repeat_interleave(num_groups, dim=2)
}
```

### Causal Mask

```rust
fn create_causal_mask(seq_len: usize) -> Tensor {
    // Upper triangular matrix of -inf
    let mut mask = Tensor::zeros([seq_len, seq_len]);
    for i in 0..seq_len {
        for j in (i+1)..seq_len {
            mask[[i, j]] = f16::NEG_INFINITY;
        }
    }
    mask
}
```

## Memory Requirements

### Model Weights (Q4_0 quantized)
- File size: ~600MB
- In-memory (dequantized to f16): ~2.2GB

### Inference (per token)
- Activations: ~20MB (batch_size=1, seq_len=1)
- KV-Cache (full context 2048 tokens):
  - Per layer: `2 * 256 * 2048 * 2 bytes = ~2MB`
  - 22 layers: `~44MB`

### Total Inference Memory (2048 context)
- Model weights (f16): ~2.2GB
- KV-cache: ~44MB
- Activations: ~20MB
- **Total**: ~2.3GB

## Comparison to Standard LLaMA

| Feature | Standard LLaMA | TinyLlama |
|---------|---------------|-----------|
| Attention | Multi-Head (MHA) | Grouped-Query (GQA) |
| FFN | SwiGLU | SwiGLU |
| Norm | RMSNorm | RMSNorm |
| Vocab | 32K | 32K |
| Parameters | 7B-70B | 1.1B |
| Layers | 32-80 | 22 |
| d_model | 4096-8192 | 2048 |

TinyLlama is a scaled-down version optimized for:
- Faster inference
- Lower memory usage
- Edge deployment
- Educational purposes

## References

- [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama)
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
