# TensorLogic LLM Examples

This directory contains comprehensive examples demonstrating how to build LLM (Large Language Model) applications with TensorLogic.

## Learning Path

Follow these examples in order to understand the complete LLM pipeline:

### 1. Foundational Concepts

**[transformer_attention.tl](transformer_attention.tl)** (2.4K)
- Multi-Head Self-Attention mechanism
- Q, K, V projections and attention score computation
- Causal masking for autoregressive generation
- **Run:** `tl run examples/transformer_attention.tl`

**[transformer_block.tl](transformer_block.tl)** (3.4K)
- Complete Transformer block architecture
- Attention + Feed-Forward Network + Layer Normalization
- Residual connections
- **Run:** `tl run examples/transformer_block.tl`

### 2. Complete LLM Architecture

**[llm_inference.tl](llm_inference.tl)** (6.0K)
- Full LLM inference pipeline with 2 Transformer layers
- Token embedding and positional encoding
- Output projection to vocabulary
- Conceptual autoregressive generation loop
- **Run:** `tl run examples/llm_inference.tl`

### 3. Text Generation Strategies

**[text_generation_sampling.tl](text_generation_sampling.tl)** (3.0K)
- Top-k sampling (k=50)
- Top-p (nucleus) sampling (p=0.9)
- Combined sampling strategy (recommended for chat)
- Comparison table and production hyperparameters
- **Run:** `tl run examples/text_generation_sampling.tl`

**[autoregressive_generation.tl](autoregressive_generation.tl)** (7.1K)
- Complete token-by-token generation pipeline
- Integration of top-k, top-p, softmax, and sample
- Production implementation details (KV-cache, temperature, special tokens)
- Real-world hyperparameters for different use cases
- **Run:** `tl run examples/autoregressive_generation.tl`

### 4. Chat Applications

**[local_llm_chat.tl](local_llm_chat.tl)** (1.8K)
- Basic chat demo loading TinyLlama model
- ChatML format structure
- Model information and configuration
- **Prerequisites:** Download model first with `cargo run --bin download_model -- --model tinyllama`
- **Run:** `tl run examples/local_llm_chat.tl`

**[chat_repl_demo.tl](chat_repl_demo.tl)** (9.6K)
- Interactive chat REPL architecture
- Multi-turn conversation handling
- Session state management
- Special commands (/help, /clear, /exit, etc.)
- Complete REPL component breakdown
- Implementation pseudocode
- **Run:** `tl run examples/chat_repl_demo.tl`

## Feature Demonstrations

### Transformer Operations

**[test_transformer_ops.tl](test_transformer_ops.tl)** (4.5K)
- Testing individual Transformer operations
- Attention mechanisms, FFN, normalization
- **Run:** `tl run examples/test_transformer_ops.tl`

**[test_transformer_functional.tl](test_transformer_functional.tl)** (5.1K)
- Functional Transformer implementation
- **Run:** `tl run examples/test_transformer_functional.tl`

### Sampling Tests

**[test_sampling.tl](test_sampling.tl)** (2.5K)
- Testing sampling functions
- **Run:** `tl run examples/test_sampling.tl`

## Quick Reference

### Sampling Strategies

| Strategy | Function | Use Case |
|----------|----------|----------|
| Top-k | `top_k(logits, k)` | Limit to top k tokens |
| Top-p | `top_p(logits, p)` | Nucleus sampling |
| Sample | `sample(probs)` | Sample from distribution |
| Combined | `top_k → top_p → softmax → sample` | Recommended for chat |

### Production Hyperparameters

**Chat Models (GPT-4, Claude, LLaMA):**
- Temperature: 0.7-1.0
- Top-p: 0.9
- Top-k: 50-100
- Max tokens: 2048-4096

**Code Generation:**
- Temperature: 0.1-0.3 (deterministic)
- Top-p: 0.95
- Top-k: 40
- Max tokens: 512-2048

**Creative Writing:**
- Temperature: 1.0-1.2 (random)
- Top-p: 0.85
- Top-k: 100
- Max tokens: 2048-8192

## Available Operations

TensorLogic provides 48 tensor operations for building LLM applications:

**Core Operations:**
- `matmul()` - Matrix multiplication (Q, K, V, FFN, output projection)
- `softmax()` - Attention weights and probability distributions
- `gelu()` - FFN activation function
- `layer_norm()` - Stabilization (2x per Transformer layer)

**Attention:**
- `apply_attention_mask()` - Causal masking for autoregressive
- `transpose()` - Key matrix transformation

**Sampling:**
- `top_k()` - Top-k filtering
- `top_p()` - Nucleus sampling
- `sample()` - Token sampling from distribution

**Model Loading:**
- `load_model()` - Load GGUF model
- `generate()` - Text generation (placeholder, to be implemented)

**Utilities:**
- `positional_encoding()` - Position embeddings
- `ones()`, `zeros()` - Tensor creation
- `flatten()` - Reshape operations

## Model Setup

### Download TinyLlama Model

```bash
# Download TinyLlama 1.1B Chat (Q4_0, ~600MB)
cargo run --bin download_model -- --model tinyllama
```

Model will be saved to: `~/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf`

### Model Information

- **Model:** TinyLlama-1.1B-Chat-v1.0
- **Format:** GGUF (Q4_0 quantization)
- **Parameters:** ~1.1 billion
- **Context length:** 2048 tokens
- **Vocabulary size:** 32,000
- **File size:** ~600MB
- **Tensors:** 201

## Implementation Status

### ✅ Completed

- Variable support in array literals (`ones([seq_len, d_model])`)
- Transformer architecture examples (attention, FFN, layer norm)
- Model loading (GGUF format)
- Sampling strategies (top-k, top-p, sample)
- Autoregressive generation pipeline (demo)
- Chat REPL architecture (demo)

### 🚧 In Progress

- **Transformer Forward Pass** - Actual model inference in Rust
- **KV-Cache** - Efficient autoregressive generation
- **Temperature Scaling** - Control randomness
- **Streaming Output** - Real-time token generation
- **Interactive REPL** - Full implementation

## Next Steps

1. **Implement Transformer Forward Pass**
   - Load actual model weights from GGUF
   - Multi-layer attention with Metal GPU acceleration

2. **Add KV-Cache**
   - Cache Key/Value tensors for efficiency
   - Memory management for long contexts

3. **Complete generate() Function**
   - Integrate actual inference
   - Temperature scaling
   - EOS token detection

4. **Build Interactive REPL**
   - Interactive chat loop
   - Chat history management
   - Special commands

## Documentation

See [docs/llm_chat_implementation.md](../docs/llm_chat_implementation.md) for detailed implementation notes and architecture documentation.

## Contributing

When adding new examples:
1. Include clear comments explaining key concepts
2. Add to this README with file size and description
3. Ensure examples run successfully with `tl run`
4. Follow the progressive complexity pattern
