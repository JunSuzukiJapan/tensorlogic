# GGUF Quantized Models Guide

This document explains how to load and use GGUF format quantized models (llama.cpp compatible) in TensorLogic.

## About GGUF Format

GGUF (GGML Universal Format) is an efficient quantization format for large language models developed by the llama.cpp project.

### Key Features

- Memory efficiency through 4-bit/8-bit quantization (up to 8x compression)
- Block-based quantization maintains accuracy
- Compatible with llama.cpp, Ollama, LM Studio, and more

### Quantization Formats Supported by TensorLogic

- ✅ **Q4_0**: 4-bit quantization (highest compression)
- ✅ **Q8_0**: 8-bit quantization (balanced precision and compression)
- ✅ **F16**: 16-bit floating point (high precision)
- ✅ **F32**: 32-bit floating point (highest precision)

## Basic Usage

### 1. Load Quantized Model

Automatically dequantized to f16 and loaded to Metal GPU:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. Get Tensors from Model

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## Choosing Quantization Format

### Q4_0 (4-bit)

- **Memory**: Minimal usage (~1/8 of original model)
- **Speed**: Fastest inference
- **Accuracy**: Slight degradation (usually acceptable)
- **Use Cases**: Chatbots, general text generation

### Q8_0 (8-bit)

- **Memory**: Moderate usage (~1/4 of original model)
- **Speed**: Fast
- **Accuracy**: High (nearly equivalent to F16)
- **Use Cases**: High-quality generation, coding assistants

### F16 (16-bit)

- **Memory**: ~1/2 of original model
- **Speed**: Standard
- **Accuracy**: TensorLogic native format, Metal GPU optimized
- **Use Cases**: When highest quality is required

## Practical Example: Token Embeddings

```tensorlogic
// Get token embeddings from LLama model
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// Get embedding vector from token ID
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## Memory Savings from Quantization

Example: LLama-7B model (7 billion parameters):

| Format     | Memory Usage | Compression |
|------------|--------------|-------------|
| F32 (orig) | ~28 GB       | 1x          |
| F16        | ~14 GB       | 2x          |
| Q8_0       | ~7 GB        | 4x          |
| Q4_0       | ~3.5 GB      | 8x          |

TensorLogic converts all formats to f16 on load and executes efficiently on Metal GPU.

## Downloading and Installing Models

### 1. Download GGUF Models from HuggingFace

Example: https://huggingface.co/TheBloke

### 2. Recommended Models (for beginners)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. Load in TensorLogic

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## Dequantization Mechanism

### Q4_0 Case

1. Group 32 4-bit values as a block
2. One f16 scale factor per block
3. Dequantization formula: `float_value = (quantized_value - 8) * scale`
4. Execute as f16 on Metal GPU

### Q8_0 Case

1. Group 32 8-bit values as a block
2. One f16 scale factor per block
3. Dequantization formula: `float_value = quantized_value * scale`
4. Execute as f16 on Metal GPU

## Performance Comparison (Inference Speed)

Inference speed on Metal GPU (M1 Max):

- **Q4_0**: ~50 tokens/sec (fastest)
- **Q8_0**: ~45 tokens/sec
- **F16**: ~40 tokens/sec (highest quality)

※ Varies by model size and complexity

## Frequently Asked Questions

### Q: What's the accuracy difference between quantized and non-quantized models?

A: Q4_0 shows ~2-3% quality degradation, Q8_0 is nearly negligible

### Q: Which quantization format should I choose?

A: Memory constrained → Q4_0, Quality focused → Q8_0, Highest quality → F16

### Q: Can I create quantized models in TensorLogic?

A: Currently read-only. Use SafeTensors format for saving

### Q: Are K-quants (Q4_K, Q5_K, etc.) supported?

A: Currently only Q4_0 and Q8_0. K-quants planned for future versions

## Practical Example: Simple Inference

```tensorlogic
// Load model (automatically placed on Metal GPU)
let model = load_model("mistral-7b-q4_0.gguf")

// Get token embedding table
let embeddings = model.get_tensor("token_embd.weight")

// Execute inference (using TensorLogic's standard operations)
let input_ids = tensor<int>([1, 2, 3, 4], device: gpu)
let embedded = embeddings[input_ids, :]
print("Embedded shape:", embedded.shape)
```

## Important Notes

- Quantized models are read-only (cannot save from TensorLogic)
- Use non-quantized models (F16/F32) for training
- Q4/Q8 optimized for inference only
- All quantization formats are automatically dequantized to f16 and loaded to GPU

## References

- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF models](https://huggingface.co/TheBloke)
- [Model Loading Guide](model_loading.md)
