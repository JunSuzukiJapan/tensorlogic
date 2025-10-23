# Model Loading Guide

This document explains how to load and use PyTorch and HuggingFace models in TensorLogic. It supports SafeTensors format (PyTorch compatible) and GGUF format (quantized LLMs).

## Basic Usage

### 1. Load SafeTensors Model (saved from PyTorch)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. Load GGUF Model (quantized LLM)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. Get Tensors from Model

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## Practical Example: Linear Layer Inference

Perform inference using model weights and biases:

```tensorlogic
fn forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // Linear transformation: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## Preparing PyTorch Models

Save your model in SafeTensors format using Python:

```python
import torch
from safetensors.torch import save_file

# Create PyTorch model
model = MyModel()

# Get model weights as dictionary
tensors = {name: param for name, param in model.named_parameters()}

# Save in SafeTensors format
save_file(tensors, "model.safetensors")
```

Then load in TensorLogic:

```tensorlogic
model = load_model("model.safetensors")
```

## Supported Formats

### 1. SafeTensors (.safetensors)

- PyTorch and HuggingFace compatible
- Supports F32, F64, F16, BF16 data types
- All data automatically converted to f16
- Loaded directly to Metal GPU

### 2. GGUF (.gguf)

- llama.cpp format quantized models
- Supports Q4_0, Q8_0, F32, F16
- Loaded directly to Metal GPU

### 3. CoreML (.mlmodel, .mlpackage)

- Apple Neural Engine optimized models
- iOS/macOS only

## Complete Linear Model Example

```tensorlogic
// Input data (batch size 4, feature dimension 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// Weight matrix (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// Bias (2 dimensions)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// Execute inference
let output = forward(X, W, b)

// Print results
print("Output shape:", output.shape)
print("Output:", output)
```

## Saving Models

You can save TensorLogic models in SafeTensors format:

```tensorlogic
save_model(model, "output.safetensors")
```

This enables interoperability with PyTorch and HuggingFace.

## Important Notes

- TensorLogic executes all operations in f16 (Metal GPU optimized)
- Other data types are automatically converted to f16 during loading
- Integer types (i8, i32, etc.) are not supported (floating-point only)
- Large models are automatically loaded to Metal GPU memory

## Related Documentation

- [GGUF Quantized Models](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [Getting Started Guide](../claudedocs/getting_started.md)
