# CoreML & Neural Engine Integration Guide

This guide explains how to use CoreML models in TensorLogic and perform high-speed inference on the Apple Neural Engine.

## About CoreML and Neural Engine

### CoreML

- Apple's proprietary machine learning framework
- Optimized exclusively for iOS/macOS
- Automatically leverages Neural Engine, GPU, and CPU
- .mlmodel / .mlmodelc format

### Neural Engine

- AI-dedicated chip exclusive to Apple Silicon
- Up to 15.8 TOPS (M1 Pro/Max)
- Ultra-low power consumption (1/10 or less compared to GPU)
- Optimized for f16 operations

### Integration with TensorLogic

- All f16 operations (Neural Engine optimized)
- Seamless integration with Metal GPU
- Automatic model format detection

## Creating CoreML Models

CoreML models are typically created with Python's coreMLtools:

```python
import coremltools as ct
import torch

# Create PyTorch model
model = MyModel()
model.eval()

# Create traced model
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Neural Engine optimization
    compute_precision=ct.precision.FLOAT16  # f16 precision
)

# Save
mlmodel.save("model.mlpackage")
```

## Using in TensorLogic

### 1. Load CoreML Model (macOS only)

```tensorlogic
model = load_model("model.mlpackage")
// or
model = load_model("model.mlmodelc")
```

### 2. Check Metadata

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Neural Engine Optimization Best Practices

### 1. Data Type: Use f16

✅ Recommended: `compute_precision=ct.precision.FLOAT16`
❌ Not recommended: FLOAT32 (executed on GPU)

### 2. Model Format: Use mlprogram format

✅ Recommended: `convert_to="mlprogram"`
❌ Not recommended: `convert_to="neuralnetwork"` (legacy format)

### 3. Batch Size: 1 is optimal

✅ Recommended: `batch_size=1`
⚠️ Note: `batch_size>1` may execute on GPU

### 4. Input Size: Fixed size is optimal

✅ Recommended: `shape=[1, 3, 224, 224]`
⚠️ Note: Variable sizes have limited optimization

## Supported Operations

### Operations executed fast on Neural Engine

- ✅ Convolutions (conv2d, depthwise_conv)
- ✅ Fully connected layers (linear, matmul)
- ✅ Pooling (max_pool, avg_pool)
- ✅ Normalization (batch_norm, layer_norm)
- ✅ Activation functions (relu, gelu, sigmoid, tanh)
- ✅ Element-wise operations (add, mul, sub, div)

### Operations executed on GPU

- ⚠️ Custom operations
- ⚠️ Complex control flow
- ⚠️ Non-standard activation functions

## Performance Comparison

ResNet-50 inference (224x224 image):

| Device             | Latency | Power   | Efficiency |
|-------------------|---------|---------|------------|
| Neural Engine     | ~3ms    | ~0.5W   | Highest    |
| Metal GPU (M1)    | ~8ms    | ~5W     | Medium     |
| CPU (M1)          | ~50ms   | ~2W     | Low        |

※ Neural Engine has overwhelming power efficiency, especially for continuous inference

## Practical Example: Image Classification Model

```tensorlogic
// Load CoreML model
let model = load_model("resnet50.mlpackage")

// Prepare image data (224x224x3, f16 format)
let image = load_image("cat.jpg")
let preprocessed = preprocess(image)  // normalize, resize

// Execute inference on Neural Engine
let output = model.predict(preprocessed)
let class_id = argmax(output)
print("Predicted class:", class_id)
```

## TensorLogic → CoreML Export

Export TensorLogic-trained models to CoreML:

### 1. Save TensorLogic Model in SafeTensors Format

```tensorlogic
save_model(tl_model, "model.safetensors")
```

### 2. Load & Convert to CoreML in Python

```python
import torch
from safetensors.torch import load_file
import coremltools as ct

# Load SafeTensors
weights = load_file("model.safetensors")

# Load weights into PyTorch model
model = MyModel()
model.load_state_dict(weights)
model.eval()

# Convert to CoreML
example_input = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16
)
mlmodel.save("model.mlpackage")
```

### 3. Load CoreML Model in TensorLogic

```tensorlogic
model = load_model("model.mlpackage")
```

## Neural Engine Constraints

### 1. macOS/iOS Only

- Requires Apple Silicon
- M1/M2/M3/M4 series

### 2. Inference Only

- Training uses Metal GPU
- Only inference on Neural Engine

### 3. Model Size Limits

- Recommended: < 1GB
- Consider split inference for large models

### 4. Operation Constraints

- f16 only (f32 automatically uses GPU)
- Optimal for standard convolutional neural networks
- Transformers also supported (recent models)

## Model Format Selection

### Recommended Formats by Use Case

#### Training: SafeTensors

- PyTorch compatible
- Weight saving/loading
- Train on Metal GPU

#### Inference (iOS/macOS): CoreML

- Neural Engine optimization
- Ultra-low power consumption
- App integration

#### Inference (General): GGUF

- Quantization support
- Cross-platform
- Memory efficient

#### Development/Debug: SafeTensors

- Direct weight access
- Flexible modifications

## References

- [CoreML Official Documentation](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Neural Engine Guide](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Model Loading Guide](model_loading.md)
