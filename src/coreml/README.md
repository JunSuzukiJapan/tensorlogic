# CoreML Integration Module

Apple CoreML integration for TensorLogic, enabling Neural Engine acceleration on Apple Silicon devices.

## Overview

This module provides seamless integration between TensorLogic tensors and CoreML models, allowing you to leverage the Neural Engine for high-performance inference.

## Architecture

```
┌─────────────────┐
│ TensorLogic     │
│ Tensor (f16)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Conversion      │
│ Layer           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CoreML          │
│ MLMultiArray    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neural Engine   │
│ (Hardware)      │
└─────────────────┘
```

## Modules

### `model.rs`
CoreML model wrapper with:
- Model loading (.mlmodelc files)
- Inference execution
- Input/output shape validation
- Error handling

### `conversion.rs`
Tensor ↔ MLMultiArray conversion:
- `tensor_to_mlmultiarray()`: TensorLogic → CoreML
- `mlmultiarray_to_tensor()`: CoreML → TensorLogic
- Batch conversion support

## Quick Start

```rust
use tensorlogic::coreml::CoreMLModel;
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

// Load CoreML model
let model = CoreMLModel::load("model.mlmodelc")?;

// Create input tensor
let device = MetalDevice::new()?;
let input = Tensor::ones(&device, vec![1, 3, 224, 224])?;

// Run inference on Neural Engine
let output = model.predict(&input)?;

println!("Output: {:?}", output.shape().dims());
```

## Platform Support

| Platform | Status | Execution |
|----------|--------|-----------|
| macOS (Apple Silicon) | ✅ Full | Neural Engine |
| macOS (Intel) | ✅ Full | CPU/GPU |
| Linux | ⚠️ Limited | Fallback (Metal GPU) |
| Windows | ⚠️ Limited | Fallback (Metal GPU) |

## Implementation Status

### ✅ Complete
- [x] MLModel loading (macOS)
- [x] Input shape validation
- [x] Error handling
- [x] Tensor conversion layer
- [x] Performance benchmarks
- [x] Cross-platform builds

### 🔄 In Progress
- [ ] MLModel.prediction() full integration
- [ ] MLMultiArray data copy (actual data)
- [ ] Model metadata extraction

### 📋 Future
- [ ] Neural Engine utilization metrics
- [ ] Automatic model optimization
- [ ] Custom layer support

## Error Handling

The module provides comprehensive error types:

```rust
pub enum CoreMLError {
    ModelLoadError(String),      // Model file issues
    ModelCompileError(String),    // Compilation failures
    InferenceError(String),       // Runtime inference errors
    ConversionError(String),      // Data conversion issues
    InvalidInputShape { ... },    // Shape mismatches
    UnsupportedDataType(String),  // Type incompatibilities
    TensorError(TensorError),     // Underlying tensor errors
    IoError(std::io::Error),      // File system errors
}
```

## Performance

Benchmarks comparing Metal GPU vs CoreML inference:

```bash
cargo bench --bench coreml_benchmark
```

Expected performance (Apple M1):
- **Metal GPU**: 50-150 GFLOPS (matrix operations)
- **Neural Engine**: 11 TOPS (neural network inference)

## Dependencies

```toml
[dependencies]
objc2 = "0.5"
objc2-core-ml = { version = "0.2", features = ["MLMultiArray", "MLModel"] }
objc2-foundation = { version = "0.2", features = ["NSArray", "NSValue", "NSError", "NSString"] }
```

## Conditional Compilation

The module uses `#[cfg(target_os = "macos")]` for platform-specific code:

```rust
#[cfg(target_os = "macos")]
{
    // Actual CoreML/Neural Engine code
    let ml_model = MLModel::modelWithContentsOfURL_error(&url)?;
}

#[cfg(not(target_os = "macos"))]
{
    // Fallback or placeholder implementation
}
```

## Testing

Run CoreML-specific tests:

```bash
cargo test --lib coreml
```

All tests work on both macOS and non-macOS platforms through conditional compilation.

## API Limitations

**objc2-core-ml 0.2 Constraints**:

The current version of `objc2-core-ml` provides minimal bindings. Some advanced CoreML features require additional work:

- **MLFeatureValue**: Not yet exposed in objc2-core-ml 0.2
- **MLFeatureProvider**: Requires custom implementation
- **MLMultiArray data access**: Limited to basic operations

**Workarounds**:
- Model loading: ✅ Fully functional
- Shape validation: ✅ Fully functional
- Data conversion: 🔄 Validation and logging (full copy pending)
- Prediction: 🔄 Framework in place (full integration pending)

## Examples

### Basic Inference

```rust
let model = CoreMLModel::load("mobilenet.mlmodelc")?;
let output = model.predict(&input_image)?;
```

### Batch Processing

```rust
let inputs = vec![image1, image2, image3];
let outputs = model.predict_batch(&inputs)?;
```

### Error Handling

```rust
match model.predict(&input) {
    Ok(output) => process(output),
    Err(CoreMLError::InvalidInputShape { expected, actual }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(e) => eprintln!("Inference error: {}", e),
}
```

## Contributing

Improvements welcome, especially:
- Full MLModel.prediction() integration
- MLMultiArray data copy optimization
- Additional CoreML feature support

## References

- [Apple CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [objc2-core-ml Crate](https://docs.rs/objc2-core-ml)
- [Neural Engine Optimization](https://developer.apple.com/documentation/coreml/optimizing_a_model_on_the_neural_engine)

---

**Module Version**: 0.1.0
**Last Updated**: 2025-10-20
**Status**: Production-ready for model loading and shape validation, inference integration in progress
