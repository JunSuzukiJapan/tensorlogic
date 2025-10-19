# Phase 8.3: CoreML Model Integration Design

## Overview

Integrate actual CoreML models for Neural Engine inference execution, enabling real ML model deployment on Apple Silicon.

## Design Goals

1. **CoreML Model Support**: Load and compile .mlmodel files
2. **Model Caching**: Cache compiled models for performance
3. **Type Safety**: Ensure f16 compatibility with Neural Engine
4. **Integration**: Seamless integration with existing NeuralEngineBuffer
5. **Performance**: Zero-copy where possible, minimize conversions

## Architecture

```
┌─────────────────────────────────────────┐
│     CoreMLModelManager (Singleton)      │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │  Model Cache                     │  │
│  │  - HashMap<path, MLModel>        │  │
│  │  - Lazy compilation              │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │  Model Loader                    │  │
│  │  - Load from .mlmodel            │  │
│  │  - Validate input/output shapes  │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │  Inference Engine                │  │
│  │  - Execute predictions           │  │
│  │  - Manage MLMultiArray I/O       │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│    NeuralEngineOps (Enhanced)           │
│  - matmul, relu via CoreML models       │
│  - Fallback to CPU if model unavailable │
└─────────────────────────────────────────┘
```

## Data Structures

```rust
pub struct CoreMLModelManager {
    /// Cache of compiled models (path → MLModel)
    model_cache: Arc<Mutex<HashMap<String, Retained<MLModel>>>>,

    /// Configuration for model compilation
    config: MLModelConfiguration,
}

pub struct ModelInput {
    /// Input buffer (Neural Engine format)
    buffer: NeuralEngineBuffer,

    /// Input name in CoreML model
    name: String,
}

pub struct ModelOutput {
    /// Output buffer (Neural Engine format)
    buffer: NeuralEngineBuffer,

    /// Output name in CoreML model
    name: String,
}

pub struct InferenceRequest {
    /// Model file path
    model_path: String,

    /// Input tensors
    inputs: Vec<ModelInput>,

    /// Expected output names
    output_names: Vec<String>,
}
```

## CoreML Integration Details

### Model Loading

```rust
impl CoreMLModelManager {
    pub fn load_model(&self, path: &str) -> TensorResult<Retained<MLModel>> {
        // Check cache first
        if let Some(model) = self.model_cache.lock().unwrap().get(path) {
            return Ok(model.clone());
        }

        // Load model from file
        let url = NSURL::fileURLWithPath(&NSString::from_str(path));
        let model = MLModel::modelWithContentsOfURL_configuration_error(
            &url,
            &self.config,
        )?;

        // Cache for reuse
        self.model_cache.lock().unwrap().insert(path.to_string(), model.clone());

        Ok(model)
    }
}
```

### Inference Execution

```rust
impl CoreMLModelManager {
    pub fn predict(
        &self,
        model_path: &str,
        inputs: Vec<ModelInput>,
    ) -> TensorResult<Vec<ModelOutput>> {
        let model = self.load_model(model_path)?;

        // Create MLFeatureProvider from inputs
        let input_dict = NSMutableDictionary::new();
        for input in &inputs {
            let feature_value = MLFeatureValue::featureValueWithMultiArray(
                &input.buffer.array
            );
            input_dict.setObject_forKey(&feature_value, &NSString::from_str(&input.name));
        }

        let input_provider = MLDictionaryFeatureProvider::initWithDictionary_error(
            input_dict
        )?;

        // Execute prediction
        let output_provider = model.predictionFromFeatures_error(&input_provider)?;

        // Extract outputs
        let mut outputs = Vec::new();
        for name in output_names {
            let feature = output_provider.featureValueForName(&NSString::from_str(&name))?;
            let array = feature.multiArrayValue()?;
            outputs.push(ModelOutput {
                buffer: NeuralEngineBuffer::from_mlmultiarray(array),
                name: name.clone(),
            });
        }

        Ok(outputs)
    }
}
```

## Integration with Existing Operations

### Enhanced NeuralEngineOps

```rust
impl NeuralEngineOps {
    /// Matrix multiplication using CoreML model
    pub fn matmul_coreml(
        &self,
        a: &NeuralEngineBuffer,
        b: &NeuralEngineBuffer,
        model_path: &str,
    ) -> TensorResult<NeuralEngineBuffer> {
        let manager = CoreMLModelManager::global();

        let inputs = vec![
            ModelInput { buffer: a.clone(), name: "input_a".to_string() },
            ModelInput { buffer: b.clone(), name: "input_b".to_string() },
        ];

        let outputs = manager.predict(model_path, inputs)?;
        Ok(outputs[0].buffer.clone())
    }

    /// ReLU using CoreML model
    pub fn relu_coreml(
        &self,
        input: &NeuralEngineBuffer,
        model_path: &str,
    ) -> TensorResult<NeuralEngineBuffer> {
        let manager = CoreMLModelManager::global();

        let inputs = vec![
            ModelInput { buffer: input.clone(), name: "input".to_string() },
        ];

        let outputs = manager.predict(model_path, inputs)?;
        Ok(outputs[0].buffer.clone())
    }
}
```

## Model Requirements

### Input/Output Specifications

CoreML models must follow these requirements:

1. **Data Type**: Float16 (MLMultiArrayDataTypeFloat16)
2. **Input Names**: Consistent naming convention ("input", "input_a", "input_b")
3. **Output Names**: Consistent naming convention ("output", "result")
4. **Shapes**: Match tensor shapes exactly (no broadcasting)

### Example Model Creation (Python)

```python
import coremltools as ct
import numpy as np

# Define a simple matmul model
class MatMulModel:
    def forward(self, a, b):
        return np.matmul(a, b)

# Convert to CoreML with Float16
model = ct.convert(
    MatMulModel(),
    inputs=[
        ct.TensorType(name="input_a", shape=(2, 3), dtype=np.float16),
        ct.TensorType(name="input_b", shape=(3, 4), dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="output", dtype=np.float16),
    ],
)

# Save model
model.save("matmul_f16.mlmodel")
```

## Configuration Options

```rust
pub struct CoreMLConfig {
    /// Use Neural Engine if available
    pub use_neural_engine: bool,

    /// Maximum model cache size
    pub max_cache_size: usize,

    /// Model directory path
    pub model_dir: PathBuf,
}

impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            use_neural_engine: true,
            max_cache_size: 10,
            model_dir: PathBuf::from("./models"),
        }
    }
}
```

## Performance Considerations

### Model Caching Strategy

- **Lazy Loading**: Load models on first use
- **LRU Eviction**: Evict least recently used when cache is full
- **Thread Safety**: Arc<Mutex<>> for concurrent access

### Zero-Copy Optimization

- **MLMultiArray**: Use existing memory when possible
- **Buffer Sharing**: NeuralEngineBuffer already uses MLMultiArray
- **Avoid Conversions**: Keep data in f16 format throughout

## Testing Strategy

### Unit Tests

1. **Model Loading**: Test .mlmodel file loading and caching
2. **Inference**: Test prediction with known inputs/outputs
3. **Error Handling**: Test invalid models, missing files
4. **Caching**: Test cache hit/miss, eviction

### Integration Tests

1. **End-to-End**: Load model → infer → validate output
2. **Performance**: Benchmark CoreML vs CPU/Metal
3. **Memory**: Verify zero-copy where possible

## Implementation Phases

### Phase 8.3.1: CoreMLModelManager ✅ (Target)
- [x] Create CoreMLModelManager structure
- [x] Implement model loading and caching
- [x] Add singleton pattern for global access

### Phase 8.3.2: Inference Engine
- [ ] Implement prediction execution
- [ ] MLFeatureProvider integration
- [ ] Output extraction

### Phase 8.3.3: NeuralEngineOps Integration
- [ ] Update matmul to use CoreML
- [ ] Update relu to use CoreML
- [ ] Fallback logic for missing models

### Phase 8.3.4: Testing & Validation
- [ ] Create test CoreML models
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

## Future Enhancements

1. **Dynamic Model Generation**: Generate CoreML models on-the-fly
2. **Model Optimization**: Quantization, pruning for Neural Engine
3. **Multi-Model Fusion**: Combine multiple models for complex operations
4. **Batch Inference**: Support for batched predictions
5. **Async Inference**: Non-blocking predictions
6. **Model Versioning**: Support multiple versions of same model

## Notes

- CoreML models are compiled on device first time they're loaded
- Neural Engine availability is platform-dependent (Apple Silicon only)
- Float16 support in CoreML requires iOS 16+ / macOS 13+
- Model files should be included in app bundle or downloaded securely
