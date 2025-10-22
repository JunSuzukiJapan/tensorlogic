# TensorLogic

A production-ready f16 tensor library for Apple Silicon with automatic differentiation and optimizers, featuring Metal GPU and Neural Engine acceleration.

> **⚠️ Experimental Project Notice**
>
> This is an **experimental research project** with the following constraints:
> - **Apple Silicon Only**: Requires macOS with M-series chips (M1/M2/M3/M4)
> - **f16 (half-precision) Only**: All floating-point operations use 16-bit floats exclusively
> - **Metal Framework Required**: GPU acceleration relies on Apple's Metal framework
>
> Not intended for production use on non-Apple hardware or general-purpose computing.

## Overview

TensorLogic is a unified tensor algebra library designed specifically for Apple Silicon (M-series chips), providing seamless integration between Metal GPU and Neural Engine through CoreML. All operations maintain f16 (half-precision) throughout for optimal Neural Engine compatibility and performance.

**📚 [Getting Started Guide](claudedocs/getting_started.md)** | **🤖 [LLM Learning Guide](examples/LLM_GUIDE.md)** | **📖 [Optimizer Tutorial](claudedocs/optimizer_tutorial.md)** | **📦 [Model Loading](examples/model_loading.tl)** | **⚡ [GGUF Quantization](examples/gguf_quantized_models.tl)** | **🧠 [CoreML & Neural Engine](examples/coreml_neural_engine.tl)** | **🔧 [Full Specification](claudedocs/f16_neural_engine_metal_spec.md)**

## 📖 Documentation

**English**: [Model Loading](docs/en/model_loading.md) | [GGUF Quantization](docs/en/gguf_quantization.md) | [CoreML & Neural Engine](docs/en/coreml_neural_engine.md)

**日本語**: [モデルローディング](docs/ja/model_loading.md) | [GGUF量子化](docs/ja/gguf_quantization.md) | [CoreML & Neural Engine](docs/ja/coreml_neural_engine.md)

**Deutsch**: [Modell-Laden](docs/de/model_loading.md) | [GGUF Quantisierung](docs/de/gguf_quantization.md) | [CoreML & Neural Engine](docs/de/coreml_neural_engine.md)

**Español**: [Carga de Modelos](docs/es/model_loading.md) | [Cuantización GGUF](docs/es/gguf_quantization.md) | [CoreML & Neural Engine](docs/es/coreml_neural_engine.md)

**Français**: [Chargement de Modèles](docs/fr/model_loading.md) | [Quantification GGUF](docs/fr/gguf_quantization.md) | [CoreML & Neural Engine](docs/fr/coreml_neural_engine.md)

**Italiano**: [Caricamento Modelli](docs/it/model_loading.md) | [Quantizzazione GGUF](docs/it/gguf_quantization.md) | [CoreML & Neural Engine](docs/it/coreml_neural_engine.md)

**한국어**: [모델 로딩](docs/ko/model_loading.md) | [GGUF 양자화](docs/ko/gguf_quantization.md) | [CoreML & Neural Engine](docs/ko/coreml_neural_engine.md)

**Português**: [Carregamento de Modelos](docs/pt/model_loading.md) | [Quantização GGUF](docs/pt/gguf_quantization.md) | [CoreML & Neural Engine](docs/pt/coreml_neural_engine.md)

**Русский**: [Загрузка Моделей](docs/ru/model_loading.md) | [Квантование GGUF](docs/ru/gguf_quantization.md) | [CoreML & Neural Engine](docs/ru/coreml_neural_engine.md)

**中文**: [模型加载](docs/zh/model_loading.md) | [GGUF量化](docs/zh/gguf_quantization.md) | [CoreML & Neural Engine](docs/zh/coreml_neural_engine.md)

## Key Features

### ✅ Production Ready

- **Tensor Operations** (Phase 1-3)
  - Element-wise operations: add, sub, mul, div
  - Matrix multiplication with GPU optimization
  - Activation functions: ReLU, GELU, Softmax
  - Broadcasting (NumPy-compatible)
  - Reductions: sum, mean, max, min (global + dimension-specific)
  - Einstein summation with pattern optimization
  - In-place operations for memory efficiency

- **Automatic Differentiation** (Phase 5-6) ⚡
  - Dynamic computation graph (PyTorch-style)
  - Full backward pass implementation
  - Gradient computation for all operations
  - Second-order derivatives (Hessian foundation)
  - Gradient checking with numerical validation
  - Create graph mode for higher-order derivatives

- **Optimizers** (Phase 9.1) 🚀
  - **SGD**: Basic gradient descent, momentum, Nesterov
  - **Adam**: Adaptive learning rates with AMSGrad support
  - **AdamW**: Decoupled weight decay for better regularization
  - Learning rate scheduling
  - State save/load for checkpointing
  - Multi-parameter group support

- **Model Loading & Quantization** (Phase 1-3) 🆕 NEW
  - **SafeTensors**: PyTorch/HuggingFace compatible (F32, F64, F16, BF16)
  - **GGUF**: Quantized LLM models (Q4_0, Q8_0, F16, F32)
  - **Automatic dequantization**: All formats → f16 → Metal GPU
  - **Memory efficient**: 4-bit models use ~8x less memory
  - **llama.cpp compatible**: Load Llama, Mistral, Phi models
  - Examples: [Model Loading](examples/model_loading.tl) | [GGUF Guide](examples/gguf_quantized_models.tl)

- **Device Acceleration** (Phase 2, 4, 7)
  - Metal GPU acceleration for all operations
  - Neural Engine integration (foundation complete)
  - Zero-copy Metal ↔ Neural Engine conversion
  - Buffer pooling for memory optimization
  - Fused operations (add+relu, mul+relu, affine)
  - Metal GPU gradient kernels
  - ExecutionPlanner for automatic device selection

- **Python Integration** (Phase 1-3) 🐍 NEW
  - Python module import: `python import numpy as np`
  - Call Python functions: `python.call("np.sum", x)`
  - Seamless Tensor ↔ NumPy conversion (f16 ↔ f32)
  - NumPy, PyTorch, SciKit-Learn integration
  - Python bindings via PyO3
  - Jupyter-ready architecture

### 🚧 Advanced Features (Optional)

- **Neural Engine Inference**: Full CoreML model integration (foundation complete, deferred)
- **Learning Rate Schedulers**: Cosine, step decay, warmup (coming soon)
- **Additional Optimizers**: RMSprop, Adagrad (future work)

## Quick Start

See the [**Getting Started Guide**](claudedocs/getting_started.md) for comprehensive tutorials.

### Installation

**Option 1: CLI Binary**

Install the TensorLogic interpreter as a binary:

```bash
cargo install --git https://github.com/JunSuzukiJapan/tensorlogic.git
```

This installs the `tl` command.

Or build from source:

```bash
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
cargo build --release
# Binary will be at: target/release/tl
```

**Option 2: Python Module** 🐍 NEW

Install TensorLogic as a Python package:

```bash
# Install maturin
pip install maturin

# Build and install wheel
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
maturin build --features python-extension --release
pip install target/wheels/tensorlogic-*.whl
```

Use in Python:

```python
import tensorlogic as tl
import numpy as np

# Create interpreter
interp = tl.Interpreter()

# Execute TensorLogic code
interp.execute("""
    main {
        python import numpy as np

        tensor x: float16[3] = [1.0, 2.0, 3.0]
        tensor sum_x: float16[1] = python.call("np.sum", x)

        print("Sum:", sum_x)
    }
""")
# Output: ✓ Python import: numpy (as np)
#         Sum: [6.0000]
```

### Basic Usage

TensorLogic is an interpreted language for tensor operations and neural network training.

#### 1. Tensor Declaration

Declare tensors with the `float16` type (TensorLogic uses 16-bit floating point for GPU efficiency):

```tensorlogic
// Scalar tensor
tensor x: float16[1] = [2.0]

// Vector tensor
tensor v: float16[3] = [1.0, 2.0, 3.0]

// Matrix tensor
tensor W: float16[2, 3] = [[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]

// Learnable parameter (for training)
tensor w: float16[1] learnable = [0.5]
```

#### 2. Tensor Operations

Perform element-wise and matrix operations:

```tensorlogic
main {
    // Element-wise operations
    result := a + b      // Addition
    result := a - b      // Subtraction
    result := a * b      // Multiplication
    result := a / b      // Division

    // Matrix operations
    result := A @ B      // Matrix multiplication
    result := a ** 2     // Power

    // Activation functions
    result := relu(x)
    result := gelu(x)

    // Reductions
    sum := sum(tensor)
    mean := mean(tensor)
}
```

#### 3. Importing External Files

Import declarations from other TensorLogic files:

```tensorlogic
// Import tensor and function definitions from another file
import "path/to/module.tl"

main {
    // Use imported tensors and functions
    result := imported_tensor * 2
}
```

**Features**:
- Relative path resolution (relative to the importing file)
- Duplicate import prevention (same file won't be imported twice)
- No main block execution (only declarations are imported)

**Example**: See [examples/import_test/](examples/import_test/)

#### 4. Control Flow

Use conditionals and loops:

```tensorlogic
main {
    // If statement
    if x > 0 {
        y := x * 2
    }

    // If-else statement
    if x > 0 {
        y := x * 2
    } else {
        y := x * 0.5
    }

    // For loop
    for i in 0..10 {
        x := x + i
    }

    // While loop
    while x < 100 {
        x := x * 1.1
    }
}
```

#### 5. Training with Gradient Descent

Train models using the `learn` statement with support for local variables:

```tensorlogic
// Declare learnable parameters
tensor W: float16[1] learnable = [0.5]

main {
    // Training data
    tensor x1: float16[1] = [1.0]
    tensor y1: float16[1] = [3.0]
    tensor x2: float16[1] = [-2.0]  // Negative numbers supported
    tensor y2: float16[1] = [-6.0]

    // Train model with local variables in learn block
    learn {
        // Local variables for intermediate computations
        pred1 := x1 * W
        pred2 := x2 * W

        // Compute loss
        err1 := pred1 - y1
        err2 := pred2 - y2
        loss := err1 * err1 + err2 * err2

        objective: loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }

    print("Learned W:", W)  // Should be close to 3.0
}
```

**Features**:
- **Local variables**: Use `:=` inside `learn` blocks for intermediate computations
- **Negative numbers**: Full support for negative numeric literals
- **Multiple optimizers**: SGD, Adam, AdamW with customizable hyperparameters

Available optimizers:
- `sgd(lr: 0.1)` - Stochastic Gradient Descent
- `adam(lr: 0.001)` - Adam optimizer
- `adamw(lr: 0.001, weight_decay: 0.01)` - AdamW with weight decay

#### 6. Run Your Program

```bash
# Run TensorLogic script
tl run your_script.tl

# With debug mode
tl run your_script.tl --debug

# Start REPL (interactive mode)
tl repl

# Show version
tl --version

# Show help
tl --help
```

### Complete Example

Here's a complete linear regression example ([examples/tutorial_01_linear_regression.tl](examples/tutorial_01_linear_regression.tl)):

```tensorlogic
// Declare learnable parameters
tensor w: float16[1] learnable = [0.5]
tensor b: float16[1] learnable = [0.5]

main {
    // Training: minimize loss function
    // Loss = w^2 + b^2 (converges to w=0, b=0)

    learn {
        objective: w * w + b * b,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

Run it:
```bash
tl run examples/tutorial_01_linear_regression.tl
```

### More Examples

- [Simple Linear Model](examples/simple_linear_model.tl) - Training and inference example with local variables
- [Tutorial 01: Linear Regression](examples/tutorial_01_linear_regression.tl) - Basic optimization
- [Tutorial 02: Multi-Parameter Optimization](examples/tutorial_02_logistic_regression.tl) - Multiple parameters
- [Tutorial 03: Neural Network Weights](examples/tutorial_03_neural_network.tl) - Weight regularization
- [Tutorial 04: Logic Programming](examples/tutorial_04_logic_programming.tl) - Neural-symbolic integration
- [Import Test](examples/import_test/) - External file imports with circular dependency detection
- [**LLM Learning Guide**](examples/LLM_GUIDE.md) - 🤖 Build language models: embeddings, attention, next-token prediction
- [Getting Started Guide](claudedocs/getting_started.md) - Comprehensive tutorials
- [Language Reference](docs/en/language_reference.md) - Complete syntax reference

## Python Integration 🐍

TensorLogic can seamlessly integrate with Python libraries like NumPy, PyTorch, and SciKit-Learn.

### Importing Python Modules

```tensorlogic
main {
    // Import Python modules with optional aliases
    python import numpy as np
    python import torch
    python import sklearn.preprocessing as preprocessing
}
```

### Calling Python Functions

```tensorlogic
main {
    python import numpy as np

    // Create TensorLogic tensor
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[3] = [4.0, 5.0, 6.0]

    // Call NumPy functions
    tensor sum_result: float16[3] = python.call("np.add", x, y)
    tensor mean_x: float16[1] = python.call("np.mean", x)
    tensor max_y: float16[1] = python.call("np.max", y)

    print("Add:", sum_result)   // [5.0, 7.0, 9.0]
    print("Mean:", mean_x)      // [2.0]
    print("Max:", max_y)        // [6.0]
}
```

### Tensor ↔ NumPy Conversion

TensorLogic automatically converts between f16 tensors and NumPy arrays:

- **TensorLogic → NumPy**: f16 → f32 (small precision loss)
- **NumPy → TensorLogic**: f32/f64 → f16
- GPU tensors are automatically moved to CPU for conversion

### Using from Python

```python
import tensorlogic as tl
import numpy as np

# Create interpreter
interp = tl.Interpreter()

# Execute TensorLogic with Python integration
code = """
main {
    python import numpy as np

    tensor data: float16[5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor normalized: float16[5] = python.call("preprocessing.normalize", data)

    print("Normalized:", normalized)
}
"""

interp.execute(code)
```

See [examples/python_integration_test.tl](examples/python_integration_test.tl) for more examples.

## Jupyter Notebook Support 📊

TensorLogic includes a Jupyter kernel for interactive development in Jupyter notebooks.

### Installation

```bash
# Install the TensorLogic Jupyter kernel
jupyter kernelspec install --user jupyter/tensorlogic

# Verify installation
jupyter kernelspec list
```

### Usage

1. **Start Jupyter**:
```bash
jupyter notebook
# or
jupyter lab
```

2. **Create a new notebook** and select "TensorLogic" as the kernel

3. **Write TensorLogic code** in cells:

```tensorlogic
// Cell 1: Declare tensors
tensor W: float16[1] learnable = [0.5]
```

```tensorlogic
// Cell 2: Train model
tensor x: float16[1] = [2.0]
tensor y: float16[1] = [6.0]

learn {
    pred := x * W
    loss := (pred - y) * (pred - y)

    objective: loss,
    optimizer: sgd(lr: 0.1),
    epochs: 50
}

print("Trained W:", W)  // Should be ~3.0
```

4. **Run cells** with `Shift+Enter`

### Features

- **Interactive execution**: Run TensorLogic code cell-by-cell
- **Variable persistence**: Variables persist across cells within a session
- **Real-time output**: See training progress and results immediately
- **Mixed workflows**: Combine with Python cells for data preprocessing/visualization

### Example Notebook

See [examples/jupyter_tutorial.ipynb](examples/jupyter_tutorial.ipynb) for a complete tutorial.

## Architecture

### Device Hierarchy

```
Device
├── Metal GPU (MTLDevice)
│   ├── MetalBuffer (f16 shared memory)
│   ├── KernelExecutor (compute pipeline cache)
│   └── Compute Shaders (.metal files)
├── Neural Engine (CoreML)
│   ├── NeuralEngineBuffer (MLMultiArray wrapper)
│   ├── NeuralEngineOps (matmul, relu)
│   └── Model Integration (future)
└── CPU (fallback)
    └── f16 operations
```

### Operation Flow

1. **Creation**: Tensors created on Metal GPU or CPU
2. **Computation**: Operations execute on Metal GPU via compute shaders
3. **Conversion**: Seamless Metal ↔ Neural Engine buffer conversion
4. **Fallback**: Automatic CPU fallback for unsupported operations

## Testing & Performance

### Test Results

**286/286 tests passing** ✅ (with `--test-threads=1`)

- Parser: 18 tests (all 6 types: float16, int16, int32, int64, bool, complex16)
- Type checker: 20 tests
- Interpreter: 45 tests
- Tensor operations: 95 tests
- Autograd: 32 tests
- Optimizers: 27 tests (SGD, Adam, AdamW + schedulers)
- CoreML integration: 16 tests
- Performance tests: 10 tests
- GPU operations: 23 tests

**Important**: Metal GPU tests require single-threaded execution to avoid race conditions:

```bash
# Run all tests (required for GPU tests)
cargo test --lib -- --test-threads=1

# Run specific test
cargo test --lib test_name -- --test-threads=1
```

**Note**: Without `--test-threads=1`, some GPU tests may fail due to concurrent Metal resource access. This is expected behavior and not a bug in the library itself.

### Hardware Support

- **Apple M4 Pro** (primary development)
- **Apple M1/M2/M3** (compatible)
- macOS 13+ (for Neural Engine features)
- All operations use f16 for optimal performance

## Project Structure

```
tensorlogic/
├── src/
│   ├── device/              # Device management
│   │   ├── metal_device.rs  # Metal GPU device
│   │   ├── metal_buffer.rs  # Metal f16 buffers
│   │   ├── kernel_executor.rs  # Compute pipeline management
│   │   ├── neural_engine_buffer.rs  # MLMultiArray wrapper
│   │   └── neural_engine_ops.rs     # Neural Engine operations
│   ├── tensor/              # Tensor implementation
│   │   ├── tensor.rs        # Core tensor type
│   │   └── shape.rs         # Shape and stride management
│   ├── ops/                 # Operations
│   │   ├── elementwise.rs   # Element-wise operations
│   │   ├── matmul.rs        # Matrix multiplication
│   │   ├── activations.rs   # Activation functions
│   │   ├── broadcast.rs     # Broadcasting
│   │   ├── reduce.rs        # Reduction operations
│   │   └── einsum.rs        # Einstein summation
│   ├── error.rs             # Error types
│   ├── lib.rs               # Library entry point
│   └── main.rs              # Demo program
├── shaders/
│   └── elementwise.metal    # Metal compute shaders
├── claudedocs/              # Design documents
│   └── f16_neural_engine_metal_spec.md
└── Cargo.toml
```

## Dependencies

- **metal** 0.29 - Metal GPU framework
- **objc** 0.2 - Objective-C runtime (legacy)
- **objc2** 0.5 - Modern Objective-C bindings
- **objc2-core-ml** 0.2 - CoreML framework (MLMultiArray, MLModel)
- **objc2-foundation** 0.2 - Foundation framework (NSArray, NSValue)
- **half** 2.4 - f16 (half-precision float) support
- **thiserror** 1.0 - Error handling
- **anyhow** 1.0 - Error context

## Development Roadmap

### ✅ Completed

- **Phase 1-2**: Metal GPU foundation and acceleration
- **Phase 3**: Advanced operations (matmul, activations, broadcasting, einsum)
- **Phase 4**: Neural Engine integration (foundation)
- **Phase 5-6**: Full autograd with computation graph
- **Phase 7**: Optimization (zero-copy, buffer pool, fusion, GPU gradients)
- **Phase 8**: Advanced features (in-place ops, planner, gradient checking, higher-order derivatives)
- **Phase 9.1**: Production optimizers (SGD, Adam, AdamW)

### 🎯 Optional Future Work

- **Learning Rate Schedulers**: Cosine annealing, step decay, warmup
- **Additional Optimizers**: RMSprop, Adagrad, AdaDelta
- **Neural Engine Inference**: Full CoreML model integration (foundation complete)
- **Distributed Training**: Multi-device support
- **Model Zoo**: Pre-trained models optimized for Apple Silicon

See [Full Specification](claudedocs/f16_neural_engine_metal_spec.md) for detailed roadmap.

## Contributing

This is currently a research/development project. Contributions, issues, and feature requests are welcome!

### Development Setup

1. Clone the repository
2. Install Rust toolchain (1.70+)
3. Install Xcode Command Line Tools
4. Run tests: `cargo test -- --test-threads=1`
5. Run demo: `cargo run`

## License

Dual-licensed under:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

This project was inspired by:
- **[Andrej Karpathy's "Intro to Large Language Models"](https://www.youtube.com/watch?v=rkBLPYqPkP4)** - Insights on efficient neural computation and optimization
- **[Tensor-Logic Paper (arXiv:2510.12269)](https://arxiv.org/abs/2510.12269)** - Theoretical foundation for integrating tensor algebra with logic programming

Built with [Claude Code](https://claude.com/claude-code) using:
- Apple's Metal framework for GPU acceleration
- Apple's CoreML for Neural Engine integration
- Rust's excellent type system and safety guarantees

## Contact

Project maintained by Jun Suzuki
- GitHub: [@JunSuzukiJapan](https://github.com/JunSuzukiJapan)

---

**Note**: This library is optimized for Apple Silicon and requires macOS. Neural Engine features require macOS 13+ or iOS 16+.
