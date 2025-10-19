# TensorLogic

A production-ready f16 tensor library for Apple Silicon with automatic differentiation and optimizers, featuring Metal GPU and Neural Engine acceleration.

## Overview

TensorLogic is a unified tensor algebra library designed specifically for Apple Silicon (M-series chips), providing seamless integration between Metal GPU and Neural Engine through CoreML. All operations maintain f16 (half-precision) throughout for optimal Neural Engine compatibility and performance.

**📚 [Getting Started Guide](claudedocs/getting_started.md)** | **📖 [Optimizer Tutorial](claudedocs/optimizer_tutorial.md)** | **🔧 [Full Specification](claudedocs/f16_neural_engine_metal_spec.md)**

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

- **Optimizers** (Phase 9.1) 🚀 NEW
  - **SGD**: Basic gradient descent, momentum, Nesterov
  - **Adam**: Adaptive learning rates with AMSGrad support
  - **AdamW**: Decoupled weight decay for better regularization
  - Learning rate scheduling
  - State save/load for checkpointing
  - Multi-parameter group support

- **Device Acceleration** (Phase 2, 4, 7)
  - Metal GPU acceleration for all operations
  - Neural Engine integration (foundation complete)
  - Zero-copy Metal ↔ Neural Engine conversion
  - Buffer pooling for memory optimization
  - Fused operations (add+relu, mul+relu, affine)
  - Metal GPU gradient kernels
  - ExecutionPlanner for automatic device selection

### 🚧 Advanced Features (Optional)

- **Neural Engine Inference**: Full CoreML model integration (foundation complete, deferred)
- **Learning Rate Schedulers**: Cosine, step decay, warmup (coming soon)
- **Additional Optimizers**: RMSprop, Adagrad (future work)

## Quick Start

See the [**Getting Started Guide**](claudedocs/getting_started.md) for comprehensive tutorials.

### Installation

```bash
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
cargo build --release
```

### Run Example

```bash
# Simple training example
cargo run --example simple_training --release
```

### Basic Usage

```rust
use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, Adam};
use tensorlogic::autograd::AutogradContext;
use half::f16;

fn main() -> TensorResult<()> {
    // Create parameter
    let mut w = Tensor::from_vec(vec![f16::from_f32(0.5)], vec![1])?;
    w.set_requires_grad(true);

    // Create optimizer
    let mut optimizer = Adam::new(vec![w.clone()], 0.01);

    // Training loop
    for epoch in 0..100 {
        optimizer.zero_grad();
        AutogradContext::clear();

        // Forward pass: y = w * x
        let x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
        let y = w.mul(&x)?;

        // Backward pass
        let mut loss = y.clone();
        loss.backward()?;

        // Update weights
        optimizer.step()?;
    }

    Ok(())
}
```

For more examples, see:
- [Getting Started Guide](claudedocs/getting_started.md) - Complete tutorial with examples
- [Optimizer Tutorial](claudedocs/optimizer_tutorial.md) - In-depth optimizer guide
- [Examples Directory](examples/) - Working code examples

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

**127/127 tests passing** ✅ (121 lib + 6 integration)

- Phase 1-3: Tensor operations and GPU acceleration (95 tests)
- Phase 4: Neural Engine integration (8 tests)
- Phase 5-6: Autograd framework (6 tests)
- Phase 7: Optimization features (8 tests)
- Phase 8: Advanced features (6 tests)
- Phase 9.1: Optimizers (19 tests)

Run tests:
```bash
cargo test -- --test-threads=1
```

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

Built with [Claude Code](https://claude.com/claude-code) using:
- Apple's Metal framework for GPU acceleration
- Apple's CoreML for Neural Engine integration
- Rust's excellent type system and safety guarantees

## Contact

Project maintained by Jun Suzuki
- GitHub: [@JunSuzukiJapan](https://github.com/JunSuzukiJapan)

---

**Note**: This library is optimized for Apple Silicon and requires macOS. Neural Engine features require macOS 13+ or iOS 16+.
