# TensorLogic

A high-performance f16 tensor library for Apple Silicon, leveraging Metal GPU and Neural Engine (CoreML) for optimized computation.

## Overview

TensorLogic is a unified tensor algebra library designed specifically for Apple Silicon (M-series chips), providing seamless integration between Metal GPU and Neural Engine through CoreML. All operations maintain f16 (half-precision) throughout for optimal Neural Engine compatibility.

## Key Features

### ✅ Implemented (Phase 1-5)

- **Metal GPU Foundation** (Phase 1-2)
  - f16 buffer management with shared memory mode
  - Device initialization and automatic device selection
  - Basic arithmetic operations (add, sub, mul, div)
  - Compute shaders for element-wise operations

- **Advanced Operations** (Phase 3)
  - Matrix multiplication with 2D GPU kernels (16x16 threadgroups)
  - Activation functions: ReLU, GELU (tanh approx), Softmax
  - Broadcasting operations (NumPy-compatible)
  - Reduction operations: sum, mean, max, min (global and per-dimension)
  - Einstein summation (einsum) with pattern optimization
    - Supports: `ij,jk->ik`, `ij->ji`, `ii->`, `i,j->ij`, batch matmul, etc.

- **Neural Engine Integration** (Phase 4)
  - CoreML MLMultiArray wrapper (NeuralEngineBuffer)
  - Bidirectional Metal ↔ Neural Engine buffer conversion
  - Neural Engine operation framework (matmul, relu)
  - Ready for CoreML model integration

- **Autograd Framework** (Phase 5) ⚡ NEW
  - Dynamic computation graph (PyTorch-style)
  - Automatic differentiation with gradient functions
  - Gradient computation for all operations:
    - Basic ops: Add, Sub, Mul, Div with broadcasting support
    - Advanced ops: MatMul, ReLU, GELU, Softmax
  - Gradient API: `requires_grad()`, `backward()`, `zero_grad()`, `grad()`
  - Integration test suite ready for full implementation

### 🚧 Planned (Phase 6+)

- **Phase 6: Full Autograd Integration** - Connect gradients with computation graph, implement full backward pass
- **Phase 7: Optimization** - Zero-copy conversion, buffer pooling, operation fusion, Metal GPU gradient kernels

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.70+ with Cargo
- Xcode Command Line Tools

### Installation

```bash
git clone https://github.com/JunSuzukiJapan/tensorlogic.git
cd tensorlogic
cargo build --release
```

### Running the Demo

```bash
cargo run --release
```

## Usage Examples

### Basic Operations

```rust
use tensorlogic::prelude::*;
use half::f16;

// Create Metal device
let device = MetalDevice::new()?;

// Create tensors on Metal GPU
let a = Tensor::from_vec_metal(
    &device,
    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
    vec![3],
)?;

let b = Tensor::from_vec_metal(
    &device,
    vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
    vec![3],
)?;

// Element-wise operations (executed on GPU)
let c = a.add(&b)?;  // [5.0, 7.0, 9.0]
let d = a.mul(&b)?;  // [4.0, 10.0, 18.0]
```

### Matrix Operations

```rust
// Matrix multiplication on Metal GPU
let a = Tensor::from_vec_metal(
    &device,
    vec![f16::from_f32(1.0), f16::from_f32(2.0),
         f16::from_f32(3.0), f16::from_f32(4.0)],
    vec![2, 2],
)?;

let b = Tensor::from_vec_metal(
    &device,
    vec![f16::from_f32(5.0), f16::from_f32(6.0),
         f16::from_f32(7.0), f16::from_f32(8.0)],
    vec![2, 2],
)?;

let c = a.matmul(&b)?;  // [19.0, 22.0, 43.0, 50.0]
```

### Einstein Summation

```rust
// Matrix multiplication via einsum
let c = Tensor::einsum("ij,jk->ik", &[&a, &b])?;

// Transpose
let transposed = Tensor::einsum("ij->ji", &[&a])?;

// Trace (diagonal sum)
let trace = Tensor::einsum("ii->", &[&a])?;

// Outer product
let outer = Tensor::einsum("i,j->ij", &[&v1, &v2])?;
```

### Neural Engine Integration

```rust
use tensorlogic::device::{MetalBuffer, NeuralEngineBuffer, NeuralEngineOps};

// Create Metal buffer
let metal_buf = MetalBuffer::from_f16_slice(device.metal_device(), &data)?;

// Convert to Neural Engine
let ne_buf = metal_buf.to_neural_engine(&vec![2, 2])?;

// Perform operations on Neural Engine
let result = NeuralEngineOps::matmul(&ne_a, &ne_b, 2, 2, 2)?;
```

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

## Performance

### Test Results

**79/79 tests passing** ✅ (74 lib + 5 integration)

- Phase 1-2: Metal foundation and GPU acceleration (28 tests)
- Phase 3: Advanced operations (25 additional tests, 53 total)
- Phase 4: Neural Engine integration (8 additional tests, 61 total)
- Phase 5: Autograd framework (13 additional tests, 74 lib total + 5 integration)

### Tested On

- Apple M4 Pro (macOS 15.0)
- All operations maintain f16 precision
- Metal GPU acceleration for compute-intensive operations

### Benchmarks

Run benchmarks with:
```bash
cargo bench
```

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

### ✅ Phase 1: Metal Foundation (Complete)
- Metal device initialization
- f16 buffer management
- Basic arithmetic shaders
- Tensor type and shape management
- CPU fallback

### ✅ Phase 2: Metal GPU Acceleration (Complete)
- Compute shader library
- KernelExecutor with pipeline caching
- Element-wise GPU operations
- Thread group optimization

### ✅ Phase 3: Advanced Operations (Complete)
- Matrix multiplication (2D GPU kernels)
- Activation functions (ReLU, GELU, Softmax)
- Broadcasting (NumPy-compatible)
- Reduction operations (sum, mean, max, min, per-dimension)
- Einstein summation with pattern optimization

### ✅ Phase 4: Neural Engine Integration (Complete)
- CoreML integration (objc2-core-ml)
- NeuralEngineBuffer (MLMultiArray wrapper)
- Metal ↔ Neural Engine conversion
- Neural Engine operation framework

### 🔄 Phase 5: Autograd (Planned)
- Computation graph construction
- Automatic differentiation
- Backpropagation on Metal GPU
- Gradient accumulation

### ⏳ Phase 6: Optimization (Future)
- Zero-copy Metal ↔ Neural Engine conversion
- Buffer pooling and memory optimization
- Operation fusion
- Automatic device placement
- GPU kernels for reductions

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
