# Getting Started with TensorLogic

TensorLogic is a tensor computation library for Apple Silicon, featuring f16-only operations with Metal GPU and Neural Engine acceleration.

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic = "0.1"
half = "2.4"
```

### Basic Example

```rust
use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, Adam};
use tensorlogic::autograd::AutogradContext;
use half::f16;

fn main() -> TensorResult<()> {
    // Create a parameter
    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    // Create optimizer
    let mut optimizer = Adam::new(vec![x.clone()], 0.1);

    // Training loop
    for _ in 0..10 {
        optimizer.zero_grad();
        AutogradContext::clear();

        // Forward pass
        let mut y = x.mul(&x)?;  // y = x²

        // Backward pass
        y.backward()?;

        // Update
        optimizer.step()?;
    }

    Ok(())
}
```

## Project Structure

```
tensorlogic/
├── src/
│   ├── tensor/          # Tensor data structure
│   ├── ops/             # Operations (add, mul, matmul, etc.)
│   ├── autograd/        # Automatic differentiation
│   ├── optim/           # Optimizers (SGD, Adam, AdamW)
│   ├── device/          # Metal GPU & Neural Engine
│   └── planner/         # Execution planning
├── examples/            # Working examples
├── claudedocs/          # Documentation
│   ├── optimizer_tutorial.md           # Complete optimizer guide
│   ├── f16_neural_engine_metal_spec.md # Full specification
│   └── phase4_current_status.md        # Neural Engine status
└── tests/              # Integration tests
```

## Core Concepts

### 1. Tensors

All tensors use f16 (half precision) for optimal Apple Silicon performance:

```rust
use half::f16;

// Create from Vec
let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
let tensor = Tensor::from_vec(data, vec![2])?;

// Create with specific shape
let matrix = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3])?;

// Enable gradients
let mut params = Tensor::from_vec(vec![f16::ZERO; 10], vec![10])?;
params.set_requires_grad(true);
```

### 2. Operations

#### Element-wise Operations
```rust
let a = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
let b = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;

let sum = a.add(&b)?;       // a + b
let diff = a.sub(&b)?;      // a - b
let prod = a.mul(&b)?;      // a * b
let quot = a.div(&b)?;      // a / b
```

#### Matrix Operations
```rust
// Matrix multiplication
let a = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3])?;
let b = Tensor::from_vec(vec![f16::ONE; 6], vec![3, 2])?;
let c = a.matmul(&b)?;  // [2,3] @ [3,2] → [2,2]

// Broadcasting
let x = Tensor::from_vec(vec![f16::from_f32(5.0)], vec![1])?;
let broadcasted = x.broadcast_to(vec![10])?;
```

#### Activations
```rust
let x = Tensor::from_vec(vec![f16::from_f32(-1.0)], vec![1])?;

let relu_out = x.relu()?;     // max(0, x)
let gelu_out = x.gelu()?;     // GELU activation
let softmax_out = x.softmax()?; // Softmax normalization
```

#### Reductions
```rust
let x = Tensor::from_vec(vec![f16::from_f32(1.0); 10], vec![10])?;

let total = x.sum()?;                  // Sum all elements
let avg = x.mean()?;                   // Mean of all elements
let max_val = x.max()?;                // Maximum value
let dim_sum = x.sum_dim(0, false)?;    // Sum along dimension
```

### 3. Automatic Differentiation

```rust
use tensorlogic::autograd::AutogradContext;

// Enable gradients
let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
x.set_requires_grad(true);

// Forward pass
let y = x.mul(&x)?;  // y = x²

// Backward pass
let mut loss = y.clone();
loss.backward()?;

// Access gradients
let grad = x.grad().unwrap();
println!("dy/dx = {:.2}", grad.to_vec()[0].to_f32());  // 4.0
```

#### Computation Graph

```rust
// Clear graph before each iteration
AutogradContext::clear();

// Build computation
x.set_requires_grad(true);
let h = x.matmul(&w1)?.add(&b1)?.relu()?;
let logits = h.matmul(&w2)?.add(&b2)?;

// Compute gradients
let mut loss = cross_entropy(&logits, &labels)?;
loss.backward()?;

// Access gradients
w1.grad()  // dL/dw1
w2.grad()  // dL/dw2
```

### 4. Optimizers

#### SGD
```rust
use tensorlogic::optim::SGD;

let params = vec![w1.clone(), w2.clone()];
let mut optimizer = SGD::new(params, 0.01);

// With momentum
let mut optimizer = SGD::with_momentum(params, 0.01, 0.9);

// Full options
let mut optimizer = SGD::with_options(
    params,
    0.01,    // learning_rate
    0.9,     // momentum
    0.0,     // dampening
    false,   // nesterov
    0.0001,  // weight_decay
);
```

#### Adam
```rust
use tensorlogic::optim::Adam;

let mut optimizer = Adam::new(params, 0.001);

// With weight decay
let mut optimizer = Adam::with_weight_decay(params, 0.001, 0.01);

// With AMSGrad
let mut optimizer = Adam::with_amsgrad(params, 0.001);
```

#### AdamW (Recommended for transformers)
```rust
use tensorlogic::optim::AdamW;

let mut optimizer = AdamW::new(params, 0.001);  // default wd=0.01

// Custom weight decay
let mut optimizer = AdamW::with_weight_decay(params, 0.001, 0.05);
```

### 5. Training Loop

```rust
use tensorlogic::optim::Optimizer;

fn train(
    params: Vec<Tensor>,
    optimizer: &mut dyn Optimizer,
    data: &[(Tensor, Tensor)],
) -> TensorResult<()> {
    for (inputs, targets) in data {
        // 1. Clear gradients
        optimizer.zero_grad();
        AutogradContext::clear();

        // 2. Forward pass
        let outputs = model_forward(inputs, &params)?;

        // 3. Compute loss
        let mut loss = compute_loss(&outputs, targets)?;

        // 4. Backward pass
        loss.backward()?;

        // 5. Update parameters
        optimizer.step()?;
    }
    Ok(())
}
```

## Device Support

### Metal GPU

TensorLogic automatically uses Metal GPU for all operations when available:

```rust
// Automatically uses Metal GPU
let x = Tensor::from_vec(data, shape)?;
let y = x.matmul(&w)?;  // Runs on GPU
```

### Neural Engine

Neural Engine integration is available for certain operations:

```rust
use tensorlogic::device::NeuralEngineOps;

// Check availability
if NeuralEngineOps::is_available() {
    println!("{}", NeuralEngineOps::info());
}

// Neural Engine operations (experimental)
let result = NeuralEngineOps::matmul(&a, &b)?;
```

**Note:** Full Neural Engine inference is in development. See [Phase 4 Status](phase4_current_status.md) for details.

### CPU Fallback

CPU implementations are available for all operations when Metal/Neural Engine are unavailable.

## Examples

### Run Provided Examples

```bash
# Simple training example
cargo run --example simple_training

# See examples/README.md for more
```

### Example: Linear Regression

```rust
use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, Adam};
use tensorlogic::autograd::AutogradContext;
use half::f16;

fn main() -> TensorResult<()> {
    // Model: y = wx + b
    let mut w = Tensor::from_vec(vec![f16::from_f32(0.1)], vec![1])?;
    let mut b = Tensor::from_vec(vec![f16::ZERO], vec![1])?;
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    // Optimizer
    let mut optimizer = Adam::new(vec![w.clone(), b.clone()], 0.01);

    // Training data: y = 3x + 2
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![5.0, 8.0, 11.0, 14.0, 17.0];

    // Training
    for epoch in 0..100 {
        optimizer.zero_grad();
        AutogradContext::clear();

        let mut total_loss = Tensor::from_vec(vec![f16::ZERO], vec![1])?;

        for (&x_val, &y_val) in x_data.iter().zip(&y_data) {
            let x = Tensor::from_vec(vec![f16::from_f32(x_val)], vec![1])?;
            let y_true = Tensor::from_vec(vec![f16::from_f32(y_val)], vec![1])?;

            // Prediction: y_pred = w*x + b
            let y_pred = w.mul(&x)?.add(&b)?;

            // Loss: (y_pred - y_true)²
            let diff = y_pred.sub(&y_true)?;
            let loss = diff.mul(&diff)?;

            total_loss = total_loss.add(&loss)?;
        }

        let mut avg_loss = total_loss.mul_scalar_(1.0 / x_data.len() as f32)?;
        avg_loss.backward()?;

        optimizer.step()?;

        if epoch % 20 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, avg_loss.to_vec()[0].to_f32());
        }
    }

    // Get updated parameters from optimizer
    let final_w = optimizer.get_params_mut()[0].to_vec()[0].to_f32();
    let final_b = optimizer.get_params_mut()[1].to_vec()[0].to_f32();

    println!("\nFinal model: y = {:.2}x + {:.2}", final_w, final_b);
    println!("Target:      y = 3.00x + 2.00");

    Ok(())
}
```

## Performance

### f16 Benefits on Apple Silicon

- **2x faster** than f32 on Metal GPU
- **2x less memory** usage
- **Neural Engine compatibility** (requires f16)
- **Optimal for M-series chips** with Unified Memory

### Optimization Tips

1. **Batch operations**: Larger tensors → better GPU utilization
2. **Reduce CPU↔GPU transfers**: Keep data on GPU when possible
3. **Use in-place operations**: `add_()`, `mul_()`, etc. when safe
4. **Metal shaders**: All operations use optimized Metal kernels

## Documentation

- **[Optimizer Tutorial](optimizer_tutorial.md)**: Complete guide to SGD, Adam, AdamW
- **[Full Specification](f16_neural_engine_metal_spec.md)**: Detailed architecture and design
- **[Phase 4 Status](phase4_current_status.md)**: Neural Engine integration progress
- **[Examples README](../examples/README.md)**: Running provided examples

## Testing

```bash
# Run all tests
cargo test -- --test-threads=1

# Run specific test
cargo test test_adam -- --test-threads=1

# Run with release optimizations
cargo test --release -- --test-threads=1
```

## Features Summary

### ✅ Production Ready

- ✅ Tensor operations (add, sub, mul, div, matmul)
- ✅ Activations (ReLU, GELU, Softmax)
- ✅ Broadcasting and reductions
- ✅ Automatic differentiation (forward + backward)
- ✅ Optimizers (SGD, Adam, AdamW)
- ✅ Metal GPU acceleration
- ✅ Second-order derivatives (foundation)
- ✅ Fused operations (add+relu, mul+relu, affine)
- ✅ In-place operations
- ✅ Execution planning
- ✅ Gradient checking

### 🚧 In Development

- 🚧 Neural Engine inference (foundation complete)
- 🚧 Full Hessian APIs (foundation complete)
- 🚧 Learning rate schedulers
- 🚧 Additional optimizers (RMSprop, etc.)

## Getting Help

- Check the [Optimizer Tutorial](optimizer_tutorial.md) for training guidance
- See [Examples](../examples/) for working code
- Review test files in `tests/` for usage patterns

## License

Dual-licensed under MIT/Apache-2.0

---

**Next Steps:**

1. Read the [Optimizer Tutorial](optimizer_tutorial.md)
2. Run the [simple training example](../examples/simple_training.rs)
3. Review the [full specification](f16_neural_engine_metal_spec.md)
4. Start building your own models!
