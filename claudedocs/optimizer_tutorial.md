# TensorLogic Optimizer Tutorial

Complete guide to using optimizers for training neural networks in TensorLogic.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Optimizer Overview](#optimizer-overview)
3. [SGD Optimizer](#sgd-optimizer)
4. [Adam Optimizer](#adam-optimizer)
5. [AdamW Optimizer](#adamw-optimizer)
6. [Training Loop Pattern](#training-loop-pattern)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)

## Quick Start

```rust
use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, Adam};
use tensorlogic::autograd::AutogradContext;
use half::f16;

fn main() -> TensorResult<()> {
    // 1. Create model parameters
    let mut w = Tensor::from_vec(vec![f16::ONE; 10], vec![10])?;
    w.set_requires_grad(true);

    // 2. Create optimizer
    let mut optimizer = Adam::new(vec![w.clone()], 0.001);

    // 3. Training loop
    for epoch in 0..100 {
        // Clear gradients
        optimizer.zero_grad();
        AutogradContext::clear();

        // Forward pass
        let output = forward(&w)?;

        // Compute loss
        let mut loss = compute_loss(&output)?;

        // Backward pass
        loss.backward()?;

        // Update parameters
        optimizer.step()?;
    }

    Ok(())
}
```

## Optimizer Overview

TensorLogic provides three production-ready optimizers:

| Optimizer | Best For | Learning Rate | Key Features |
|-----------|----------|---------------|--------------|
| **SGD** | Simple problems, when you need stability | 0.001-0.1 | Basic gradient descent, momentum, Nesterov |
| **Adam** | Most deep learning tasks | 0.0001-0.01 | Adaptive learning rates, fast convergence |
| **AdamW** | When you need regularization | 0.0001-0.01 | Decoupled weight decay, better generalization |

### Common Optimizer Interface

All optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer {
    fn step(&mut self) -> TensorResult<()>;           // Update parameters
    fn zero_grad(&mut self);                           // Clear gradients
    fn get_lr(&self) -> f32;                          // Get learning rate
    fn set_lr(&mut self, lr: f32);                    // Set learning rate
    fn state_dict(&self) -> OptimizerState;           // Save state
    fn load_state_dict(&mut self, state: OptimizerState) -> TensorResult<()>;
    fn add_param_group(&mut self, group: ParamGroup); // Add parameter group
    fn num_param_groups(&self) -> usize;              // Get group count
}
```

## SGD Optimizer

### Basic SGD

Stochastic Gradient Descent: `θ_{t+1} = θ_t - η ∇L(θ_t)`

```rust
use tensorlogic::optim::SGD;

// Create SGD optimizer with learning rate 0.01
let params = vec![w1.clone(), w2.clone(), b1.clone(), b2.clone()];
let mut optimizer = SGD::new(params, 0.01);

// Training loop
for _ in 0..epochs {
    optimizer.zero_grad();
    let mut loss = forward_and_loss()?;
    loss.backward()?;
    optimizer.step()?;
}
```

### SGD with Momentum

Momentum accelerates convergence: `v_{t+1} = μ v_t + ∇L(θ_t)`

```rust
// SGD with momentum=0.9
let mut optimizer = SGD::with_momentum(params, 0.01, 0.9);
```

**Benefits:**
- Faster convergence on smooth loss landscapes
- Reduces oscillations in narrow valleys
- Helps escape local minima

**Hyperparameters:**
- `momentum`: typically 0.9 or 0.99
- Higher momentum → more acceleration, but less responsive to changes

### SGD with All Options

Full control over SGD behavior:

```rust
let mut optimizer = SGD::with_options(
    params,
    0.01,      // learning_rate
    0.9,       // momentum
    0.0,       // dampening (reduces momentum contribution)
    true,      // nesterov (look-ahead momentum)
    0.0001,    // weight_decay (L2 regularization)
);
```

**Nesterov Momentum:**
- Look-ahead gradient estimation
- More accurate than standard momentum
- Formula: `θ_{t+1} = θ_t - η [∇L(θ_t) + μ v_t]`

**Weight Decay:**
- L2 regularization: prevents overfitting
- Formula: `∇L(θ) ← ∇L(θ) + λθ`
- Typical values: 0.0001 to 0.01

## Adam Optimizer

Adaptive Moment Estimation with per-parameter learning rates.

### Basic Adam

```rust
use tensorlogic::optim::Adam;

let mut optimizer = Adam::new(params, 0.001);
```

**Default hyperparameters:**
- Learning rate: 0.001
- β₁ (momentum): 0.9
- β₂ (variance): 0.999
- ε (numerical stability): 0.001 (optimized for f16)

### Adam Algorithm

1. **First moment** (mean of gradients):
   ```
   m_{t+1} = β₁ m_t + (1-β₁) ∇L(θ_t)
   ```

2. **Second moment** (variance of gradients):
   ```
   v_{t+1} = β₂ v_t + (1-β₂) [∇L(θ_t)]²
   ```

3. **Bias correction**:
   ```
   m̂ = m / (1 - β₁^t)
   v̂ = v / (1 - β₂^t)
   ```

4. **Parameter update**:
   ```
   θ_{t+1} = θ_t - η m̂ / (√v̂ + ε)
   ```

### Adam with Custom Hyperparameters

```rust
let mut optimizer = Adam::with_options(
    params,
    0.001,          // learning_rate
    (0.9, 0.999),   // betas (β₁, β₂)
    0.001,          // epsilon
);
```

### Adam with Weight Decay

```rust
let mut optimizer = Adam::with_weight_decay(params, 0.001, 0.01);
```

**Note:** In standard Adam, weight decay is applied to gradients, which can be suboptimal. Consider using AdamW instead.

### Adam with AMSGrad

AMSGrad variant uses max of historical second moments:

```rust
let mut optimizer = Adam::with_amsgrad(params, 0.001);
```

**Benefits:**
- Better convergence in some cases
- Prevents learning rate from growing too large
- Useful when Adam shows divergence

## AdamW Optimizer

Adam with **decoupled weight decay** for better regularization.

### Basic AdamW

```rust
use tensorlogic::optim::AdamW;

// AdamW with default weight_decay=0.01
let mut optimizer = AdamW::new(params, 0.001);
```

### Key Difference from Adam

**Adam (coupled weight decay):**
```rust
// Weight decay affects gradients
gradient ← gradient + weight_decay * param
param ← param - lr * adaptive_gradient
```

**AdamW (decoupled weight decay):**
```rust
// Weight decay applied directly to parameters
param ← param - lr * adaptive_gradient
param ← param - lr * weight_decay * param  // Separate step
```

### AdamW with Custom Weight Decay

```rust
let mut optimizer = AdamW::with_weight_decay(params, 0.001, 0.05);
```

**Recommended weight decay values:**
- Small models: 0.01
- Large models (transformers): 0.01 - 0.1
- Fine-tuning: 0.001 - 0.01

### AdamW with AMSGrad

```rust
let mut optimizer = AdamW::with_amsgrad(params, 0.001);
```

## Training Loop Pattern

Complete training loop with best practices:

```rust
use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, AdamW};
use tensorlogic::autograd::AutogradContext;

fn train_model<M, O>(
    model: &mut M,
    optimizer: &mut O,
    train_data: &[(Tensor, Tensor)],
    num_epochs: usize,
) -> TensorResult<Vec<f32>>
where
    M: Model,
    O: Optimizer,
{
    let mut loss_history = Vec::new();

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for (inputs, targets) in train_data {
            // 1. Clear gradients
            optimizer.zero_grad();
            AutogradContext::clear();

            // 2. Forward pass
            let outputs = model.forward(inputs)?;

            // 3. Compute loss
            let mut loss = compute_loss(&outputs, targets)?;
            let loss_value = loss.to_vec()[0].to_f32();
            epoch_loss += loss_value;

            // 4. Backward pass
            loss.backward()?;

            // 5. Update parameters
            optimizer.step()?;
        }

        let avg_loss = epoch_loss / train_data.len() as f32;
        loss_history.push(avg_loss);

        println!("Epoch {}: loss = {:.4}", epoch, avg_loss);
    }

    Ok(loss_history)
}
```

## Advanced Features

### Multiple Parameter Groups

Different learning rates for different parts of the model:

```rust
use tensorlogic::optim::{ParamGroup, AdamW};

let mut optimizer = AdamW::new(vec![], 0.0);  // Empty initialization

// Add first group (feature extractor with small LR)
let group1 = ParamGroup::new(feature_params, 0.0001);
optimizer.add_param_group(group1);

// Add second group (classifier with larger LR)
let group2 = ParamGroup::new(classifier_params, 0.001);
optimizer.add_param_group(group2);

// Add third group with weight decay
let group3 = ParamGroup::with_weight_decay(decoder_params, 0.001, 0.01);
optimizer.add_param_group(group3);
```

### Learning Rate Scheduling

Manual learning rate adjustment:

```rust
// Training loop with LR decay
for epoch in 0..num_epochs {
    train_epoch(&mut model, &mut optimizer, train_data)?;

    // Step decay: reduce LR every 10 epochs
    if epoch % 10 == 0 && epoch > 0 {
        let current_lr = optimizer.get_lr();
        let new_lr = current_lr * 0.5;
        optimizer.set_lr(new_lr);
        println!("Learning rate adjusted to: {:.6}", new_lr);
    }
}
```

**Common LR schedules:**
- **Step decay**: multiply by 0.1 or 0.5 every N epochs
- **Exponential decay**: `lr = lr₀ * γ^epoch`
- **Cosine annealing**: `lr = lr_min + (lr_max - lr_min) * (1 + cos(πt/T)) / 2`

### Saving and Loading Optimizer State

```rust
// Save optimizer state
let state = optimizer.state_dict();
save_to_file("optimizer_checkpoint.bin", &state)?;

// Load optimizer state
let state = load_from_file("optimizer_checkpoint.bin")?;
optimizer.load_state_dict(state)?;
```

**State includes:**
- Step count
- Momentum buffers (SGD)
- First and second moment estimates (Adam/AdamW)
- Max second moments (AMSGrad)

### Gradient Clipping

Prevent exploding gradients:

```rust
fn clip_gradients(params: &[Tensor], max_norm: f32) -> TensorResult<()> {
    let mut total_norm = 0.0;

    // Calculate total gradient norm
    for param in params {
        if let Some(grad) = param.grad() {
            let grad_data = grad.to_vec();
            for g in grad_data {
                total_norm += g.to_f32().powi(2);
            }
        }
    }
    total_norm = total_norm.sqrt();

    // Clip if necessary
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        for param in params {
            if let Some(mut grad) = param.grad() {
                grad.mul_scalar_(clip_coef)?;
            }
        }
    }

    Ok(())
}

// Use in training loop
loss.backward()?;
clip_gradients(&params, 1.0)?;  // Clip to max_norm=1.0
optimizer.step()?;
```

## Best Practices

### Choosing an Optimizer

**Use SGD when:**
- ✅ You have a simple, convex problem
- ✅ You want maximum stability
- ✅ You're fine-tuning a pretrained model
- ✅ You have lots of time for hyperparameter tuning

**Use Adam when:**
- ✅ You're training a neural network from scratch
- ✅ You want fast convergence
- ✅ You have sparse gradients (NLP, embeddings)
- ✅ You don't have time for extensive LR tuning

**Use AdamW when:**
- ✅ You're training transformers or large models
- ✅ You need better generalization (regularization)
- ✅ You want to prevent overfitting
- ✅ You're following modern best practices

### Hyperparameter Guidelines

**Learning Rate:**
- SGD: 0.001 - 0.1 (start with 0.01)
- SGD+Momentum: 0.001 - 0.01 (can be smaller than plain SGD)
- Adam/AdamW: 0.0001 - 0.01 (start with 0.001)

**Momentum (SGD):**
- Standard: 0.9
- Heavy ball: 0.99
- Start with 0.9, increase if convergence is slow

**Weight Decay:**
- Small models: 0.0001 - 0.001
- Large models: 0.01 - 0.1
- Too much → underfitting, too little → overfitting

**Adam/AdamW Betas:**
- β₁: typically 0.9 (can increase to 0.95 for smoother updates)
- β₂: typically 0.999 (decrease to 0.99 for faster adaptation)

### Common Pitfalls

**❌ Forgetting to clear gradients:**
```rust
// WRONG: Gradients accumulate
loss.backward()?;
optimizer.step()?;
loss.backward()?;  // Gradients add up!
```

```rust
// CORRECT: Clear before each backward
optimizer.zero_grad();
AutogradContext::clear();
loss.backward()?;
optimizer.step()?;
```

**❌ Learning rate too high:**
```
Symptoms: Loss explodes, NaN values, divergence
Solution: Reduce LR by 10x or 100x
```

**❌ Learning rate too low:**
```
Symptoms: Very slow convergence, plateaus early
Solution: Increase LR by 2x or 5x
```

**❌ Wrong optimizer for task:**
```
Training transformers with SGD → Slow convergence
Training simple regression with Adam → Overcomplicated

Solution: Match optimizer to problem complexity
```

### Debugging Training

**Loss not decreasing:**
1. Check learning rate (too high or too low)
2. Verify gradients are flowing: `param.grad().is_some()`
3. Check loss function implementation
4. Verify data normalization

**Loss exploding:**
1. Reduce learning rate
2. Enable gradient clipping
3. Check for numerical instabilities (division by zero, log(0))
4. Use smaller batch sizes

**Slow convergence:**
1. Increase learning rate
2. Use Adam instead of SGD
3. Add momentum to SGD
4. Check batch size (larger batches → faster but less stable)

## Performance Tips

1. **Batch size:**
   - Larger batches → faster training, more stable gradients
   - Smaller batches → better generalization, more noisy updates
   - Typical: 32, 64, 128, 256

2. **Gradient accumulation:**
   ```rust
   let accumulation_steps = 4;
   for (i, batch) in train_data.iter().enumerate() {
       let loss = forward_and_loss(batch)?;
       loss.backward()?;

       if (i + 1) % accumulation_steps == 0 {
           optimizer.step()?;
           optimizer.zero_grad();
       }
   }
   ```

3. **Mixed precision (f16):**
   - TensorLogic uses f16 by default
   - 2x faster, 2x less memory than f32
   - Optimized for Apple Silicon Neural Engine

4. **Warmup and decay:**
   ```rust
   let warmup_steps = 1000;
   let total_steps = 10000;

   for step in 0..total_steps {
       // Linear warmup
       if step < warmup_steps {
           let lr = base_lr * (step as f32 / warmup_steps as f32);
           optimizer.set_lr(lr);
       }
       // Cosine decay
       else {
           let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
           let lr = base_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
           optimizer.set_lr(lr);
       }

       train_step()?;
   }
   ```

## Examples

See the `examples/` directory for complete working examples:

- **`optimizer_comparison.rs`**: Benchmark comparing SGD, Adam, and AdamW
- **`mnist_training.rs`**: Full MNIST training loop with Adam
- **`learning_rate_scheduling.rs`**: Advanced LR scheduling techniques

## References

- **SGD:** Robbins & Monro (1951)
- **Momentum:** Polyak (1964)
- **Nesterov:** Nesterov (1983)
- **Adam:** Kingma & Ba (2014) - [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- **AdamW:** Loshchilov & Hutter (2017) - [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **AMSGrad:** Reddi et al. (2018) - [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)
