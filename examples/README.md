# TensorLogic Examples

This directory contains practical examples demonstrating TensorLogic features.

## Running Examples

```bash
# Run a specific example
cargo run --example simple_training

# Run with release optimizations
cargo run --release --example simple_training
```

## Available Examples

### `simple_training.rs`
Basic optimization example showing:
- Parameter initialization
- Adam optimizer usage
- Training loop with gradient computation
- Loss tracking and convergence

**Problem:** Minimize `f(x) = (x - 5)²`
**Optimizer:** Adam with learning rate 0.1
**Expected result:** Converges to x ≈ 5.0

## Example Structure

Each example follows this pattern:

1. **Problem Definition**: Clear mathematical problem statement
2. **Parameter Setup**: Initialize tensors with `requires_grad=true`
3. **Optimizer Creation**: Choose appropriate optimizer (SGD, Adam, AdamW)
4. **Training Loop**:
   ```rust
   for iteration in 0..num_iterations {
       optimizer.zero_grad();
       AutogradContext::clear();

       let loss = compute_loss()?;
       loss.backward()?;

       optimizer.step()?;
   }
   ```
5. **Results**: Display final parameters and loss

## Next Steps

For comprehensive tutorials, see:
- [Optimizer Tutorial](../claudedocs/optimizer_tutorial.md)
- [Main Specification](../claudedocs/f16_neural_engine_metal_spec.md)
