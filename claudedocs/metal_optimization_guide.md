# Metal GPU Optimization Guide

Complete guide for optimizing TensorLogic applications on Apple Silicon Metal GPUs.

## Quick Reference

| Optimization | When to Use | Expected Improvement |
|--------------|-------------|---------------------|
| Fused Operations | MatMul + Activation, Add + ReLU | 30-50% faster |
| Buffer Pooling | Repeated operations | 20-30% less overhead |
| Large Batches | Model inference | 2-5x throughput |
| Half-Precision (f16) | Default | 2x bandwidth vs f32 |

## 1. Fused Operations

### What is Kernel Fusion?

Kernel fusion combines multiple operations into a single GPU kernel, eliminating:
- Intermediate buffer allocations
- Kernel launch overhead (~0.15-0.20ms per launch)
- Memory traffic (50% reduction)

### Available Fused Operations

#### MatMul + Activation
```rust
use tensorlogic::ops::Activation;

// Fused: 1 kernel launch
let result = a.matmul_with_activation(&b, Activation::ReLU)?;

// Unfused: 2 kernel launches + intermediate buffer
let temp = a.matmul(&b)?;
let result = temp.relu()?;
```

**Performance**:
- Small matrices (64×64): ~30% faster
- Large matrices (1024×1024): ~15% faster
- Saves 0.15-0.20ms kernel launch overhead

#### Element-wise + ReLU
```rust
// Add + ReLU (fused)
let result = a.fused_add_relu(&b)?;

// Mul + ReLU (fused)
let result = a.fused_mul_relu(&b)?;

// vs unfused
let temp = a.add(&b)?;
let result = temp.relu()?;
```

**Performance**:
- Small tensors (<10K): ~40% faster
- Large tensors (>1M): ~20% faster

#### Affine Transformation
```rust
// x * scale + bias (fused)
let result = x.fused_affine(&scale, &bias)?;

// vs unfused
let temp = x.mul(&scale)?;
let result = temp.add(&bias)?;
```

**Use Cases**:
- Batch normalization
- Layer normalization
- Custom scaling/shifting

### Activation Types

```rust
pub enum Activation {
    None = 0,   // No activation (just matmul)
    ReLU = 1,   // ReLU: max(x, 0)
    GELU = 2,   // GELU approximation
}
```

**GELU Approximation**:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

## 2. Buffer Pooling

### How Buffer Pooling Works

TensorLogic maintains a pool of reusable Metal buffers organized by size:
```
HashMap<size, Vec<Buffer>>
```

When you allocate a buffer:
1. Check pool for existing buffer of that size
2. Reuse if available (fast path)
3. Create new buffer if needed (slow path)

When buffer is dropped:
1. Return to pool (if under capacity)
2. Make available for reuse

### Monitoring Buffer Pool

```rust
use tensorlogic::device::MetalDevice;

let device = MetalDevice::new()?;

// Get pool statistics
let stats = device.buffer_pool_stats();

println!("Buffer Pool Stats:");
println!("  Total buffers: {}", stats.total_pooled);
println!("  Size classes: {}", stats.size_classes);
println!("  Total memory: {} bytes", stats.total_memory);
println!("  Allocations: {}", stats.allocation_count);
println!("  Reuses: {}", stats.reuse_count);

// Calculate reuse rate
let reuse_rate = 100.0 * stats.reuse_count as f64 / stats.allocation_count as f64;
println!("  Reuse rate: {:.1}%", reuse_rate);
```

**Good Reuse Rates**:
- Training loop: 80-95% (lots of reuse)
- Batch inference: 60-80% (moderate reuse)
- One-off operations: 0-20% (little reuse)

### Buffer Pool Configuration

```rust
// Default: 10 buffers per size class
let device = MetalDevice::new()?;

// Custom capacity
let metal_device = metal::Device::system_default().unwrap();
let pool = BufferPool::with_capacity(&metal_device, 20);
```

**Tuning**:
- More buffers → More memory used, better reuse
- Fewer buffers → Less memory used, more allocations
- Default (10) works well for most cases

### Memory Management

```rust
// Shrink pool to fit memory constraints
device.buffer_pool().shrink_to_fit(max_memory_bytes);

// Clear entire pool
device.buffer_pool().clear();

// Clear specific size
device.buffer_pool().clear_size(1024);
```

## 3. Batch Processing

### Why Batching Matters

GPU performance scales with data size:

| Operation | Size | Latency | Throughput |
|-----------|------|---------|------------|
| MatMul | 64×64 | 0.43ms | 1.2 GFLOPS |
| MatMul | 256×256 | 0.56ms | 60 GFLOPS |
| MatMul | 1024×1024 | 4.37ms | 491 GFLOPS |

**Key Insight**: Larger operations → Better GPU utilization

### Batching Strategies

#### 1. Batch Model Inference

```rust
// Bad: Process one at a time
for input in inputs {
    let output = model.forward(&input)?;
    results.push(output);
}

// Good: Batch all inputs
let batch = Tensor::cat(&inputs, 0)?;  // Concatenate along batch dim
let batch_output = model.forward(&batch)?;
let results = batch_output.split(0)?;  // Split along batch dim
```

**Performance**: 2-5x throughput improvement

#### 2. Batch Element-wise Operations

```rust
// Bad: Many small operations
let mut results = Vec::new();
for (a, b) in tensors_a.iter().zip(tensors_b.iter()) {
    results.push(a.add(b)?);
}

// Good: Single large operation
let stacked_a = Tensor::stack(&tensors_a, 0)?;
let stacked_b = Tensor::stack(&tensors_b, 0)?;
let result = stacked_a.add(&stacked_b)?;
let results = result.unstack(0)?;
```

#### 3. Optimal Batch Sizes

| Operation Type | Min Efficient Size | Optimal Size |
|----------------|-------------------|--------------|
| Element-wise | 10K elements | 100K-1M elements |
| MatMul | 128×128 | 256×256 or larger |
| Activation | 10K elements | 100K-1M elements |
| Reduction | 1K elements | 100K-1M elements |

**Rule of Thumb**: Aim for >100K elements per operation

## 4. Precision and Data Types

### Half-Precision (f16) Benefits

TensorLogic uses `half::f16` by default:

**Advantages**:
- 2× memory bandwidth vs f32
- 2× memory capacity
- Faster data transfers
- Apple GPU optimized for f16

**Precision**:
- Range: ±65,504
- Precision: ~3-4 decimal digits
- Sufficient for most ML tasks

### When to Use Full Precision

```rust
// f16 is default (recommended)
let tensor = Tensor::from_vec_gpu(&device, data_f16, shape)?;

// f32 for high-precision requirements
// (Note: Currently f16 only, f32 support planned)
```

**Use f32 when**:
- Numerical stability critical
- Very large/small values (>65K or <0.0001)
- Accumulating many small values

## 5. Memory Transfer Optimization

### Minimize Host↔Device Transfers

```rust
// Bad: Many small transfers
for i in 0..100 {
    let cpu_data = compute_on_cpu(i);
    let gpu_tensor = Tensor::from_vec_gpu(&device, cpu_data, shape)?;
    let result = gpu_tensor.relu()?;
    let cpu_result = result.to_vec();
}

// Good: Batch transfers
let all_cpu_data = (0..100).map(compute_on_cpu).flatten().collect();
let gpu_tensor = Tensor::from_vec_gpu(&device, all_cpu_data, batch_shape)?;
let result = gpu_tensor.relu()?;
let cpu_results = result.to_vec();
```

### Keep Data on GPU

```rust
// Bad: Unnecessary round-trips
let gpu_a = Tensor::from_vec_gpu(&device, data_a, shape)?;
let cpu_temp = gpu_a.relu()?.to_vec();  // GPU → CPU
let gpu_b = Tensor::from_vec_gpu(&device, cpu_temp, shape)?;  // CPU → GPU
let result = gpu_b.add(&gpu_c)?;

// Good: Stay on GPU
let gpu_a = Tensor::from_vec_gpu(&device, data_a, shape)?;
let gpu_b = gpu_a.relu()?;  // Stay on GPU
let result = gpu_b.add(&gpu_c)?;  // Stay on GPU
```

## 6. Performance Profiling

### Benchmarking Operations

```rust
use std::time::Instant;

let iterations = 100;
let warmup = 10;

// Warmup
for _ in 0..warmup {
    let _ = a.matmul(&b)?;
}

// Benchmark
let start = Instant::now();
for _ in 0..iterations {
    let _ = a.matmul(&b)?;
}
let duration = start.elapsed();

let avg_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
println!("Average: {:.4} ms", avg_ms);
```

### Calculating GFLOPS

```rust
// Matrix multiplication: 2*M*N*K FLOPs
let m = a.dims()[0];
let k = a.dims()[1];
let n = b.dims()[1];
let flops = 2.0 * m as f64 * n as f64 * k as f64;

let gflops = flops / (avg_time_seconds * 1e9);
println!("Performance: {:.2} GFLOPS", gflops);
```

### Calculating Bandwidth

```rust
// Element-wise operation: read 2 inputs + write 1 output
let elements = tensor.numel();
let bytes = elements * std::mem::size_of::<half::f16>() * 3;

let bandwidth_gbps = bytes as f64 / (avg_time_seconds * 1e9);
println!("Bandwidth: {:.2} GB/s", bandwidth_gbps);
```

## 7. Common Patterns

### Neural Network Forward Pass

```rust
// Optimized forward pass
impl Model {
    fn forward(&self, x: &Tensor) -> TensorResult<Tensor> {
        // Use fused operations
        let h1 = x.matmul_with_activation(&self.w1, Activation::ReLU)?;
        let h2 = h1.matmul_with_activation(&self.w2, Activation::ReLU)?;
        let output = h2.matmul_with_activation(&self.w3, Activation::None)?;
        Ok(output)
    }
}

// Batch inference
let batch_input = Tensor::stack(&inputs, 0)?;
let batch_output = model.forward(&batch_input)?;
```

### Training Loop

```rust
// Optimized training loop
for epoch in 0..epochs {
    for batch in dataloader {
        // Forward pass with fused ops
        let predictions = model.forward(&batch.x)?;

        // Compute loss
        let loss = mse_loss(&predictions, &batch.y)?;

        // Backward pass (buffers automatically pooled)
        loss.backward()?;

        // Optimizer step (reuses buffers)
        optimizer.step()?;
        optimizer.zero_grad()?;
    }
}

// Check buffer pool efficiency
let stats = device.buffer_pool_stats();
println!("Reuse rate: {:.1}%",
    100.0 * stats.reuse_count as f64 / stats.allocation_count as f64);
```

### Image Processing Pipeline

```rust
// Efficient image batch processing
fn process_images(images: Vec<Tensor>) -> TensorResult<Vec<Tensor>> {
    // Stack images into batch
    let batch = Tensor::stack(&images, 0)?;  // [N, H, W, C]

    // Fused operations for preprocessing
    let normalized = batch.fused_affine(&scale, &bias)?;

    // Model inference on entire batch
    let features = model.forward(&normalized)?;

    // Split back to individual results
    features.unstack(0)
}
```

## 8. Performance Checklist

Before deploying, verify:

- [ ] Using fused operations for matmul + activation?
- [ ] Batch size ≥ 16 for model inference?
- [ ] Operations process ≥100K elements when possible?
- [ ] Minimizing host↔device transfers?
- [ ] Keeping intermediate results on GPU?
- [ ] Buffer pool reuse rate >60% for training?
- [ ] Benchmarked critical operations?
- [ ] Profiled memory usage under load?

## 9. Troubleshooting

### Low Performance

**Symptom**: Operations slower than expected

**Checklist**:
1. Check batch size (should be ≥16)
2. Verify using fused operations
3. Check buffer pool reuse rate
4. Confirm data stays on GPU
5. Measure kernel launch overhead

### High Memory Usage

**Symptom**: Memory pressure, out of memory errors

**Solutions**:
```rust
// Shrink buffer pool
device.buffer_pool().shrink_to_fit(max_bytes);

// Reduce batch size
let smaller_batch_size = current_size / 2;

// Clear pool between epochs
device.buffer_pool().clear();
```

### Slow Transfers

**Symptom**: High host↔device transfer time

**Solutions**:
1. Batch transfers (transfer more at once)
2. Keep data on GPU between operations
3. Use Metal shared memory mode (default)
4. Pre-allocate tensors when possible

## 10. Advanced Topics

### Custom Fused Kernels

To add your own fused operation:

1. **Write Metal shader** (`shaders/my_ops.metal`):
```metal
kernel void fused_my_op_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Your fused operation logic
    half x = input[id];
    output[id] = /* fused computation */;
}
```

2. **Implement Rust API** (`src/ops/my_ops.rs`):
```rust
impl Tensor {
    pub fn fused_my_op(&self) -> TensorResult<Self> {
        match self.device() {
            Device::Metal(_) => self.fused_my_op_metal(),
            Device::CPU => self.fused_my_op_cpu(),
            _ => Err(TensorError::UnsupportedDevice),
        }
    }
}
```

3. **Test correctness**:
```rust
#[test]
fn test_fused_vs_unfused() {
    let fused = tensor.fused_my_op()?;
    let unfused = tensor.op1()?.op2()?;
    assert_tensors_close(&fused, &unfused, 1e-3);
}
```

### Profile-Guided Optimization

```bash
# Run comprehensive benchmarks
cargo bench --bench metal_performance > baseline.txt

# Make optimization changes
# ...

# Re-run benchmarks
cargo bench --bench metal_performance > optimized.txt

# Compare results
diff baseline.txt optimized.txt
```

## Conclusion

Following this guide will help you achieve optimal Metal GPU performance:

- **491 GFLOPS** peak compute (M4 Pro)
- **93 GB/s** peak memory bandwidth
- **20-30%** overhead reduction with buffer pooling
- **30-50%** speedup with kernel fusion

For questions or issues, see the [benchmark results](./metal_gpu_optimization_summary.md) or file an issue on GitHub.
