# Testing Strategy

## Overview

TensorLogic has a comprehensive test suite with 544 tests achieving 85% code coverage. Tests are categorized into **GPU tests** and **CPU tests** to avoid resource contention and ensure reliable execution.

## Test Categories

### GPU Tests (9 files, ~140 tests)

Tests that use GPU resources via `MetalDevice::new()`. These tests are marked with `#[serial]` to run sequentially and avoid GPU resource contention and deadlocks.

**Files:**
- `test_gpu_operations.rs` - Core GPU operations (21 tests)
- `test_attention_math.rs` - Attention mechanisms (30 tests)
- `test_f16_activations.rs` - F16 activation functions (18 tests)
- `test_f16_basic_ops.rs` - F16 basic operations (15 tests)
- `test_model_loading.rs` - Model loading and weights (18 tests)
- `test_rope_application.rs` - RoPE positional encoding (13 tests)
- `test_embedding.rs` - Embedding operations
- `test_error_handling.rs` - Error handling with GPU
- `test_rope.rs` - RoPE implementation

### CPU Tests (9 files, ~404 tests)

Tests that run entirely on CPU without GPU resources. These tests can run in parallel for faster execution.

**Files:**
- `test_advanced_indexing.rs` - Gather, scatter, embedding (50 tests)
- `test_broadcasting.rs` - Broadcasting operations (35 tests)
- `test_reduce_ops.rs` - Sum, mean, max, min (40 tests)
- `test_reshape_ops.rs` - Reshape, flatten, transpose (50 tests)
- `test_memory_device.rs` - Memory management (50 tests)
- `test_integration.rs` - End-to-end workflows (30 tests)
- `test_attention_mask.rs` - Attention mask logic
- `test_einsum.rs` - Einstein summation
- `test_optimizers.rs` - Optimization algorithms

## Running Tests

### Run All Tests (Recommended)
```bash
cargo test
```
This automatically runs GPU tests sequentially and CPU tests in parallel.

### Run Only GPU Tests
```bash
cargo test --test test_gpu_operations
cargo test --test test_attention_math
cargo test --test test_f16_activations
# ... etc
```

### Run Only CPU Tests (Parallel)
```bash
cargo test --test test_broadcasting
cargo test --test test_reduce_ops
cargo test --test test_reshape_ops
# ... etc
```

### Run Specific Test
```bash
cargo test --test test_gpu_operations test_matmul_basic_2x2_f32
```

### Control Parallelism

**Default behavior:**
- GPU tests: Sequential (enforced by `#[serial]`)
- CPU tests: Parallel (uses all available cores)

**Override parallel execution:**
```bash
# Force single-threaded for debugging
cargo test -- --test-threads=1

# Limit parallel threads
cargo test -- --test-threads=4
```

## Test Organization

### GPU Test Structure

All GPU tests include:
1. `use serial_test::serial;` import
2. `#[serial]` attribute on each test function
3. Comment explaining sequential execution requirement

Example:
```rust
use serial_test::serial;

#[test]
#[serial]  // GPU resource contention prevention
fn test_matmul_basic() {
    let device = MetalDevice::new().unwrap();
    // ... test code
}
```

### CPU Test Structure

CPU tests use standard `#[test]` attribute:
```rust
#[test]
fn test_broadcasting_scalar_to_vector() {
    // ... test code (no GPU)
}
```

## Why This Separation?

### GPU Resource Contention
- Multiple GPU operations running simultaneously can cause:
  - **Deadlocks**: Competing for Metal command buffers
  - **Resource exhaustion**: Limited GPU memory
  - **Flaky tests**: Non-deterministic failures
  - **Performance degradation**: Context switching overhead

### Solution: Sequential GPU Tests
- `#[serial]` attribute forces sequential execution
- Prevents GPU resource conflicts
- Ensures deterministic test results
- Maintains test reliability

### Benefits of Parallel CPU Tests
- **Faster execution**: ~8x speedup on 8-core machines
- **Better resource utilization**: CPU cores not shared with GPU
- **Scalability**: Execution time stays constant as tests grow

## Coverage Metrics

**Overall Coverage: 85%**

| Category | Coverage |
|----------|----------|
| Basic Operations | 95% |
| Core Tensor Ops | 90-95% |
| Memory Management | 85% |
| Integration | 80% |
| Error Handling | 90% |

## Adding New Tests

### GPU Test Checklist
1. ✅ Add to appropriate `test_*.rs` file
2. ✅ Include `use serial_test::serial;` at top
3. ✅ Add `#[serial]` attribute before `#[test]`
4. ✅ Document why GPU is needed
5. ✅ Test with `cargo test --test <filename>`

### CPU Test Checklist
1. ✅ Add to appropriate `test_*.rs` file
2. ✅ Use standard `#[test]` attribute
3. ✅ Ensure no `MetalDevice::new()` calls
4. ✅ Verify parallel execution: `cargo test --test <filename> -- --test-threads=8`

## Troubleshooting

### Test Hangs or Deadlocks
**Symptom**: Tests freeze, never complete
**Cause**: GPU resource contention
**Solution**: Verify `#[serial]` attribute is present on GPU tests

### Flaky Test Failures
**Symptom**: Tests pass/fail randomly
**Cause**: Race conditions in parallel GPU execution
**Solution**: Add `#[serial]` to the test

### Slow Test Execution
**Symptom**: Tests take too long
**Cause**: Too many sequential tests, or unnecessary serialization
**Solution**:
- Verify CPU-only tests don't have `#[serial]`
- Run CPU tests separately: `cargo test --test test_broadcasting`
- Use `--test-threads=N` to control parallelism

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run CPU Tests (Parallel)
  run: |
    cargo test --test test_broadcasting -- --test-threads=8
    cargo test --test test_reduce_ops -- --test-threads=8
    cargo test --test test_reshape_ops -- --test-threads=8

- name: Run GPU Tests (Sequential)
  run: |
    cargo test --test test_gpu_operations -- --test-threads=1
    cargo test --test test_attention_math -- --test-threads=1
```

### Local Development
```bash
# Quick CPU-only check (fast)
cargo test --test test_broadcasting --test test_reduce_ops

# Full validation including GPU (slower)
cargo test
```

## Performance Benchmarks

**Test Suite Execution Times:**
- CPU tests (parallel, 8 threads): ~5 seconds
- GPU tests (sequential): ~3 seconds
- **Total**: ~8 seconds

**Without separation (all parallel):**
- Frequent deadlocks and hangs
- Unreliable execution
- Test failures due to resource contention
