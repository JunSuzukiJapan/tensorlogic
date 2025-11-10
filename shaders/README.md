# TensorLogic Metal Shaders

This directory contains the Metal GPU kernels used by TensorLogic.

## Structure

### Active Shaders

- **[unified.metal](unified.metal)** - Main shader file containing all active GPU kernels (5108 lines, 185 kernels)
  - This is the only shader file currently used by the system
  - All operations are compiled from this single file for better optimization

### Archived Shaders

- **[archive/](archive/)** - Historical shader files that have been consolidated into unified.metal
  - These files are kept for reference but are not used by the build system
  - If you need to add new operations, add them directly to unified.metal

## Shader Categories in unified.metal

The unified.metal file contains the following categories of operations:

### 1. Matrix Operations (~800 lines)
- **Tiled matrix multiplication**: 16x16 and 32x32 tile sizes
  - `matmul_tiled_f16/f32` - Standard tiled matmul
  - `matmul_tiled_32x32_f16/f32` - Large tile matmul for big matrices
- **Matrix multiplication with transpose**: Fused transpose-matmul for efficiency
  - `matmul_transposed_b_tiled_f16/f32` - A @ B.T with tiling
- **Simple matrix multiplication**: No tiling (for small matrices)
  - `matmul_simple_f16/f32`
- **Fused operations**: Matmul with bias, activation
  - `matmul_bias_f16/f32`
  - `fused_matmul_activation_f16/f32`

**Performance**: Tiled implementations achieve 4-5x speedup over naive matmul

### 2. Element-wise Operations (~600 lines)
- **Arithmetic**: add, subtract, multiply, divide
- **Scalar operations**: add_scalar, mul_scalar, div_scalar, sub_scalar
- **Mathematical functions**: exp, log, sqrt, pow, abs, sign
- **Trigonometric**: sin, cos, tan
- **Utility**: clamp, neg, fill, reciprocal

Both f16 and f32 versions available for all operations.

### 3. Activation Functions (~400 lines)
- **ReLU family**: relu, leaky_relu, elu, selu
- **GELU**: Gaussian Error Linear Unit (used in transformers)
- **Sigmoid family**: sigmoid, tanh, hard_sigmoid
- **Advanced**: softplus, swish, mish, hard_swish

All activation functions support both forward and backward passes.

### 4. Reduction Operations (~800 lines)
- **Sum**: Global and per-dimension reduction
  - `sum_global_f16/f32` - Full tensor sum
  - `sum_dim_f16/f32` - Sum along specific dimension
- **Mean**: Global and per-dimension averaging
- **Max/Min**: Global and per-dimension extrema
  - `max_global_f16/f32`
  - `min_global_f16/f32`
- **Softmax**: Temperature-scaled and fused variants
  - `softmax_f16/f32` - Standard softmax
  - `fused_div_softmax_f16/f32` - Fused divide + softmax
- **ArgMax/ArgMin**: Index of max/min values

### 5. Normalization (~600 lines)
- **RMSNorm**: Root Mean Square Normalization (used in LLaMA/TinyLlama)
  - `rms_norm_f16/f32`
  - Formula: `output = (x / rms(x)) * weight`
  - Critical for modern LLM architectures
- **LayerNorm**: Standard layer normalization
  - `layer_norm_f16/f32`
- **BatchNorm**: Batch normalization
  - `batch_norm_f16/f32`

### 6. Embedding Operations (~100 lines)
- **Embedding lookup**: Extract rows from embedding table
  - `embedding_lookup_f16/f32`
  - Used for token → embedding conversion in LLMs

### 7. Advanced Operations (~1000 lines)
- **RoPE**: Rotary Position Embedding
  - `apply_rope_f16/f32`
  - Critical for positional encoding in LLMs
- **Attention mechanisms**:
  - `apply_attention_mask_f16/f32`
  - `fused_attention_scores_f16/f32`
- **Einsum operations**: General tensor contractions
  - `einsum_ihd_jhd_ihj_f16` - Query @ Key.T pattern
  - `einsum_ihj_jhd_ihd_f16` - Attention @ Value pattern
- **Fused operations**: Combined ops for performance
  - `fused_add_relu_f16/f32`
  - `fused_mul_add_f16/f32`
  - `fused_affine_f16/f32`

### 8. Tensor Shape Operations (~400 lines)
- **Broadcast**: Expand tensors to compatible shapes
  - `broadcast_f16/f32`
- **Concat**: Concatenate tensors along dimension
  - `concat_f16/f32`
- **Permute/Transpose**: Reorder dimensions
  - `permute_f16/f32`
  - `transpose_f16/f32`
- **Reshape utilities**: Change tensor views

### 9. Sampling and Generation (~400 lines)
- **Temperature-based sampling**: Scale logits before sampling
  - `temperature_sample_f16/f32`
- **Top-k sampling**: Sample from top-k highest probability tokens
  - `topk_sample_f16/f32`
- **Top-p sampling**: Nucleus sampling
  - `cumulative_sample_f16/f32`
- **Utilities**: argmax, find_max, divide_by_sum

### 10. Gradient Operations (Autograd) (~800 lines)
- **Backward passes**: Gradient computation for all operations
  - `add_backward_f16/f32`
  - `mul_backward_f16/f32`
  - `exp_backward_f16/f32`
  - `cos_backward_f16/f32`
  - etc.
- **Gradient accumulation**: Efficient backpropagation

## Performance Characteristics

### Memory Hierarchy
- **Thread registers** (fastest): ~10-20 TB/s
- **Threadgroup memory** (fast): ~400 GB/s on Apple Silicon
- **Global memory** (slow): ~90 GB/s on Apple Silicon

### Optimization Strategies
1. **Tiling**: Cache data in threadgroup memory
2. **Fusion**: Combine multiple operations into single kernel
3. **Precision**: Use f16 when possible for 2x throughput
4. **Occupancy**: 16x16 tiles for optimal thread utilization

### Typical Speedups
- Tiled matmul: **4-5x** vs naive implementation
- Fused operations: **2-3x** vs separate kernels
- f16 operations: **2x** throughput vs f32

## Precision Support

Most operations support both precisions:
- **f16 (half)**: Better performance (2x), lower memory, sufficient for most ML workloads
- **f32 (float)**: Higher precision when needed (gradients, accumulation)

## Adding New Operations

To add a new GPU kernel:

1. Add the kernel function to **unified.metal** in the appropriate section
2. Implement both f16 and f32 versions if applicable
3. Add forward and backward (gradient) versions if needed
4. Update this README with the new operation
5. Add tests in `examples/tests/` to verify correctness

## Testing

Comprehensive tests are available in `examples/tests/`:
- `test_arithmetic_ops.tl` - Element-wise operations
- `test_activation_ops.tl` - Activation functions
- `test_math_ops.tl` - Mathematical functions
- `test_reduction_ops.tl` - Reduction operations
- `test_shape_ops.tl` - Shape manipulation
- `test_broadcast_ops.tl` - Broadcasting
- `test_matmul_kernel.tl` - Matrix multiplication

Run all tests:
```bash
./examples/tests/run_all_shader_tests.sh
```

## Recent Fixes

### MatMul Tile Loading Bug (2025-11-10)
Fixed critical bug in `matmul_transposed_b_tiled_*` kernels where tile loading was incorrect:

**Before** (broken):
```metal
uint col = threadgroup_position_in_grid.x * TILE_SIZE + tx;
uint b_row = col;  // Varies per thread - WRONG!
```

**After** (fixed):
```metal
uint b_row = threadgroup_position_in_grid.x * TILE_SIZE + ty;  // Constant per threadgroup
```

This bug caused all matmul results to be zero for multi-token generation. See `claudedocs/matmul_kernel_fix_summary.md` for details.

## References

- Metal Shading Language Specification: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple GPU Architecture: https://developer.apple.com/metal/
- Candle (reference implementation): https://github.com/huggingface/candle

---

**Last Updated**: 2025-11-10
**Total Kernels**: 185
**Lines of Code**: 5108
**Supported Precisions**: f16, f32
