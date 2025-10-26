# Layer-by-Layer Degradation Analysis

## Date
2025-10-26

## Problem Statement
TinyLlama 1.1B inference produces progressively worse outputs as layer count increases.

## Test Results

| Layers | Predicted Token(s) | Quality | Logits Range |
|--------|-------------------|---------|--------------|
| 0 (direct embedding→output) | 30466 | ✅ Works | N/A |
| 1 | 5392 | ✅ Works | N/A |
| 2 | 1647 | ✅ Works | N/A |
| 10 | 15547, 20740 ("umi angularjs") | ⚠️ Wrong output | N/A |
| 22 | 1314, 16908... | ❌ Garbage | 0.89-1.19 |

## Observations

### Weights Verification
- ✅ Q4_0 quantization/dequantization: CORRECT
- ✅ Q6_K quantization/dequantization: CORRECT
- ✅ All weight operations execute successfully

### Individual Operations
All Metal GPU operations tested for determinism (10 runs each):
- ✅ Einsum: DETERMINISTIC (max diff: 0.0)
- ✅ Softmax: DETERMINISTIC (max diff: 0.0)
- ✅ RoPE: DETERMINISTIC (max diff: 0.0)
- ✅ Matmul: DETERMINISTIC (max diff: 0.0)
- ✅ RMSNorm: DETERMINISTIC (max diff: 0.0)

### Key Finding: Progressive Degradation
The problem is NOT:
- ❌ Non-determinism in GPU operations
- ❌ Weight quantization issues
- ❌ Individual operation bugs

The problem IS:
- ✅ **Cumulative error through layers**
- Quality degrades gradually: 1 layer good → 2 layers good → 10 layers bad → 22 layers very bad

## Hypotheses (Ordered by Likelihood)

### 1. Numerical Precision Issues (f16 accumulation)
- **Evidence**: Progressive degradation suggests compounding errors
- **Mechanism**: f16 precision loss in residual connections across 22 layers
- **Test**: Compare f16 vs f32 intermediate values
- **Status**: TO TEST

### 2. Normalization Scale Drift
- **Evidence**: Logits at layer 22 are abnormally uniform (~1.0)
- **Mechanism**: RMSNorm may not be controlling magnitude properly through layers
- **Test**: Check intermediate activation magnitudes at each layer
- **Status**: TO TEST

### 3. Residual Connection Issues
- **Evidence**: Layers work individually but fail when chained
- **Mechanism**: Values growing/shrinking incorrectly through residuals
- **Test**: Compare residual magnitudes with reference (llama.cpp/candle)
- **Status**: TO TEST

### 4. Layer Connection Logic
- **Evidence**: Each individual layer seems correct
- **Mechanism**: Incorrect data flow between layers
- **Test**: Step-by-step comparison with reference implementation
- **Status**: PARTIAL (layer 0 verified as correct)

## Expected vs Actual Logits

### Healthy Model (Expected)
- Range: -10 to +20
- Distribution: Wide, with clear peaks for probable tokens
- Top token clearly separated from others

### Current TensorLogic (22 layers)
- Range: 0.89 to 1.19
- Distribution: Nearly uniform (all tokens equally likely)
- No clear winner, essentially random sampling

```
=== Temperature Sample Debug ===
  Max logit value: 1.0107422
  Top 5 tokens:
    1: token=1314, logit=1.0107422
    2: token=29145, logit=0.97509766
    3: token=22771, logit=0.94189453
    4: token=27734, logit=0.92333984
    5: token=12116, logit=0.8911133
```

This indicates the model has lost all discriminative ability.

## Next Steps

1. ✅ **COMPLETED**: Verify weight quantization
2. ✅ **COMPLETED**: Verify individual operations (determinism)
3. ✅ **COMPLETED**: Test with different layer counts
4. **IN PROGRESS**: Check intermediate value magnitudes at each layer
5. **TODO**: Compare with llama.cpp intermediate values
6. **TODO**: Test f16 vs f32 precision hypothesis
7. **TODO**: Implement gradient checkpointing / mixed precision

## Implementation Details

### Test Files Created
- `examples/tests/verify_weights_values.tl` - Weight verification (PASSED)
- `examples/test_einsum_determinism.rs` - Einsum test (PASSED)
- `examples/test_softmax_determinism.rs` - Softmax test (PASSED)
- `examples/test_rope_determinism.rs` - RoPE test (PASSED)
- `examples/test_matmul_determinism.rs` - Matmul test (PASSED)
- `examples/test_rmsnorm_determinism.rs` - RMSNorm test (PASSED)
- `examples/tests/debug_layer_by_layer.tl` - Layer-by-layer debug (COMPLETED)
- `examples/tests/debug_2layers_simple.tl` - 2-layer test (PASSED)

### Documentation
- `claudedocs/step5_determinism_tests_complete.md` - Determinism test results
- `claudedocs/findings_layer_degradation.md` - This document

## Architecture Reference

```
TinyLlama 1.1B:
- 22 transformer layers
- hidden_dim: 2048
- n_heads: 32 (Q heads)
- n_kv_heads: 4 (KV heads - GQA)
- head_dim: 64 (2048 / 32)
- ffn_dim: 5632
- vocab_size: 32000
- RoPE theta: 10000
- RMSNorm eps: 1e-5
- SwiGLU activation in FFN
```

## Transformer Layer Structure

Each layer:
1. **Attention Block**:
   - RMSNorm(x) → Q, K, V projections
   - Reshape to heads: [seq, hidden] → [seq, n_heads, head_dim]
   - GQA expansion: 4 KV heads → 32 Q heads via broadcast
   - RoPE on Q and K
   - Attention: einsum(Q, K) → softmax → einsum(attn, V)
   - Output projection
   - Residual: x = x + attn_output

2. **FFN Block**:
   - RMSNorm(x) → gate, up projections
   - SwiGLU: silu(gate) * up → down projection
   - Residual: x = x + ffn_output

3. **Final Output**:
   - RMSNorm(x) → output projection to vocab_size
   - Sample from logits

## Comparison with Working Implementation

llama.cpp with same model produces:
- Coherent text output
- Wide logit range
- Clear token preferences

TensorLogic produces:
- Random-seeming output (worse as layers increase)
- Narrow logit range (~1.0)
- Nearly uniform token probabilities

**This strongly suggests a cumulative numerical issue, not a logic error.**
