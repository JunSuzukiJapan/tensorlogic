# Candle Migration Phase 2: Shape Tensor Elimination

## Problem

After Phase 1 (reducing 50→2 GPU syncs during prefill), generation phase still hung completely. Investigation revealed **22 GPU syncs per token** during generation.

## Root Cause

[chat_full_22layers_f16.tl](../examples/chat_full_22layers_f16.tl) generation loop called `shape()` **22 times per token** (lines 333-546 before fix):

```rust
// OLD CODE - 22 syncs per token
let KV0_shp = shape(KV0)  // GPU sync!
let pos0 = KV0_shp[0]
let nK0 = apply_rope_k(nK0_raw, 1.0, pos0)
// ... repeated for all 22 layers

// Each shape() call triggers wait_until_completed()
// in src/interpreter/eval.rs:700 and :776
```

**Total per token: 22 shape() syncs + 1 sampling sync = 23 GPU syncs**

This completely broke batching during generation.

## Solution: Candle-Style CPU Position Tracking

Candle tracks position as CPU variable and passes `index_pos: usize` parameter (see [tmp/candle/candle-transformers/src/models/llama.rs:505](../tmp/candle/candle-transformers/src/models/llama.rs#L505)):

```rust
pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache)
```

### Changes Made

#### 1. CPU Position Tracking
```diff
// Initialize position from prefill sequence length
+ let current_pos = seq_len

// Increment after each token
+ current_pos = current_pos + 1.0
```

#### 2. Eliminate Shape() Calls
```diff
// OLD: 22 shape() syncs per token
- let KV0_shp = shape(KV0)
- let pos0 = KV0_shp[0]
- let nK0 = apply_rope_k(nK0_raw, 1.0, pos0)

// NEW: Use CPU-tracked position (no sync)
+ let nK0 = apply_rope_k(nK0_raw, 1.0, current_pos)
```

Applied to all 22 layers (Lines 335-521).

#### 3. CPU Sampling (Already implemented in Phase 1)
```rust
// Always use CPU sampling (Candle-style)
fn temperature_sample_gpu<T: FloatType>(...) {
    // No GPU sampling kernels
    self.temperature_sample_cpu(logits, temperature, vocab_size)
}
```

## Files Modified

- [examples/chat_full_22layers_f16.tl](../examples/chat_full_22layers_f16.tl): Eliminated 22 shape() calls, added CPU position tracking
- [src/interpreter/builtin_sampling.rs](../src/interpreter/builtin_sampling.rs): Already CPU-only (Phase 1)
- [src/tensor/tensor_io.rs](../src/tensor/tensor_io.rs): Clarified single sync point (Phase 1)

## Expected Improvement

**Before:** 23 GPU syncs per token (22 shape() + 1 sampling)
**After:** 1 GPU sync per token (sampling only)
**Reduction:** 95.7% fewer syncs per token

## Current Status: ⚠️ Issue Persists

Despite eliminating all shape() calls, **generation still hangs completely** at "Assistant:" prompt.

### Verification
- ✅ All 22 shape() calls removed from generation loop
- ✅ Only 1 shape() remains in prefill (line 151 - runs once)
- ✅ Test functions have shape() but don't execute during generation
- ❌ Generation still times out after 300 seconds

## Next Investigation Steps

1. **Verify sampling execution path** - Is temperature_sample actually being called?
2. **Check for deadlock** - Could be in interpreter loop, not GPU syncs
3. **Profile generation phase** - Where is time actually spent?
4. **Test with minimal example** - Isolate issue to specific component

Possible causes:
- Infinite loop in generation logic
- Deadlock in interpreter evaluation
- Issue with `detokenize_single()` or `print()`
- Memory corruption during concat operations

## Comparison Table

| Implementation | Position Tracking | Syncs/Token | Status |
|----------------|-------------------|-------------|--------|
| Candle | CPU variable (`index_pos`) | 1 | ✅ Working |
| TensorLogic (Before) | GPU shape() × 22 | 23 | ❌ Hanging |
| TensorLogic (After) | CPU variable (`current_pos`) | 1 | ❌ Still hanging |

The architecture now matches Candle's approach, but a deeper runtime issue remains.
