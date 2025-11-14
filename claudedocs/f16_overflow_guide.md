# F16 Overflow Guide

## Understanding F16 Precision Limits

Half-precision floating point (f16) has a **maximum value of ±65,504**. This guide explains how this limitation affects TensorLogic operations and how to handle overflow situations.

## Table of Contents

1. [F16 vs F32 Comparison](#f16-vs-f32-comparison)
2. [When Overflow Occurs](#when-overflow-occurs)
3. [Sum Operation Deep Dive](#sum-operation-deep-dive)
4. [Practical Solutions](#practical-solutions)
5. [Best Practices](#best-practices)

---

## F16 vs F32 Comparison

| Type | Bytes | Range | Precision | Use Case |
|------|-------|-------|-----------|----------|
| `f16` | 2 | ±65,504 | ~3 decimal digits | Memory-constrained, GPU-accelerated ML |
| `f32` | 4 | ±3.4×10³⁸ | ~7 decimal digits | General computation, large reductions |

### Key Insight

**F16 is 50% smaller but has a 10³⁴ times smaller range than f32.**

---

## When Overflow Occurs

### Overflow Triggers

F16 overflow happens when:
1. **Large tensor reductions**: `sum()` on 1M+ elements
2. **Cumulative operations**: Running sums, integrations
3. **Large vocabulary logits**: 32K+ token predictions (e.g., LM head)

### Example: LM Head Overflow

```tl
// TinyLlama-1.1B chat model
let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
let tokens = tokenize(tokenizer, prompt, false)  // 34 tokens
let embeddings = embedding(emb_weight, tokens)   // [34, 2048] f16

// After 22 transformer layers...
let x_final = rms_norm(layer22_output, output_norm)  // [34, 2048] f16
let logits = linear(x_final, lm_head)  // [34, 32000] f16, values ≈ ±20

// Attempting sum():
let logits_sum = sum(logits)
// Calculation: 34 × 32000 × 20 (avg) = 21,760,000
// Result: inf (exceeds f16 max of 65,504)
```

### Overflow Cascade

```
Input: [34, 32000] f16 with values ≈ ±20
→ Total elements: 1,088,000
→ Sum calculation: 1,088,000 × 20 = 21,760,000
→ F16 max: 65,504
→ 21,760,000 > 65,504 → OVERFLOW → inf
```

---

## Sum Operation Deep Dive

### Design: Type Preservation (like Candle)

```rust
// TensorLogic follows Candle's design
pub fn sum(&self) -> TensorResult<T> {
    // Returns same type as input
    // Tensor<f16>.sum() → f16
    // Tensor<f32>.sum() → f32
}
```

### Internal Accumulation

Even though the result type is f16, **accumulation uses f32 internally** for accuracy:

```rust
// Pseudo-code of internal implementation
fn sum_f16_tensor(tensor: &Tensor<f16>) -> f16 {
    let mut accumulator: f32 = 0.0;  // Accumulate in f32
    for value in tensor.iter() {
        accumulator += value.to_f32();
    }
    f16::from_f32(accumulator)  // ⚠️ Overflow can happen here!
}
```

### Why Type Preservation?

1. **Consistency**: Predictable type behavior
2. **User Control**: Explicit conversions when needed
3. **Framework Compatibility**: Matches PyTorch, Candle, JAX

---

## Practical Solutions

### Solution 1: Use F32 Models (Recommended)

**When loading models:**

```tl
// Instead of default f16:
let model = load_model(model_path)  // f16 (default)

// Use f32 variant (future feature):
// let model = load_model_f32(model_path)  // f32 (no overflow)
```

**Trade-offs:**
- ✅ No overflow issues
- ✅ Higher precision
- ❌ 2× memory usage
- ❌ Slower on Metal GPU

### Solution 2: Avoid Sum on Large Tensors

**Don't use `sum()` for diagnostics on large tensors:**

```tl
// ❌ Bad: Sum for debugging
let logits = linear(x, lm_head)
print("Logits sum:", sum(logits))  // May be inf!

// ✅ Good: Use max/min for debugging
print("Logits range:", min(logits), "to", max(logits))
print("Logits shape:", shape(logits))
```

### Solution 3: Use Appropriate Operations (Candle-Compatible Method)

**For LLM inference, use temperature_sample() - DO NOT use sum():**

```tl
// ✅ CORRECT: Candle-compatible LLM workflow (used in all TensorLogic chat demos)
let logits = linear(x_final, lm_head)               // [seq_len, vocab_size] f16
let next_token = temperature_sample(logits, 0.8)    // Internally converts to f32, applies softmax, samples

// ✅ CORRECT: Greedy decoding (temperature=0.0)
let next_token = temperature_sample(logits, 0.0)    // Argmax, no randomness

// ❌ WRONG: Never sum() logits for inference
let logits_sum = sum(logits)  // inf - meaningless for LLM inference!
```

**Why temperature_sample() is safe:**
- Internally converts f16 → f32 (same as Candle's `to_dtype(DType::F32)`)
- Applies temperature scaling and softmax normalization
- Samples from probability distribution or argmax
- **No overflow** because softmax normalizes to [0, 1]

**Comparison with Candle:**
```rust
// Candle approach (from candle-transformers/src/generation/mod.rs)
let logits = logits.to_dtype(DType::F32)?;         // Convert to f32
let prs = candle_nn::ops::softmax_last_dim(&logits)?;
let next_token = self.sample_multinomial(&prs)?;

// TensorLogic approach (equivalent behavior)
let next_token = temperature_sample(logits, temperature)
// Internally: sync_and_read_f32() -> temperature scaling -> softmax -> sample
```

### Solution 4: Reduce Dimensionality First

**Instead of global sum, use dimensional reductions:**

```tl
// ❌ Overflow risk: Global sum
let total = sum(logits)  // 1,088,000 elements → inf

// ✅ Safe: Per-sequence sum
let per_seq_sum = sum_dim(logits, dim=1, keepdim=false)  // [34] f16
// Each sum is only 32,000 elements, safer

// ✅ Safest: Two-stage reduction
let per_seq_max = max_dim(logits, dim=1)  // [34] f16
let final_max = max(per_seq_max)          // scalar, no overflow
```

---

## Best Practices

### ✅ Do

1. **Use f32 for large reductions**
   ```tl
   // When available:
   // let sum_f32 = sum(to_f32(large_tensor))
   ```

2. **Use dimensionality-aware operations**
   ```tl
   let probs = softmax(logits, dim=-1)  // Per-row normalization
   let max_val = max(tensor)             // Single max value
   ```

3. **Check for infinity in results**
   ```tl
   let result = sum(tensor)
   if is_inf(result) {
       print("Warning: Sum overflow, result is inf")
   }
   ```

4. **Understand your data range**
   ```tl
   print("Value range:", min(tensor), "to", max(tensor))
   print("Estimated sum:", numel(tensor), "×", mean(tensor))
   ```

### ❌ Don't

1. **Don't sum large f16 tensors for computation**
   ```tl
   // ❌ Bad
   let loss = sum(squared_errors)  // May overflow

   // ✅ Good
   let loss = mean(squared_errors)  // Mean is safer
   ```

2. **Don't assume f16 = f32 with less precision**
   ```tl
   // F16 is not just "less precise f32"
   // It has a fundamentally different range!
   ```

3. **Don't use sum() for logging large tensors**
   ```tl
   // ❌ Bad diagnostic
   print("Logits sum:", sum(logits))

   // ✅ Good diagnostic
   print("Logits stats:", min(logits), max(logits), mean(logits))
   ```

---

## Comparison with Other Frameworks

### Candle (Hugging Face)

```rust
// Candle: Same behavior
let t: Tensor = ...; // f16 tensor
let sum = t.sum()?;  // Returns f16, can overflow
```

### PyTorch

```python
# PyTorch: Same behavior
t = torch.randn(1000000, dtype=torch.float16)
t.sum()  # Can overflow to inf
```

### TensorLogic

```tl
// TensorLogic: Same behavior (consistent with Candle)
let t = tensor([...])  // f16 tensor
let s = sum(t)         // Returns f16, can overflow
```

**All three prioritize type consistency over automatic overflow prevention.**

---

## FAQ

### Q: Why not auto-convert f16 sum to f32?

**A:** Type consistency. Users expect `Tensor<f16>.sum()` → `f16`, not `f32`. Automatic conversion would:
- Break type expectations
- Hide precision conversions
- Make code behavior less predictable

### Q: How do I know if my sum will overflow?

**A:** Rough calculation:
```tl
let estimated_sum = numel(tensor) * mean(tensor)
if estimated_sum > 65000 {
    print("Warning: Likely to overflow in f16")
}
```

### Q: Can I still use f16 for LLMs?

**A:** Yes! Most LLM operations work fine with f16:
- ✅ Matrix multiplications (individual results within range)
- ✅ Softmax (normalized to [0, 1])
- ✅ Layer norm, RMS norm (normalized)
- ⚠️ LM head sum (for debugging only, use softmax for inference)

### Q: What about mean(), max(), min()?

- `mean()`: Can overflow intermediate sum, but less likely (division helps)
- `max()`: No overflow (single max value)
- `min()`: No overflow (single min value)

---

## Examples

### Example 1: Safe LLM Inference

```tl
// Complete inference without overflow
fn generate_token(x: float16[?, ?], lm_head: float16[?, ?]) -> int {
    let logits = linear(x, lm_head)        // [seq_len, vocab_size] f16
    let probs = softmax(logits, dim=-1)    // Normalized, safe
    let next_token = argmax(probs)         // Integer token ID
    next_token
}
```

### Example 2: Debugging Large Tensors

```tl
// Comprehensive tensor diagnostics without overflow
fn debug_tensor(t: float16[?, ?]) {
    print("Shape:", shape(t))
    print("Elements:", numel(t))
    print("Range:", min(t), "to", max(t))
    print("Mean:", mean(t))  // May overflow, but less likely

    // Don't do this for large tensors:
    // print("Sum:", sum(t))  // ❌ May be inf
}
```

### Example 3: Training Loss (Future)

```tl
// When training features are available
fn compute_loss(predictions: float16[?, ?], targets: float16[?, ?]) -> float32 {
    let errors = predictions - targets
    let squared_errors = errors * errors

    // Use mean instead of sum to avoid overflow
    let loss_f16 = mean(squared_errors)

    // Convert to f32 for stability
    to_f32(loss_f16)
}
```

---

## Summary

| Operation | F16 Safe? | Alternative |
|-----------|-----------|-------------|
| `sum(large_tensor)` | ❌ Overflow risk | `mean()`, `max()`, dimensional sums |
| `sum(small_tensor)` | ✅ Safe if < 10K elements | - |
| `mean()` | ⚠️ Intermediate overflow | `to_f32(tensor).mean()` |
| `max()`, `min()` | ✅ Always safe | - |
| `softmax()` | ✅ Normalized output | - |
| `linear()` | ✅ Individual results safe | - |

**Golden Rule**: If you need `sum()` on >10,000 f16 elements, consider if you really need sum, or if max/mean/softmax would work better.

---

## Further Reading

- [Candle Documentation](https://github.com/huggingface/candle)
- [IEEE 754 Half Precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
- [TensorLogic Full Specification](f16_neural_engine_metal_spec.md)
