# GGUF Quantization Bug Fixes

## Summary

Fixed critical bugs in Q4_0 and Q6_K dequantization that were causing `inf` and `NaN` values in loaded tensors.

## Problem

### Initial Symptoms
- All generated tokens were ID 0 (`<unk>`)
- `output.weight` tensor contained `inf` values
- Logits contained `NaN` values after matmul
- Model appeared corrupted

### Root Cause
1. **Q4_0 Index Ordering Bug**: Wrong memory layout (interleaved vs. sequential)
2. **Q6_K Index Ordering Bug**: Wrong memory layout and incorrect scale indexing
3. **Q6_K Block Size**: Wrong block structure (208 bytes vs. 210 bytes)

## Investigation Process

### Step 1: Verification
- ✅ llama.cpp loads same GGUF file correctly
- ✅ Candle loads same GGUF file correctly
- ❌ TensorLogic produces inf/NaN
- **Conclusion**: Bug in TensorLogic's dequantization, not file corruption

### Step 2: Quantization Type Identification
- Model contains 3 types: F32 (45 tensors), Q4_0 (155 tensors), Q6_K (1 tensor)
- `token_embd.weight`: Q4_0 → Loaded correctly after fix
- `output.weight`: Q6_K → Needed fix

## Fixes

### Fix 1: Q4_0 Dequantization (src/model/formats/gguf.rs:54-111)

**Wrong Implementation (Interleaved)**:
```rust
for i in 0..16 {
    let byte = data[values_offset + i];
    let q_low = (byte & 0x0F) as i8 - 8;
    result.push(f_low);   // ❌ Position: 0, 2, 4, 6...
    let q_high = ((byte >> 4) & 0x0F) as i8 - 8;
    result.push(f_high);  // ❌ Position: 1, 3, 5, 7...
}
// Result: [low0, high0, low1, high1, ...] WRONG!
```

**Correct Implementation (Sequential)**:
```rust
for j in 0..16 {
    let byte = data[values_offset + j];

    // Lower 4 bits → first half of block
    let x0 = ((byte & 0x0F) as i8 - 8) as f32;
    result[base_idx + j] = half::f16::from_f32(x0 * scale_f32);

    // Upper 4 bits → second half of block
    let x1 = ((byte >> 4) as i8 - 8) as f32;
    result[base_idx + j + 16] = half::f16::from_f32(x1 * scale_f32);
}
// Result: [low0-15, high0-15] CORRECT!
```

**Reference**: Candle's `BlockQ4_0::to_float()` in `candle-core/src/quantized/k_quants.rs`

### Fix 2: Q6_K Dequantization (src/model/formats/gguf.rs:113-198)

**Block Structure (210 bytes)**:
```c
typedef struct {
    uint8_t ql[128];      // Lower 4 bits of quantized values
    uint8_t qh[64];       // Upper 2 bits of quantized values
    int8_t  scales[16];   // Scales
    ggml_half d;          // Super-block scale (2 bytes)
} block_q6_K;
```

**Key Changes**:
1. Corrected block size: 208 → 210 bytes
2. Fixed index ordering: Sequential access to correct positions
3. Corrected scale indexing: `sc[is + 0/2/4/6]` instead of complex offset calculation
4. Each 6-bit value = (4 bits from ql) | (2 bits from qh << 4)

**Output Layout**:
```
y[l + 0]  = d * scales[is + 0] * q1;  // Positions 0-31
y[l + 32] = d * scales[is + 2] * q2;  // Positions 32-63
y[l + 64] = d * scales[is + 4] * q3;  // Positions 64-95
y[l + 96] = d * scales[is + 6] * q4;  // Positions 96-127
```

**Reference**: llama.cpp's `dequantize_row_q6_K()` in `ggml-quants.c`

## Verification

### Before Fix
```
output.weight top 20 values: [inf, inf, inf, ...]
Logits: [NaN, NaN, NaN, ...]
Generated tokens: [0, 0, 0, 0, 0]
```

### After Fix
```
output.weight top 20 values: [0.1533, 0.1417, 0.1377, ...]
Logits: [0.1255, 0.1212, 0.1115, ...]
Generated tokens: [valid token IDs]
```

## Lessons Learned

1. **Don't Assume Weight Tying**: Initial workaround automatically replaced broken `output.weight` with `token_embd.weight` - this hid the real bug
2. **Compare with Reference**: llama.cpp and Candle provided correct implementations
3. **Index Ordering Matters**: Quantization formats have specific memory layouts that must be followed exactly
4. **Verify with Known-Good Tools**: Testing same file with llama.cpp proved file was not corrupted

## Files Modified
- `src/model/formats/gguf.rs`: Fixed Q4_0 and Q6_K dequantization
- `src/interpreter/mod.rs`: Added `print_top_k()` builtin for debugging

## Testing
```bash
# Test Q4_0 (token_embd.weight)
./target/release/tl run examples/test_generation.tl

# Test Q6_K (output.weight)
./target/release/tl run examples/debug_output_weight.tl

# Compare with llama.cpp
llama-cli --model ~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf --prompt "Hello"
```
