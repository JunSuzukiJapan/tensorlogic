# Loop-Based Generation Strategy for TensorLogic Chat Demo

## Current Situation Analysis

### Current Approach (Manual Expansion)
- **Token 1**: 22 layers manually written (lines 429-483)
- **Token 2**: 22 layers manually written (lines 486-525)
- **Token 3**: 22 layers manually written (lines 527-540)
- **Total**: 66 manual layer calls for 3 tokens
- **Problem**: 10 tokens would require 220 manual layer calls

### Key Limitation
Each token generation requires:
1. Processing ALL previous tokens through 22 layers
2. Appending new token to sequence
3. Growing input size: token₁[1×2048] → token₂[2×2048] → token₃[3×2048]

## Loop Strategy Design

### Two-Level Loop Structure

```
Outer Loop: Token Generation (max 10 tokens)
├─ Inner Loop: Process through 22 Transformer Layers
│  ├─ Load layer i weights dynamically: model.blk[i]
│  ├─ Apply attention + FFN
│  └─ Pass to next layer
├─ Sample next token from final layer output
├─ Check EOS (token_id == 2)
└─ Append to sequence if not EOS
```

### Key Insight: TensorLogic Dynamic Access
From `test_for_loop.tl`:
```rust
for i in range(5) {
    let attn_norm = model.blk[i].attn_norm.weight
    let attn_q = model.blk[i].attn_q.weight
}
```

✅ We CAN access weights dynamically: `model.blk[i].attn_norm.weight`
✅ No need to pre-load weights into individual variables

## Implementation Plan

### Phase 1: Single-Layer Loop Function

Create a function that processes ONE token through ALL 22 layers:

```rust
fn process_through_all_layers(
    input_emb: float16[?, ?],    // Current sequence embeddings
    model: Model                  // Model with all weights
) -> float16[?, ?] {
    let x = input_emb

    // Loop through 22 transformer layers
    for layer_idx in range(22) {
        // Dynamically load layer weights
        let norm1_w = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight

        let norm2_w = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        // Attention block with residual
        let normed1 = rms_norm(x, norm1_w)
        let Q = linear(normed1, W_q)
        let K = linear(normed1, W_k)
        let V = linear(normed1, W_v)
        let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
        let x_after_attn = x + attn_out

        // FFN block with residual
        let normed2 = rms_norm(x_after_attn, norm2_w)
        let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
        x = x_after_attn + ffn_out
    }

    x  // Return final layer output
}
```

### Phase 2: Token Generation Loop

```rust
main {
    // ... (initialization, model loading, tokenizer, prompt)

    // Initial token IDs from prompt
    let gen_tokens = initial_tokens  // e.g., [1, 15043]

    let max_tokens = 10
    let eos_token_id = 2
    let generated_count = 0

    // Token generation loop
    for token_step in range(max_tokens) {
        // 1. Get embeddings for ALL tokens so far
        let current_emb = embedding(gen_tokens, emb_weight)

        // 2. Process through all 22 layers
        let final_output = process_through_all_layers(current_emb, model)

        // 3. Extract last token's output
        let last_pos = get_last_position(final_output)  // [1, 2048]

        // 4. Final norm and projection
        let normed = rms_norm(last_pos, output_norm_weight)
        let logits = linear(normed, output_weight)

        // 5. Sample next token
        let next_token = sample_token(logits, temperature)

        // 6. Check EOS
        let next_token_int = to_int(next_token)
        if next_token_int == eos_token_id {
            print("[EOS detected at token", token_step + 1, "]")
            break
        }

        // 7. Append to sequence
        gen_tokens = append(gen_tokens, next_token)
        generated_count = generated_count + 1

        // 8. Incremental decode and display
        let decoded = detokenize(tokenizer, gen_tokens, false)
        print(decoded, "")
    }

    // Final output
    print("")
    print("Generated", generated_count, "tokens")
}
```

## Critical Challenges & Solutions

### Challenge 1: Growing Sequence Size
**Problem**: Each iteration processes more tokens
- Token 1: [1×2048] input
- Token 2: [2×2048] input
- Token 3: [3×2048] input

**Solution**: ✅ TensorLogic handles dynamic shapes
- RoPE, attention, FFN all support dynamic sequence length
- `shape(Q)[0]` gets current sequence length dynamically

### Challenge 2: Array Operations
**Problem**: Need to append tokens: `gen_tokens = append(gen_tokens, next_token)`

**Solution Options**:
1. **If TensorLogic has array append**: Use built-in function
2. **If not**: Work around by:
   - Keep count of generated tokens
   - Reconstruct embeddings each iteration from token IDs
   - Use tokenizer to manage full sequence

**Investigation Needed**: Check if TensorLogic has:
- `append(array, element)` function
- `concat(array1, array2)` function
- Array/list type support

### Challenge 3: Extracting Last Token
**Problem**: Need to extract last position from [seq_len, d_model]

**Solution**: Use existing `slice` operation:
```rust
// Get last token's hidden state
let shape_tensor = shape(final_output)
let seq_len = shape_tensor[0]
let last_idx = seq_len - 1.0
let last_pos = slice(final_output, last_idx, last_idx + 1.0, 0)
```

### Challenge 4: Loop Variable Scope
**Problem**: Variables defined inside loop might not persist

**Solution**:
- Use mutable variables that persist across iterations
- Or re-initialize from model state each iteration

## Implementation Steps

### Step 1: Verify Dynamic Access ✓
```bash
# Already confirmed from test_for_loop.tl:
# model.blk[i].attn_norm.weight works!
```

### Step 2: Implement Helper Functions
1. `get_last_token(tensor, axis)` - extract last position
2. Check if `append()` exists, or find workaround

### Step 3: Refactor Layer Processing
Replace manual expansion with loop-based `process_through_all_layers()`

### Step 4: Implement Generation Loop
Outer loop for token generation with EOS detection

### Step 5: Test Incrementally
1. Test single layer loop (process through 1 layer correctly)
2. Test all layers loop (22 layers)
3. Test generation loop (2 tokens)
4. Test full generation (10 tokens with EOS)

## Expected Benefits

### Code Size Reduction
- **Before**: ~500 lines with manual expansion
- **After**: ~150-200 lines with loops
- **Reduction**: 60-70% smaller

### Scalability
- Easy to change: `max_tokens = 50` → generate 50 tokens
- Easy to add: Different models with different layer counts
- Easy to modify: Change architecture without manual expansion

### Maintainability
- Single source of truth for layer processing
- Bugs fixed in one place affect all layers
- Clear separation: layer loop vs generation loop

## Risk Assessment

### High Risk
- ❌ Array append might not exist → Need workaround
- ⚠️ Loop variable scope unclear → Test carefully

### Medium Risk
- ⚠️ Dynamic weight loading performance → Measure if slower
- ⚠️ For loop semantics might differ from expectations

### Low Risk
- ✅ Dynamic shape handling already works
- ✅ Dynamic model access confirmed working
- ✅ All tensor operations support dynamic sizes

## Next Actions

1. ✅ **Test array operations**:
   - Check if `append()` exists
   - Check if `concat()` exists
   - Test alternative approaches

2. ✅ **Implement `process_through_all_layers()` function**:
   - Start with 2-3 layers to verify loop works
   - Expand to all 22 layers
   - Test with different sequence lengths

3. ✅ **Implement generation loop**:
   - Start with 2-3 tokens
   - Add EOS detection
   - Expand to 10 tokens

4. ✅ **Add incremental decoding**:
   - Display tokens as they're generated
   - Not just final result

## Success Criteria

- ✅ Generate 10 tokens using loops (not manual expansion)
- ✅ EOS detection stops generation early
- ✅ Incremental display shows progress
- ✅ Code < 200 lines in main{}
- ✅ Works with different prompts
- ✅ Maintainable and clear structure
