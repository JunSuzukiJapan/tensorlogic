# Loop-Based vs Manual Expansion: Comparison

## Side-by-Side Comparison

### Manual Expansion Approach
```rust
// Token 1 generation
let emb_t1 = embedding(emb_weight, [prompt_tokens, token_1])

// Layer 0
let x0_t1 = transformer_layer(emb_t1, W_q_0, W_k_0, W_v_0, W_o_0,
                               attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
// Layer 1
let x1_t1 = transformer_layer(x0_t1, W_q_1, W_k_1, W_v_1, W_o_1,
                               attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)
// ... Layer 2-20 (18 more copies)
// Layer 21
let x21_t1 = transformer_layer(x20_t1, W_q_21, W_k_21, W_v_21, W_o_21,
                                attn_norm_21, W_gate_21, W_up_21, W_down_21, ffn_norm_21)

let logits_t1 = linear(rms_norm(slice_last(x21_t1, 0), output_norm), output_weight)
let token_2 = temperature_sample(logits_t1, 0.8)

// Token 2 generation (REPEAT ALL 22 LAYERS)
let emb_t2 = embedding(emb_weight, [prompt_tokens, token_1, token_2])
let x0_t2 = transformer_layer(emb_t2, W_q_0, ...)
// ... (22 more layers manually written)

// Token 3 generation (REPEAT ALL 22 LAYERS AGAIN)
// ... (22 more layers manually written)

// For 10 tokens: 220 manual layer calls needed!
```

### Loop-Based Approach
```rust
let gen_tokens = prompt_tokens

for token_step in range(10) {
    // Get embeddings for all generated tokens
    let current_emb = embedding(emb_weight, gen_tokens)
    let x = current_emb

    // Process through ALL 22 layers automatically
    for layer_idx in range(22) {
        // Dynamic weight loading - ONE line per weight type!
        let attn_norm = model.blk[layer_idx].attn_norm.weight
        let W_q = model.blk[layer_idx].attn_q.weight
        let W_k = model.blk[layer_idx].attn_k.weight
        let W_v = model.blk[layer_idx].attn_v.weight
        let W_o = model.blk[layer_idx].attn_output.weight
        let ffn_norm = model.blk[layer_idx].ffn_norm.weight
        let W_gate = model.blk[layer_idx].ffn_gate.weight
        let W_up = model.blk[layer_idx].ffn_up.weight
        let W_down = model.blk[layer_idx].ffn_down.weight

        // Transform through layer
        let x_norm1 = rms_norm(x, attn_norm)
        let Q = linear(x_norm1, W_q)
        let K = linear(x_norm1, W_k)
        let V = linear(x_norm1, W_v)
        let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
        let x1 = x + attn_out
        let x_norm2 = rms_norm(x1, ffn_norm)
        let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
        x = x1 + ffn_out
    }

    // Sample next token
    let logits = linear(rms_norm(slice_last(x, 0), output_norm), output_weight)
    let next_token_id = temperature_sample(logits, 0.8)

    // EOS check
    if next_token_id == eos_token_id {
        break
    }

    // Append and continue
    gen_tokens = append(gen_tokens, next_token_id)

    // Incremental display
    print(detokenize(tokenizer, gen_tokens, false))
}
```

## Metrics

| Metric | Manual Expansion | Loop-Based | Improvement |
|--------|-----------------|------------|-------------|
| **Lines of Code** | ~500 lines | ~260 lines | 48% reduction |
| **Max Tokens** | 3 (hardcoded) | 10 (configurable) | 3.3× more |
| **Layer Calls** | 66 (manual) | 220 (automatic) | 3.3× more |
| **Maintainability** | Very Low | High | ✓ |
| **Scalability** | Not Scalable | Highly Scalable | ✓ |
| **Bug Fix Effort** | Update 66 places | Update 1 place | 66× easier |
| **Model Size Change** | Full rewrite | Change one number | ✓ |
| **Add Feature** | Update every token | Update one loop | ✓ |

## Real-World Impact

### Scenario 1: Generate 50 tokens
- **Manual**: Would need 1,100 layer calls written by hand (impossible)
- **Loop**: Change `range(10)` → `range(50)` (1 character change)

### Scenario 2: Use 32-layer model
- **Manual**: Rewrite everything with 32 layers × N tokens
- **Loop**: Change `range(22)` → `range(32)` (2 character change)

### Scenario 3: Fix bug in attention
- **Manual**: Update bug fix in 66+ places (error-prone)
- **Loop**: Update bug fix in 1 place (guaranteed consistency)

### Scenario 4: Add KV caching
- **Manual**: Modify 66+ layer calls to include cache
- **Loop**: Modify single layer loop body

## Key Insights

### What Made This Possible

1. **Dynamic Model Access**: `model.blk[layer_idx].weight_name`
   - Discovered this works in TensorLogic
   - Eliminates need to preload all weights

2. **Token Append**: `append(token_ids, token_id)`
   - Allows growing sequence dynamically
   - Essential for autoregressive generation

3. **Nested Loops**: `for` inside `for`
   - Outer loop: token generation
   - Inner loop: layer processing

4. **Early Exit**: `break` statement
   - EOS detection stops generation early
   - Saves computation for shorter responses

### Limitations Discovered

1. **`range()` requires constants**
   - Can't use `range(max_tokens)` with variable
   - Workaround: Use constant `range(10)`

2. **Function parameter types**
   - Model type can't be annotated
   - Solution: Omit type annotation

3. **Tensor shapes**
   - `slice_last` returns 1D, need 2D
   - Solution: `reshape(tensor, [1.0, 2048.0])`

## Conclusion

Loop-based generation is **dramatically superior** to manual expansion:
- **Smaller code**: 48% reduction
- **More capable**: 3.3× more tokens
- **Maintainable**: Single source of truth
- **Scalable**: Easy to extend

This establishes TensorLogic as a viable platform for LLM development and experimentation.
