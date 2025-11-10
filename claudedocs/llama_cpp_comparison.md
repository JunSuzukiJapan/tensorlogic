# llama.cpp vs TensorLogic: Output Comparison

## Test Setup

**Model**: tinyllama-1.1b-chat-q4_0.gguf
**Prompt**:
```
<|system|>
You are a friendly chatbot.</s>
<|user|>
Hello!</s>
<|assistant|>

```
**Parameters**:
- n_predict: 10 tokens
- temperature: 0.1
- seed: 42 (for reproducibility)

## Results

### llama.cpp Output (✅ CORRECT)

```
<|assistant|>
Absolutely! Here's a sample
```

**Analysis**:
- ✅ Meaningful English text
- ✅ Contextually appropriate response
- ✅ Grammar is correct
- ✅ Follows chat template
- **This is how the model SHOULD behave**

### TensorLogic Output (❌ BROKEN)

#### With q4_0 model:
```
<|assistant|>
ividualankaclosañouvelniejypdisambiguation
```

#### With f16 model:
```
<|assistant|>
celzh Foжноappa StanleyновоahuinhoSequence
```

**Analysis**:
- ❌ Completely nonsensical text
- ❌ Mix of random characters from different languages
- ❌ No grammatical structure
- ❌ No semantic meaning
- **The Transformer implementation is fundamentally broken**

## Technical Comparison

| Aspect | llama.cpp | TensorLogic |
|--------|-----------|-------------|
| Model Loading | ✅ Works | ✅ Works |
| Tokenization | ✅ Works | ✅ Works |
| Detokenization | ✅ Works | ✅ Works |
| Embedding Lookup | ✅ Works | ✅ Works |
| **Transformer Forward Pass** | ✅ Works | ❌ **BROKEN** |
| Logits Generation | ✅ Works | ❌ Wrong distribution |
| Sampling | ✅ Works | ✅ Works (but on wrong logits) |

## Root Cause Analysis

The problem is **NOT** in:
- ✅ Loop implementation (manual expansion has same issue)
- ✅ Model loading
- ✅ Tokenizer
- ✅ Sampling function
- ✅ Shape handling (all shapes are correct)

The problem **IS** in:
- ❌ **Transformer forward pass implementation**
- ❌ Attention mechanism
- ❌ Position encoding
- ❌ Layer normalization
- ❌ Or some combination of the above

## Specific Issues to Investigate

### 1. Causal Attention Mask
**Hypothesis**: TensorLogic may not be applying causal masking correctly.

In autoregressive generation, each token should only attend to:
- Itself
- All previous tokens
- **NOT** future tokens

**Evidence**:
- llama.cpp uses proper causal masking
- TensorLogic's `tinyllama_gqa_attention` may not mask future tokens

**Test**: Check if attention scores include future positions

### 2. RoPE (Rotary Position Embedding)
**Hypothesis**: RoPE may be applied incorrectly.

RoPE should:
- Be applied to Q and K (not V) ✓ (TensorLogic does this)
- Use correct position indices for each token
- Rotate embeddings based on position

**Evidence**:
- TensorLogic applies `rope(Q_heads)` and `rope(K_heads)`
- But may not be passing position information correctly

**Test**: Verify position indices are correct for growing sequences

### 3. RMSNorm Scale
**Hypothesis**: RMSNorm epsilon or scale factor may be wrong.

**Evidence**:
- Logits show reasonable magnitude (7-10)
- But distribution is completely wrong
- Could be subtle numerical issue

**Test**: Compare RMSNorm output values with llama.cpp

### 4. Linear Layer Transpose
**Hypothesis**: Matrix multiplication transpose flags may be incorrect.

In Transformer:
- `linear(x, W)` should compute `x @ W^T` or `x @ W` depending on weight layout
- GGUF format has specific weight layouts

**Evidence**:
- Shapes are correct ([1, 32000] for logits)
- But values are wrong
- Could be transpose issue

**Test**: Verify weight matrix orientation

### 5. Grouped Query Attention (GQA) Expansion
**Hypothesis**: GQA broadcasting from 4 to 32 heads may be wrong.

TensorLogic does:
```rust
let K_with_group = reshape(K_rope, [seq_len_f, 4.0, 1.0, 64.0])
let K_broadcast = broadcast_to(K_with_group, [seq_len_f, 4.0, 8.0, 64.0])
let K_expanded = reshape(K_broadcast, [seq_len_f, 32.0, 64.0])
```

**Evidence**:
- Complex reshaping and broadcasting
- Easy to get dimensions wrong
- Critical for correct attention

**Test**: Compare expanded K/V shapes and values with llama.cpp

## Debugging Strategy

### Phase 1: Isolate Attention (PRIORITY)
1. Extract Q, K, V from first token, first layer
2. Compare with llama.cpp values
3. Check attention scores
4. Verify causal mask application

### Phase 2: Check Position Encoding
1. Verify RoPE position indices
2. Compare rotated Q/K with llama.cpp
3. Test with different sequence lengths

### Phase 3: Validate Layer Norm
1. Compare RMSNorm outputs
2. Check epsilon and scale
3. Verify numerical stability

### Phase 4: Test Linear Layers
1. Compare Q/K/V projection outputs
2. Verify attention output projection
3. Check FFN layer outputs

### Phase 5: End-to-End Layer Test
1. Feed known input through one layer
2. Compare every intermediate value
3. Identify exact divergence point

## Recommended Actions

### Immediate (High Priority)
1. **Add causal attention mask** to `tinyllama_gqa_attention`
   - This is the most likely culprit
   - Easy to fix
   - High impact

2. **Verify RoPE implementation**
   - Check position index calculation
   - Compare with llama.cpp's rope implementation

### Short Term (Medium Priority)
3. **Create unit tests for each component**
   - Test RMSNorm with known inputs
   - Test attention with small matrices
   - Test GQA expansion logic

4. **Add numerical debugging**
   - Print intermediate tensor values
   - Compare with llama.cpp at each step
   - Identify divergence point

### Long Term (Low Priority)
5. **Performance optimization**
   - Only after correctness is achieved
   - KV caching
   - Batched inference

## Update: Causal Mask Added

### Implementation
- ✅ Added `causal_mask(seq_len)` function to interpreter (builtin_nn.rs)
- ✅ Applied causal mask to attention scores in chat demo
- ✅ Code compiles and runs without errors

### Results After Adding Causal Mask
```
<|assistant|>
 daßdaleslcer VALUESérieoleanapatotfer
```

### Analysis
- ❌ **Output still nonsensical** - different garbage but still not meaningful text
- ❌ **Causal mask alone did NOT fix the problem**
- ✅ Code runs successfully with proper masking applied
- ✅ All tensor shapes are correct
- ⚠️ **Problem must be in other components**: RoPE, GQA expansion, or numerical issues

### Next Investigation Priorities
1. **RoPE Position Encoding** - Most likely remaining issue
   - Verify position indices are correct for growing sequences
   - Compare RoPE outputs with llama.cpp
   - Check if RoPE is using correct dimensions and frequencies

2. **GQA Head Expansion** - Complex reshaping could be wrong
   - Verify K/V expansion from 4 to 32 heads
   - Check reshape and broadcast operations
   - Compare expanded K/V shapes and values with llama.cpp

3. **Numerical Debugging** - Compare intermediate values
   - Extract Q, K, V after RoPE
   - Compare attention scores before/after softmax
   - Check final logits distribution

## Conclusion

**Loop-based generation is 100% successful** - it works exactly as designed.

**Causal mask implementation is working** - but did not solve the output problem.

**The root cause is likely in RoPE or GQA** - these are the most complex remaining components.

**Next step**: Deep debug RoPE position encoding and GQA head expansion with numerical comparisons to llama.cpp.
