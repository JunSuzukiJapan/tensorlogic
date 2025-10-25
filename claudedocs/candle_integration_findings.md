# Candle Integration Investigation - Findings

## Session Goal
User requested: "candleのみを使って、２２層のtransformerを書いて、それをTensorLogicから呼び出せるようにして それを使ったTensorLogicと、現状のTensorLogicで"Hello"という入力に対する出力が変わるか試してみて"

Translation: Write a 22-layer transformer using only candle, make it callable from TensorLogic, and test whether the output changes for "Hello" input between candle-based and current TensorLogic.

## Critical Discovery: Non-Deterministic Token Generation

### Problem
Both 10-layer and 22-layer demos exhibit non-deterministic behavior when generating tokens:

**10-layer demo (3 consecutive runs with same input "Hello"):**
- Run 1: Token 1 = 0 (`<unk>`)
- Run 2: Token 1 = 22893
- Run 3: Token 1 = 8807

**Simplified 1-layer test (consistent behavior):**
- All runs: Token 1 = 2354 (deterministic)

### Analysis

**Working:**
- 1-layer forward pass: Deterministic (5/5 samples = token 2354)
- GGUF loading: Fixed and working correctly
- Individual operations (embedding, linear, RoPE, softmax): Appear correct

**Broken:**
- 10-layer demo: Non-deterministic token generation
- 22-layer demo: Non-deterministic, often generates token 0

**Key Difference:**
- 1-layer: No KV cache, simple forward pass
- 10-layer: Uses KV cache with complex memory management
- 22-layer: No KV cache but many layers

### Hypothesis

Possible causes for non-determinism:
1. **Buffer Pool Corruption**: Reused buffers may contain stale data
2. **Uninitialized Memory**: Metal buffers not properly zero-initialized
3. **Memory Synchronization**: Race conditions in GPU operations
4. **Numerical Instability**: Accumulating errors with deep networks

The fact that 1-layer is deterministic while 10-layer (with KV cache) is not suggests the issue is related to:
- Memory reuse patterns in buffer pool
- KV cache concatenation/reshaping operations
- Multi-layer state propagation

## Candle Integration Status

### Attempted Approach
1. Added candle dependencies with Metal features to Cargo.toml
2. Created `src/candle_llama.rs` with TinyLlama configuration
3. Created `src/interpreter/builtin_candle.rs` for TensorLogic integration

### Blockers Encountered

**API Compatibility Issues:**
1. `VarBuilder::from_gguf()` doesn't exist in candle 0.8.4
2. `Tensor.i()` indexing method not available
3. `rand::distributions::WeightedIndex` not available in rand 0.9.2
4. Model/Tokenizer API mismatches with TensorLogic's implementation

**Architecture Mismatch:**
- TensorLogic Model doesn't store file path
- Need to pass model path separately for candle loading
- Tokenizer API differences (encode/decode methods)
- Value enum doesn't support List or Scalar types

### Resolution
Reverted candle integration changes to focus on fixing the core non-determinism issue first. Comparison with candle is meaningless if TensorLogic's output is non-deterministic.

## Recommendations

### Priority 1: Fix Non-Determinism (BLOCKING)
1. Add buffer pool debugging to track buffer reuse
2. Implement deterministic seed for temperature_sample
3. Add memory initialization verification for Metal buffers
4. Test with buffer pool disabled to isolate issue
5. Add debug logging for logits/softmax intermediate values

### Priority 2: Candle Integration (AFTER FIX)
Once TensorLogic generates deterministic output:
1. Use candle's GGUF loader directly instead of VarBuilder
2. Create simplified comparison script (1-layer first)
3. Match tokenization format between implementations
4. Compare logits before sampling (eliminate temperature randomness)
5. Verify numerical equivalence layer by layer

## Test Files Created

- `examples/tests/simple_forward.tl` - 1-layer test (working, deterministic)
- `examples/tests/test_candle_compare.tl` - Comparison template (incomplete)
- `examples/tests/debug_first_token.tl` - Debug script (incomplete)

## Next Steps

**Immediate:**
1. Debug non-deterministic behavior in 10-layer demo
2. Identify root cause (buffer pool vs KV cache vs numerical)
3. Implement fix and verify determinism across multiple runs

**After Fix:**
1. Verify 22-layer demo works deterministically
2. Re-attempt candle integration with corrected API usage
3. Implement proper comparison (logits-level, not token-level)
4. Work towards ChatGPT-like REPL goal

## User Context

**User's Ultimate Goal**: "最終的には、ChatGPTのようなreplが目的です" (ChatGPT-like REPL)

**Target Implementation**: `chat_demo_full_22_layers.tl` for 22-layer full model

**Comparison Goal**: Validate TensorLogic's output against candle/llama.cpp for "Hello" input

**Current Blocker**: Non-deterministic token generation makes comparison impossible
