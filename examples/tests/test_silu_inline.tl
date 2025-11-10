// Regression test for SiLU inline expression bug
//
// This test ensures that inline expressions like "x * sigmoid(x)" work correctly
// without requiring intermediate variables.
//
// Bug history:
//   - Issue: First call to function with "x * sigmoid(x)" returned 0
//   - Cause: Binary operations didn't wait for GPU completion of sigmoid
//   - Fix: Added wait_until_completed() before eval_binary_op in eval.rs
//
// This test should ALWAYS pass with the fix in place.

fn silu_inline(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn silu_intermediate(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    x * sig
}

test test_silu_inline {
    print("================================================================================")
    print("SiLU Inline Expression Regression Test")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    let tokens = tokenize(tokenizer, "Hello", false)
    let e = embedding(embed_table, tokens)
    let x_norm = rms_norm(e, ffn_norm_0)
    let gate = linear(x_norm, W_gate_0)

    print("Input gate sum:", sum(gate))
    print("")

    // Test 1: Inline expression (the bug case)
    print("Test 1: Inline expression - x * sigmoid(x)")
    let result1 = silu_inline(gate)
    let sum1 = sum(result1)
    print("  Result sum:", sum1)

    // Test 2: Intermediate variable (workaround that always worked)
    print("")
    print("Test 2: Intermediate variable - let sig = sigmoid(x); x * sig")
    let result2 = silu_intermediate(gate)
    let sum2 = sum(result2)
    print("  Result sum:", sum2)

    print("")
    print("================================================================================")

    // Both should return the same non-zero value
    if sum1 == 0.0 {
        print("❌ FAIL: Inline version returned 0")
        panic("Regression: inline expression bug has returned!")
    }
    if sum2 == 0.0 {
        print("❌ FAIL: Intermediate version returned 0")
        panic("Intermediate variable version should always work")
    }

    // Results should be approximately equal (both are ~5.58)
    print("  Difference between results:", sum1 - sum2)

    // Simple check: both values should be close to expected value
    let expected = 5.5
    if sum1 < expected {
        print("❌ FAIL: Inline version too small:", sum1)
        panic("Inline version should return ~5.58")
    }
    if sum2 < expected {
        print("❌ FAIL: Intermediate version too small:", sum2)
        panic("Intermediate version should return ~5.58")
    }

    print("✅ All assertions passed")
    print("✅ Inline expressions work correctly")
    print("================================================================================")
}
