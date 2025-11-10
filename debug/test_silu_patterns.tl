// Test: Different patterns of silu to identify exact bug condition

// Pattern 1: Original buggy form
fn silu_v1(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

// Pattern 2: Workaround (known to work)
fn silu_v2(x: float16[?, ?]) -> float16[?, ?] {
    let sig_x = sigmoid(x)
    let result = x * sig_x
    result
}

// Pattern 3: Just bind sigmoid result
fn silu_v3(x: float16[?, ?]) -> float16[?, ?] {
    let sig_x = sigmoid(x)
    x * sig_x
}

// Pattern 4: Just bind multiplication result
fn silu_v4(x: float16[?, ?]) -> float16[?, ?] {
    let result = x * sigmoid(x)
    result
}

// Pattern 5: No function, direct expression
// (to test if it's specific to function returns)

main {
    print("================================================================================")
    print("SiLU Pattern Bug Test")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    // Use actual FFN workflow like in chat demo
    let tokens = tokenize(tokenizer, "Hello", false)
    let e = embedding(embed_table, tokens)
    let x_norm = rms_norm(e, ffn_norm_0)
    let gate = linear(x_norm, W_gate_0)

    print("Input gate sum:", sum(gate))
    print("")

    print("[1/5] Pattern 1: x * sigmoid(x) - direct return")
    let result1 = silu_v1(gate)
    let sum1 = sum(result1)
    print("  Result sum:", sum1)
    if sum1 == 0.0 {
        print("  ❌ FAIL: Returns 0 (BUG)")
    }
    if sum1 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[2/5] Pattern 2: Full workaround (sig_x + result variables)")
    let result2 = silu_v2(gate)
    let sum2 = sum(result2)
    print("  Result sum:", sum2)
    if sum2 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum2 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[3/5] Pattern 3: Only bind sigmoid result")
    let result3 = silu_v3(gate)
    let sum3 = sum(result3)
    print("  Result sum:", sum3)
    if sum3 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum3 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[4/5] Pattern 4: Only bind multiplication result")
    let result4 = silu_v4(gate)
    let sum4 = sum(result4)
    print("  Result sum:", sum4)
    if sum4 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum4 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("[5/5] Pattern 5: No function (direct in main)")
    let result5 = gate * sigmoid(gate)
    let sum5 = sum(result5)
    print("  Result sum:", sum5)
    if sum5 == 0.0 {
        print("  ❌ FAIL: Returns 0")
    }
    if sum5 != 0.0 {
        print("  ✅ PASS: Returns non-zero")
    }
    print("")

    print("================================================================================")
    print("Analysis:")
    print("  If pattern 1 fails but pattern 5 passes → bug is function-specific")
    print("  If pattern 3 passes → binding sigmoid is enough")
    print("  If pattern 4 passes → binding multiplication is enough")
    print("================================================================================")
}
