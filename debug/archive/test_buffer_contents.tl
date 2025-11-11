// Test: Check actual buffer contents at each step

fn silu_buggy_instrumented(x: float16[?, ?]) -> float16[?, ?] {
    let sig_result = sigmoid(x)
    print("  After sigmoid, sum =", sum(sig_result))

    let mul_result = x * sig_result
    print("  After multiply, sum =", sum(mul_result))

    mul_result
}

fn silu_direct(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Buffer Contents Debug Test")
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

    print("Test 1: Instrumented version (with intermediate variable)")
    let result1 = silu_buggy_instrumented(gate)
    print("Final result sum:", sum(result1))
    print("")

    print("Test 2: Direct version (buggy)")
    let result2 = silu_direct(gate)
    print("Final result sum:", sum(result2))
    print("")

    print("Test 3: Direct expression in main")
    let result3 = gate * sigmoid(gate)
    print("Final result sum:", sum(result3))
    print("")

    print("================================================================================")
    if sum(result2) == 0.0 {
        print("❌ Bug confirmed: Direct return in function returns 0")
    }
    if sum(result2) != 0.0 {
        print("✅ Bug fixed: Direct return works!")
    }
    print("================================================================================")
}
