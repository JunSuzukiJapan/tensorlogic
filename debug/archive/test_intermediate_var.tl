// Test: Use intermediate variable without print

fn silu_inline(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn silu_intermediate(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    x * sig
}

main {
    print("================================================================================")
    print("Intermediate Variable Test")
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

    print("Test 1: silu_inline (x * sigmoid(x) directly)")
    let result1 = silu_inline(gate)
    print("  Result sum:", sum(result1))
    print("")

    print("Test 2: silu_intermediate (sig = sigmoid(x), then x * sig)")
    let result2 = silu_intermediate(gate)
    print("  Result sum:", sum(result2))
    print("")

    print("================================================================================")
    if sum(result1) == 0.0 {
        print("❌ Inline version returns 0")
    }
    if sum(result1) != 0.0 {
        print("✅ Inline version works!")
    }
    if sum(result2) != 0.0 {
        print("✅ Intermediate version works!")
    }
    print("================================================================================")
}
