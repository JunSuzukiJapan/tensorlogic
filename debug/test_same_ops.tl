// Test: Call a function with same operations (mul + sigmoid) first

fn other_silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn silu_buggy(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Same Operations Test")
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

    print("First: Call other_silu")
    let result1 = other_silu(gate)
    print("  other_silu result sum:", sum(result1))
    print("")

    print("Second: Call silu_buggy")
    let result2 = silu_buggy(gate)
    print("  silu_buggy result sum:", sum(result2))
    print("")

    print("================================================================================")
    if sum(result1) == 0.0 {
        print("❌ other_silu (first call) returns 0")
    }
    if sum(result1) != 0.0 {
        print("✅ other_silu (first call) returns non-zero")
    }

    if sum(result2) == 0.0 {
        print("❌ silu_buggy (second call) returns 0")
    }
    if sum(result2) != 0.0 {
        print("✅ silu_buggy (second call) returns non-zero")
    }
    print("================================================================================")
}
