// Test: Call the buggy pattern three times to see if only first fails

fn silu_buggy(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Three Calls Test")
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

    print("Call 1:")
    let result1 = silu_buggy(gate)
    print("  Result sum:", sum(result1))
    print("")

    print("Call 2:")
    let result2 = silu_buggy(gate)
    print("  Result sum:", sum(result2))
    print("")

    print("Call 3:")
    let result3 = silu_buggy(gate)
    print("  Result sum:", sum(result3))
    print("")

    print("================================================================================")
}
