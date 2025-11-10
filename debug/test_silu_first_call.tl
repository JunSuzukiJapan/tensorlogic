// Test: Call buggy silu pattern as the FIRST operation

fn silu_direct(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("SiLU First Call Test")
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

    // This is the FIRST call to any silu function
    print("Calling silu_direct as FIRST operation...")
    let result = silu_direct(gate)
    print("Returned from silu_direct")
    print("Result sum:", sum(result))
    print("")

    print("================================================================================")
    if sum(result) == 0.0 {
        print("❌ Bug confirmed: First call returns 0")
    }
    if sum(result) != 0.0 {
        print("✅ Bug fixed: First call works!")
    }
    print("================================================================================")
}
