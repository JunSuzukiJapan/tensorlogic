// Test: Print values INSIDE the function before returning

fn silu_with_print(x: float16[?, ?]) -> float16[?, ?] {
    let result = x * sigmoid(x)
    print("  Inside function, result sum:", sum(result))
    result
}

main {
    print("================================================================================")
    print("Internal Print Test")
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

    print("Calling function...")
    let result = silu_with_print(gate)
    print("Returned from function")
    print("Outside function, result sum:", sum(result))
    print("")

    print("================================================================================"
)
}
