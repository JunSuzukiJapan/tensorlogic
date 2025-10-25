main {
    print("=== Testing token_embd.weight shape ===")

    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let embed_table = get_tensor(model, "token_embd.weight")

    print("token_embd.weight shape:", shape(embed_table))
    print("Expected: [32000, 2048] for [vocab_size, d_model]")

    // Also check output.weight which should be the same
    let output_weight = get_tensor(model, "output.weight")
    print("output.weight shape:", shape(output_weight))
}
