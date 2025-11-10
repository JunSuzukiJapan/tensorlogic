// Minimal test for matmul operation
// Check if basic matmul works correctly

main {
    print("=== Simple Matmul Test ===")
    print("")

    // Load model to get Metal device initialized
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Get embedding for a single token as input
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let single_token = tokenize(tokenizer, "Hello", false)
    let x = embedding(embed_table, single_token)
    print("Input shape:", shape(x))
    print("Input sample:", x[0, 0], x[0, 1], x[0, 2])

    // Get weight matrix
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    print("Weight shape:", shape(W_q_0))

    // Do matmul: x @ W_q_0.T
    // x: [1, 2048], W_q_0: [2048, 2048] → result: [1, 2048]
    let result = matmul(x, transpose(W_q_0))
    print("Result shape:", shape(result))
    print("Result sample:", result[0, 0], result[0, 1], result[0, 2])

    if result[0, 0] == 0.0 && result[0, 1] == 0.0 && result[0, 2] == 0.0 {
        print("❌ BUG: Matmul result is all zeros!")
    } else {
        print("✅ Matmul works!")
    }
}
