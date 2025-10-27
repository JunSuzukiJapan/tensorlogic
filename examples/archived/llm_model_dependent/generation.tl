// Test: Verify chat generation works with fixed Q6_K

main {
    print("=== Testing Chat Generation ==")
    print("")

    // Load model
    print("[1/2] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_weight = get_tensor(model, "token_embd.weight")
    let embed_table = transpose(embed_weight)
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Loaded")
    print("")

    // Test generation
    print("[2/2] Testing generation...")
    let tokens = tokenize(tokenizer, "Hello", true)
    print("      Input: \"Hello\"")
    print("      Tokens:", tokens)

    let embeddings = embedding(embed_table, tokens)
    let logits = matmul(embeddings, output_weight)

    print("      Logits shape:", shape(logits))

    // Check top logits
    print("")
    print("      Top 10 logits:")
    let k = to_int(10.0)
    let dummy = print_top_k(logits, k)

    // Sample next token
    let next_greedy = to_int(argmax(logits))
    let next_temp = temperature_sample(logits, 0.8)

    print("")
    print("Result:")
    print("  Greedy next token:", next_greedy)
    print("  Temperature next token:", next_temp)

    // Detokenize
    let response_tokens = append(tokens, next_temp)
    let response = detokenize(tokenizer, response_tokens, true)
    print("  Generated text:", response)
    print("")
}
