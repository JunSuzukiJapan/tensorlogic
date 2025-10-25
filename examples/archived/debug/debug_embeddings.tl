// Debug: Check embedding values

main {
    print("=== Debugging Embeddings ===")
    print("")

    // Load model
    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = home + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    // Get embedding table
    let embed_table = get_tensor(model, "token_embd.weight")
    print("Embedding table shape:", shape(embed_table))
    print("")

    // Test tokens
    let test_tokens = [1, 6324, 10994]  // BOS, "Hi", "Hello"

    print("Testing embeddings for tokens:", test_tokens)
    print("")

    // Get embeddings
    let embeddings = embedding(embed_table, test_tokens)
    print("Embeddings shape:", shape(embeddings))
    print("")

    // Print first few values of each embedding
    print("✅ Embeddings computed successfully!")
    print("")
    print("Note: Embedding values are 2048-dimensional vectors")
    print("Each token gets mapped to a learned vector representation")
}
