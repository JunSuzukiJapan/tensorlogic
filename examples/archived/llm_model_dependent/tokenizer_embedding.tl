// Test tokenizer + embedding integration

main {
    print("=== Tokenizer + Embedding Test ===")
    print("")

    // Load model and tokenizer
    print("[1/3] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model and tokenizer loaded")
    print("")

    // Get embedding table (and transpose it)
    print("[2/3] Getting embedding table...")
    let embed_table_raw = get_tensor(model, "token_embd.weight")
    print("      Raw embedding shape:", shape(embed_table_raw))
    
    // Transpose from [2048, 32000] to [32000, 2048]
    let embed_table = transpose(embed_table_raw)
    print("      Transposed shape:", shape(embed_table))
    print("")

    // Tokenize text
    print("[3/3] Testing tokenize → embed...")
    let text = "Hello world"
    print("      Input text: \"", text, "\"")

    let token_ids = tokenize(tokenizer, text, true)
    print("      Token IDs:", token_ids)

    // Get embeddings
    let embeddings = embedding(embed_table, token_ids)
    print("      Embeddings shape:", shape(embeddings))
    print("")

    print("✅ Tokenizer + Embedding integration test complete!")
    print("")
    print("Summary:")
    print("  • Vocabulary size: 32000")
    print("  • Embedding dimension: 2048")
    print("  • Input tokens:", token_ids)
    print("  • Output embeddings:", shape(embeddings))
}
