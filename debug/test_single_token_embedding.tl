// ============================================================================
// Test: Single Token Embedding Support
// ============================================================================
//
// This test verifies that embedding() now accepts:
// 1. TokenIds (original)
// 2. TokenIdArray (original)
// 3. Integer (NEW - single token)
//
// Usage:
//   timeout 30 ./target/release/tl run:debug/test_single_token_embedding.tl
//
// ============================================================================

main {
    // Load TinyLlama model
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Get embedding table
    let embed_table = get_tensor(model, "token_embd.weight")
    print("Embedding table shape: ")
    print(embed_table.shape())
    print("\n")

    // Test 1: Original TokenIds (multiple tokens)
    print("Test 1: TokenIds (multiple tokens)\n")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let token_ids = tokenize(tokenizer, "Hello", false)
    print("Token IDs: ")
    print(token_ids)
    print("\n")

    let embeddings_multi = embedding(embed_table, token_ids)
    print("Result shape: ")
    print(embeddings_multi.shape())
    print("\n\n")

    // Test 2: NEW - Single token as integer
    print("Test 2: Integer (single token)\n")
    let single_token = 15043  // "Hello" token
    print("Token ID: ")
    print(single_token)
    print("\n")

    let embedding_single = embedding(embed_table, single_token)
    print("Result shape: ")
    print(embedding_single.shape())
    print("\n")

    print("✓ SUCCESS: Single token embedding works!\n")
    print("  Both TokenIds and Integer types are accepted by embedding()\n")
    print("\n✓ All tests completed!\n")
}
