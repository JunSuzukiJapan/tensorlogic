// Test loop-based generation approach
// Verify: for loop, append, break, dynamic model access

main {
    print("=== Loop Generation Test ===")
    print("")

    // Test 1: Basic for loop with break
    print("[1/3] Testing for loop with break...")
    for i in range(10) {
        print("  Token", i)
        if i == 3 {
            print("  Breaking at i=3")
            break
        }
    }
    print("  ✓ Loop with break works")
    print("")

    // Test 2: Token append
    print("[2/3] Testing token append...")

    // Load tokenizer to get TokenIds
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let token_ids = tokenize(tokenizer, "Hello", false)
    print("  Initial tokens:", token_ids)

    let new_token = 848  // Some token
    token_ids = append(token_ids, new_token)
    print("  After append:", token_ids)

    // Verify it works
    let decoded = detokenize(tokenizer, token_ids, false)
    print("  Decoded:", decoded)
    print("  ✓ Token append works")
    print("")

    // Test 3: Dynamic model layer access
    print("[3/3] Testing dynamic model access...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)
    print("  ✓ Model loaded")

    // Access first 3 layers dynamically
    for i in range(3) {
        let norm_w = model.blk[i].attn_norm.weight
        let norm_shape = shape(norm_w)
        print("  Layer", i, "attn_norm shape:", norm_shape)
    }
    print("  ✓ Dynamic layer access works")
    print("")

    print("=== All Tests Passed ===")
    print("")
}
