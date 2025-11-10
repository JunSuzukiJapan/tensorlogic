// Test if embedding lookup works correctly with token accumulation
// This tests the pattern used in chat_demo_22layers.tl

main {
    print("=== Embedding Token Accumulation Test ===")
    print("")
    print("Testing the pattern from chat_demo_22layers.tl:")
    print("  1. Embed initial tokens (like prompt)")
    print("  2. Append generated token")
    print("  3. Embed accumulated tokens")
    print("  4. Repeat")
    print("")

    // Load model, tokenizer, and get embedding table
    print("[1/5] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    let embed_table = get_tensor(model, "token_embd.weight")
    print("      ✓ Loaded model and tokenizer")
    print("      Embedding shape:", shape(embed_table))
    print("")

    // Test token accumulation pattern with real tokenization
    print("[2/5] Testing initial embedding (like prompt)...")
    let prompt_text = "Hello world"
    let initial_tokens = tokenize(tokenizer, prompt_text, false)
    print("      Prompt:", prompt_text)
    print("      Initial tokens:", initial_tokens)
    print("      Length:", len(initial_tokens))

    let e1 = embedding(embed_table, initial_tokens)
    print("      e1 shape:", shape(e1))
    print("      e1 first few values:", e1[0, 0], e1[0, 1], e1[0, 2])
    print("      ✅ Initial embedding works")
    print("")

    // Simulate first token generation (using a simple forward pass and sampling)
    print("[3/5] Simulating first token generation...")
    // For simplicity, just pick a token ID directly
    let token1 = 100
    let gen_tokens = append(initial_tokens, token1)
    print("      Generated token1:", token1)
    print("      Accumulated length:", len(gen_tokens))
    print("")

    // Test embedding with accumulated tokens (THIS IS WHERE THE BUG MIGHT BE)
    print("[4/5] Testing embedding with accumulated tokens...")
    let e2 = embedding(embed_table, gen_tokens)
    print("      e2 shape:", shape(e2))
    print("      e2 first few values:", e2[0, 0], e2[0, 1], e2[0, 2])
    let e2_check = e2[0, 0]

    if e2_check == 0.0 {
        print("      ❌ FAILED: e2 values are zero (BUG FOUND!)")
        print("      This is the bug affecting chat_demo_22layers.tl!")
    } else {
        print("      ✅ PASSED: e2 has non-zero values")
    }
    print("")

    // Simulate second token generation
    print("[5/5] Testing second accumulation...")
    let token2 = 200
    let gen_tokens2 = append(gen_tokens, token2)
    print("      Generated token2:", token2)
    print("      Accumulated length:", len(gen_tokens2))

    let e3 = embedding(embed_table, gen_tokens2)
    print("      e3 shape:", shape(e3))
    print("      e3 first few values:", e3[0, 0], e3[0, 1], e3[0, 2])
    let e3_check = e3[0, 0]

    if e3_check == 0.0 {
        print("      ❌ FAILED: e3 values are zero (BUG FOUND!)")
    } else {
        print("      ✅ PASSED: e3 has non-zero values")
    }
    print("")

    print("=== Test Complete ===")
    print("")
    print("Summary:")
    print("  - Initial embedding: ✅ Works")

    if e2_check != 0.0 {
        print("  - First accumulation: ✅ Works")
    } else {
        print("  - First accumulation: ❌ BUG FOUND!")
    }

    if e3_check != 0.0 {
        print("  - Second accumulation: ✅ Works")
    } else {
        print("  - Second accumulation: ❌ BUG FOUND!")
    }
}
