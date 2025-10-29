// Test model loading and tokenizer with f32

main {
    print("=== Testing f32 Model Loading ===")
    print("")

    // Test 1: Load model
    print("[1] Loading model...")
    let home = env("HOME")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    print("  ✓ Model loaded")
    print("")

    // Test 2: Load tokenizer
    print("[2] Loading tokenizer...")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")
    print("  ✓ Tokenizer loaded")
    print("")

    // Test 3: Get a tensor
    print("[3] Getting embedding tensor...")
    let tok_embd = get_tensor(model, "token_embd.weight")
    let emb_shape = shape(tok_embd)
    print("  ✓ Embedding tensor loaded, shape:", emb_shape)
    print("")

    // Test 4: Get layer 0 weights
    print("[4] Getting layer 0 attention weights...")
    let layer_0_q = get_tensor(model, "blk.0.attn_q.weight")
    let q_shape = shape(layer_0_q)
    print("  ✓ Layer 0 Q loaded, shape:", q_shape)
    print("")

    // Test 5: Tokenize
    print("[5] Tokenizing...")
    let text = "Hello!"
    let tokens = tokenize(tokenizer, text, true)
    print("  ✓ Tokenized:", text)
    print("")

    // Test 6: Embedding lookup
    print("[6] Embedding lookup...")
    let x = embedding(tok_embd, tokens)
    let x_shape = shape(x)
    print("  ✓ Embedding looked up, shape:", x_shape)
    print("")

    // Test 7: Linear with actual model weights
    print("[7] Linear with actual weights...")
    let out = linear(x, layer_0_q)
    let out_shape = shape(out)
    print("  ✓ Linear completed, shape:", out_shape)
    print("")

    print("=== All model loading tests completed successfully ===")
}
