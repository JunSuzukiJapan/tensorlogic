// Minimal transformer test to debug hang

main {
    print("=== Minimal Transformer Test ===")

    // Load model
    print("[1/3] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model_f32(model_path)
    print("      ✓ Model loaded")

    // Load tokenizer
    print("[2/3] Loading tokenizer...")
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Tokenizer loaded")

    // Get weights
    print("[3/3] Testing operations...")
    let embed_table = model.token_embd.weight
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    print("      Got weights")

    // Test embedding
    let tokens = tokenize(tokenizer, "Hello", false)
    print("      Tokens:", tokens)

    let x = embedding(embed_table, tokens)
    print("      ✓ Embedding done")

    // Test rms_norm
    print("      Testing rms_norm...")
    let x_norm = rms_norm(x, attn_norm_0)
    print("      ✓ rms_norm done")

    // Test linear
    print("      Testing linear...")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let Q = linear(x_norm, W_q_0)
    print("      ✓ linear done")

    print("=== Test Complete ===")
}
