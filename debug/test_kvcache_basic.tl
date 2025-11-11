// Basic KVCache test

main {
    print("=== KVCache Basic Test ===")
    print("")

    // Create KV cache for 22 layers
    print("[1] Creating KV cache...")
    let cache = KVCache::new_f16(22)
    print("    ✓ Cache created:", cache)
    print("")

    // Load model
    print("[2] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("    ✓ Model loaded")
    print("")

    // Test cache inspection
    print("[3] Cache inspection:")
    print("    Cache:", cache)
    print("")

    print("=== Test Complete ===")
}
