// Debug RoPE with 29 vs 30 tokens
// Find why NaN appears at 30 tokens

main {
    print("=== RoPE Debug: 29 vs 30 tokens ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    // Test 1: 29 tokens (works)
    print("[Test 1] RoPE with 29 tokens")
    let test_29 = ones([29, 32, 64])  // [seq_len=29, n_heads=32, head_dim=64]
    let rope_29 = rope(test_29)
    print("  Input shape: [29, 32, 64]")
    print("  Output (first 5):", rope_29)
    print("  Output shape:", shape(rope_29))
    print("")

    // Test 2: 30 tokens (fails)
    print("[Test 2] RoPE with 30 tokens")
    let test_30 = ones([30, 32, 64])  // [seq_len=30, n_heads=32, head_dim=64]
    let rope_30 = rope(test_30)
    print("  Input shape: [30, 32, 64]")
    print("  Output (first 5):", rope_30)
    print("  Output shape:", shape(rope_30))
    print("")

    // Test 3: 31 tokens
    print("[Test 3] RoPE with 31 tokens")
    let test_31 = ones([31, 32, 64])
    let rope_31 = rope(test_31)
    print("  Input shape: [31, 32, 64]")
    print("  Output (first 5):", rope_31)
    print("")

    // Test 4: Different head configurations
    print("[Test 4] RoPE with 30 tokens, 4 heads (KV heads)")
    let test_30_kv = ones([30, 4, 64])  // [seq_len=30, n_kv_heads=4, head_dim=64]
    let rope_30_kv = rope(test_30_kv)
    print("  Input shape: [30, 4, 64]")
    print("  Output (first 5):", rope_30_kv)
    print("")

    print("=== Check for NaN/Inf above ===")
}
