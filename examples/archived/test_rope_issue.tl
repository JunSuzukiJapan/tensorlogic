// Test rope() twice to isolate the issue

main {
    print("=== Testing RoPE Twice ===")
    print("")

    let seq_len = 20.0
    let num_heads = 32.0
    let head_dim = 64.0
    let hidden = 2048.0

    // First rope() call
    print("[1] First RoPE...")
    let x1 = f32::ones([seq_len, hidden])
    let x1_heads = reshape(x1, [seq_len, num_heads, head_dim])
    let x1_rope = rope(x1_heads)
    print("  ✓ First RoPE OK")
    print("")

    // Second rope() call
    print("[2] Second RoPE...")
    let kv_heads = 4.0
    let k_heads = f32::ones([seq_len, kv_heads, head_dim])
    let k_rope = rope(k_heads)
    print("  ✓ Second RoPE OK")
    print("")

    // Try simple operation after two ropes
    print("[3] Simple reshape after two RoPEs...")
    let k_reshaped = reshape(k_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  ✓ Reshape OK")
    print("")

    print("=== All operations completed ===")
}
