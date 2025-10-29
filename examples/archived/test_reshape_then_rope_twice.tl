// Test reshape() → rope() twice

main {
    print("=== Testing Reshape → RoPE Twice ===")
    print("")

    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // First: reshape → rope
    print("[1] Reshape → RoPE...")
    let x1 = f32::ones([seq_len, hidden])
    let x1_heads = reshape(x1, [seq_len, num_heads, head_dim])
    let x1_rope = rope(x1_heads)
    print("  ✓ First Reshape → RoPE OK")
    print("")

    // Second: reshape → rope (DIFFERENT input, but same shape)
    print("[2] Another Reshape → RoPE...")
    let x2 = f32::ones([seq_len, hidden])
    let x2_heads = reshape(x2, [seq_len, num_heads, head_dim])
    let x2_rope = rope(x2_heads)
    print("  ✓ Second Reshape → RoPE OK")
    print("")

    // Now try broadcast after second reshape→rope
    print("[3] Broadcast after second reshape→rope...")
    let x2_4d = reshape(x2_rope, [seq_len, num_heads, 1.0, head_dim])
    let x2_bc = broadcast_to(x2_4d, [seq_len, num_heads, 8.0, head_dim])
    print("  ✓ Broadcast OK")
    print("")

    print("=== All operations completed ===")
}
