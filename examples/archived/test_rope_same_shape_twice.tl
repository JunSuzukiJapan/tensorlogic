// Test rope() on same shape twice - this might be the issue

main {
    print("=== Testing RoPE on Same Shape Twice ===")
    print("")

    let seq_len = 20.0
    let num_heads = 32.0
    let head_dim = 64.0

    // First rope() with [20, 32, 64]
    print("[1] First RoPE on [20, 32, 64]...")
    let x1 = f32::ones([seq_len, num_heads, head_dim])
    let x1_rope = rope(x1)
    print("  ✓ First RoPE OK")
    print("")

    // Second rope() ALSO with [20, 32, 64] (SAME SHAPE!)
    print("[2] Second RoPE ALSO on [20, 32, 64]...")
    let x2 = f32::ones([seq_len, num_heads, head_dim])
    let x2_rope = rope(x2)
    print("  ✓ Second RoPE OK")
    print("")

    // Now try broadcast after second rope
    print("[3] Broadcast after second RoPE...")
    let x2_4d = reshape(x2_rope, [seq_len, num_heads, 1.0, head_dim])
    let x2_bc = broadcast_to(x2_4d, [seq_len, num_heads, 8.0, head_dim])
    print("  ✓ Broadcast OK")
    print("")

    print("=== All operations completed ===")
}
