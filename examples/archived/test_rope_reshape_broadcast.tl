// Test rope() → reshape() → broadcast_to() combination

main {
    print("=== Testing RoPE → Reshape → Broadcast ===")
    print("")

    let seq_len = 20.0
    let kv_heads = 4.0
    let head_dim = 64.0

    // First test: without rope
    print("[1] Without RoPE: reshape → broadcast...")
    let k1 = f32::ones([seq_len, kv_heads, head_dim])
    let k1_4d = reshape(k1, [seq_len, kv_heads, 1.0, head_dim])
    let k1_bc = broadcast_to(k1_4d, [seq_len, kv_heads, 8.0, head_dim])
    print("  ✓ Without RoPE OK")
    print("")

    // Second test: with rope
    print("[2] With RoPE: rope → reshape → broadcast...")
    let k2 = f32::ones([seq_len, kv_heads, head_dim])
    print("  2a: Applying RoPE...")
    let k2_rope = rope(k2)
    print("  2b: Reshaping...")
    let k2_4d = reshape(k2_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  2c: Broadcasting (THIS IS WHERE IT HANGS)...")
    let k2_bc = broadcast_to(k2_4d, [seq_len, kv_heads, 8.0, head_dim])
    print("  ✓ With RoPE OK")
    print("")

    print("=== All operations completed ===")
}
