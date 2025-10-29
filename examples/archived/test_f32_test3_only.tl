// Test only the problematic Test 3 from attention_ops

main {
    print("=== Testing Only Test 3 (Key Expansion) ===")
    print("")

    // Real sizes from TinyLlama
    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // Test 3: Key expansion (4 heads → 32 heads)
    print("[3] Key expansion...")
    let kv_heads = 4.0
    print("  3a: Creating K tensor [", seq_len, ",", kv_heads, ",", head_dim, "]...")
    let k_heads = f32::ones([seq_len, kv_heads, head_dim])
    print("  3b: Applying RoPE to K...")
    let k_rope = rope(k_heads)
    print("  3c: Reshaping to [", seq_len, ",", kv_heads, ", 1,", head_dim, "]...")
    let k_exp = reshape(k_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  3d: Broadcasting to [", seq_len, ",", kv_heads, ", 8,", head_dim, "]...")
    let k_broadcast = broadcast_to(k_exp, [seq_len, kv_heads, 8.0, head_dim])
    print("  3e: Reshaping to [", seq_len, ",", num_heads, ",", head_dim, "]...")
    let k_expanded = reshape(k_broadcast, [seq_len, num_heads, head_dim])
    print("  ✓ Key expansion OK")
    print("")

    print("=== Test 3 completed successfully ===")
}
