// Test 1, 2, 3 sequentially to find interference

main {
    print("=== Testing Tests 1, 2, 3 Sequentially ===")
    print("")

    // Real sizes
    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // Test 1: Reshape to heads
    print("[1] Reshaping to heads...")
    let x = f32::ones([seq_len, hidden])
    let x_heads = reshape(x, [seq_len, num_heads, head_dim])
    print("  ✓ Reshape OK, shape:", shape(x_heads))
    print("")

    // Test 2: RoPE
    print("[2] Applying RoPE...")
    let x_rope = rope(x_heads)
    print("  ✓ RoPE OK")
    print("")

    // Test 3: Key expansion (4 heads → 32 heads)
    print("[3] Key expansion...")
    let kv_heads = 4.0
    print("  3a: Creating K tensor...")
    let k_heads = f32::ones([seq_len, kv_heads, head_dim])
    print("  3b: Applying RoPE to K...")
    let k_rope = rope(k_heads)
    print("  3c: Reshaping...")
    let k_exp = reshape(k_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  3d: Broadcasting...")
    let k_broadcast = broadcast_to(k_exp, [seq_len, kv_heads, 8.0, head_dim])
    print("  3e: Final reshape...")
    let k_expanded = reshape(k_broadcast, [seq_len, num_heads, head_dim])
    print("  ✓ Key expansion OK")
    print("")

    print("=== All tests completed successfully ===")
}
