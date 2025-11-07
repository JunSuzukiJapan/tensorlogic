// Test reshape → broadcast_to interaction bug

main {
    print("=== Testing reshape → broadcast_to Bug ===")
    print("")

    let seq_len = 20.0
    let kv_heads = 4.0
    let head_dim = 64.0

    // Test 1: Direct ones → broadcast (works)
    print("[1] Direct f32::ones() → broadcast...")
    let k_exp_direct = f32::ones([seq_len, kv_heads, 1.0, head_dim])
    let k_bc_direct = broadcast_to(k_exp_direct, [seq_len, kv_heads, 8.0, head_dim])
    print("  ✓ Direct approach OK")
    print("")

    // Test 2: ones → reshape → broadcast (hangs?)
    print("[2] f32::ones() → reshape → broadcast...")
    let k_3d = f32::ones([seq_len, kv_heads, head_dim])
    print("  2a: Created [", seq_len, ",", kv_heads, ",", head_dim, "]")

    print("  2b: Reshaping to [", seq_len, ",", kv_heads, ", 1,", head_dim, "]...")
    let k_4d = reshape(k_3d, [seq_len, kv_heads, 1.0, head_dim])
    print("  2c: Reshape completed")

    print("  2d: Broadcasting to [", seq_len, ",", kv_heads, ", 8,", head_dim, "]...")
    let k_bc = broadcast_to(k_4d, [seq_len, kv_heads, 8.0, head_dim])
    print("  2e: Broadcast completed")

    print("  ✓ Reshape → broadcast OK")
    print("")

    // Test 3: ones → rope → reshape → broadcast (hangs?)
    print("[3] f32::ones() → RoPE → reshape → broadcast...")
    let k_3d2 = f32::ones([seq_len, kv_heads, head_dim])
    print("  3a: Created tensor")

    print("  3b: Applying RoPE...")
    let k_rope = rope(k_3d2)
    print("  3c: RoPE completed")

    print("  3d: Reshaping...")
    let k_4d2 = reshape(k_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  3e: Reshape completed")

    print("  3f: Broadcasting...")
    let k_bc2 = broadcast_to(k_4d2, [seq_len, kv_heads, 8.0, head_dim])
    print("  3g: Broadcast completed")

    print("  ✓ RoPE → reshape → broadcast OK")
    print("")

    print("=== Test completed ===")
}
