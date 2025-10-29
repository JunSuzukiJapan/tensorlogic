// Test broadcast_to() with different sizes

main {
    print("=== Testing broadcast_to() Issue ===")
    print("")

    // Test 1: Small broadcast (from test 6 that worked)
    print("[1] Small broadcast...")
    let x_small = f32::ones([10, 4, 1, 64])
    let bc_small = broadcast_to(x_small, [10.0, 4.0, 8.0, 64.0])
    print("  ✓ Small broadcast OK")
    print("")

    // Test 2: Medium broadcast
    print("[2] Medium broadcast (seq=20)...")
    let x_med = f32::ones([20, 4, 1, 64])
    let bc_med = broadcast_to(x_med, [20.0, 4.0, 8.0, 64.0])
    print("  ✓ Medium broadcast OK")
    print("")

    // Test 3: Exact problem size
    print("[3] Problem size broadcast (the one that hangs)...")
    let seq_len = 20.0
    let kv_heads = 4.0
    let head_dim = 64.0

    print("  Creating tensor [", seq_len, ",", kv_heads, ", 1,", head_dim, "]...")
    let k_exp = f32::ones([seq_len, kv_heads, 1.0, head_dim])

    print("  Broadcasting to [", seq_len, ",", kv_heads, ", 8,", head_dim, "]...")
    let k_broadcast = broadcast_to(k_exp, [seq_len, kv_heads, 8.0, head_dim])

    print("  ✓ Problem size broadcast OK")
    print("")

    print("=== Broadcast tests completed ===")
}
