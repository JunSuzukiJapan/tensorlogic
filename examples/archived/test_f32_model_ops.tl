// Test model-specific f32 operations

main {
    print("=== Testing f32 Model Operations ===")
    print("")

    // Test 1: linear()
    print("[1] Testing linear()...")
    let x = f32::ones([1, 256])
    let w = f32::ones([512, 256])
    let out = linear(x, w)
    print("  ✓ linear() OK, shape:", shape(out))
    print("")

    // Test 2: rms_norm()
    print("[2] Testing rms_norm()...")
    let x_norm = f32::ones([1, 256])
    let w_norm = f32::ones([256])
    let normed = rms_norm(x_norm, w_norm)
    print("  ✓ rms_norm() OK, shape:", shape(normed))
    print("")

    // Test 3: reshape()
    print("[3] Testing reshape()...")
    let x_reshape = f32::ones([10, 256])
    let reshaped = reshape(x_reshape, [10.0, 4.0, 64.0])
    print("  ✓ reshape() OK, shape:", shape(reshaped))
    print("")

    // Test 4: rope()
    print("[4] Testing rope()...")
    let x_rope = f32::ones([10, 4, 64])
    let roped = rope(x_rope)
    print("  ✓ rope() OK, shape:", shape(roped))
    print("")

    // Test 5: einsum() simple
    print("[5] Testing einsum() simple...")
    let a = f32::ones([5, 3, 4])
    let b = f32::ones([5, 3, 4])
    let ein_out = einsum("ihd,jhd->ihj", a, b)
    print("  ✓ einsum() simple OK, shape:", shape(ein_out))
    print("")

    // Test 6: broadcast_to()
    print("[6] Testing broadcast_to()...")
    let x_bc = f32::ones([10, 4, 1, 64])
    let bc_out = broadcast_to(x_bc, [10.0, 4.0, 8.0, 64.0])
    print("  ✓ broadcast_to() OK, shape:", shape(bc_out))
    print("")

    print("=== All model operations completed successfully ===")
}
