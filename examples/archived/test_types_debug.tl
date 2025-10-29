// Debug type checking

main {
    print("=== Type Debug Test ===")
    print("")

    // Test 1: Check what arange(3) returns
    print("[Test 1] arange(3) alone:")
    let idx = arange(3)
    let s_idx = shape(idx)
    print("  Shape:", s_idx)
    print("")

    // Test 2: Check what zeros with f32 shape returns
    print("[Test 2] zeros([3.0]):")
    let z_f32 = zeros([3.0])
    let s_z_f32 = shape(z_f32)
    print("  Shape:", s_z_f32)
    print("")

    // Test 3: Check what zeros with integer shape returns
    print("[Test 3] zeros([3]):")
    let z_int = zeros([3])
    let s_z_int = shape(z_int)
    print("  Shape:", s_z_int)
    print("")

    // Test 4: Try simple arithmetic to see types
    print("[Test 4] Type compatibility test:")
    let a = zeros([3.0])
    let b = arange(3)
    print("  a shape:", shape(a))
    print("  b shape:", shape(b))
    let c = a + b
    print("  a + b shape:", shape(c))
    print("")

    print("=== Tests complete ===")
}
