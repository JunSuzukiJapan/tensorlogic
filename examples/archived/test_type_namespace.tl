// Test type namespace syntax

main {
    print("=== Testing Type Namespace Syntax ===")
    print("")

    // Test 1: f32::zeros
    print("[Test 1] f32::zeros([3, 2]):")
    let z_f32 = f32::zeros([3, 2])
    print("  Shape:", shape(z_f32))
    print("")

    // Test 2: f16::zeros
    print("[Test 2] f16::zeros([3, 2]):")
    let z_f16 = f16::zeros([3, 2])
    print("  Shape:", shape(z_f16))
    print("")

    // Test 3: f32::ones
    print("[Test 3] f32::ones([2, 3]):")
    let o_f32 = f32::ones([2, 3])
    print("  Shape:", shape(o_f32))
    print("")

    // Test 4: f16::ones
    print("[Test 4] f16::ones([2, 3]):")
    let o_f16 = f16::ones([2, 3])
    print("  Shape:", shape(o_f16))
    print("")

    // Test 5: Type compatibility
    print("[Test 5] f32 type compatibility:")
    let a = f32::zeros([3])
    let b = f32::ones([3])
    let c = a + b
    print("  ✓ f32::zeros + f32::ones succeeded")
    print("")

    // Test 6: arange (always f32)
    print("[Test 6] f32::arange(5):")
    let r = f32::arange(5)
    print("  Shape:", shape(r))
    print("")

    print("=== All Tests Passed ===")
}
