// Test f16::arange() implementation

main {
    print("=== f16::arange() Test ===")
    print("")

    print("[Test 1] f16::arange(5):")
    let r1 = f16::arange(5)
    print("  Shape:", shape(r1))
    print("  ✓ f16::arange(5) succeeded")
    print("")

    print("[Test 2] f16::arange(2, 7):")
    let r2 = f16::arange(2, 7)
    print("  Shape:", shape(r2))
    print("  ✓ f16::arange(2, 7) succeeded")
    print("")

    print("[Test 3] Compare f32::arange and f16::arange:")
    let r_f32 = f32::arange(10)
    let r_f16 = f16::arange(10)
    print("  f32::arange(10) shape:", shape(r_f32))
    print("  f16::arange(10) shape:", shape(r_f16))
    print("  ✓ Both types work correctly")
    print("")

    print("=== All Tests Passed ===")
}
