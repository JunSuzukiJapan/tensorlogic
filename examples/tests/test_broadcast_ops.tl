// Comprehensive test for broadcast Metal shader operations
// Tests: broadcasting scalar to tensor, broadcasting smaller tensors to larger ones

main {
    print("=== Broadcast Operations Test ===")
    print("")

    // Test 1: Scalar broadcast with addition
    print("[1/5] Testing scalar broadcast (addition)...")
    let a1 = ones([2, 3])
    let result1 = a1 + 5.0
    print("      ones([2, 3]) + 5.0")
    print("      Result:", result1[0, 0], result1[0, 1], result1[1, 2])

    if result1[0, 0] == 6.0 && result1[1, 2] == 6.0 {
        print("      ✅ Scalar broadcast addition works")
    } else {
        print("      ❌ Scalar broadcast addition failed")
    }
    print("")

    // Test 2: Scalar broadcast with multiplication
    print("[2/5] Testing scalar broadcast (multiplication)...")
    let a2 = ones([2, 3]) * 2.0
    let result2 = a2 * 3.0
    print("      (ones * 2) * 3")
    print("      Result:", result2[0, 0], result2[0, 1], result2[1, 2])

    if result2[0, 0] == 6.0 && result2[1, 2] == 6.0 {
        print("      ✅ Scalar broadcast multiplication works")
    } else {
        print("      ❌ Scalar broadcast multiplication failed")
    }
    print("")

    // Test 3: Scalar broadcast with subtraction
    print("[3/5] Testing scalar broadcast (subtraction)...")
    let a3 = ones([2, 3]) * 10.0
    let result3 = a3 - 3.0
    print("      (ones * 10) - 3")
    print("      Result:", result3[0, 0], result3[0, 1], result3[1, 2])

    if result3[0, 0] == 7.0 && result3[1, 2] == 7.0 {
        print("      ✅ Scalar broadcast subtraction works")
    } else {
        print("      ❌ Scalar broadcast subtraction failed")
    }
    print("")

    // Test 4: Scalar broadcast with division
    print("[4/5] Testing scalar broadcast (division)...")
    let a4 = ones([2, 3]) * 12.0
    let result4 = a4 / 4.0
    print("      (ones * 12) / 4")
    print("      Result:", result4[0, 0], result4[0, 1], result4[1, 2])

    if result4[0, 0] == 3.0 && result4[1, 2] == 3.0 {
        print("      ✅ Scalar broadcast division works")
    } else {
        print("      ❌ Scalar broadcast division failed")
    }
    print("")

    // Test 5: Complex broadcast expression
    print("[5/5] Testing complex broadcast expression...")
    let a5 = ones([2, 2])
    let result5 = (a5 + 2.0) * 3.0 - 1.0
    print("      (ones + 2) * 3 - 1")
    print("      Result:", result5[0, 0], result5[0, 1], result5[1, 1])

    // (1 + 2) * 3 - 1 = 3 * 3 - 1 = 9 - 1 = 8
    if result5[0, 0] == 8.0 && result5[1, 1] == 8.0 {
        print("      ✅ Complex broadcast expression works")
    } else {
        print("      ❌ Complex broadcast expression failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested broadcast operations:")
    print("  ✓ Scalar broadcast addition")
    print("  ✓ Scalar broadcast multiplication")
    print("  ✓ Scalar broadcast subtraction")
    print("  ✓ Scalar broadcast division")
    print("  ✓ Complex broadcast expressions")
    print("")
    print("All broadcast shader operations tested!")
}
