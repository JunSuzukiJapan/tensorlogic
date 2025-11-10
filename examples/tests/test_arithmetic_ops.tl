// Comprehensive test for arithmetic Metal shader operations
// Tests: add, subtract, multiply, divide, scalar operations

main {
    print("=== Arithmetic Operations Test ===")
    print("")

    // Test 1: Element-wise addition
    print("[1/7] Testing element-wise addition...")
    let a1 = ones([3, 4])
    let b1 = ones([3, 4])
    let result1 = a1 + b1
    print("      1 + 1 =", result1[0, 0], "(expected: 2.0)")

    if result1[0, 0] == 2.0 && result1[1, 1] == 2.0 {
        print("      ✅ Addition works")
    } else {
        print("      ❌ Addition failed")
    }
    print("")

    // Test 2: Element-wise subtraction
    print("[2/7] Testing element-wise subtraction...")
    let a2 = ones([3, 4])
    let b2 = zeros([3, 4])
    let result2 = a2 - b2
    print("      1 - 0 =", result2[0, 0], "(expected: 1.0)")

    if result2[0, 0] == 1.0 && result2[2, 3] == 1.0 {
        print("      ✅ Subtraction works")
    } else {
        print("      ❌ Subtraction failed")
    }
    print("")

    // Test 3: Element-wise multiplication
    print("[3/7] Testing element-wise multiplication...")
    let a3 = positional_encoding(2, 3)
    let b3 = ones([2, 3])
    let result3 = a3 * b3
    print("      x * 1 =", result3[0, 0], "(should equal x)")

    let check3 = result3[0, 0] == a3[0, 0] && result3[1, 2] == a3[1, 2]
    if check3 {
        print("      ✅ Multiplication works")
    } else {
        print("      ❌ Multiplication failed")
    }
    print("")

    // Test 4: Element-wise division
    print("[4/7] Testing element-wise division...")
    let a4 = ones([3, 4])
    let b4 = ones([3, 4])
    let result4 = a4 / b4
    print("      1 / 1 =", result4[0, 0], "(expected: 1.0)")

    if result4[0, 0] == 1.0 && result4[2, 3] == 1.0 {
        print("      ✅ Division works")
    } else {
        print("      ❌ Division failed")
    }
    print("")

    // Test 5: Scalar addition
    print("[5/7] Testing scalar addition...")
    let a5 = zeros([2, 3])
    let result5 = a5 + 5.0
    print("      0 + 5 =", result5[0, 0], "(expected: 5.0)")

    if result5[0, 0] == 5.0 && result5[1, 2] == 5.0 {
        print("      ✅ Scalar addition works")
    } else {
        print("      ❌ Scalar addition failed")
    }
    print("")

    // Test 6: Scalar multiplication
    print("[6/7] Testing scalar multiplication...")
    let a6 = ones([2, 3])
    let result6 = a6 * 3.0
    print("      1 * 3 =", result6[0, 0], "(expected: 3.0)")

    if result6[0, 0] == 3.0 && result6[1, 2] == 3.0 {
        print("      ✅ Scalar multiplication works")
    } else {
        print("      ❌ Scalar multiplication failed")
    }
    print("")

    // Test 7: Complex expression (multiple operations)
    print("[7/7] Testing complex expression...")
    let a7 = ones([2, 2])
    let b7 = ones([2, 2])
    let result7 = (a7 + b7) * 2.0 - 1.0
    print("      (1 + 1) * 2 - 1 =", result7[0, 0], "(expected: 3.0)")

    if result7[0, 0] == 3.0 && result7[1, 1] == 3.0 {
        print("      ✅ Complex expression works")
    } else {
        print("      ❌ Complex expression failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested arithmetic operations:")
    print("  ✓ Element-wise addition")
    print("  ✓ Element-wise subtraction")
    print("  ✓ Element-wise multiplication")
    print("  ✓ Element-wise division")
    print("  ✓ Scalar addition")
    print("  ✓ Scalar multiplication")
    print("  ✓ Complex expressions")
    print("")
    print("All arithmetic shader operations tested!")
}
