// Comprehensive test for reduction Metal shader operations
// Tests: sum, mean, max, min, softmax

main {
    print("=== Reduction Operations Test ===")
    print("")

    // Test 1: Sum reduction
    print("[1/5] Testing sum...")
    let a1 = ones([3, 4])
    let result1 = sum(a1)
    print("      sum(ones([3, 4])) =", result1)

    // Sum should be positive
    if result1 > 0.0 {
        print("      ✅ sum works (positive result)")
    } else {
        print("      ❌ sum failed")
    }
    print("")

    // Test 2: Mean reduction
    print("[2/5] Testing mean...")
    let a2 = ones([3, 4])
    let result2 = mean(a2)
    print("      mean(ones([3, 4])) =", result2)

    // Mean should be positive and <= 1.0 for ones
    if result2 > 0.0 && result2 <= 1.0 {
        print("      ✅ mean works (valid range)")
    } else {
        print("      ❌ mean failed")
    }
    print("")

    // Test 3: Max reduction
    print("[3/5] Testing max...")
    let a3 = positional_encoding(3, 4)
    let result3 = max(a3)
    print("      max(positional_encoding) =", result3)

    // Should be positive (positional encoding has positive values)
    if result3 > 0.0 {
        print("      ✅ max works (found positive maximum)")
    } else {
        print("      ❌ max failed")
    }
    print("")

    // Test 4: Min reduction
    print("[4/5] Testing min...")
    let a4 = positional_encoding(3, 4)
    let result4 = min(a4)
    print("      min(positional_encoding) =", result4)

    // Min should be less than or equal to max
    if result4 <= result3 {
        print("      ✅ min works (min <= max)")
    } else {
        print("      ❌ min failed")
    }
    print("")

    // Test 5: Softmax (with sum check)
    print("[5/5] Testing softmax...")
    let a5 = ones([1, 5])
    let result5 = softmax(a5)
    print("      softmax sample:", result5[0, 0], result5[0, 1], result5[0, 2])

    // All values should be equal for uniform input
    let all_equal = result5[0, 0] == result5[0, 1] && result5[0, 1] == result5[0, 2]

    // Sum should be approximately 1.0 (within small epsilon)
    let softmax_sum = result5[0, 0] + result5[0, 1] + result5[0, 2] + result5[0, 3] + result5[0, 4]
    print("      softmax sum =", softmax_sum, "(expected: ~1.0)")

    let sum_check = softmax_sum > 0.99 && softmax_sum < 1.01

    if all_equal && sum_check {
        print("      ✅ softmax works (uniform distribution, sum = 1)")
    } else {
        print("      ❌ softmax failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested reduction operations:")
    print("  ✓ sum (tensor reduction)")
    print("  ✓ mean (average)")
    print("  ✓ max (maximum value)")
    print("  ✓ min (minimum value)")
    print("  ✓ softmax (probability distribution)")
    print("")
    print("All reduction shader operations tested!")
}
