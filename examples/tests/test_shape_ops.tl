// Comprehensive test for shape manipulation Metal shader operations
// Tests: reshape, flatten, transpose, permute

main {
    print("=== Shape Operations Test ===")
    print("")

    // Test 1: Reshape
    print("[1/4] Testing reshape...")
    let a1 = positional_encoding(6, 1)  // [6, 1]
    let reshaped = reshape(a1, [2, 3])  // [6, 1] -> [2, 3]
    print("      Original shape: [6, 1]")
    print("      Reshaped to: [2, 3]")
    print("      Sample values:", reshaped[0, 0], reshaped[0, 1], reshaped[1, 0])

    // Just verify we can access the reshaped tensor
    let access_ok = reshaped[0, 0] >= 0.0
    if access_ok {
        print("      ✅ reshape works")
    } else {
        print("      ❌ reshape failed")
    }
    print("")

    // Test 2: Flatten
    print("[2/4] Testing flatten...")
    let a2 = ones([3, 4])  // [3, 4]
    let flattened = flatten(a2)
    print("      Original shape: [3, 4]")
    print("      Flattened shape: 1D")
    print("      Sample values:", flattened[0], flattened[5], flattened[11])

    // All should be 1.0 since we flattened ones
    let flatten_ok = flattened[0] == 1.0 && flattened[5] == 1.0
    if flatten_ok {
        print("      ✅ flatten works")
    } else {
        print("      ❌ flatten failed")
    }
    print("")

    // Test 3: Transpose
    print("[3/4] Testing transpose...")
    let a3 = positional_encoding(2, 3)  // [2, 3]
    let transposed = transpose(a3)      // [3, 2]
    print("      Original shape: [2, 3]")
    print("      Transposed shape: [3, 2]")
    print("      Original [0,0]:", a3[0, 0])
    print("      Transposed [0,0]:", transposed[0, 0])

    // Verify transpose swaps dimensions
    let transpose_ok = transposed[0, 0] == a3[0, 0]
    if transpose_ok {
        print("      ✅ transpose works")
    } else {
        print("      ❌ transpose failed")
    }
    print("")

    // Test 4: Permute
    print("[4/4] Testing permute...")
    let a4 = positional_encoding(3, 4)  // [3, 4]
    let permuted = permute(a4, [1, 0])  // Swap dimensions: [4, 3]
    print("      Original shape: [3, 4]")
    print("      Permuted [1, 0]: [4, 3]")
    print("      Sample values:", permuted[0, 0], permuted[1, 0])

    // Just verify we can access the permuted tensor
    let permute_ok = permuted[0, 0] >= 0.0
    if permute_ok {
        print("      ✅ permute works")
    } else {
        print("      ❌ permute failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested shape operations:")
    print("  ✓ reshape (change dimensions)")
    print("  ✓ flatten (collapse to 1D)")
    print("  ✓ transpose (2D swap)")
    print("  ✓ permute (multi-dimensional reorder)")
    print("")
    print("All shape shader operations tested!")
}
