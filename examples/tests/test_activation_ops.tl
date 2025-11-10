// Comprehensive test for activation function Metal shaders
// Tests: ReLU, GELU, tanh, sigmoid (if available)

main {
    print("=== Activation Functions Test ===")
    print("")

    // Test 1: ReLU activation
    print("[1/4] Testing ReLU...")
    let a1 = positional_encoding(3, 3)
    let neg1 = zeros([3, 3]) - a1  // Create negative values
    let result1 = relu(neg1)
    print("      ReLU on negatives:", result1[0, 0], "(expected: 0.0)")

    let pos_result1 = relu(a1)
    let pos_check = pos_result1[0, 0] == a1[0, 0]
    let neg_check = result1[0, 0] == 0.0 && result1[1, 1] == 0.0

    if pos_check && neg_check {
        print("      ✅ ReLU works (zeros negatives, keeps positives)")
    } else {
        print("      ❌ ReLU failed")
    }
    print("")

    // Test 2: GELU activation
    print("[2/4] Testing GELU...")
    let a2 = zeros([2, 3])
    let result2 = gelu(a2)
    print("      GELU(0) =", result2[0, 0], "(expected: ~0.0)")

    let a2b = ones([2, 3])
    let result2b = gelu(a2b)
    print("      GELU(1) =", result2b[0, 0], "(expected: ~0.841)")

    // GELU(0) should be ~0, GELU(1) should be ~0.841
    let check2_zero = result2[0, 0] < 0.1 && result2[0, 0] > -0.1
    let check2_one = result2b[0, 0] > 0.7 && result2b[0, 0] < 1.0

    if check2_zero && check2_one {
        print("      ✅ GELU works")
    } else {
        print("      ❌ GELU failed")
    }
    print("")

    // Test 3: tanh activation
    print("[3/4] Testing tanh...")
    let a3 = zeros([2, 3])
    let result3 = tanh(a3)
    print("      tanh(0) =", result3[0, 0], "(expected: 0.0)")

    let a3b = ones([2, 3])
    let result3b = tanh(a3b)
    print("      tanh(1) =", result3b[0, 0], "(expected: ~0.762)")

    // tanh(0) = 0, tanh(1) ≈ 0.762
    let check3_zero = result3[0, 0] < 0.1 && result3[0, 0] > -0.1
    let check3_one = result3b[0, 0] > 0.7 && result3b[0, 0] < 0.8

    if check3_zero && check3_one {
        print("      ✅ tanh works")
    } else {
        print("      ❌ tanh failed")
    }
    print("")

    // Test 4: Activation composition
    print("[4/4] Testing activation composition...")
    let a4 = ones([2, 2])
    let step1 = gelu(a4)
    let step2 = tanh(step1)
    let step3 = relu(step2)
    print("      relu(tanh(gelu(1))) =", step3[0, 0])

    // Should be positive since all activations preserve positives
    if step3[0, 0] > 0.0 && step3[1, 1] > 0.0 {
        print("      ✅ Activation composition works")
    } else {
        print("      ❌ Activation composition failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested activation functions:")
    print("  ✓ ReLU (Rectified Linear Unit)")
    print("  ✓ GELU (Gaussian Error Linear Unit)")
    print("  ✓ tanh (Hyperbolic Tangent)")
    print("  ✓ Activation composition")
    print("")
    print("All activation shader operations tested!")
}
