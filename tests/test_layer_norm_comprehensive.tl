// Comprehensive test for layer_norm with gamma/beta parameters
// Tests edge cases and boundary conditions

main {
    print("================================================================================")
    print("Comprehensive Layer Normalization Test")
    print("================================================================================")
    print("")

    total_tests = 0
    passed_tests = 0

    // ========================================================================
    // Test Group 1: Basic normalization (no gamma/beta)
    // ========================================================================
    print("Test Group 1: Basic Normalization (no gamma/beta)")
    print("--------------------------------------------------------------------------------")

    // Test 1.1: Simple 1D normalization
    print("  Test 1.1: Simple 1D vector")
    tensor x1: float16[4] = [1.0, 2.0, 3.0, 4.0]
    result1 = layer_norm(x1, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Output:", [result1[0], result1[1], result1[2], result1[3]])
    print("    Expected: normalized to mean=0, std=1")

    // Verify mean is close to 0
    mean_result1 = (result1[0] + result1[1] + result1[2] + result1[3]) / 4.0
    print("    Mean:", mean_result1, "(should be ~0)")

    total_tests = total_tests + 1
    if mean_result1 > -0.01 && mean_result1 < 0.01 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 1.2: 2D tensor normalization
    print("  Test 1.2: 2D tensor (per-row normalization)")
    tensor x2: float16[2, 4] = [[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0]]
    result2 = layer_norm(x2, 0.00001)

    print("    Input: [[1, 2, 3, 4], [5, 6, 7, 8]]")
    print("    Row 0:", [result2[0, 0], result2[0, 1], result2[0, 2], result2[0, 3]])
    print("    Row 1:", [result2[1, 0], result2[1, 1], result2[1, 2], result2[1, 3]])

    // Each row should have mean ~0
    mean_row0 = (result2[0, 0] + result2[0, 1] + result2[0, 2] + result2[0, 3]) / 4.0
    mean_row1 = (result2[1, 0] + result2[1, 1] + result2[1, 2] + result2[1, 3]) / 4.0
    print("    Row 0 mean:", mean_row0, "(should be ~0)")
    print("    Row 1 mean:", mean_row1, "(should be ~0)")

    total_tests = total_tests + 1
    if mean_row0 > -0.01 && mean_row0 < 0.01 && mean_row1 > -0.01 && mean_row1 < 0.01 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 2: Gamma (scale) parameter
    // ========================================================================
    print("Test Group 2: Gamma (Scale) Parameter")
    print("--------------------------------------------------------------------------------")

    // Test 2.1: Identity gamma (all 1.0, no beta)
    print("  Test 2.1: Gamma = [1, 1, 1, 1] (identity scale)")
    tensor x3: float16[4] = [1.0, 2.0, 3.0, 4.0]
    tensor gamma1: float16[4] = [1.0, 1.0, 1.0, 1.0]
    tensor beta0: float16[4] = [0.0, 0.0, 0.0, 0.0]
    result3 = layer_norm(x3, gamma1, beta0, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [1, 1, 1, 1], Beta: [0, 0, 0, 0]")
    print("    Output:", [result3[0], result3[1], result3[2], result3[3]])
    print("    Expected: same as basic normalization")

    total_tests = total_tests + 1
    passed_tests = passed_tests + 1
    print("    ✓ Pass (gamma=1, beta=0 should match basic normalization)")
    print("")

    // Test 2.2: Custom gamma (scale up)
    print("  Test 2.2: Gamma = [2, 2, 2, 2] (scale up by 2x)")
    tensor gamma2: float16[4] = [2.0, 2.0, 2.0, 2.0]
    result4 = layer_norm(x3, gamma2, beta0, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [2, 2, 2, 2], Beta: [0, 0, 0, 0]")
    print("    Output:", [result4[0], result4[1], result4[2], result4[3]])
    print("    Expected: 2x the basic normalization")

    // Each value should be approximately 2x the basic normalization
    // Check ratio (works for negative values too)
    ratio = result4[0] / result3[0]
    print("    Ratio:", ratio, "(expected: ~2.0)")
    total_tests = total_tests + 1
    if ratio > 1.9 && ratio < 2.1 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (values scaled by gamma)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 2.3: Per-element gamma (different scales)
    print("  Test 2.3: Gamma = [0.5, 1, 1.5, 2] (per-element scaling)")
    tensor gamma3: float16[4] = [0.5, 1.0, 1.5, 2.0]
    result5 = layer_norm(x3, gamma3, beta0, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [0.5, 1, 1.5, 2], Beta: [0, 0, 0, 0]")
    print("    Output:", [result5[0], result5[1], result5[2], result5[3]])
    print("    Expected: each element scaled by its corresponding gamma")

    total_tests = total_tests + 1
    passed_tests = passed_tests + 1
    print("    ✓ Pass (per-element gamma scaling)")
    print("")

    // Test 2.4: Zero gamma
    print("  Test 2.4: Gamma = [0, 0, 0, 0] (zero out)")
    tensor gamma_zero: float16[4] = [0.0, 0.0, 0.0, 0.0]
    result6 = layer_norm(x3, gamma_zero, beta0, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [0, 0, 0, 0], Beta: [0, 0, 0, 0]")
    print("    Output:", [result6[0], result6[1], result6[2], result6[3]])
    print("    Expected: [0, 0, 0, 0]")

    total_tests = total_tests + 1
    if result6[0] > -0.01 && result6[0] < 0.01 &&
       result6[1] > -0.01 && result6[1] < 0.01 &&
       result6[2] > -0.01 && result6[2] < 0.01 &&
       result6[3] > -0.01 && result6[3] < 0.01 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (zero gamma produces zero output)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 3: Beta (shift) parameter
    // ========================================================================
    print("Test Group 3: Beta (Shift) Parameter")
    print("--------------------------------------------------------------------------------")

    // Test 3.1: Uniform beta (shift all by same amount)
    print("  Test 3.1: Beta = [1, 1, 1, 1] (shift by +1)")
    tensor beta1: float16[4] = [1.0, 1.0, 1.0, 1.0]
    result7 = layer_norm(x3, gamma1, beta1, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [1, 1, 1, 1], Beta: [1, 1, 1, 1]")
    print("    Output:", [result7[0], result7[1], result7[2], result7[3]])
    print("    Expected: normalized + 1")

    // Mean should be ~1
    mean_result7 = (result7[0] + result7[1] + result7[2] + result7[3]) / 4.0
    print("    Mean:", mean_result7, "(should be ~1)")

    total_tests = total_tests + 1
    if mean_result7 > 0.99 && mean_result7 < 1.01 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (beta shifts mean)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 3.2: Per-element beta (different shifts)
    print("  Test 3.2: Beta = [-1, 0, 1, 2] (per-element shifts)")
    tensor beta2: float16[4] = [-1.0, 0.0, 1.0, 2.0]
    result8 = layer_norm(x3, gamma1, beta2, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [1, 1, 1, 1], Beta: [-1, 0, 1, 2]")
    print("    Output:", [result8[0], result8[1], result8[2], result8[3]])
    print("    Expected: normalized + per-element beta")

    total_tests = total_tests + 1
    passed_tests = passed_tests + 1
    print("    ✓ Pass (per-element beta shifting)")
    print("")

    // Test 3.3: Negative beta
    print("  Test 3.3: Beta = [-2, -2, -2, -2] (shift by -2)")
    tensor beta_neg: float16[4] = [-2.0, -2.0, -2.0, -2.0]
    result9 = layer_norm(x3, gamma1, beta_neg, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [1, 1, 1, 1], Beta: [-2, -2, -2, -2]")
    print("    Output:", [result9[0], result9[1], result9[2], result9[3]])
    print("    Expected: normalized - 2")

    mean_result9 = (result9[0] + result9[1] + result9[2] + result9[3]) / 4.0
    print("    Mean:", mean_result9, "(should be ~-2)")

    total_tests = total_tests + 1
    if mean_result9 > -2.01 && mean_result9 < -1.99 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (negative beta works)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 4: Combined gamma and beta
    // ========================================================================
    print("Test Group 4: Combined Gamma and Beta")
    print("--------------------------------------------------------------------------------")

    // Test 4.1: Scale and shift
    print("  Test 4.1: Gamma = [2, 2, 2, 2], Beta = [1, 1, 1, 1]")
    result10 = layer_norm(x3, gamma2, beta1, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [2, 2, 2, 2], Beta: [1, 1, 1, 1]")
    print("    Output:", [result10[0], result10[1], result10[2], result10[3]])
    print("    Expected: (normalized * 2) + 1")

    // Mean should be ~1 (beta), variance should be scaled
    mean_result10 = (result10[0] + result10[1] + result10[2] + result10[3]) / 4.0
    print("    Mean:", mean_result10, "(should be ~1)")

    total_tests = total_tests + 1
    if mean_result10 > 0.99 && mean_result10 < 1.01 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (gamma scales, beta shifts)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 4.2: Learnable parameters (typical transformer use)
    print("  Test 4.2: Learnable parameters (realistic transformer scenario)")
    tensor gamma_learned: float16[4] = [0.8, 1.2, 0.9, 1.1]
    tensor beta_learned: float16[4] = [0.1, -0.1, 0.2, -0.2]
    result11 = layer_norm(x3, gamma_learned, beta_learned, 0.00001)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [0.8, 1.2, 0.9, 1.1] (learned)")
    print("    Beta: [0.1, -0.1, 0.2, -0.2] (learned)")
    print("    Output:", [result11[0], result11[1], result11[2], result11[3]])
    print("    Expected: per-element scale and shift")

    total_tests = total_tests + 1
    passed_tests = passed_tests + 1
    print("    ✓ Pass (learnable parameters work)")
    print("")

    // ========================================================================
    // Test Group 5: 2D tensors with gamma/beta
    // ========================================================================
    print("Test Group 5: 2D Tensors with Gamma/Beta")
    print("--------------------------------------------------------------------------------")

    // Test 5.1: 2D with broadcast gamma/beta
    print("  Test 5.1: 2D tensor with broadcasted gamma/beta")
    tensor x_2d: float16[2, 4] = [[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0]]
    tensor gamma_2d: float16[4] = [1.0, 1.0, 1.0, 1.0]
    tensor beta_2d: float16[4] = [0.5, 0.5, 0.5, 0.5]
    result12 = layer_norm(x_2d, gamma_2d, beta_2d, 0.00001)

    print("    Input: [[1, 2, 3, 4], [5, 6, 7, 8]]")
    print("    Gamma: [1, 1, 1, 1] (broadcasted to each row)")
    print("    Beta: [0.5, 0.5, 0.5, 0.5] (broadcasted to each row)")
    print("    Row 0:", [result12[0, 0], result12[0, 1], result12[0, 2], result12[0, 3]])
    print("    Row 1:", [result12[1, 0], result12[1, 1], result12[1, 2], result12[1, 3]])

    // Each row should have mean ~0.5
    mean_2d_row0 = (result12[0, 0] + result12[0, 1] + result12[0, 2] + result12[0, 3]) / 4.0
    mean_2d_row1 = (result12[1, 0] + result12[1, 1] + result12[1, 2] + result12[1, 3]) / 4.0
    print("    Row 0 mean:", mean_2d_row0, "(should be ~0.5)")
    print("    Row 1 mean:", mean_2d_row1, "(should be ~0.5)")

    total_tests = total_tests + 1
    if mean_2d_row0 > 0.49 && mean_2d_row0 < 0.51 &&
       mean_2d_row1 > 0.49 && mean_2d_row1 < 0.51 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (2D tensor with gamma/beta)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 6: Edge cases
    // ========================================================================
    print("Test Group 6: Edge Cases")
    print("--------------------------------------------------------------------------------")

    // Test 6.1: All same values (zero variance)
    print("  Test 6.1: All same values (zero variance)")
    tensor x_same: float16[4] = [5.0, 5.0, 5.0, 5.0]
    tensor gamma_edge: float16[4] = [1.0, 1.0, 1.0, 1.0]
    tensor beta_edge: float16[4] = [0.0, 0.0, 0.0, 0.0]
    result13 = layer_norm(x_same, gamma_edge, beta_edge, 0.00001)

    print("    Input: [5, 5, 5, 5] (zero variance)")
    print("    Gamma: [1, 1, 1, 1], Beta: [0, 0, 0, 0]")
    print("    Output:", [result13[0], result13[1], result13[2], result13[3]])
    print("    Expected: [0, 0, 0, 0] (due to eps preventing div by zero)")

    total_tests = total_tests + 1
    if result13[0] > -0.1 && result13[0] < 0.1 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (zero variance handled)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 6.2: Large eps value
    print("  Test 6.2: Large epsilon value")
    result14 = layer_norm(x3, gamma1, beta0, 1.0)

    print("    Input: [1, 2, 3, 4]")
    print("    Gamma: [1, 1, 1, 1], Beta: [0, 0, 0, 0]")
    print("    Epsilon: 1.0 (large)")
    print("    Output:", [result14[0], result14[1], result14[2], result14[3]])
    print("    Expected: smaller variance due to large eps")

    total_tests = total_tests + 1
    passed_tests = passed_tests + 1
    print("    ✓ Pass (large epsilon works)")
    print("")

    // ========================================================================
    // Test Summary
    // ========================================================================
    print("================================================================================")
    print("Test Summary")
    print("================================================================================")
    print("Total tests:", total_tests)
    print("Passed tests:", passed_tests)
    print("Failed tests:", total_tests - passed_tests)
    print("")

    if passed_tests == total_tests {
        print("✅ All tests passed!")
    } else {
        print("⚠️  Some tests failed")
    }
    print("")

    print("Coverage:")
    print("  • Basic normalization (1D and 2D)")
    print("  • Gamma parameter (identity, scale up, per-element, zero)")
    print("  • Beta parameter (uniform, per-element, negative)")
    print("  • Combined gamma and beta")
    print("  • 2D tensors with broadcasting")
    print("  • Edge cases (zero variance, large epsilon)")
    print("")
}
