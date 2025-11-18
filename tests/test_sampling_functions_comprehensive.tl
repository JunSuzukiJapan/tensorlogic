// Comprehensive test for sampling functions: argmax, top_k, top_p
// Tests edge cases and boundary conditions

main {
    print("================================================================================")
    print("Comprehensive Sampling Functions Test")
    print("================================================================================")
    print("")

    total_tests = 0
    passed_tests = 0

    // ========================================================================
    // Test Group 1: argmax - Edge Cases
    // ========================================================================
    print("Test Group 1: argmax - Edge Cases")
    print("--------------------------------------------------------------------------------")

    // Test 1.1: Single element tensor
    print("  Test 1.1: Single element")
    tensor single: float16[1] = [5.0]
    result = argmax(single)
    print("    argmax([5.0]) =", result, "(expected: 0)")
    total_tests = total_tests + 1
    if result == 0 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 1.2: All same values (tie - should return first)
    print("  Test 1.2: All same values (tie)")
    tensor ties: float16[5] = [3.0, 3.0, 3.0, 3.0, 3.0]
    result = argmax(ties)
    print("    argmax([3.0, 3.0, 3.0, 3.0, 3.0]) =", result, "(expected: 0 or any)")
    total_tests = total_tests + 1
    if result >= 0 && result < 5 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (returns valid index)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 1.3: Negative values
    print("  Test 1.3: Negative values")
    tensor negatives: float16[5] = [-5.0, -2.0, -10.0, -1.0, -3.0]
    result = argmax(negatives)
    print("    argmax([-5, -2, -10, -1, -3]) =", result, "(expected: 3)")
    total_tests = total_tests + 1
    if result == 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 1.4: Mixed positive and negative
    print("  Test 1.4: Mixed positive and negative")
    tensor mixed: float16[6] = [-1.0, 0.5, -0.5, 2.0, 1.0, -2.0]
    result = argmax(mixed)
    print("    argmax([-1, 0.5, -0.5, 2, 1, -2]) =", result, "(expected: 3)")
    total_tests = total_tests + 1
    if result == 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 1.5: Maximum at beginning
    print("  Test 1.5: Maximum at beginning")
    tensor first_max: float16[5] = [10.0, 1.0, 2.0, 3.0, 4.0]
    result = argmax(first_max)
    print("    argmax([10, 1, 2, 3, 4]) =", result, "(expected: 0)")
    total_tests = total_tests + 1
    if result == 0 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 2: top_k - Edge Cases
    // ========================================================================
    print("Test Group 2: top_k - Edge Cases")
    print("--------------------------------------------------------------------------------")

    tensor base_logits: float16[10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    // Test 2.1: k=1 (keep only top value)
    print("  Test 2.1: k=1 (keep only best)")
    filtered = top_k(base_logits, 1)
    result = argmax(filtered)
    print("    top_k(logits, k=1) then argmax =", result, "(expected: 9)")
    print("    Index 9:", filtered[9], "(should be 1.0)")
    print("    Index 8:", filtered[8], "(should be -inf)")
    total_tests = total_tests + 1
    if result == 9 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 2.2: k=vocab_size (keep all)
    print("  Test 2.2: k=vocab_size (keep all)")
    filtered = top_k(base_logits, 10)
    print("    top_k(logits, k=10)")
    print("    Index 0:", filtered[0], "(should be 0.1, not -inf)")
    print("    Index 9:", filtered[9], "(should be 1.0)")
    total_tests = total_tests + 1
    // Check if values are preserved (not -inf)
    if filtered[0] > -1.0 && filtered[9] > 0.9 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (all values kept)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 2.3: k > vocab_size (should keep all)
    print("  Test 2.3: k > vocab_size")
    filtered = top_k(base_logits, 20)
    print("    top_k(logits, k=20) where vocab_size=10")
    print("    Index 0:", filtered[0], "(should be kept)")
    print("    Index 9:", filtered[9], "(should be kept)")
    total_tests = total_tests + 1
    if filtered[0] > -1.0 && filtered[9] > 0.9 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (all values kept)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 2.4: k with negative values
    print("  Test 2.4: k with negative logits")
    tensor neg_logits: float16[5] = [-5.0, -2.0, -10.0, -1.0, -3.0]
    filtered = top_k(neg_logits, 2)
    result = argmax(filtered)
    print("    top_k([-5, -2, -10, -1, -3], k=2)")
    print("    argmax =", result, "(expected: 3, value=-1)")
    print("    Index 3:", filtered[3], "(should be -1)")
    print("    Index 0:", filtered[0], "(should be -inf)")
    total_tests = total_tests + 1
    if result == 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 3: top_p - Edge Cases
    // ========================================================================
    print("Test Group 3: top_p - Edge Cases")
    print("--------------------------------------------------------------------------------")

    // Test 3.1: p=1.0 (keep all)
    print("  Test 3.1: p=1.0 (keep all)")
    tensor p_logits: float16[5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    filtered = top_p(p_logits, 1.0)
    print("    top_p(logits, p=1.0)")
    print("    Index 0:", filtered[0], "(should be kept)")
    print("    Index 4:", filtered[4], "(should be kept)")
    total_tests = total_tests + 1
    if filtered[0] > 0.0 && filtered[4] > 4.0 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (all values kept)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 3.2: Very skewed distribution (one dominant value)
    print("  Test 3.2: Very skewed distribution")
    tensor skewed: float16[5] = [0.1, 0.2, 0.3, 0.4, 10.0]
    filtered = top_p(skewed, 0.95)
    result = argmax(filtered)
    print("    top_p([0.1, 0.2, 0.3, 0.4, 10.0], p=0.95)")
    print("    argmax =", result, "(expected: 4)")
    print("    Index 4:", filtered[4], "(should be 10.0)")
    total_tests = total_tests + 1
    if result == 4 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 3.3: Uniform distribution (all equal)
    print("  Test 3.3: Uniform distribution")
    tensor uniform: float16[4] = [1.0, 1.0, 1.0, 1.0]
    filtered = top_p(uniform, 0.5)
    print("    top_p([1, 1, 1, 1], p=0.5)")
    print("    After softmax, each has 25% probability")
    print("    p=0.5 should keep 2-3 values")
    // Count how many are not -inf
    kept_count = 0
    if filtered[0] > 0.0 {
        kept_count = kept_count + 1
    }
    if filtered[1] > 0.0 {
        kept_count = kept_count + 1
    }
    if filtered[2] > 0.0 {
        kept_count = kept_count + 1
    }
    if filtered[3] > 0.0 {
        kept_count = kept_count + 1
    }
    print("    Kept values:", kept_count, "(expected: 2-3)")
    total_tests = total_tests + 1
    if kept_count >= 2 && kept_count <= 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 3.4: p with negative logits
    print("  Test 3.4: p with negative logits")
    tensor neg_p_logits: float16[5] = [-5.0, -2.0, -10.0, -1.0, -3.0]
    filtered = top_p(neg_p_logits, 0.8)
    result = argmax(filtered)
    print("    top_p([-5, -2, -10, -1, -3], p=0.8)")
    print("    argmax =", result, "(expected: 3, highest is -1)")
    total_tests = total_tests + 1
    if result == 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // ========================================================================
    // Test Group 4: Integration - Real Sampling Scenarios
    // ========================================================================
    print("Test Group 4: Integration - Real Sampling Scenarios")
    print("--------------------------------------------------------------------------------")

    // Test 4.1: Greedy vs Top-K vs Top-P comparison
    print("  Test 4.1: Strategy comparison")
    tensor real_logits: float16[8] = [
        0.5, 1.2, 0.8, 2.3, 1.5, 3.1, 0.9, 1.8
    ]

    greedy = argmax(real_logits)

    topk_filtered = top_k(real_logits, 3)
    topk_result = argmax(topk_filtered)

    topp_filtered = top_p(real_logits, 0.7)
    topp_result = argmax(topp_filtered)

    print("    Greedy (argmax):", greedy)
    print("    Top-K (k=3) then argmax:", topk_result)
    print("    Top-P (p=0.7) then argmax:", topp_result)
    print("    All should select from high-value indices")

    total_tests = total_tests + 1
    // All should select reasonable tokens (indices 3, 5, or 7 are highest)
    if greedy == 5 && topk_result >= 3 && topp_result >= 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (all strategies select reasonable tokens)")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 4.2: Temperature=0 equivalent to greedy
    print("  Test 4.2: Greedy should match highest logit position")
    tensor verify_logits: float16[6] = [0.1, 0.3, 0.2, 0.8, 0.5, 0.4]
    greedy = argmax(verify_logits)
    print("    Logits: [0.1, 0.3, 0.2, 0.8, 0.5, 0.4]")
    print("    argmax =", greedy, "(expected: 3)")
    total_tests = total_tests + 1
    if greedy == 3 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass")
    } else {
        print("    ✗ Fail")
    }
    print("")

    // Test 4.3: Top-K then Top-P (nested filtering)
    print("  Test 4.3: Nested filtering (Top-K then Top-P)")
    tensor nest_logits: float16[10] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]
    // First apply top-k to reduce to top 5
    step1 = top_k(nest_logits, 5)
    // Then apply top-p to further filter
    step2 = top_p(step1, 0.8)
    final_result = argmax(step2)

    print("    Original: [0.1, ..., 1.0]")
    print("    After top_k(k=5): keeps indices 5-9")
    print("    After top_p(p=0.8): further filters")
    print("    Final argmax =", final_result, "(should be 9)")
    total_tests = total_tests + 1
    if final_result == 9 {
        passed_tests = passed_tests + 1
        print("    ✓ Pass (nested filtering works)")
    } else {
        print("    ✗ Fail")
    }
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
    print("  • argmax: single element, ties, negatives, mixed, position variations")
    print("  • top_k: k=1, k=vocab_size, k>vocab_size, negative logits")
    print("  • top_p: p=1.0, skewed distribution, uniform, negative logits")
    print("  • Integration: strategy comparison, greedy verification, nested filtering")
    print("")
}
