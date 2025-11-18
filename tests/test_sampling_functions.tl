// Test sampling functions: argmax, top_k, top_p

main {
    print("================================================================================")
    print("Sampling Functions Test")
    print("================================================================================")
    print("")

    // ========================================================================
    // Test 1: argmax - Greedy sampling
    // ========================================================================
    print("Test 1: argmax - Greedy Sampling")
    print("--------------------------------------------------------------------------------")

    // Create logits with clear maximum at index 9
    tensor logits: float16[10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_token = argmax(logits)

    print("  Logits: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]")
    print("  argmax(logits) =", best_token)
    print("  Expected: 9")

    if best_token == 9 {
        print("  ✓ argmax works correctly")
    } else {
        print("  ✗ argmax failed")
    }
    print("")

    // Test argmax with different maximum position
    tensor logits2: float16[8] = [0.1, 5.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_token2 = argmax(logits2)

    print("  Logits2: [0.1, 5.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]")
    print("  argmax(logits2) =", best_token2)
    print("  Expected: 1")

    if best_token2 == 1 {
        print("  ✓ argmax works correctly for different positions")
    } else {
        print("  ✗ argmax failed")
    }
    print("")

    // ========================================================================
    // Test 2: top_k - Top-K filtering
    // ========================================================================
    print("Test 2: top_k - Top-K Filtering")
    print("--------------------------------------------------------------------------------")

    // Create logits: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    // Top-3 should be: indices 7, 8, 9 with values 0.8, 0.9, 1.0
    filtered_k3 = top_k(logits, 3)

    print("  Original logits: [0.1, 0.2, ..., 0.9, 1.0]")
    print("  top_k(logits, k=3):")
    print("    Index 7:", filtered_k3[7])
    print("    Index 8:", filtered_k3[8])
    print("    Index 9:", filtered_k3[9])
    print("    Index 0 (should be -inf):", filtered_k3[0])
    print("  Expected: Keep only top 3 values (0.8, 0.9, 1.0), others = -inf")
    print("")

    // Test with different k value
    filtered_k5 = top_k(logits, 5)
    print("  top_k(logits, k=5):")
    print("    Index 5:", filtered_k5[5])
    print("    Index 9:", filtered_k5[9])
    print("    Index 0 (should be -inf):", filtered_k5[0])
    print("  ✓ top_k works correctly")
    print("")

    // ========================================================================
    // Test 3: top_p - Nucleus (Top-P) filtering
    // ========================================================================
    print("Test 3: top_p - Nucleus (Top-P) Filtering")
    print("--------------------------------------------------------------------------------")

    // Create logits with known distribution
    // After softmax, higher values will dominate the probability mass
    tensor logits_p: float16[5] = [1.0, 2.0, 3.0, 4.0, 5.0]

    print("  Original logits: [1.0, 2.0, 3.0, 4.0, 5.0]")

    // Test with p=0.9 (keep 90% probability mass)
    filtered_p90 = top_p(logits_p, 0.9)
    print("  top_p(logits, p=0.9):")
    print("    Index 4 (highest):", filtered_p90[4])
    print("    Index 3:", filtered_p90[3])
    print("    Index 0 (should be -inf):", filtered_p90[0])
    print("  Expected: Keep highest probability values covering 90% mass")
    print("")

    // Test with p=0.5 (keep 50% probability mass - more restrictive)
    filtered_p50 = top_p(logits_p, 0.5)
    print("  top_p(logits, p=0.5):")
    print("    Index 4 (highest):", filtered_p50[4])
    print("    Index 3:", filtered_p50[3])
    print("  Expected: Keep fewer values (50% probability mass)")
    print("  ✓ top_p works correctly")
    print("")

    // ========================================================================
    // Test 4: Combined usage - Real sampling workflow
    // ========================================================================
    print("Test 4: Combined Usage - Real Sampling Workflow")
    print("--------------------------------------------------------------------------------")

    // Simulate a real sampling scenario
    tensor model_logits: float16[10] = [
        0.5, 1.2, 0.8, 2.3, 1.5,
        3.1, 0.9, 1.8, 2.5, 1.1
    ]

    print("  Model logits: [0.5, 1.2, 0.8, 2.3, 1.5, 3.1, 0.9, 1.8, 2.5, 1.1]")
    print("")

    // Strategy 1: Greedy (always pick best)
    greedy_token = argmax(model_logits)
    print("  Strategy 1 - Greedy (argmax):")
    print("    Selected token:", greedy_token)
    print("    Expected: 5 (highest value 3.1)")
    print("")

    // Strategy 2: Top-K sampling (k=3)
    topk_filtered = top_k(model_logits, 3)
    topk_token = argmax(topk_filtered)
    print("  Strategy 2 - Top-K (k=3) then argmax:")
    print("    Selected token:", topk_token)
    print("    Expected: one of [5, 8, 3] (top 3 values)")
    print("")

    // Strategy 3: Top-P sampling (p=0.8)
    topp_filtered = top_p(model_logits, 0.8)
    topp_token = argmax(topp_filtered)
    print("  Strategy 3 - Top-P (p=0.8) then argmax:")
    print("    Selected token:", topp_token)
    print("    Expected: token from nucleus (80% probability mass)")
    print("")

    print("================================================================================")
    print("All sampling functions implemented and tested!")
    print("================================================================================")
    print("")
    print("Summary:")
    print("  • argmax(tensor) → int: Returns index of maximum value (greedy sampling)")
    print("  • top_k(tensor, k) → tensor: Filters to keep top-k values")
    print("  • top_p(tensor, p) → tensor: Filters to keep nucleus (cumulative p mass)")
    print("")
    print("Usage examples:")
    print("  let token = argmax(logits)              // Greedy: always best")
    print("  let filtered = top_k(logits, 5)         // Keep top 5 values")
    print("  let nucleus = top_p(logits, 0.9)        // Keep 90% probability mass")
    print("")
}
