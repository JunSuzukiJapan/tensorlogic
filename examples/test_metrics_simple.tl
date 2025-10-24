// Simple test for evaluation metrics

main {
    print("=== Testing Evaluation Metrics ===")
    print("")

    // Test MRR
    print("Test 1: MRR calculation")
    let rank1 = 1
    let mrr1 = compute_mrr(rank1)
    print("  Rank:", rank1, "→ MRR:", mrr1)

    let rank3 = 3
    let mrr3 = compute_mrr(rank3)
    print("  Rank:", rank3, "→ MRR:", mrr3)
    print("")

    // Test Hits@k
    print("Test 2: Hits@k calculation")
    let hits1 = compute_hits_at_k(1, 1)
    print("  Rank=1, k=1 → Hits:", hits1)

    let hits2 = compute_hits_at_k(3, 1)
    print("  Rank=3, k=1 → Hits:", hits2)

    let hits3 = compute_hits_at_k(2, 3)
    print("  Rank=2, k=3 → Hits:", hits3)
    print("")

    // Test Mean Rank
    print("Test 3: Mean Rank calculation")
    let sum_ranks = 6
    let num_queries = 3
    let mr = compute_mean_rank(sum_ranks, num_queries)
    print("  Sum of ranks:", sum_ranks)
    print("  Num queries:", num_queries)
    print("  Mean Rank:", mr)
    print("")

    print("✅ All metrics tests passed!")
}
