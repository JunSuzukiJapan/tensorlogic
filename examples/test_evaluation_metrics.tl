// Knowledge Graph Evaluation Metrics Demo
// Phase 13.2 & 13.4: Ranking and Evaluation

main {
    print("======================================================================")
    print("   EVALUATION METRICS FOR KNOWLEDGE GRAPH EMBEDDINGS")
    print("======================================================================")
    print("")

    print("📊 Evaluation Metrics:")
    print("  - Mean Reciprocal Rank (MRR)")
    print("  - Hits@k (Hits@1, Hits@3, Hits@10)")
    print("  - Mean Rank (MR)")
    print("")

    // ========================================================================
    // Part 1: Understanding Ranks
    // ========================================================================
    print("======================================================================")
    print("PART 1: UNDERSTANDING RANKS")
    print("======================================================================")
    print("")

    print("Scenario: Predicting tail for (alice, lives_in, ?)")
    print("")
    print("Candidates (sorted by score, descending):")
    print("  Rank 1: tokyo     (score: 0.95)")
    print("  Rank 2: osaka     (score: 0.82)")
    print("  Rank 3: kyoto     (score: 0.71)")
    print("  Rank 4: nagoya    (score: 0.45)")
    print("  Rank 5: sapporo   (score: 0.23)")
    print("")
    print("Ground truth: tokyo")
    print("Tokyo's rank: 1 (best possible!)")
    print("")

    // Compute rank: rank = number of higher-scored candidates + 1
    let num_higher_than_tokyo = 0  // No candidates scored higher than tokyo
    let tokyo_rank = num_higher_than_tokyo + 1
    print("Computed rank for tokyo:", tokyo_rank)
    print("  Formula: rank = num_higher + 1 = 0 + 1 = 1")
    print("")

    // ========================================================================
    // Part 2: Mean Reciprocal Rank (MRR)
    // ========================================================================
    print("======================================================================")
    print("PART 2: MEAN RECIPROCAL RANK (MRR)")
    print("======================================================================")
    print("")

    print("Formula: MRR = (1/N) × Σ(1/rank_i)")
    print("  where N = number of queries")
    print("  rank_i = rank of correct answer for query i")
    print("")

    print("Example with 3 queries:")
    print("")

    // Query 1
    print("Query 1: (alice, lives_in, ?)")
    print("  Ground truth: tokyo")
    print("  Rank: 1")
    let mrr_1 = compute_mrr(1)
    print("  Reciprocal Rank: 1/1 =", mrr_1)
    print("")

    // Query 2
    print("Query 2: (bob, works_at, ?)")
    print("  Ground truth: company_b")
    print("  Rank: 3")
    let mrr_2 = compute_mrr(3)
    print("  Reciprocal Rank: 1/3 =", mrr_2)
    print("")

    // Query 3
    print("Query 3: (charlie, born_in, ?)")
    print("  Ground truth: kyoto")
    print("  Rank: 2")
    let mrr_3 = compute_mrr(2)
    print("  Reciprocal Rank: 1/2 =", mrr_3)
    print("")

    // Compute average MRR
    let total_rr = mrr_1 + mrr_2 + mrr_3
    let avg_mrr = total_rr / 3.0
    print("Mean Reciprocal Rank (MRR):")
    print("  = (1.0 + 0.333 + 0.5) / 3")
    print("  =", avg_mrr)
    print("")

    print("Interpretation:")
    print("  MRR = 1.0  → Perfect predictions (all rank 1)")
    print("  MRR = 0.5  → Good predictions (avg rank ~2)")
    print("  MRR = 0.1  → Poor predictions (avg rank ~10)")
    print("")

    // ========================================================================
    // Part 3: Hits@k Metric
    // ========================================================================
    print("======================================================================")
    print("PART 3: HITS@K METRIC")
    print("======================================================================")
    print("")

    print("Formula: Hits@k = (1/N) × Σ(I(rank_i ≤ k))")
    print("  where I(condition) = 1 if true, 0 if false")
    print("")

    print("Example: Same 3 queries from above")
    print("")

    // Hits@1
    print("--- Hits@1 ---")
    let hits1_q1 = compute_hits_at_k(1, 1)  // rank=1, k=1 → 1
    let hits1_q2 = compute_hits_at_k(3, 1)  // rank=3, k=1 → 0
    let hits1_q3 = compute_hits_at_k(2, 1)  // rank=2, k=1 → 0
    print("  Query 1 (rank=1):", hits1_q1)
    print("  Query 2 (rank=3):", hits1_q2)
    print("  Query 3 (rank=2):", hits1_q3)
    
    let temp1 = hits1_q1 + hits1_q2
    let hits1_total = temp1 + hits1_q3
    print("  Total hits:", hits1_total)
    print("  Hits@1 = 1/3 = 0.33")
    print("")

    // Hits@3
    print("--- Hits@3 ---")
    let hits3_q1 = compute_hits_at_k(1, 3)  // rank=1, k=3 → 1
    let hits3_q2 = compute_hits_at_k(3, 3)  // rank=3, k=3 → 1
    let hits3_q3 = compute_hits_at_k(2, 3)  // rank=2, k=3 → 1
    print("  Query 1 (rank=1):", hits3_q1)
    print("  Query 2 (rank=3):", hits3_q2)
    print("  Query 3 (rank=2):", hits3_q3)
    
    let temp3a = hits3_q1 + hits3_q2
    let hits3_total = temp3a + hits3_q3
    print("  Total hits:", hits3_total)
    print("  Hits@3 = 3/3 = 1.00")
    print("")

    // Hits@10
    print("--- Hits@10 ---")
    let hits10_q1 = compute_hits_at_k(1, 10)
    let hits10_q2 = compute_hits_at_k(3, 10)
    let hits10_q3 = compute_hits_at_k(2, 10)
    print("  All queries have rank ≤ 10")
    
    let temp10a = hits10_q1 + hits10_q2
    let hits10_total = temp10a + hits10_q3
    print("  Total hits:", hits10_total)
    print("  Hits@10 = 3/3 = 1.00")
    print("")

    print("Interpretation:")
    print("  Hits@1 = 0.33  → 33% of answers are rank 1")
    print("  Hits@3 = 1.00  → 100% of answers in top-3")
    print("  Hits@10 = 1.00 → 100% of answers in top-10")
    print("")

    // ========================================================================
    // Part 4: Mean Rank (MR)
    // ========================================================================
    print("======================================================================")
    print("PART 4: MEAN RANK (MR)")
    print("======================================================================")
    print("")

    print("Formula: MR = (1/N) × Σ(rank_i)")
    print("  Average rank of correct answers")
    print("")

    let sum_of_ranks = 1 + 3 + 2  // ranks from our 3 queries
    let num_queries = 3
    let mean_rank = compute_mean_rank(sum_of_ranks, num_queries)

    print("Ranks: 1, 3, 2")
    print("Mean Rank = (1 + 3 + 2) / 3 =", mean_rank)
    print("")

    print("Interpretation:")
    print("  MR = 1.0   → Perfect (all rank 1)")
    print("  MR = 5.0   → Good (answers in top-5 on average)")
    print("  MR = 50.0  → Poor (answers ranked low)")
    print("")

    print("Note: MR is sensitive to outliers")
    print("      MRR is more robust for evaluation")
    print("")

    // ========================================================================
    // Part 5: Filtered vs Raw Evaluation
    // ========================================================================
    print("======================================================================")
    print("PART 5: FILTERED VS RAW EVALUATION")
    print("======================================================================")
    print("")

    print("RAW EVALUATION:")
    print("  - Rank among ALL candidate entities")
    print("  - May penalize known true facts")
    print("")

    print("Example: Query (alice, lives_in, ?)")
    print("  Known facts: alice lives_in tokyo")
    print("               alice lives_in osaka (also true)")
    print("")
    print("  Raw ranking:")
    print("    Rank 1: tokyo  ✓ (correct)")
    print("    Rank 2: osaka  ✓ (also correct, but treated as wrong!)")
    print("    Rank 3: kyoto")
    print("")

    print("FILTERED EVALUATION:")
    print("  - Remove all known true facts except the target")
    print("  - Only rank against genuinely negative candidates")
    print("")
    print("  Filtered ranking (remove osaka from candidates):")
    print("    Rank 1: tokyo  ✓ (correct)")
    print("    Rank 2: kyoto")
    print("    Rank 3: nagoya")
    print("")

    print("Best Practice: Use FILTERED evaluation")
    print("  More accurate measure of model's predictive power")
    print("")

    // ========================================================================
    // Part 6: Complete Evaluation Workflow
    // ========================================================================
    print("======================================================================")
    print("PART 6: COMPLETE EVALUATION WORKFLOW")
    print("======================================================================")
    print("")

    print("Steps for Evaluating Link Prediction:")
    print("")

    print("1. PREPARE TEST SET")
    print("   - Hold out some triples for testing")
    print("   - Ensure no overlap with training data")
    print("")

    print("2. FOR EACH TEST TRIPLE (h, r, t):")
    print("")

    print("   a) Generate candidates")
    print("      - All entities (for tail prediction)")
    print("      - Or all entities (for head prediction)")
    print("")

    print("   b) Score each candidate")
    print("      - score_i = model(h, r, candidate_i)")
    print("")

    print("   c) Rank candidates by score")
    print("      - Sort in descending order")
    print("      - Find rank of correct answer t")
    print("")

    print("   d) (Optional) Filter known facts")
    print("      - Remove other true triples from ranking")
    print("      - Re-compute rank")
    print("")

    print("   e) Record rank")
    print("      - Store for metric computation")
    print("")

    print("3. COMPUTE METRICS")
    print("   - MRR = average of (1/rank)")
    print("   - Hits@k = percentage with rank ≤ k")
    print("   - MR = average rank")
    print("")

    print("4. REPORT RESULTS")
    print("   - Typically report:")
    print("     * MRR")
    print("     * Hits@1, Hits@3, Hits@10")
    print("     * MR (optional)")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY")
    print("======================================================================")
    print("")

    print("✅ Implemented Metrics:")
    print("  ✓ compute_rank(target_score, num_higher)")
    print("  ✓ compute_mrr(rank)")
    print("  ✓ compute_hits_at_k(rank, k)")
    print("  ✓ compute_mean_rank(sum_ranks, num_queries)")
    print("")

    print("📊 Key Metrics:")
    print("  - MRR: Emphasizes top-ranked predictions")
    print("  - Hits@k: Percentage of correct answers in top-k")
    print("  - MR: Average rank (sensitive to outliers)")
    print("")

    print("🎯 Best Practices:")
    print("  - Use filtered evaluation")
    print("  - Report MRR and Hits@1/3/10")
    print("  - Evaluate on separate test set")
    print("")

    print("✅ Evaluation metrics demo completed!")
}
