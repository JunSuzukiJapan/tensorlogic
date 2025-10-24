// Complete Relation Prediction Demo
// Phase 13: 関係予測機能の総合デモンストレーション

main {
    print("======================================================================")
    print("   RELATION PREDICTION - COMPLETE DEMONSTRATION")
    print("======================================================================")
    print("")

    print("📚 Phase 13 Features:")
    print("  ✓ Link prediction (head/tail)")
    print("  ✓ Candidate ranking")
    print("  ✓ Evaluation metrics (MRR, Hits@k, MR)")
    print("  ✓ Complete workflow")
    print("")

    // ========================================================================
    // Scenario Setup
    // ========================================================================
    print("======================================================================")
    print("SCENARIO: KNOWLEDGE GRAPH COMPLETION")
    print("======================================================================")
    print("")

    print("Knowledge Graph:")
    print("  Known facts:")
    print("    - (alice, lives_in, tokyo)")
    print("    - (bob, lives_in, osaka)")
    print("    - (charlie, works_at, company_a)")
    print("")

    print("Queries to answer:")
    print("    1. Where does david live? → (david, lives_in, ?)")
    print("    2. Who lives in kyoto? → (?, lives_in, kyoto)")
    print("    3. Where does alice work? → (alice, works_at, ?)")
    print("")

    // Create embeddings
    let alice = zeros([8])
    let bob = zeros([8])
    let charlie = zeros([8])
    let david = zeros([8])

    let tokyo = zeros([8])
    let osaka = zeros([8])
    let kyoto = zeros([8])
    let company_a = zeros([8])
    let company_b = zeros([8])

    let lives_in = zeros([8])
    let works_at = zeros([8])

    print("Initialized 8-dimensional embeddings for:")
    print("  People: alice, bob, charlie, david")
    print("  Places: tokyo, osaka, kyoto, company_a, company_b")
    print("  Relations: lives_in, works_at")
    print("")

    // ========================================================================
    // Query 1: Tail Prediction
    // ========================================================================
    print("======================================================================")
    print("QUERY 1: TAIL PREDICTION")
    print("======================================================================")
    print("")

    print("Question: Where does david live?")
    print("Query: (david, lives_in, ?)")
    print("")

    print("Step 1: Generate candidates")
    print("  Candidates: tokyo, osaka, kyoto")
    print("")

    print("Step 2: Score each candidate using TransE")
    let score_tokyo = predict_tail_transe(david, lives_in, tokyo, "L2")
    let score_osaka = predict_tail_transe(david, lives_in, osaka, "L2")
    let score_kyoto = predict_tail_transe(david, lives_in, kyoto, "L2")

    print("  tokyo:", score_tokyo)
    print("  osaka:", score_osaka)
    print("  kyoto:", score_kyoto)
    print("")

    print("Step 3: Rank candidates (by score, descending)")
    print("  Assume after sorting:")
    print("    Rank 1: tokyo  (highest score)")
    print("    Rank 2: osaka")
    print("    Rank 3: kyoto")
    print("")

    print("Step 4: Return top prediction")
    print("  Answer: david lives_in tokyo")
    print("")

    // ========================================================================
    // Query 2: Head Prediction
    // ========================================================================
    print("======================================================================")
    print("QUERY 2: HEAD PREDICTION")
    print("======================================================================")
    print("")

    print("Question: Who lives in kyoto?")
    print("Query: (?, lives_in, kyoto)")
    print("")

    print("Step 1: Generate candidates")
    print("  Candidates: alice, bob, charlie, david")
    print("")

    print("Step 2: Score each candidate using TransE")
    let score_alice_kyoto = predict_head_transe(alice, lives_in, kyoto, "L2")
    let score_bob_kyoto = predict_head_transe(bob, lives_in, kyoto, "L2")
    let score_charlie_kyoto = predict_head_transe(charlie, lives_in, kyoto, "L2")
    let score_david_kyoto = predict_head_transe(david, lives_in, kyoto, "L2")

    print("  alice:", score_alice_kyoto)
    print("  bob:", score_bob_kyoto)
    print("  charlie:", score_charlie_kyoto)
    print("  david:", score_david_kyoto)
    print("")

    print("Step 3: Rank candidates")
    print("  Assume after sorting:")
    print("    Rank 1: charlie")
    print("    Rank 2: david")
    print("    Rank 3: alice")
    print("    Rank 4: bob")
    print("")

    print("Step 4: Return top prediction")
    print("  Answer: charlie lives_in kyoto")
    print("")

    // ========================================================================
    // Query 3: Cross-relation Prediction
    // ========================================================================
    print("======================================================================")
    print("QUERY 3: CROSS-RELATION PREDICTION")
    print("======================================================================")
    print("")

    print("Question: Where does alice work?")
    print("Query: (alice, works_at, ?)")
    print("")

    print("Step 1: Generate candidates")
    print("  Candidates: company_a, company_b")
    print("")

    print("Step 2: Score candidates using DistMult")
    let score_company_a = predict_tail_distmult(alice, works_at, company_a)
    let score_company_b = predict_tail_distmult(alice, works_at, company_b)

    print("  company_a:", score_company_a)
    print("  company_b:", score_company_b)
    print("")

    print("Step 3: Select top-ranked")
    print("  Answer: alice works_at company_b (hypothetical)")
    print("")

    // ========================================================================
    // Evaluation Example
    // ========================================================================
    print("======================================================================")
    print("EVALUATION ON TEST SET")
    print("======================================================================")
    print("")

    print("Suppose we have ground truth for 5 test queries:")
    print("")

    // Test query 1
    print("Test 1: (person_1, lives_in, ?)")
    print("  Ground truth: city_x")
    print("  Predicted rank: 1")
    let test1_mrr = compute_mrr(1)
    let test1_hits1 = compute_hits_at_k(1, 1)
    let test1_hits10 = compute_hits_at_k(1, 10)
    print("  MRR:", test1_mrr, "| Hits@1:", test1_hits1, "| Hits@10:", test1_hits10)
    print("")

    // Test query 2
    print("Test 2: (person_2, works_at, ?)")
    print("  Ground truth: company_y")
    print("  Predicted rank: 5")
    let test2_mrr = compute_mrr(5)
    let test2_hits1 = compute_hits_at_k(5, 1)
    let test2_hits10 = compute_hits_at_k(5, 10)
    print("  MRR:", test2_mrr, "| Hits@1:", test2_hits1, "| Hits@10:", test2_hits10)
    print("")

    // Test query 3
    print("Test 3: (?, lives_in, city_z)")
    print("  Ground truth: person_3")
    print("  Predicted rank: 2")
    let test3_mrr = compute_mrr(2)
    let test3_hits1 = compute_hits_at_k(2, 1)
    let test3_hits10 = compute_hits_at_k(2, 10)
    print("  MRR:", test3_mrr, "| Hits@1:", test3_hits1, "| Hits@10:", test3_hits10)
    print("")

    // Compute averages
    print("Overall Metrics (3 queries):")
    let sum_mrr = test1_mrr + test2_mrr + test3_mrr
    let avg_mrr = sum_mrr / 3.0
    print("  Average MRR:", avg_mrr)

    let temp_hits1 = test1_hits1 + test2_hits1
    let sum_hits1 = temp_hits1 + test3_hits1
    print("  Total Hits@1:", sum_hits1, "/ 3")

    let temp_hits10 = test1_hits10 + test2_hits10
    let sum_hits10 = temp_hits10 + test3_hits10
    print("  Total Hits@10:", sum_hits10, "/ 3")

    let sum_ranks = 1 + 5 + 2
    let mean_rank = compute_mean_rank(sum_ranks, 3)
    print("  Mean Rank:", mean_rank)
    print("")

    // ========================================================================
    // Applications Summary
    // ========================================================================
    print("======================================================================")
    print("PRACTICAL APPLICATIONS")
    print("======================================================================")
    print("")

    print("1. KNOWLEDGE BASE AUTO-COMPLETION")
    print("   Input: Incomplete database with missing info")
    print("   Output: Predicted missing facts")
    print("   Example: Fill in unknown locations, relationships")
    print("")

    print("2. QUESTION ANSWERING")
    print("   Input: Natural language question")
    print("   Output: Answer from knowledge graph")
    print("   Example: 'Where does X live?' → Query + Predict")
    print("")

    print("3. RECOMMENDATION SYSTEMS")
    print("   Input: User preferences (known relations)")
    print("   Output: Recommended items (predicted relations)")
    print("   Example: User likes A, B → Predict user likes C")
    print("")

    print("4. LINK VALIDATION")
    print("   Input: Triple to verify")
    print("   Output: Plausibility score")
    print("   Example: Is (person, works_at, mars) plausible? NO")
    print("")

    print("5. KNOWLEDGE DISCOVERY")
    print("   Input: Existing knowledge graph")
    print("   Output: Novel relationships to investigate")
    print("   Example: Discover hidden connections between entities")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY: PHASE 13 COMPLETE")
    print("======================================================================")
    print("")

    print("✅ Implemented Features:")
    print("  ✓ Link prediction (head/tail)")
    print("    - predict_tail_transe/distmult")
    print("    - predict_head_transe/distmult")
    print("  ✓ Ranking functionality")
    print("    - Score-based candidate ranking")
    print("  ✓ Evaluation metrics")
    print("    - MRR, Hits@k, Mean Rank")
    print("  ✓ Complete workflows demonstrated")
    print("")

    print("📊 Key Concepts:")
    print("  - Link prediction fills knowledge gaps")
    print("  - Ranking determines most likely candidates")
    print("  - Metrics measure prediction quality")
    print("  - Real-world applications across domains")
    print("")

    print("🎯 Next Steps:")
    print("  - Phase 14: GNN integration")
    print("  - ComplEx algorithm implementation")
    print("  - Production deployment considerations")
    print("")

    print("======================================================================")
    print("   RELATION PREDICTION DEMO COMPLETE")
    print("======================================================================")
}
