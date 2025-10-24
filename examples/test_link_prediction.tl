// Link Prediction Demo - Phase 13.1
// Demonstrates head and tail prediction for knowledge graph triples

main {
    print("======================================================================")
    print("   LINK PREDICTION DEMONSTRATION (Phase 13.1)")
    print("======================================================================")
    print("")

    print("📚 Link Prediction Task:")
    print("  Given partial triple, predict missing entity")
    print("  - Head prediction: (?, relation, tail)")
    print("  - Tail prediction: (head, relation, ?)")
    print("")

    // ========================================================================
    // Setup: Create entity and relation embeddings
    // ========================================================================
    print("--- Setup: Entity and Relation Embeddings ---")
    print("")

    // Entity embeddings (4 entities, 8 dimensions each)
    let alice = zeros([8])
    let bob = zeros([8])
    let tokyo = zeros([8])
    let osaka = zeros([8])
    let kyoto = zeros([8])

    // Relation embedding
    let lives_in = zeros([8])

    print("Initialized embeddings:")
    print("  Entities: alice, bob, tokyo, osaka, kyoto")
    print("  Relation: lives_in")
    print("  Dimension: 8")
    print("")

    // ========================================================================
    // Part 1: Tail Prediction (head, relation, ?)
    // ========================================================================
    print("======================================================================")
    print("PART 1: TAIL PREDICTION")
    print("======================================================================")
    print("")

    print("Query: (alice, lives_in, ?)")
    print("Candidates: tokyo, osaka, kyoto")
    print("")

    // Score each candidate using TransE
    print("--- TransE Scoring ---")
    let score_tokyo_transe = predict_tail_transe(alice, lives_in, tokyo, "L2")
    let score_osaka_transe = predict_tail_transe(alice, lives_in, osaka, "L2")
    let score_kyoto_transe = predict_tail_transe(alice, lives_in, kyoto, "L2")

    print("  tokyo:", score_tokyo_transe)
    print("  osaka:", score_osaka_transe)
    print("  kyoto:", score_kyoto_transe)
    print("")

    // Score each candidate using DistMult
    print("--- DistMult Scoring ---")
    let score_tokyo_distmult = predict_tail_distmult(alice, lives_in, tokyo)
    let score_osaka_distmult = predict_tail_distmult(alice, lives_in, osaka)
    let score_kyoto_distmult = predict_tail_distmult(alice, lives_in, kyoto)

    print("  tokyo:", score_tokyo_distmult)
    print("  osaka:", score_osaka_distmult)
    print("  kyoto:", score_kyoto_distmult)
    print("")

    print("Interpretation:")
    print("  Higher score = more likely to be the correct answer")
    print("  Rank candidates by score to get prediction")
    print("")

    // ========================================================================
    // Part 2: Head Prediction (?, relation, tail)
    // ========================================================================
    print("======================================================================")
    print("PART 2: HEAD PREDICTION")
    print("======================================================================")
    print("")

    print("Query: (?, lives_in, tokyo)")
    print("Candidates: alice, bob")
    print("")

    // Score each candidate using TransE
    print("--- TransE Scoring ---")
    let score_alice_transe = predict_head_transe(alice, lives_in, tokyo, "L2")
    let score_bob_transe = predict_head_transe(bob, lives_in, tokyo, "L2")

    print("  alice:", score_alice_transe)
    print("  bob:", score_bob_transe)
    print("")

    // Score each candidate using DistMult
    print("--- DistMult Scoring ---")
    let score_alice_distmult = predict_head_distmult(alice, lives_in, tokyo)
    let score_bob_distmult = predict_head_distmult(bob, lives_in, tokyo)

    print("  alice:", score_alice_distmult)
    print("  bob:", score_bob_distmult)
    print("")

    // ========================================================================
    // Part 3: Prediction Workflow
    // ========================================================================
    print("======================================================================")
    print("PART 3: COMPLETE PREDICTION WORKFLOW")
    print("======================================================================")
    print("")

    print("Steps for Link Prediction:")
    print("")
    print("1. IDENTIFY QUERY TYPE")
    print("   - Head prediction: (?, r, t)")
    print("   - Tail prediction: (h, r, ?)")
    print("   - Relation prediction: (h, ?, t)")
    print("")

    print("2. GENERATE CANDIDATES")
    print("   - All entities in the knowledge graph")
    print("   - Or: Top-k most frequent entities")
    print("   - Or: Entities of specific type")
    print("")

    print("3. SCORE EACH CANDIDATE")
    print("   - Use trained embedding model (TransE, DistMult, etc.)")
    print("   - Compute score for each (h, r, t) candidate")
    print("   - score = model_function(emb[h], emb[r], emb[t])")
    print("")

    print("4. RANK CANDIDATES")
    print("   - Sort candidates by score (descending)")
    print("   - Top-ranked = most likely answer")
    print("")

    print("5. FILTER (Optional)")
    print("   - Remove known facts from ranking")
    print("   - Focus on discovering new knowledge")
    print("")

    print("6. EVALUATE")
    print("   - If ground truth known:")
    print("     * Check rank of correct answer")
    print("     * Compute metrics: MRR, Hits@1, Hits@10")
    print("")

    // ========================================================================
    // Part 4: Applications
    // ========================================================================
    print("======================================================================")
    print("PART 4: PRACTICAL APPLICATIONS")
    print("======================================================================")
    print("")

    print("Use Case 1: KNOWLEDGE BASE COMPLETION")
    print("  Problem: Database has missing information")
    print("  Example: Person X lives in ? (country unknown)")
    print("  Solution: Predict most likely country using embeddings")
    print("")

    print("Use Case 2: RECOMMENDATION")
    print("  Problem: Recommend items to users")
    print("  Example: User might like ? (based on past preferences)")
    print("  Solution: Predict items with high scores")
    print("")

    print("Use Case 3: QUESTION ANSWERING")
    print("  Problem: Answer factual questions")
    print("  Example: Where does Alice live?")
    print("  Solution: Query = (alice, lives_in, ?), predict tail")
    print("")

    print("Use Case 4: ANOMALY DETECTION")
    print("  Problem: Find unusual/incorrect facts")
    print("  Example: Is (bob, lives_in, mars) plausible?")
    print("  Solution: If score is very low, likely incorrect")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY")
    print("======================================================================")
    print("")

    print("✅ Implemented Functions:")
    print("  ✓ predict_tail_transe(head, rel, tail_candidate, model)")
    print("  ✓ predict_head_transe(head_candidate, rel, tail, model)")
    print("  ✓ predict_tail_distmult(head, rel, tail_candidate)")
    print("  ✓ predict_head_distmult(head_candidate, rel, tail)")
    print("")

    print("📊 Key Concepts:")
    print("  - Link prediction fills missing information")
    print("  - Scoring functions rank candidates")
    print("  - Higher scores = more plausible triples")
    print("  - Applications: KB completion, QA, recommendations")
    print("")

    print("🎯 Next Steps:")
    print("  - Ranking with multiple candidates")
    print("  - Filtering known facts")
    print("  - Evaluation metrics (MRR, Hits@k)")
    print("")

    print("✅ Link prediction demo completed!")
}
