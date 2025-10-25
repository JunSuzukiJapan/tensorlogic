// Knowledge Graph Embedding Models Comparison
// TransE vs DistMult vs ComplEx

main {
    print("======================================================================")
    print("   KNOWLEDGE GRAPH EMBEDDINGS: MODEL COMPARISON")
    print("======================================================================")
    print("")

    print("🎯 Comparing Three State-of-the-Art Models:")
    print("  1. TransE - Translational distance model")
    print("  2. DistMult - Bilinear diagonal model")
    print("  3. ComplEx - Complex embeddings model")
    print("")

    // ========================================================================
    // Setup: Same embeddings for all models
    // ========================================================================
    print("======================================================================")
    print("SETUP: KNOWLEDGE GRAPH")
    print("======================================================================")
    print("")

    print("Entities:")
    print("  - alice, bob")
    print("  - tokyo, paris")
    print("")

    print("Relations:")
    print("  - friend_of (symmetric)")
    print("  - lives_in (antisymmetric)")
    print("")

    // Standard embeddings for TransE and DistMult
    let alice = zeros([8])
    let bob = zeros([8])
    let tokyo = zeros([8])
    let paris = zeros([8])

    let friend_of = zeros([8])
    let lives_in = zeros([8])

    // Complex embeddings for ComplEx
    let alice_re = zeros([8])
    let alice_im = zeros([8])
    let bob_re = zeros([8])
    let bob_im = zeros([8])
    let tokyo_re = zeros([8])
    let tokyo_im = zeros([8])
    let paris_re = zeros([8])
    let paris_im = zeros([8])

    let friend_of_re = zeros([8])
    let friend_of_im = zeros([8])
    let lives_in_re = zeros([8])
    let lives_in_im = zeros([8])

    print("Initialized 8-dimensional embeddings")
    print("")

    // ========================================================================
    // Part 1: Scoring Function Comparison
    // ========================================================================
    print("======================================================================")
    print("PART 1: SCORING FUNCTIONS")
    print("======================================================================")
    print("")

    print("Triple: (alice, friend_of, bob)")
    print("")

    print("--- TransE ---")
    print("Formula: -||h + r - t||")
    print("")
    let score_transe = transe_score(alice, friend_of, bob, "L2")
    print("  Score:", score_transe)
    print("")
    print("Intuition:")
    print("  - Treats relation as translation: h + r ≈ t")
    print("  - Distance-based (lower distance = higher score)")
    print("  - Good for antisymmetric relations")
    print("")

    print("--- DistMult ---")
    print("Formula: Σ(h ⊙ r ⊙ t)")
    print("")
    let score_distmult = distmult_score(alice, friend_of, bob)
    print("  Score:", score_distmult)
    print("")
    print("Intuition:")
    print("  - Element-wise multiplication and sum")
    print("  - Symmetric: score(h,r,t) = score(t,r,h)")
    print("  - Good for symmetric relations")
    print("")

    print("--- ComplEx ---")
    print("Formula: Re(<h, r, conj(t)>)")
    print("")
    let score_complex = complex_score(
        alice_re, alice_im,
        friend_of_re, friend_of_im,
        bob_re, bob_im
    )
    print("  Score:", score_complex)
    print("")
    print("Intuition:")
    print("  - Uses complex numbers (real + imaginary)")
    print("  - Complex conjugate enables asymmetry")
    print("  - Best of both worlds!")
    print("")

    // ========================================================================
    // Part 2: Relation Type Handling
    // ========================================================================
    print("======================================================================")
    print("PART 2: RELATION TYPE HANDLING")
    print("======================================================================")
    print("")

    print("--- Symmetric Relations (friend_of) ---")
    print("")
    print("Property: If (alice, friend_of, bob) then (bob, friend_of, alice)")
    print("")

    print("TransE:")
    let transe_ab = transe_score(alice, friend_of, bob, "L2")
    let transe_ba = transe_score(bob, friend_of, alice, "L2")
    print("  (alice, friend_of, bob):", transe_ab)
    print("  (bob, friend_of, alice):", transe_ba)
    print("  Can differ → ❌ Not ideal for symmetric relations")
    print("")

    print("DistMult:")
    let distmult_ab = distmult_score(alice, friend_of, bob)
    let distmult_ba = distmult_score(bob, friend_of, alice)
    print("  (alice, friend_of, bob):", distmult_ab)
    print("  (bob, friend_of, alice):", distmult_ba)
    print("  Always same → ✅ Perfect for symmetric relations")
    print("")

    print("ComplEx:")
    let complex_ab = complex_score(
        alice_re, alice_im,
        friend_of_re, friend_of_im,
        bob_re, bob_im
    )
    let complex_ba = complex_score(
        bob_re, bob_im,
        friend_of_re, friend_of_im,
        alice_re, alice_im
    )
    print("  (alice, friend_of, bob):", complex_ab)
    print("  (bob, friend_of, alice):", complex_ba)
    print("  Can be same or different → ✅ Handles both!")
    print("")

    print("--- Antisymmetric Relations (lives_in) ---")
    print("")
    print("Property: (alice, lives_in, tokyo) doesn't imply (tokyo, lives_in, alice)")
    print("")

    print("TransE:")
    let transe_at = transe_score(alice, lives_in, tokyo, "L2")
    let transe_ta = transe_score(tokyo, lives_in, alice, "L2")
    print("  (alice, lives_in, tokyo):", transe_at)
    print("  (tokyo, lives_in, alice):", transe_ta)
    print("  Different scores → ✅ Good for antisymmetric")
    print("")

    print("DistMult:")
    let distmult_at = distmult_score(alice, lives_in, tokyo)
    let distmult_ta = distmult_score(tokyo, lives_in, alice)
    print("  (alice, lives_in, tokyo):", distmult_at)
    print("  (tokyo, lives_in, alice):", distmult_ta)
    print("  Always same → ❌ Cannot model antisymmetric")
    print("")

    print("ComplEx:")
    let complex_at = complex_score(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        tokyo_re, tokyo_im
    )
    let complex_ta = complex_score(
        tokyo_re, tokyo_im,
        lives_in_re, lives_in_im,
        alice_re, alice_im
    )
    print("  (alice, lives_in, tokyo):", complex_at)
    print("  (tokyo, lives_in, alice):", complex_ta)
    print("  Can differ → ✅ Handles antisymmetric!")
    print("")

    // ========================================================================
    // Part 3: Link Prediction Comparison
    // ========================================================================
    print("======================================================================")
    print("PART 3: LINK PREDICTION")
    print("======================================================================")
    print("")

    print("Query: (alice, lives_in, ?)")
    print("Candidates: tokyo, paris")
    print("")

    print("TransE Predictions:")
    let transe_tokyo = predict_tail_transe(alice, lives_in, tokyo, "L2")
    let transe_paris = predict_tail_transe(alice, lives_in, paris, "L2")
    print("  tokyo:", transe_tokyo)
    print("  paris:", transe_paris)
    print("")

    print("DistMult Predictions:")
    let distmult_tokyo = predict_tail_distmult(alice, lives_in, tokyo)
    let distmult_paris = predict_tail_distmult(alice, lives_in, paris)
    print("  tokyo:", distmult_tokyo)
    print("  paris:", distmult_paris)
    print("")

    print("ComplEx Predictions:")
    let complex_tokyo = predict_tail_complex(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        tokyo_re, tokyo_im
    )
    let complex_paris = predict_tail_complex(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        paris_re, paris_im
    )
    print("  tokyo:", complex_tokyo)
    print("  paris:", complex_paris)
    print("")

    print("All models can rank candidates by score")
    print("Different models may give different rankings!")
    print("")

    // ========================================================================
    // Part 4: Model Characteristics
    // ========================================================================
    print("======================================================================")
    print("PART 4: MODEL CHARACTERISTICS")
    print("======================================================================")
    print("")

    print("┌──────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Aspect       │ TransE      │ DistMult    │ ComplEx     │")
    print("├──────────────┼─────────────┼─────────────┼─────────────┤")
    print("│ Parameters   │ d           │ d           │ 2d          │")
    print("│ Symmetric    │ ❌          │ ✅          │ ✅          │")
    print("│ Antisymmetric│ ✅          │ ❌          │ ✅          │")
    print("│ Composition  │ ✅          │ ❌          │ ✅          │")
    print("│ Complexity   │ Simple      │ Simple      │ Moderate    │")
    print("│ Intuition    │ High        │ Moderate    │ Low         │")
    print("└──────────────┴─────────────┴─────────────┴─────────────┘")
    print("")

    // ========================================================================
    // Part 5: When to Use Each Model
    // ========================================================================
    print("======================================================================")
    print("PART 5: WHEN TO USE EACH MODEL")
    print("======================================================================")
    print("")

    print("--- Use TransE When: ---")
    print("  ✓ Most relations are antisymmetric")
    print("  ✓ You want interpretable embeddings")
    print("  ✓ Computational efficiency is critical")
    print("  ✓ You need to explain predictions")
    print("  ✓ Relation composition is important")
    print("")
    print("Example domains:")
    print("  - Hierarchical relationships (parent, boss)")
    print("  - Temporal relationships (before, after)")
    print("  - Location relationships (located_in, capital_of)")
    print("")

    print("--- Use DistMult When: ---")
    print("  ✓ Most relations are symmetric")
    print("  ✓ You want fast training/inference")
    print("  ✓ Simple implementation needed")
    print("  ✓ Data is limited (fewer parameters)")
    print("")
    print("Example domains:")
    print("  - Social networks (colleague, friend)")
    print("  - Equivalence relations (similar_to, synonym_of)")
    print("  - Collaborative filtering (likes, related_to)")
    print("")

    print("--- Use ComplEx When: ---")
    print("  ✓ Mixed symmetric/antisymmetric relations")
    print("  ✓ State-of-the-art performance needed")
    print("  ✓ Sufficient training data available")
    print("  ✓ Can afford 2x parameters")
    print("  ✓ Want best accuracy regardless of complexity")
    print("")
    print("Example domains:")
    print("  - General knowledge graphs (Freebase, Wikidata)")
    print("  - Biomedical knowledge (proteins, diseases, drugs)")
    print("  - Multi-domain applications")
    print("")

    // ========================================================================
    // Part 6: Performance Comparison
    // ========================================================================
    print("======================================================================")
    print("PART 6: TYPICAL PERFORMANCE (BENCHMARK RESULTS)")
    print("======================================================================")
    print("")

    print("FB15k Benchmark (Freebase subset):")
    print("")
    print("  Model       MRR    Hits@1  Hits@10")
    print("  ────────────────────────────────────")
    print("  TransE      ~0.38   ~27%    ~64%")
    print("  DistMult    ~0.42   ~30%    ~69%")
    print("  ComplEx     ~0.69   ~60%    ~84%")
    print("")
    print("  → ComplEx significantly outperforms!")
    print("")

    print("WN18 Benchmark (WordNet subset):")
    print("")
    print("  Model       MRR    Hits@1  Hits@10")
    print("  ────────────────────────────────────")
    print("  TransE      ~0.49   ~11%    ~94%")
    print("  DistMult    ~0.82   ~73%    ~94%")
    print("  ComplEx     ~0.94   ~94%    ~95%")
    print("")
    print("  → ComplEx again performs best!")
    print("")

    print("Note: Actual performance varies by:")
    print("  - Dataset characteristics")
    print("  - Hyperparameter tuning")
    print("  - Training strategy")
    print("  - Negative sampling method")
    print("")

    // ========================================================================
    // Part 7: Practical Recommendations
    // ========================================================================
    print("======================================================================")
    print("PART 7: PRACTICAL RECOMMENDATIONS")
    print("======================================================================")
    print("")

    print("1. START SIMPLE")
    print("   - Begin with TransE or DistMult")
    print("   - Understand data characteristics")
    print("   - Establish baseline performance")
    print("")

    print("2. ANALYZE RELATIONS")
    print("   - Count symmetric vs antisymmetric")
    print("   - Check relation type distribution")
    print("   - This guides model selection")
    print("")

    print("3. TRY MULTIPLE MODELS")
    print("   - Different datasets favor different models")
    print("   - ComplEx usually best but not always")
    print("   - Compare on validation set")
    print("")

    print("4. CONSIDER ENSEMBLE")
    print("   - Combine predictions from multiple models")
    print("   - Often outperforms single best model")
    print("   - Weighted average by validation performance")
    print("")

    print("5. HYPERPARAMETER TUNING")
    print("   - Embedding dimension (64, 128, 256, 512)")
    print("   - Learning rate (0.001 - 0.01)")
    print("   - Batch size (128 - 512)")
    print("   - Negative samples per positive (1 - 10)")
    print("   - Regularization weight (0.0001 - 0.01)")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY")
    print("======================================================================")
    print("")

    print("✅ Three Powerful Models:")
    print("  1. TransE - Best for antisymmetric, interpretable")
    print("  2. DistMult - Best for symmetric, simple")
    print("  3. ComplEx - Best overall, most expressive")
    print("")

    print("🎯 Key Takeaway:")
    print("  No single model is always best!")
    print("  Choose based on:")
    print("    - Relation types in your data")
    print("    - Performance requirements")
    print("    - Computational constraints")
    print("    - Interpretability needs")
    print("")

    print("📚 All Models Available in TensorLogic:")
    print("  ✓ transe_score() + predict_tail/head_transe()")
    print("  ✓ distmult_score() + predict_tail/head_distmult()")
    print("  ✓ complex_score() + predict_tail/head_complex()")
    print("  ✓ Shared loss functions and evaluation metrics")
    print("")

    print("======================================================================")
    print("   MODEL COMPARISON COMPLETE")
    print("======================================================================")
}
