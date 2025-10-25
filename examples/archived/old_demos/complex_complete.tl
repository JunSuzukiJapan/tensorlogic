// ComplEx: Complete Demonstration
// Phase 12.4: ComplEx (Complex Embeddings) Implementation

main {
    print("======================================================================")
    print("   COMPLEX EMBEDDINGS (ComplEx) - COMPLETE DEMONSTRATION")
    print("======================================================================")
    print("")

    print("📚 ComplEx Overview:")
    print("  - Uses complex-valued vectors (real + imaginary parts)")
    print("  - Unifies symmetric and antisymmetric relation modeling")
    print("  - Generalizes DistMult while adding expressiveness")
    print("  - State-of-the-art performance on many benchmarks")
    print("")

    // ========================================================================
    // Part 1: Complex Embeddings Basics
    // ========================================================================
    print("======================================================================")
    print("PART 1: COMPLEX EMBEDDINGS BASICS")
    print("======================================================================")
    print("")

    print("--- Complex Numbers Representation ---")
    print("")
    print("Each entity/relation has TWO embeddings:")
    print("  - Real part (h_re)")
    print("  - Imaginary part (h_im)")
    print("")
    print("Example: alice = (alice_re, alice_im)")
    print("  where alice_re, alice_im ∈ ℝ^d")
    print("")

    // Create embeddings (dimension 8)
    let alice_re = zeros([8])
    let alice_im = zeros([8])
    let bob_re = zeros([8])
    let bob_im = zeros([8])
    let charlie_re = zeros([8])
    let charlie_im = zeros([8])

    let tokyo_re = zeros([8])
    let tokyo_im = zeros([8])
    let osaka_re = zeros([8])
    let osaka_im = zeros([8])
    let kyoto_re = zeros([8])
    let kyoto_im = zeros([8])

    let lives_in_re = zeros([8])
    let lives_in_im = zeros([8])
    let friend_of_re = zeros([8])
    let friend_of_im = zeros([8])
    let works_at_re = zeros([8])
    let works_at_im = zeros([8])

    print("Initialized 8-dimensional complex embeddings:")
    print("  People: alice, bob, charlie (each has _re and _im)")
    print("  Places: tokyo, osaka, kyoto")
    print("  Relations: lives_in, friend_of, works_at")
    print("")

    print("--- Complex Conjugate ---")
    print("")
    print("Key operation in ComplEx:")
    print("  For z = a + bi")
    print("  conj(z) = a - bi")
    print("")
    print("This conjugate makes ComplEx ASYMMETRIC,")
    print("unlike DistMult which is symmetric.")
    print("")

    // ========================================================================
    // Part 2: ComplEx Scoring Function
    // ========================================================================
    print("======================================================================")
    print("PART 2: COMPLEX SCORING FUNCTION")
    print("======================================================================")
    print("")

    print("--- Formula ---")
    print("")
    print("Score(h, r, t) = Re(<h, r, conj(t)>)")
    print("")
    print("Expanded:")
    print("  = Σ_i [")
    print("      h_re[i] · r_re[i] · t_re[i]")
    print("    + h_re[i] · r_im[i] · t_im[i]")
    print("    + h_im[i] · r_re[i] · t_im[i]")
    print("    - h_im[i] · r_im[i] · t_re[i]")
    print("    ]")
    print("")

    print("Example: (alice, friend_of, bob)")
    print("")
    let score_alice_bob = complex_score(
        alice_re, alice_im,
        friend_of_re, friend_of_im,
        bob_re, bob_im
    )
    print("  Score:", score_alice_bob)
    print("")

    print("Example: (bob, lives_in, tokyo)")
    print("")
    let score_bob_tokyo = complex_score(
        bob_re, bob_im,
        lives_in_re, lives_in_im,
        tokyo_re, tokyo_im
    )
    print("  Score:", score_bob_tokyo)
    print("")

    // ========================================================================
    // Part 3: Link Prediction
    // ========================================================================
    print("======================================================================")
    print("PART 3: LINK PREDICTION WITH COMPLEX")
    print("======================================================================")
    print("")

    print("--- Tail Prediction: (alice, lives_in, ?) ---")
    print("")
    print("Candidates: tokyo, osaka, kyoto")
    print("")

    let score_tokyo = predict_tail_complex(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        tokyo_re, tokyo_im
    )
    print("  tokyo:", score_tokyo)

    let score_osaka = predict_tail_complex(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        osaka_re, osaka_im
    )
    print("  osaka:", score_osaka)

    let score_kyoto = predict_tail_complex(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        kyoto_re, kyoto_im
    )
    print("  kyoto:", score_kyoto)
    print("")

    print("Rank by score (higher = more likely)")
    print("  → Best prediction: highest scoring city")
    print("")

    print("--- Head Prediction: (?, friend_of, bob) ---")
    print("")
    print("Candidates: alice, charlie")
    print("")

    let score_alice_friend = predict_head_complex(
        alice_re, alice_im,
        friend_of_re, friend_of_im,
        bob_re, bob_im
    )
    print("  alice:", score_alice_friend)

    let score_charlie_friend = predict_head_complex(
        charlie_re, charlie_im,
        friend_of_re, friend_of_im,
        bob_re, bob_im
    )
    print("  charlie:", score_charlie_friend)
    print("")

    print("→ Best prediction: highest scoring person")
    print("")

    // ========================================================================
    // Part 4: Training Concept
    // ========================================================================
    print("======================================================================")
    print("PART 4: TRAINING CONCEPT")
    print("======================================================================")
    print("")

    print("--- Positive and Negative Sampling ---")
    print("")
    print("Positive triple: (alice, lives_in, tokyo)")
    print("  → True fact from knowledge graph")
    print("")

    let pos_score = complex_score(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        tokyo_re, tokyo_im
    )
    print("  Positive score:", pos_score)
    print("")

    print("Negative triple: (alice, lives_in, mars) [corrupted]")
    print("  → False fact (corrupted head or tail)")
    print("")

    let mars_re = zeros([8])
    let mars_im = zeros([8])
    let neg_score = complex_score(
        alice_re, alice_im,
        lives_in_re, lives_in_im,
        mars_re, mars_im
    )
    print("  Negative score:", neg_score)
    print("")

    print("--- Loss Function ---")
    print("")
    print("Margin Ranking Loss:")
    print("  L = max(0, margin + neg_score - pos_score)")
    print("")

    let loss = margin_ranking_loss(pos_score, neg_score, 1.0)
    print("  Loss (margin=1.0):", loss)
    print("")

    print("Goal: Maximize pos_score, minimize neg_score")
    print("  → Trained embeddings distinguish true from false facts")
    print("")

    // ========================================================================
    // Part 5: Properties and Comparison
    // ========================================================================
    print("======================================================================")
    print("PART 5: COMPLEX PROPERTIES & COMPARISON")
    print("======================================================================")
    print("")

    print("--- Key Properties ---")
    print("")
    print("1. SYMMETRIC RELATIONS")
    print("   When r_im = 0 (purely real):")
    print("   ComplEx reduces to DistMult")
    print("   Example: 'colleague', 'sibling'")
    print("")

    print("2. ANTISYMMETRIC RELATIONS")
    print("   Complex conjugate enables asymmetry:")
    print("   score(h,r,t) ≠ score(t,r,h)")
    print("   Example: 'parent_of', 'boss_of'")
    print("")

    print("3. COMPOSITION")
    print("   Can model relation composition:")
    print("   If r = r1 ∘ r2, then r ≈ r1 · r2")
    print("   Example: 'grandparent' = 'parent' ∘ 'parent'")
    print("")

    print("--- Model Comparison ---")
    print("")
    print("┌─────────────┬───────────┬───────────────┬────────────┐")
    print("│ Model       │ Symmetric │ Antisymmetric │ Parameters │")
    print("├─────────────┼───────────┼───────────────┼────────────┤")
    print("│ TransE      │     ❌    │       ✅      │     d      │")
    print("│ DistMult    │     ✅    │       ❌      │     d      │")
    print("│ ComplEx     │     ✅    │       ✅      │     2d     │")
    print("└─────────────┴───────────┴───────────────┴────────────┘")
    print("")

    print("ComplEx advantages:")
    print("  ✓ Most expressive (handles all relation types)")
    print("  ✓ State-of-the-art performance on benchmarks")
    print("  ✓ Generalizes DistMult as special case")
    print("")

    print("ComplEx tradeoffs:")
    print("  ⚠ 2x parameters (real + imaginary)")
    print("  ⚠ More complex computation")
    print("  ⚠ Less intuitive than TransE")
    print("")

    // ========================================================================
    // Part 6: Evaluation Metrics
    // ========================================================================
    print("======================================================================")
    print("PART 6: EVALUATION METRICS")
    print("======================================================================")
    print("")

    print("--- Ranking Metrics ---")
    print("")
    print("Test query: (alice, lives_in, ?)")
    print("Ground truth: tokyo")
    print("")

    print("Scenario 1: tokyo ranked #1")
    let mrr1 = compute_mrr(1)
    let hits1 = compute_hits_at_k(1, 1)
    let hits10 = compute_hits_at_k(1, 10)
    print("  MRR:", mrr1)
    print("  Hits@1:", hits1)
    print("  Hits@10:", hits10)
    print("  → Perfect prediction!")
    print("")

    print("Scenario 2: tokyo ranked #5")
    let mrr2 = compute_mrr(5)
    let hits1_s2 = compute_hits_at_k(5, 1)
    let hits10_s2 = compute_hits_at_k(5, 10)
    print("  MRR:", mrr2)
    print("  Hits@1:", hits1_s2)
    print("  Hits@10:", hits10_s2)
    print("  → Moderate performance")
    print("")

    print("Higher MRR & Hits@k → Better model")
    print("")

    // ========================================================================
    // Part 7: Applications
    // ========================================================================
    print("======================================================================")
    print("PART 7: REAL-WORLD APPLICATIONS")
    print("======================================================================")
    print("")

    print("1. KNOWLEDGE BASE COMPLETION")
    print("   Problem: Incomplete databases with missing facts")
    print("   Solution: Predict missing relationships")
    print("   Example: Suggest missing 'works_at' relations")
    print("")

    print("2. QUESTION ANSWERING")
    print("   Problem: Answer questions from knowledge graph")
    print("   Solution: Treat questions as link prediction")
    print("   Example: 'Where does Alice live?' → (alice, lives_in, ?)")
    print("")

    print("3. RELATION EXTRACTION")
    print("   Problem: Extract structured facts from text")
    print("   Solution: Score candidate triples from text")
    print("   Example: Validate (alice, colleague, bob) from documents")
    print("")

    print("4. DRUG DISCOVERY")
    print("   Problem: Predict drug-disease interactions")
    print("   Solution: Model as knowledge graph")
    print("   Example: (drug_x, treats, disease_y)?")
    print("")

    print("5. RECOMMENDATION SYSTEMS")
    print("   Problem: Recommend items to users")
    print("   Solution: Predict (user, likes, item) triples")
    print("   Example: What movies might alice like?")
    print("")

    print("6. SOCIAL NETWORK ANALYSIS")
    print("   Problem: Predict social connections")
    print("   Solution: Model friendships and influence")
    print("   Example: Who should alice connect with?")
    print("")

    // ========================================================================
    // Part 8: Implementation Notes
    // ========================================================================
    print("======================================================================")
    print("PART 8: IMPLEMENTATION NOTES")
    print("======================================================================")
    print("")

    print("--- Storage Strategy ---")
    print("")
    print("Don't use native complex numbers:")
    print("  Instead: Store as paired real tensors")
    print("")
    print("  entity_re: Tensor[d]  # Real part")
    print("  entity_im: Tensor[d]  # Imaginary part")
    print("")

    print("--- Computation ---")
    print("")
    print("Score computation:")
    print("  1. Compute 4 trilinear products")
    print("  2. Sum: term1 + term2 + term3 - term4")
    print("  3. Result is scalar score")
    print("")

    print("--- Training Tips ---")
    print("")
    print("1. INITIALIZATION")
    print("   - Random uniform or normal initialization")
    print("   - Small values (e.g., [-0.1, 0.1])")
    print("")

    print("2. REGULARIZATION")
    print("   - L2 regularization on embeddings")
    print("   - Prevents overfitting")
    print("   - λ(||h_re||² + ||h_im||² + ...)")
    print("")

    print("3. NEGATIVE SAMPLING")
    print("   - Corrupt head OR tail (not both)")
    print("   - Multiple negatives per positive")
    print("   - Filtered: exclude true facts")
    print("")

    print("4. OPTIMIZATION")
    print("   - Adam optimizer recommended")
    print("   - Learning rate: 0.001 - 0.01")
    print("   - Batch size: 128 - 512")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY: COMPLEX COMPLETE")
    print("======================================================================")
    print("")

    print("✅ Implemented Features:")
    print("  ✓ complex_score() - Complex embedding scoring")
    print("  ✓ predict_tail_complex() - Tail entity prediction")
    print("  ✓ predict_head_complex() - Head entity prediction")
    print("  ✓ Works with margin_ranking_loss()")
    print("  ✓ Works with compute_mrr(), compute_hits_at_k()")
    print("")

    print("📊 Key Concepts:")
    print("  - Complex numbers: z = a + bi")
    print("  - Complex conjugate: conj(z) = a - bi")
    print("  - Asymmetric via conjugate operation")
    print("  - Generalizes DistMult (when im=0)")
    print("  - State-of-the-art expressiveness")
    print("")

    print("🎯 Use Cases:")
    print("  ✓ When you need both symmetric AND antisymmetric relations")
    print("  ✓ When DistMult is too simple")
    print("  ✓ When you have enough data for 2x parameters")
    print("  ✓ State-of-the-art performance requirements")
    print("")

    print("📚 References:")
    print("  - Trouillon et al. 'Complex Embeddings for Simple Link Prediction'")
    print("  - ICML 2016")
    print("  - https://arxiv.org/abs/1606.06357")
    print("")

    print("======================================================================")
    print("   COMPLEX DEMONSTRATION COMPLETE")
    print("======================================================================")
}
