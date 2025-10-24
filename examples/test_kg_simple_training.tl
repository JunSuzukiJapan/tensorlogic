// Simple Knowledge Graph Embedding Training Demo
// Demonstrates scoring and loss computation for KG embeddings

main {
    print("🎓 Knowledge Graph Embedding Training Demo")
    print("==================================================")
    print("")

    // Create sample embeddings (simulating entity/relation vectors)
    let alice = zeros([4])
    let bob = zeros([4])
    let tokyo = zeros([4])
    let osaka = zeros([4])

    let lives_in_rel = zeros([4])

    print("📊 Simulated Embeddings:")
    print("  Entity embeddings: alice, bob, tokyo, osaka")
    print("  Relation embedding: lives_in")
    print("  Dimension: 4")
    print("")

    print("=== TransE Scoring ===")
    print("")

    // Positive triple: (alice, lives_in, tokyo)
    print("Positive triple: (alice, lives_in, tokyo)")
    let pos_score = transe_score(alice, lives_in_rel, tokyo, "L2")
    print("  TransE score:", pos_score)

    // Negative triple: (alice, lives_in, osaka)
    print("")
    print("Negative triple: (alice, lives_in, osaka)")
    let neg_score = transe_score(alice, lives_in_rel, osaka, "L2")
    print("  TransE score:", neg_score)

    print("")
    print("Interpretation:")
    print("  Higher score = better match")
    print("  Lower distance = higher score (TransE uses negative distance)")
    print("")

    print("=== Margin Ranking Loss ===")
    print("")

    let margin = 1.0
    let loss = margin_ranking_loss(pos_score, neg_score, margin)
    print("Margin:", margin)
    print("Loss:", loss)
    print("")
    print("Formula: loss = max(0, margin + neg_score - pos_score)")
    print("  Goal: Positive score should be > negative score + margin")
    print("")

    print("=== DistMult Scoring ===")
    print("")

    let distmult_pos = distmult_score(alice, lives_in_rel, tokyo)
    let distmult_neg = distmult_score(alice, lives_in_rel, osaka)
    print("DistMult positive:", distmult_pos)
    print("DistMult negative:", distmult_neg)
    print("")
    print("Formula: score = sum(h * r * t)")
    print("  Models symmetric relations")
    print("")

    print("=== Binary Cross Entropy Loss ===")
    print("")

    let bce_pos = binary_cross_entropy(pos_score, 1.0)
    let bce_neg = binary_cross_entropy(neg_score, 0.0)
    print("BCE for positive (target=1):", bce_pos)
    print("BCE for negative (target=0):", bce_neg)
    print("Total BCE:", bce_pos + bce_neg)
    print("")

    print("=== Training Workflow ===")
    print("")
    print("Complete training loop would include:")
    print("  1. Initialize embeddings randomly")
    print("  2. For each epoch:")
    print("     a. Sample positive triple from facts")
    print("     b. Generate negative triple (corrupt head or tail)")
    print("     c. Compute scores and loss")
    print("     d. Compute gradients")
    print("     e. Update embeddings: emb -= lr * gradient")
    print("  3. After training:")
    print("     - Positive triples have higher scores")
    print("     - Can use embeddings for link prediction")
    print("")

    print("📐 Negative Sampling Example:")
    print("")
    print("Positive: (alice, lives_in, tokyo)")
    print("Corrupt tail: (alice, lives_in, ???)")
    print("  Options: osaka, kyoto, nagoya, ...")
    print("  Strategy: Random sampling or frequency-based")
    print("")
    print("Corrupt head: (???, lives_in, tokyo)")
    print("  Options: bob, charlie, david, ...")
    print("")

    print("✅ Training demo completed!")
    print("")
    print("📝 This demo shows:")
    print("  ✓ TransE scoring (-||h + r - t||)")
    print("  ✓ DistMult scoring (sum(h * r * t))")
    print("  ✓ Margin ranking loss")
    print("  ✓ Binary cross entropy loss")
    print("  ✓ Training workflow concepts")
}
