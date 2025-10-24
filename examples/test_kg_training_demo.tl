// Knowledge Graph Embedding Training Demo
// Demonstrates the complete training workflow for KG embeddings

relation lives_in(person: entity, city: entity)

// Create entity embeddings using explicit entity lists
embedding PersonEmbed {
    entities: {alice, bob, charlie, david}
    dimension: 8
    init: xavier
}

embedding CityEmbed {
    entities: {tokyo, osaka, kyoto, nagoya}
    dimension: 8
    init: xavier
}

// Create relation embedding
relation_embedding RelEmbed {
    relations: {lives_in}
    dimension: 8
    init: xavier
}

main {
    print("🎓 Knowledge Graph Embedding Training Demo")
    print("=" * 50)
    print("")

    // Add known facts (positive triples)
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    print("📊 Training Data:")
    print("  Positive triples:")
    print("    - (alice, lives_in, tokyo)")
    print("    - (bob, lives_in, osaka)")
    print("    - (charlie, lives_in, kyoto)")
    print("")

    // Training parameters
    let learning_rate = 0.01
    let margin = 1.0
    let num_epochs = 10

    print("🔧 Training Configuration:")
    print("  Algorithm: TransE")
    print("  Loss: Margin Ranking Loss")
    print("  Learning rate:", learning_rate)
    print("  Margin:", margin)
    print("  Epochs:", num_epochs)
    print("")

    print("🔄 Training Loop Demonstration:")
    print("  (Showing conceptual training process)")
    print("")

    // Demonstrate one training iteration
    print("=== Epoch 1 Example ===")
    print("")

    // Get embeddings for a positive triple: (alice, lives_in, tokyo)
    let alice_emb = PersonEmbed[alice]
    let tokyo_emb = CityEmbed[tokyo]
    let lives_in_emb = RelEmbed[lives_in]

    print("Positive triple: (alice, lives_in, tokyo)")
    let pos_score = transe_score(alice_emb, lives_in_emb, tokyo_emb, "L2")
    print("  Positive score:", pos_score)

    // Generate negative sample by corrupting tail
    print("")
    print("Negative sampling (corrupt tail):")
    let negative_city = random_entity()
    print("  Sampled negative entity:", negative_city)

    // For demo, use osaka as negative (should be different from tokyo)
    let osaka_emb = CityEmbed[osaka]
    let neg_score = transe_score(alice_emb, lives_in_emb, osaka_emb, "L2")
    print("  Negative triple: (alice, lives_in, osaka)")
    print("  Negative score:", neg_score)

    // Compute loss
    print("")
    let loss = margin_ranking_loss(pos_score, neg_score, margin)
    print("Margin ranking loss:", loss)
    print("")

    // Gradient computation (conceptual)
    print("📐 Gradient Computation:")
    print("  In a full implementation, we would:")
    print("  1. Compute ∇loss w.r.t. embeddings")
    print("  2. Update: emb = emb - lr * ∇loss")
    print("  3. Repeat for all triples in batch")
    print("")

    // Demonstrate all scoring algorithms
    print("=== Comparing Scoring Functions ===")
    print("")

    print("TransE (translation-based):")
    let transe_pos = transe_score(alice_emb, lives_in_emb, tokyo_emb, "L2")
    let transe_neg = transe_score(alice_emb, lives_in_emb, osaka_emb, "L2")
    print("  Positive:", transe_pos, " | Negative:", transe_neg)

    print("")
    print("DistMult (symmetric):")
    let distmult_pos = distmult_score(alice_emb, lives_in_emb, tokyo_emb)
    let distmult_neg = distmult_score(alice_emb, lives_in_emb, osaka_emb)
    print("  Positive:", distmult_pos, " | Negative:", distmult_neg)

    print("")

    // Demonstrate loss functions
    print("=== Loss Function Comparison ===")
    print("")

    let mr_loss = margin_ranking_loss(transe_pos, transe_neg, margin)
    print("Margin Ranking Loss:", mr_loss)

    let bce_pos = binary_cross_entropy(transe_pos, 1.0)
    let bce_neg = binary_cross_entropy(transe_neg, 0.0)
    print("BCE Loss (positive):", bce_pos)
    print("BCE Loss (negative):", bce_neg)
    print("BCE Total:", bce_pos + bce_neg)

    print("")

    // Training workflow summary
    print("=== Full Training Workflow ===")
    print("")
    print("For each epoch:")
    print("  1. Sample positive triple (h, r, t) from known facts")
    print("  2. Generate negative triple by corrupting h or t")
    print("  3. Compute scores: score_pos, score_neg")
    print("  4. Compute loss: L = max(0, margin + score_neg - score_pos)")
    print("  5. Backpropagate gradients")
    print("  6. Update embeddings: emb = emb - lr * grad")
    print("  7. Repeat for all training triples")
    print("")

    print("🎯 Expected Outcome:")
    print("  - Positive triples get higher scores (lower distances)")
    print("  - Negative triples get lower scores (higher distances)")
    print("  - Model learns meaningful entity and relation representations")
    print("")

    print("✅ Training demo completed!")
    print("")
    print("📝 Note: This demo shows the conceptual training loop.")
    print("   Full gradient-based optimization requires autograd integration.")
}
