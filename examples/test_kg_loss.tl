// Test Knowledge Graph Embedding Loss Functions

entity Person
entity City

relation lives_in(person: Person, city: City)

// Create embeddings
embedding PersonEmbed {
    entities: {alice, bob, charlie}
    dimension: 4
    init: random
}

embedding CityEmbed {
    entities: {tokyo, osaka, kyoto}
    dimension: 4
    init: random
}

relation_embedding RelEmbed {
    relations: {lives_in}
    dimension: 4
    init: random
}

main {
    print("🧪 Testing Knowledge Graph Loss Functions")
    print("")

    // Positive triple: (alice, lives_in, tokyo)
    let alice_emb = PersonEmbed[alice]
    let tokyo_emb = CityEmbed[tokyo]

    // Negative triple: (alice, lives_in, osaka) - corrupted tail
    let osaka_emb = CityEmbed[osaka]

    // Mock relation embedding
    let rel_emb = zeros([4])

    print("=== TransE Scores ===")
    let pos_score = transe_score(alice_emb, rel_emb, tokyo_emb, "L2")
    print("Positive triple score:", pos_score)

    let neg_score = transe_score(alice_emb, rel_emb, osaka_emb, "L2")
    print("Negative triple score:", neg_score)
    print("")

    print("=== Margin Ranking Loss ===")
    let margin = 1.0
    let loss = margin_ranking_loss(pos_score, neg_score, margin)
    print("Loss with margin", margin, ":", loss)
    print("")
    print("Formula: loss = max(0, margin + neg_score - pos_score)")
    print("  → Positive scores should be higher than negative scores")
    print("  → If pos_score > neg_score + margin, loss = 0")
    print("")

    print("=== DistMult Scores ===")
    let pos_distmult = distmult_score(alice_emb, rel_emb, tokyo_emb)
    print("Positive triple score:", pos_distmult)

    let neg_distmult = distmult_score(alice_emb, rel_emb, osaka_emb)
    print("Negative triple score:", neg_distmult)
    print("")

    let distmult_loss = margin_ranking_loss(pos_distmult, neg_distmult, margin)
    print("DistMult loss:", distmult_loss)

    print("")
    print("=== Binary Cross Entropy ===")
    // Positive example (target = 1)
    let bce_pos = binary_cross_entropy(pos_score, 1.0)
    print("BCE for positive triple (target=1):", bce_pos)

    // Negative example (target = 0)
    let bce_neg = binary_cross_entropy(neg_score, 0.0)
    print("BCE for negative triple (target=0):", bce_neg)
    print("")
    print("Formula: -target * log(sigmoid(score)) - (1-target) * log(1-sigmoid(score))")

    print("")
    print("✅ Loss function test completed!")
}
