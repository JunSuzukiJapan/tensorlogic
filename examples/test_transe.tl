// Test TransE Knowledge Graph Embedding

entity Person
entity City

relation lives_in(person: Person, city: City)

// Create embeddings
embedding PersonEmbed {
    entities: {alice, bob}
    dimension: 4
    init: random
}

embedding CityEmbed {
    entities: {tokyo, osaka}
    dimension: 4
    init: random
}

relation_embedding RelEmbed {
    relations: {lives_in}
    dimension: 4
    init: random
}

main {
    print("🧪 Testing TransE Scoring Function")
    print("")

    // Get embeddings for entities and relation
    let alice_emb = PersonEmbed[alice]
    let tokyo_emb = CityEmbed[tokyo]
    // Create mock relation embedding using zeros for now
    let lives_in_emb = zeros([4])

    print("Entity embeddings:")
    print("  alice:", alice_emb)
    print("  tokyo:", tokyo_emb)
    print("  lives_in (mock):", lives_in_emb)
    print("")

    // Compute TransE score: score = -||h + r - t||
    print("TransE Scores:")

    // L2 norm (default)
    let score_l2 = transe_score(alice_emb, lives_in_emb, tokyo_emb, "L2")
    print("  L2 norm:", score_l2)

    // Note: L1 norm not yet implemented (requires Tensor.abs())

    print("")
    print("✅ TransE test completed!")
    print("")
    print("Note: Higher (less negative) scores indicate better fit")
    print("      TransE learns: h + r ≈ t")
}
