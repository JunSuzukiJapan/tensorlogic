// Test DistMult Knowledge Graph Embedding

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
    print("🧪 Testing DistMult Scoring Function")
    print("")

    // Get embeddings for entities and relation
    let alice_emb = PersonEmbed[alice]
    let tokyo_emb = CityEmbed[tokyo]
    // Create mock relation embedding
    let lives_in_emb = zeros([4])

    print("Entity embeddings:")
    print("  alice:", alice_emb)
    print("  tokyo:", tokyo_emb)
    print("  lives_in (mock):", lives_in_emb)
    print("")

    // Compute DistMult score: score = sum(h * r * t)
    print("DistMult Score:")
    let score = distmult_score(alice_emb, lives_in_emb, tokyo_emb)
    print("  score:", score)

    print("")
    print("✅ DistMult test completed!")
    print("")
    print("Note: DistMult models symmetric relations")
    print("      Score = sum of element-wise products")
}
