// Test Entity Embedding Lookup

entity Person
entity City

relation lives_in(person: Person, city: City)

// Create embeddings with explicit entities
embedding PersonEmbed {
    entities: {alice, bob, charlie}
    dimension: 8
    init: xavier
}

embedding CityEmbed {
    entities: {tokyo, osaka, kyoto}
    dimension: 4
    init: random
}

main {
    // Add some facts
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    print("=== Entity Embedding Lookup Test ===")
    print("")

    // Lookup embeddings using bracket notation with string literals
    print("Person embeddings (8-dim):")
    let alice_emb = PersonEmbed["alice"]
    print("  alice:", alice_emb)

    let bob_emb = PersonEmbed["bob"]
    print("  bob:", bob_emb)

    let charlie_emb = PersonEmbed["charlie"]
    print("  charlie:", charlie_emb)
    print("")

    // Lookup city embeddings
    print("City embeddings (4-dim):")
    let tokyo_emb = CityEmbed["tokyo"]
    print("  tokyo:", tokyo_emb)

    let osaka_emb = CityEmbed["osaka"]
    print("  osaka:", osaka_emb)

    let kyoto_emb = CityEmbed["kyoto"]
    print("  kyoto:", kyoto_emb)
    print("")

    // Variable lookup
    print("Variable lookup:")
    let person_name = "alice"
    let person_vec = PersonEmbed[person_name]
    print("  PersonEmbed[person_name] where person_name='alice':")
    print("  ", person_vec)
    print("")

    print("✅ Embedding lookup test completed!")
}
