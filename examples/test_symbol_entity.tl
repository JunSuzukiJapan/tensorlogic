// Test symbol-based entity references for unified syntax

entity Person
entity City

relation lives_in(person: Person, city: City)

embedding PersonEmbed {
    entities: {alice, bob, charlie}
    dimension: 8
    init: xavier
}

main {
    // 1. Facts use symbols
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    print("=== Symbol-based Entity Reference Test ===")
    print("")

    // 2. entity_onehot() with symbols (unified!)
    print("entity_onehot with symbols:")
    let onehot1 = entity_onehot(Person, alice)
    print("  entity_onehot(Person, alice):", onehot1)

    let onehot2 = entity_onehot(Person, bob)
    print("  entity_onehot(Person, bob):", onehot2)
    print("")

    // 3. Embedding lookup with symbols (unified!)
    print("Embedding lookup with symbols:")
    let emb1 = PersonEmbed[alice]
    print("  PersonEmbed[alice]:", emb1)

    let emb2 = PersonEmbed[bob]
    print("  PersonEmbed[bob]:", emb2)
    print("")

    // 4. Backward compatibility: strings still work
    print("Backward compatibility (strings):")
    let onehot_str = entity_onehot(Person, "charlie")
    print("  entity_onehot(Person, \"charlie\"):", onehot_str)

    let emb_str = PersonEmbed["charlie"]
    print("  PersonEmbed[\"charlie\"]:", emb_str)
    print("")

    // 5. Variables work too
    print("Variable references:")
    let person_name = "alice"
    let onehot_var = entity_onehot(Person, person_name)
    print("  entity_onehot(Person, person_name) where person_name='alice':")
    print("  ", onehot_var)

    let emb_var = PersonEmbed[person_name]
    print("  PersonEmbed[person_name]:", emb_var)
    print("")

    print("✅ Unified syntax achieved!")
    print("   Facts: lives_in(alice, tokyo)")
    print("   Onehot: entity_onehot(Person, alice)")
    print("   Embed: PersonEmbed[alice]")
}
