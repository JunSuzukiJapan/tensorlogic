// Test Entity Embedding with explicit entities first

entity Person
entity City

relation lives_in(person: Person, city: City)

// Create embeddings with explicit entities (known ahead of time)
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

    with Person {
        print("Person entities:", entity_dim("Person"))
    }

    with City {
        print("City entities:", entity_dim("City"))
    }

    print("✅ Entity embedding test completed!")
}
