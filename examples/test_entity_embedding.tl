// Test Entity Embedding with entity types

entity Person
entity City

relation lives_in(person: Person, city: City)

// Create embeddings from entity types
embedding PersonEmbed {
    entities: Person
    dimension: 8
    init: xavier
}

embedding CityEmbed {
    entities: City
    dimension: 4
    init: random
}

main {
    // Add some entities
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    with Person {
        print("Person entities:", entity_dim(Person))
    }

    with City {
        print("City entities:", entity_dim(City))
    }

    print("✅ Entity embedding test completed!")
}
