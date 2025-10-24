// Test Relation Embedding

entity Person
entity City

relation lives_in(person: Person, city: City)
relation owns(person: Person, item: City)
relation friend_of(person: Person, friend: Person)

// Create relation embedding with explicit relations
relation_embedding RelEmbed {
    relations: {lives_in, owns, friend_of}
    dimension: 8
    init: xavier
}

main {
    // Add some facts
    lives_in(alice, tokyo)
    owns(bob, laptop)
    friend_of(charlie, alice)

    print("✅ Relation embedding test completed!")
}
