// Test entity-tensor conversion functions

entity Person
entity City

relation lives_in(person: Person, city: City)

main {
    // Add some entities
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    with Person {
        // Test entity_dim() - should return 3
        let person_dim = entity_dim("Person")
        print("Person dimension:", person_dim)

        // Test entity_onehot() for each person
        let alice_vec = entity_onehot("Person", "alice")
        print("Alice one-hot:", alice_vec)

        let bob_vec = entity_onehot("Person", "bob")
        print("Bob one-hot:", bob_vec)

        let charlie_vec = entity_onehot("Person", "charlie")
        print("Charlie one-hot:", charlie_vec)
    }

    with City {
        // Test entity_dim() - should return 3
        let city_dim = entity_dim("City")
        print("City dimension:", city_dim)

        // Test entity_onehot() for cities
        let tokyo_vec = entity_onehot("City", "tokyo")
        print("Tokyo one-hot:", tokyo_vec)
    }
}
