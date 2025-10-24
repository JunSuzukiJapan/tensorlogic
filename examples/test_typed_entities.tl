// Typed entity declaration test
// Tests that entities are collected to specific types

// Data-driven entity types
entity Person
entity Item

// Relation with typed parameters
relation owns(person: Person, item: Item)

main {
    // Facts - alice/bob should go to Person, book/pen should go to Item
    owns(alice, book)
    owns(bob, pen)
    owns(alice, pen)

    // Verify entity collection with with-blocks
    with Person {
        // Should show alice, bob
    }

    with Item {
        // Should show book, pen
    }
}
