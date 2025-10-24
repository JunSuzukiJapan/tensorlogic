// Pure typed entity test (no generic entity type)

entity Person
entity Item
entity City

relation owns(person: Person, item: Item)
relation lives_in(person: Person, city: City)

main {
    // Ownership facts
    owns(alice, book)
    owns(bob, pen)
    owns(charlie, laptop)

    // Location facts
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    // Check entity counts
    with Person {
        // Expected: alice, bob, charlie (3 entities)
    }

    with Item {
        // Expected: book, pen, laptop (3 entities)
    }

    with City {
        // Expected: tokyo, osaka, kyoto (3 entities)
    }
}
