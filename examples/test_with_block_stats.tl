// Test with-block entity statistics display

// Data-driven entity type
entity Person
entity Item

// Relations
relation owns(person: entity, item: entity)

main {
    // Add some initial facts
    owns(alice, book)
    owns(alice, pen)

    // With-block should show entity statistics
    with Person {
        owns(bob, laptop)?
        owns(alice, X)?

        // Add new fact in with-block
        owns(charlie, phone)
        owns(charlie, Y)?
    }

    with Item {
        owns(Z, book)?

        // Add more items
        owns(david, tablet)
        owns(david, W)?
    }
}
