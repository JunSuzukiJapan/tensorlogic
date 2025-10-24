// Detailed typed entity test with queries

entity Person
entity Item
entity City  // Should remain empty
entity Animal  // For testing generic entity type

relation owns(person: Person, item: Item)
relation lives_in(person: Person, city: City)
relation likes(x: entity, y: entity)  // Generic entity type - backward compatibility

main {
    // Add some ownership facts
    owns(alice, book)
    owns(bob, pen)
    owns(charlie, laptop)
    owns(alice, pen)

    // Add location facts
    lives_in(alice, tokyo)
    lives_in(bob, osaka)

    // Test backward compatibility with generic entity type
    likes(alice, cat)
    likes(bob, dog)

    // Verify collection with queries
    with Person {
        owns(alice, X)?
        owns(bob, Y)?
    }

    with Item {
        owns(Who, book)?
    }

    with City {
        lives_in(Who, tokyo)?
    }

    // Animal should have cat and dog (from generic entity collection)
    with Animal {
    }
}
