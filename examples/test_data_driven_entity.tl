// Test data-driven entity collection

// Data-driven entity types (no explicit enumeration)
entity Person
entity City

// Relations with entity-typed parameters
relation knows(p1: entity, p2: entity)
relation lives_in(person: entity, city: entity)

main {
    // Add facts - entities should be automatically collected
    knows(alice, bob)
    knows(bob, charlie)
    knows(charlie, david)

    lives_in(alice, tokyo)
    lives_in(bob, london)
    lives_in(charlie, paris)
    lives_in(david, berlin)

    // Query to verify entities were collected
    knows(alice, X)?
    lives_in(Y, tokyo)?
}
