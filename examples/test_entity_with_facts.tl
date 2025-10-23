// Entity declaration with facts and queries

// Define entity types
entity Person = {alice, bob, charlie, david}
entity City = {tokyo, london, paris}

// Define relations
relation knows(p1: entity, p2: entity)
relation lives_in(person: entity, city: entity)

main {
    // Add facts
    knows(alice, bob)
    knows(bob, charlie)
    knows(charlie, david)

    lives_in(alice, tokyo)
    lives_in(bob, london)
    lives_in(charlie, paris)

    // Query with entity scoping
    with Person {
        knows(alice, bob)?
        knows(bob, X)?
    }

    with City {
        lives_in(alice, Y)?
    }
}
