// Entity declaration syntax test

// Explicit enumeration (with =)
entity Person = {alice, bob, charlie}

// Data-driven construction (without =)
entity City

main {
    // With block for entity collection
    with Person {
        knows(alice, bob)?
        knows(bob, charlie)?
    }
}
