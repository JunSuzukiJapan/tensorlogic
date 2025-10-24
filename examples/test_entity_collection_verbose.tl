// Test data-driven entity collection with verification

// Mixed entity types: explicit and data-driven
entity Color = {red, green, blue}  // Explicit
entity Animal                       // Data-driven

// Relations
relation likes(animal: entity, color: entity)
relation friend(a1: entity, a2: entity)

main {
    // Add facts about animals liking colors
    // Animals should be auto-collected into Animal entity type
    // Colors are already explicitly defined
    likes(cat, red)
    likes(dog, blue)
    likes(bird, green)
    likes(cat, green)

    // Add facts about animal friendships
    friend(cat, dog)
    friend(dog, bird)

    // Queries
    likes(cat, X)?      // What does cat like?
    likes(Y, blue)?     // What likes blue?
    friend(cat, Z)?     // Who is cat's friend?
}
