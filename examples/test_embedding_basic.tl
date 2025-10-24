// Test basic embedding declaration

embedding PersonEmbed {
    entities: {alice, bob, charlie}
    dimension: 4
    init: random
}

main {
    print("Embedding test")
    print("PersonEmbed created with 3 entities, dimension 4")
}
