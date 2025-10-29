relation Parent(x: entity, y: entity)

main {
    print("Simple Logic Test")
    print("")

    // Add facts (no <- prefix)
    Parent(Alice, Bob)
    Parent(Bob, Charlie)

    print("Facts added")
    print("")

    // Simple query (? suffix only)
    print("Query: Parent(Alice, X)?")
    Parent(Alice, X)?
}
