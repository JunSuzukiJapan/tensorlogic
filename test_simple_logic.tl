relation Parent(x: entity, y: entity)

main {
    print("Simple Logic Test")
    print("")

    // Add facts
    <- Parent(Alice, Bob)
    <- Parent(Bob, Charlie)

    print("Facts added")
    print("")

    // Simple query without rules
    print("Query: Parent(Alice, X)?")
    ?- Parent(Alice, X)
}
