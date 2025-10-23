// Test file for unified syntax implementation
// Demonstrates the new syntax without <- and ?- prefixes

relation Parent(x: entity, y: entity)
relation Grandparent(x: entity, y: entity)

main {
    print("=== Unified Syntax Test ===")
    print("")

    // Facts: No <- prefix needed
    print("Adding facts...")
    Parent(Alice, Bob)
    Parent(Bob, Charlie)
    Parent(Bob, Diana)
    Parent(Charlie, Eve)
    print("✓ Facts added")
    print("")

    // Queries: Use ? suffix only
    print("Query 1: Parent(Alice, X)?")
    Parent(Alice, X)?
    print("")

    print("Query 2: Parent(Bob, X)?")
    Parent(Bob, X)?
    print("")

    print("Query 3: Parent(X, Charlie)?")
    Parent(X, Charlie)?
    print("")

    print("=== Test Complete ===")
}
