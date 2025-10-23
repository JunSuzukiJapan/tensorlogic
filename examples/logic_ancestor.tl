// Logic Programming Example: Ancestor Relationships
//
// This example demonstrates logic programming with:
// - Relation declarations
// - Rule definitions (using <- for rules)
// - Fact assertions (no prefix needed)
// - Queries (using ? suffix)

// ============================================================================
// Relation Declarations
// ============================================================================

// Define Parent relation (binary relation between entities)
relation Parent(x: entity, y: entity)

// ============================================================================
// Rule Definitions
// ============================================================================

// Rule 1: Direct ancestry - If X is a parent of Y, then X is an ancestor of Y
Ancestor(X, Y) <- Parent(X, Y)

// Rule 2: Transitive ancestry - If X is an ancestor of Y and Y is a parent of Z,
//         then X is an ancestor of Z
Ancestor(X, Z) <- Ancestor(X, Y), Parent(Y, Z)

// ============================================================================
// Main Execution Block
// ============================================================================

main {
    print("Logic Programming Example: Ancestor Relationships")
    print("")

    // ========================================================================
    // Fact Assertions: Add facts to the knowledge base
    // ========================================================================
    print("Adding facts to knowledge base:")

    Parent(Alice, Bob)
    Parent(Bob, Charlie)
    Parent(Charlie, David)

    print("")

    // ========================================================================
    // Queries: Ask questions about the knowledge base
    // ========================================================================
    print("Running queries:")
    print("")

    // Query 1: Find all ancestors of Charlie
    print("Query 1: Who are the ancestors of Charlie?")
    Ancestor(X, Charlie)?
    print("")

    // Query 2: Check if Alice is an ancestor of David
    print("Query 2: Is Alice an ancestor of David?")
    Ancestor(Alice, David)?
    print("")

    // Query 3: Find all descendants of Alice
    print("Query 3: Who are the descendants of Alice?")
    Ancestor(Alice, Y)?
    print("")

    // Query 4: Check direct parent relationship
    print("Query 4: Is Bob a parent of Charlie?")
    Parent(Bob, Charlie)?
}
