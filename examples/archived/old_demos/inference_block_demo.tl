// Inference Block Demo
// Demonstrates the new inference block syntax

// Define relations for family relationships
relation Parent(x: entity, y: entity)
relation Sibling(x: entity, y: entity)
relation Ancestor(x: entity, y: entity)

// Define a relation with embeddings for neural-symbolic learning
relation Friend(x: entity, y: entity) embed float16[32]

main {
    print("=== Inference Block Demo ===")
    print("")

    // Traditional single inference statements (still supported)
    print("Method 1: Single inference statements")
    infer forward Parent(alice, X)?
    infer backward Parent(X, Y)?
    print("")

    // New inference block syntax
    print("Method 2: Inference block (recommended for multiple inferences)")
    infer {
        // Forward inference: Logic → Tensor
        // Query the logic engine and convert results to tensor representation
        forward Parent(alice, X)?

        // Backward inference: Tensor → Logic
        // Use neural network predictions to create logic facts
        backward Sibling(X, Y)?

        // Gradient inference: Differentiable logic
        // Propagate gradients through logic operations for learning
        gradient Friend(alice, bob)?

        // Symbolic inference: Symbolic reasoning
        // Apply symbolic rules and transformations
        symbolic Ancestor(alice, Z)?
    }

    print("")
    print("=== Inference Block Benefits ===")
    print("  • Cleaner syntax for multiple inferences")
    print("  • Clear grouping of related operations")
    print("  • Consistent with 'learn' block style")
    print("  • Sequential execution from top to bottom")
    print("")

    // Another example with different inference methods
    print("Example: Mixed inference methods in one block")
    infer {
        forward Parent(bob, X)?
        gradient Friend(bob, charlie)?
        symbolic Ancestor(X, Y)?
    }

    print("")
    print("✅ Inference block demo completed!")
}
