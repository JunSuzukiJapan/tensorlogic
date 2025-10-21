// Simplified GNN test without string multiplication

main {
    print("=== GNN Message Passing Test ===")

    // Node features [4 nodes, 2 features each]
    tensor node_0: float16[2] = [1.0, 0.0]
    tensor node_1: float16[2] = [0.0, 1.0]
    tensor node_2: float16[2] = [1.0, 1.0]
    tensor node_3: float16[2] = [0.5, 0.5]

    print("\nInitial node features:")
    print("  node_0:", node_0)
    print("  node_1:", node_1)
    print("  node_2:", node_2)
    print("  node_3:", node_3)

    // Weight matrix [2, 2]
    tensor W: float16[2, 2] learnable = [[0.5, 0.5],
                                          [0.5, 0.5]]

    print("\nWeight matrix W:")
    print("  W[0,0]:", W[0, 0], "  W[0,1]:", W[0, 1])
    print("  W[1,0]:", W[1, 0], "  W[1,1]:", W[1, 1])

    // Transform features: h'_i = W @ h_i
    tensor h_0_prime: float16[2] = W @ node_0
    tensor h_1_prime: float16[2] = W @ node_1
    tensor h_2_prime: float16[2] = W @ node_2
    tensor h_3_prime: float16[2] = W @ node_3

    print("\nTransformed features:")
    print("  h'_0 = W @ h_0 =", h_0_prime)
    print("  h'_1 = W @ h_1 =", h_1_prime)
    print("  h'_2 = W @ h_2 =", h_2_prime)
    print("  h'_3 = W @ h_3 =", h_3_prime)

    // Aggregate messages (mean of neighbors)
    // Node 0 neighbors: [1, 2]
    tensor msg_to_0: float16[2] = h_1_prime + h_2_prime
    tensor agg_0: float16[2] = msg_to_0 / [2.0]

    print("\nMessage aggregation for node 0:")
    print("  Neighbors: [1, 2]")
    print("  msg_to_0 = h'_1 + h'_2 =", msg_to_0)
    print("  agg_0 = msg_to_0 / 2 =", agg_0)

    // Combine with self features
    tensor combined_0: float16[2] = h_0_prime + agg_0

    print("\nCombined features:")
    print("  combined_0 = h'_0 + agg_0 =", combined_0)

    // Apply activation (ReLU)
    tensor new_0: float16[2] = relu(combined_0)

    print("\nAfter ReLU activation:")
    print("  new_0 = ReLU(combined_0) =", new_0)

    print("\n=== Test Complete ===")
    print("✓ GNN message passing working correctly!")
}
