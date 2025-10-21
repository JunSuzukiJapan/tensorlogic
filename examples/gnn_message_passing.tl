// Graph Neural Network: Message Passing
// Demonstrates basic GNN operations on a simple graph

// Graph structure:
//   Node 0 -- Node 1
//     |         |
//   Node 2 -- Node 3
//
// Adjacency: 0-1, 0-2, 1-3, 2-3

main {
    // Node features: [num_nodes, feature_dim] = [4, 2]
    tensor node_0: float16[2] = [1.0, 0.0]
    tensor node_1: float16[2] = [0.0, 1.0]
    tensor node_2: float16[2] = [1.0, 1.0]
    tensor node_3: float16[2] = [0.5, 0.5]

    print("=== Graph Neural Network: Message Passing ===")
    print("Initial node features:")
    print("  Node 0:", node_0)
    print("  Node 1:", node_1)
    print("  Node 2:", node_2)
    print("  Node 3:", node_3)

    // Learnable weight matrix: [feature_dim, feature_dim] = [2, 2]
    tensor W: float16[2, 2] learnable = [[0.5, 0.5],
                                          [0.5, 0.5]]

    print("Weight matrix W:", W)

    // Step 1: Transform features with W
    // h'_i = W @ h_i
    tensor h_0_prime: float16[2] = W @ node_0
    tensor h_1_prime: float16[2] = W @ node_1
    tensor h_2_prime: float16[2] = W @ node_2
    tensor h_3_prime: float16[2] = W @ node_3

    print("Transformed features (W @ h_i):")
    print("  h'_0:", h_0_prime)
    print("  h'_1:", h_1_prime)
    print("  h'_2:", h_2_prime)
    print("  h'_3:", h_3_prime)

    // Step 2: Aggregate messages from neighbors
    // For node 0: neighbors are [1, 2]
    tensor msg_to_0: float16[2] = h_1_prime + h_2_prime
    tensor agg_0: float16[2] = msg_to_0 / [2.0]  // Average

    // For node 1: neighbors are [0, 3]
    tensor msg_to_1: float16[2] = h_0_prime + h_3_prime
    tensor agg_1: float16[2] = msg_to_1 / [2.0]

    // For node 2: neighbors are [0, 3]
    tensor msg_to_2: float16[2] = h_0_prime + h_3_prime
    tensor agg_2: float16[2] = msg_to_2 / [2.0]

    // For node 3: neighbors are [1, 2]
    tensor msg_to_3: float16[2] = h_1_prime + h_2_prime
    tensor agg_3: float16[2] = msg_to_3 / [2.0]

    print("Aggregated messages (mean of neighbors):")
    print("  Agg_0:", agg_0)
    print("  Agg_1:", agg_1)
    print("  Agg_2:", agg_2)
    print("  Agg_3:", agg_3)

    // Step 3: Combine with self-features
    // h_i^(new) = σ(h'_i + agg_i)
    tensor combined_0: float16[2] = h_0_prime + agg_0
    tensor combined_1: float16[2] = h_1_prime + agg_1
    tensor combined_2: float16[2] = h_2_prime + agg_2
    tensor combined_3: float16[2] = h_3_prime + agg_3

    // Apply activation (ReLU)
    tensor new_0: float16[2] = relu(combined_0)
    tensor new_1: float16[2] = relu(combined_1)
    tensor new_2: float16[2] = relu(combined_2)
    tensor new_3: float16[2] = relu(combined_3)

    print("Updated node features (after one GNN layer):")
    print("  Node 0:", new_0)
    print("  Node 1:", new_1)
    print("  Node 2:", new_2)
    print("  Node 3:", new_3)

    print("Note: This demonstrates basic message passing")
    print("      Real GNN would iterate multiple layers")
}
