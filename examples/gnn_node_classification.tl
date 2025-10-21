// GNN Node Classification Training
// Train a simple GNN to classify nodes in a graph

// Graph: 4 nodes, 2 classes
// Node 0, 1 -> Class 0
// Node 2, 3 -> Class 1

main {
    // Initial node features: [4, 2]
    tensor h_0: float16[2] learnable = [1.0, 0.0]
    tensor h_1: float16[2] learnable = [0.8, 0.2]
    tensor h_2: float16[2] learnable = [0.2, 0.8]
    tensor h_3: float16[2] learnable = [0.0, 1.0]

    // GNN weight matrix: [2, 2]
    tensor W_gnn: float16[2, 2] learnable = [[0.5, 0.5],
                                              [0.5, 0.5]]

    // Classification weight: [2, 2] (feature_dim -> num_classes)
    tensor W_class: float16[2, 2] learnable = [[1.0, 0.0],
                                                [0.0, 1.0]]

    // Target labels (one-hot)
    tensor label_0: float16[2] = [1.0, 0.0]  // Class 0
    tensor label_1: float16[2] = [1.0, 0.0]  // Class 0
    tensor label_2: float16[2] = [0.0, 1.0]  // Class 1
    tensor label_3: float16[2] = [0.0, 1.0]  // Class 1

    print("=== GNN Node Classification Training ===")

    // === Forward Pass ===

    // Step 1: Transform features
    tensor h_0_t: float16[2] = W_gnn @ h_0
    tensor h_1_t: float16[2] = W_gnn @ h_1
    tensor h_2_t: float16[2] = W_gnn @ h_2
    tensor h_3_t: float16[2] = W_gnn @ h_3

    // Step 2: Message passing (same graph as before)
    // Node 0: neighbors [1, 2]
    tensor agg_0: float16[2] = (h_1_t + h_2_t) / [2.0]
    // Node 1: neighbors [0, 3]
    tensor agg_1: float16[2] = (h_0_t + h_3_t) / [2.0]
    // Node 2: neighbors [0, 3]
    tensor agg_2: float16[2] = (h_0_t + h_3_t) / [2.0]
    // Node 3: neighbors [1, 2]
    tensor agg_3: float16[2] = (h_1_t + h_2_t) / [2.0]

    // Step 3: Update with ReLU
    tensor emb_0: float16[2] = relu(h_0_t + agg_0)
    tensor emb_1: float16[2] = relu(h_1_t + agg_1)
    tensor emb_2: float16[2] = relu(h_2_t + agg_2)
    tensor emb_3: float16[2] = relu(h_3_t + agg_3)

    print("Node embeddings after GNN:")
    print("  Node 0:", emb_0)
    print("  Node 1:", emb_1)
    print("  Node 2:", emb_2)
    print("  Node 3:", emb_3)

    // Step 4: Classification
    tensor logits_0: float16[2] = W_class @ emb_0
    tensor logits_1: float16[2] = W_class @ emb_1
    tensor logits_2: float16[2] = W_class @ emb_2
    tensor logits_3: float16[2] = W_class @ emb_3

    // Apply softmax for probabilities
    tensor probs_0: float16[2] = softmax(logits_0)
    tensor probs_1: float16[2] = softmax(logits_1)
    tensor probs_2: float16[2] = softmax(logits_2)
    tensor probs_3: float16[2] = softmax(logits_3)

    print("Classification probabilities:")
    print("  Node 0 (should be Class 0):", probs_0)
    print("  Node 1 (should be Class 0):", probs_1)
    print("  Node 2 (should be Class 1):", probs_2)
    print("  Node 3 (should be Class 1):", probs_3)

    // === Loss Computation ===
    // Cross-entropy loss (simplified MSE for demonstration)
    tensor diff_0: float16[2] = probs_0 - label_0
    tensor diff_1: float16[2] = probs_1 - label_1
    tensor diff_2: float16[2] = probs_2 - label_2
    tensor diff_3: float16[2] = probs_3 - label_3

    tensor sq_0: float16[2] = diff_0 * diff_0
    tensor sq_1: float16[2] = diff_1 * diff_1
    tensor sq_2: float16[2] = diff_2 * diff_2
    tensor sq_3: float16[2] = diff_3 * diff_3

    tensor loss_0: float16[1] = mean(sq_0)
    tensor loss_1: float16[1] = mean(sq_1)
    tensor loss_2: float16[1] = mean(sq_2)
    tensor loss_3: float16[1] = mean(sq_3)

    tensor total_loss: float16[1] = (loss_0 + loss_1 + loss_2 + loss_3) / [4.0]

    print("Loss per node:")
    print("  Node 0:", loss_0)
    print("  Node 1:", loss_1)
    print("  Node 2:", loss_2)
    print("  Node 3:", loss_3)
    print("Total loss:", total_loss)

    // Training would use 'learn' block to minimize total_loss
    print("Note: Use 'learn' block with optimizer to train this GNN")
}
