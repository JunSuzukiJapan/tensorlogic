// ============================================================================
// GNN Node Classification Task
// ============================================================================
//
// GNNを使用してグラフのノード分類タスクを実行します。
//
// タスク設定：
//   グラフ内の各ノードをクラスに分類する（Semi-supervised Learning）
//   一部のノードのラベルが既知、残りを予測
//
// グラフ構造：
//   Node 0 -------- Node 1
//     |               |
//     |               |
//   Node 2 -------- Node 3
//
//   Ground Truth Labels:
//     - Node 0: Class 0
//     - Node 1: Class 0
//     - Node 2: Class 1
//     - Node 3: Class 1
//
// アーキテクチャ：
//   1. GNN Layer: Message Passing で近傍情報を集約
//      h_i^(new) = ReLU(W_gnn @ h_i + Σ_{j∈N(i)} W_gnn @ h_j)
//
//   2. Classification Layer: 学習した表現をクラスに射影
//      logits_i = W_class @ h_i^(new)
//      probs_i = softmax(logits_i)
//
//   3. Loss: Cross-Entropy Loss (ここではMSEで簡略化)
//      L = (1/N) Σ_i ||probs_i - label_i||^2
//
// 学習の流れ：
//   1. Forward Pass: グラフ全体で表現を計算
//   2. Loss Computation: ラベル付きノードで損失を計算
//   3. Backward Pass: 勾配を計算
//   4. Update: パラメータを更新
//
// 参考文献：
//   "Semi-Supervised Classification with Graph Convolutional Networks"
//   (Kipf & Welling, 2017)
//   https://arxiv.org/abs/1609.02907
// ============================================================================

main {
    print("=" * 70)
    print("GNN Node Classification Task Demonstration")
    print("=" * 70)

    // ========================================
    // タスク設定
    // ========================================
    print("\nTask Setup:")
    print("  Problem: Semi-supervised node classification")
    print("  Graph: 4 nodes, 4 edges")
    print("  Classes: 2 (binary classification)")
    print("\n  Ground Truth Labels:")
    print("    Node 0 → Class 0")
    print("    Node 1 → Class 0")
    print("    Node 2 → Class 1")
    print("    Node 3 → Class 1")

    // ========================================
    // 初期ノード特徴（学習可能）
    // ========================================
    // Initial node features: [4 nodes, 2 features]
    // これらは学習可能なパラメータ（embedding）

    tensor h_0: float16[2] learnable = [1.0, 0.0]    // Class 0寄りの初期値
    tensor h_1: float16[2] learnable = [0.8, 0.2]    // Class 0寄りの初期値
    tensor h_2: float16[2] learnable = [0.2, 0.8]    // Class 1寄りの初期値
    tensor h_3: float16[2] learnable = [0.0, 1.0]    // Class 1寄りの初期値

    print("\n" + "=" * 70)
    print("Model Parameters")
    print("=" * 70)

    print("\nInitial Node Features (learnable):")
    print("  h_0 =", h_0, "← Node 0 initial embedding")
    print("  h_1 =", h_1, "← Node 1 initial embedding")
    print("  h_2 =", h_2, "← Node 2 initial embedding")
    print("  h_3 =", h_3, "← Node 3 initial embedding")

    // ========================================
    // GNN層の重み行列
    // ========================================
    // GNN weight matrix: [feature_dim, feature_dim] = [2, 2]
    tensor W_gnn: float16[2, 2] learnable = [[0.5, 0.5],
                                              [0.5, 0.5]]

    print("\nGNN Layer Weight Matrix W_gnn [2, 2]:")
    print("  Row 0:", [W_gnn[0, 0], W_gnn[0, 1]])
    print("  Row 1:", [W_gnn[1, 0], W_gnn[1, 1]])
    print("  Purpose: Transform and propagate node features")

    // ========================================
    // 分類層の重み行列
    // ========================================
    // Classification weight: [feature_dim, num_classes] = [2, 2]
    tensor W_class: float16[2, 2] learnable = [[1.0, 0.0],
                                                [0.0, 1.0]]

    print("\nClassification Layer Weight Matrix W_class [2, 2]:")
    print("  Row 0:", [W_class[0, 0], W_class[0, 1]])
    print("  Row 1:", [W_class[1, 0], W_class[1, 1]])
    print("  Purpose: Map node embeddings to class logits")

    // ========================================
    // ターゲットラベル（One-Hot）
    // ========================================
    tensor label_0: float16[2] = [1.0, 0.0]  // Class 0
    tensor label_1: float16[2] = [1.0, 0.0]  // Class 0
    tensor label_2: float16[2] = [0.0, 1.0]  // Class 1
    tensor label_3: float16[2] = [0.0, 1.0]  // Class 1

    print("\nTarget Labels (one-hot encoding):")
    print("  Node 0: label =", label_0, "← Class 0")
    print("  Node 1: label =", label_1, "← Class 0")
    print("  Node 2: label =", label_2, "← Class 1")
    print("  Node 3: label =", label_3, "← Class 1")

    // ========================================
    // Forward Pass: GNN Layer
    // ========================================
    print("\n" + "=" * 70)
    print("Forward Pass: GNN Layer")
    print("=" * 70)

    // Step 1: Transform features
    print("\nStep 1: Transform Node Features")
    print("  Formula: h'_i = W_gnn @ h_i")

    tensor h_0_t: float16[2] = W_gnn @ h_0
    tensor h_1_t: float16[2] = W_gnn @ h_1
    tensor h_2_t: float16[2] = W_gnn @ h_2
    tensor h_3_t: float16[2] = W_gnn @ h_3

    print("  Transformed features:")
    print("    h'_0 =", h_0_t)
    print("    h'_1 =", h_1_t)
    print("    h'_2 =", h_2_t)
    print("    h'_3 =", h_3_t)

    // Step 2: Message passing
    print("\nStep 2: Message Passing (Aggregate Neighbors)")
    print("  Graph adjacency:")
    print("    Node 0: neighbors = [1, 2]")
    print("    Node 1: neighbors = [0, 3]")
    print("    Node 2: neighbors = [0, 3]")
    print("    Node 3: neighbors = [1, 2]")

    // Node 0: neighbors [1, 2]
    tensor agg_0: float16[2] = (h_1_t + h_2_t) / [2.0]
    print("\n  Node 0 aggregation (mean of neighbors 1, 2):", agg_0)

    // Node 1: neighbors [0, 3]
    tensor agg_1: float16[2] = (h_0_t + h_3_t) / [2.0]
    print("  Node 1 aggregation (mean of neighbors 0, 3):", agg_1)

    // Node 2: neighbors [0, 3]
    tensor agg_2: float16[2] = (h_0_t + h_3_t) / [2.0]
    print("  Node 2 aggregation (mean of neighbors 0, 3):", agg_2)

    // Node 3: neighbors [1, 2]
    tensor agg_3: float16[2] = (h_1_t + h_2_t) / [2.0]
    print("  Node 3 aggregation (mean of neighbors 1, 2):", agg_3)

    // Step 3: Update with ReLU activation
    print("\nStep 3: Combine and Activate")
    print("  Formula: h_i^(new) = ReLU(h'_i + agg_i)")

    tensor emb_0: float16[2] = relu(h_0_t + agg_0)
    tensor emb_1: float16[2] = relu(h_1_t + agg_1)
    tensor emb_2: float16[2] = relu(h_2_t + agg_2)
    tensor emb_3: float16[2] = relu(h_3_t + agg_3)

    print("\n  Node embeddings after GNN layer:")
    print("    emb_0 =", emb_0)
    print("    emb_1 =", emb_1)
    print("    emb_2 =", emb_2)
    print("    emb_3 =", emb_3)

    print("\n  Interpretation:")
    print("    - Embeddings now contain information from neighbors")
    print("    - Similar nodes should have similar embeddings")
    print("    - These embeddings are used for classification")

    // ========================================
    // Forward Pass: Classification Layer
    // ========================================
    print("\n" + "=" * 70)
    print("Forward Pass: Classification Layer")
    print("=" * 70)

    // Step 4: Classification
    print("\nStep 4: Compute Class Logits")
    print("  Formula: logits_i = W_class @ emb_i")

    tensor logits_0: float16[2] = W_class @ emb_0
    tensor logits_1: float16[2] = W_class @ emb_1
    tensor logits_2: float16[2] = W_class @ emb_2
    tensor logits_3: float16[2] = W_class @ emb_3

    print("\n  Logits (raw scores before softmax):")
    print("    logits_0 =", logits_0)
    print("    logits_1 =", logits_1)
    print("    logits_2 =", logits_2)
    print("    logits_3 =", logits_3)

    // Apply softmax for probabilities
    print("\nStep 5: Apply Softmax (convert to probabilities)")
    print("  Formula: probs_i = softmax(logits_i)")

    tensor probs_0: float16[2] = softmax(logits_0)
    tensor probs_1: float16[2] = softmax(logits_1)
    tensor probs_2: float16[2] = softmax(logits_2)
    tensor probs_3: float16[2] = softmax(logits_3)

    print("\n  Classification probabilities:")
    print("    Node 0:", probs_0, "← Should favor Class 0 [1, 0]")
    print("    Node 1:", probs_1, "← Should favor Class 0 [1, 0]")
    print("    Node 2:", probs_2, "← Should favor Class 1 [0, 1]")
    print("    Node 3:", probs_3, "← Should favor Class 1 [0, 1]")

    print("\n  Interpretation:")
    print("    - Each probability vector sums to 1.0")
    print("    - probs[0] = probability of Class 0")
    print("    - probs[1] = probability of Class 1")

    // ========================================
    // Loss Computation
    // ========================================
    print("\n" + "=" * 70)
    print("Loss Computation")
    print("=" * 70)

    print("\nLoss Function: Mean Squared Error (simplified)")
    print("  In production, use Cross-Entropy Loss:")
    print("  L = -Σ_i Σ_c label[i,c] * log(probs[i,c])")
    print("\n  For demonstration, we use MSE:")
    print("  L = (1/N) Σ_i ||probs_i - label_i||^2")

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

    print("\n  Per-node loss:")
    print("    Node 0 loss:", loss_0)
    print("    Node 1 loss:", loss_1)
    print("    Node 2 loss:", loss_2)
    print("    Node 3 loss:", loss_3)

    tensor total_loss: float16[1] = (loss_0 + loss_1 + loss_2 + loss_3) / [4.0]

    print("\n  Total loss (average):", total_loss)
    print("    Lower is better → model predictions closer to labels")

    // ========================================
    // まとめ
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nNode Classification Pipeline:")
    print("  1. Transform        → h'_i = W_gnn @ h_i")
    print("  2. Aggregate        → m_i = mean({h'_j | j ∈ neighbors(i)})")
    print("  3. Update           → emb_i = ReLU(h'_i + m_i)")
    print("  4. Classify         → logits_i = W_class @ emb_i")
    print("  5. Softmax          → probs_i = softmax(logits_i)")
    print("  6. Loss             → L = MSE(probs, labels)")

    print("\nLearnable Parameters:")
    print("  - Node features: h_0, h_1, h_2, h_3  (8 params)")
    print("  - GNN weights: W_gnn                  (4 params)")
    print("  - Classification weights: W_class     (4 params)")
    print("  Total: 16 learnable parameters")

    print("\nTraining with TensorLogic:")
    print("  Use 'learn' block to minimize total_loss:")
    print("    learn {")
    print("        objective: total_loss,")
    print("        optimizer: adam(lr: 0.01),")
    print("        epochs: 100")
    print("    }")

    print("\nExtensions:")
    print("  - Multiple GNN layers for deeper propagation")
    print("  - Attention mechanisms for weighted aggregation")
    print("  - Dropout for regularization")
    print("  - Batch normalization for stability")
    print("  - Edge features and heterogeneous graphs")

    print("\nApplications:")
    print("  - Social network analysis (user classification)")
    print("  - Molecular property prediction (drug discovery)")
    print("  - Knowledge graph completion (link prediction)")
    print("  - Recommendation systems (user-item graphs)")

    print("\n" + "=" * 70)
    print("End of Node Classification Demonstration")
    print("=" * 70)
}
