// ============================================================================
// Graph Neural Network: Message Passing Framework
// ============================================================================
//
// GNN（Graph Neural Network）の基本的なメッセージパッシングを実装します。
//
// 背景：
//   GNNはグラフ構造化されたデータ（ソーシャルネットワーク、分子、知識グラフ等）
//   から学習するためのニューラルネットワークです。
//
// グラフ構造：
//   Node 0 -------- Node 1
//     |               |
//     |               |
//   Node 2 -------- Node 3
//
//   エッジ（隣接関係）：
//     - 0 ↔ 1 (Node 0 と Node 1 は接続)
//     - 0 ↔ 2 (Node 0 と Node 2 は接続)
//     - 1 ↔ 3 (Node 1 と Node 3 は接続)
//     - 2 ↔ 3 (Node 2 と Node 3 は接続)
//
// メッセージパッシングの数式：
//   1. Transform: h'_i = W @ h_i
//   2. Aggregate: m_i = Σ_{j∈N(i)} h'_j
//   3. Combine:   h_i^(new) = σ(h'_i + m_i)
//
//   where:
//     h_i       = ノード i の特徴ベクトル
//     W         = 学習可能な重み行列
//     N(i)      = ノード i の隣接ノード集合
//     σ         = 活性化関数（ReLU等）
//
// 処理の流れ：
//   1. 各ノードの特徴をWで変換
//   2. 隣接ノードからメッセージを集約（平均）
//   3. 自身の特徴と集約メッセージを結合
//   4. 活性化関数を適用して新しい特徴を生成
//
// 参考文献：
//   "Semi-Supervised Classification with Graph Convolutional Networks"
//   (Kipf & Welling, 2017)
//   https://arxiv.org/abs/1609.02907
// ============================================================================

main {
    print("=" * 70)
    print("Graph Neural Network: Message Passing Demonstration")
    print("=" * 70)

    // ========================================
    // グラフ構造の定義
    // ========================================
    print("\nGraph Structure:")
    print("  Number of nodes: 4")
    print("  Number of edges: 4")
    print("\n  Topology:")
    print("    Node 0 -------- Node 1")
    print("      |               |")
    print("      |               |")
    print("    Node 2 -------- Node 3")
    print("\n  Adjacency List:")
    print("    Node 0: neighbors = [1, 2]")
    print("    Node 1: neighbors = [0, 3]")
    print("    Node 2: neighbors = [0, 3]")
    print("    Node 3: neighbors = [1, 2]")

    // ========================================
    // 初期ノード特徴の定義
    // ========================================
    // Node features: [num_nodes, feature_dim] = [4, 2]
    // 各ノードは2次元の特徴ベクトルを持つ

    tensor node_0: float16[2] = [1.0, 0.0]   // 1次元目に特徴あり
    tensor node_1: float16[2] = [0.0, 1.0]   // 2次元目に特徴あり
    tensor node_2: float16[2] = [1.0, 1.0]   // 両方に特徴あり
    tensor node_3: float16[2] = [0.5, 0.5]   // バランスの取れた特徴

    print("\n" + "=" * 70)
    print("Initial Node Features (h_i)")
    print("=" * 70)
    print("\nFeature Dimension: 2")
    print("\nNode features:")
    print("  Node 0: h_0 =", node_0, "← Strong in dimension 0")
    print("  Node 1: h_1 =", node_1, "← Strong in dimension 1")
    print("  Node 2: h_2 =", node_2, "← Balanced features")
    print("  Node 3: h_3 =", node_3, "← Moderate features")

    // ========================================
    // 学習可能な重み行列
    // ========================================
    // Weight matrix: [feature_dim, feature_dim] = [2, 2]
    tensor W: float16[2, 2] learnable = [[0.5, 0.5],
                                          [0.5, 0.5]]

    print("\n" + "=" * 70)
    print("Learnable Weight Matrix (W)")
    print("=" * 70)
    print("\nWeight matrix W [2, 2]:")
    print("  W[0,0]:", W[0, 0], "  W[0,1]:", W[0, 1])
    print("  W[1,0]:", W[1, 0], "  W[1,1]:", W[1, 1])
    print("\nPurpose: Transform node features to new representation space")

    // ========================================
    // Step 1: Feature Transformation
    // ========================================
    print("\n" + "=" * 70)
    print("Step 1: Transform Features (h'_i = W @ h_i)")
    print("=" * 70)

    print("\nFormula: h'_i = W @ h_i")
    print("  This projects node features to a new space")

    tensor h_0_prime: float16[2] = W @ node_0
    tensor h_1_prime: float16[2] = W @ node_1
    tensor h_2_prime: float16[2] = W @ node_2
    tensor h_3_prime: float16[2] = W @ node_3

    print("\nTransformed features:")
    print("  h'_0 = W @ h_0 =", h_0_prime)
    print("  h'_1 = W @ h_1 =", h_1_prime)
    print("  h'_2 = W @ h_2 =", h_2_prime)
    print("  h'_3 = W @ h_3 =", h_3_prime)

    // ========================================
    // Step 2: Message Aggregation
    // ========================================
    print("\n" + "=" * 70)
    print("Step 2: Aggregate Messages from Neighbors")
    print("=" * 70)

    print("\nAggregation Function: Mean (average of neighbor features)")
    print("Formula: m_i = (1/|N(i)|) Σ_{j∈N(i)} h'_j")
    print("  where N(i) is the set of neighbors of node i")

    // For node 0: neighbors are [1, 2]
    print("\nNode 0 aggregation:")
    print("  Neighbors: [1, 2]")
    print("  Messages from:")
    print("    - Node 1:", h_1_prime)
    print("    - Node 2:", h_2_prime)

    tensor msg_to_0: float16[2] = h_1_prime + h_2_prime
    tensor agg_0: float16[2] = msg_to_0 / [2.0]  // Average

    print("  Aggregated (mean):", agg_0)

    // For node 1: neighbors are [0, 3]
    print("\nNode 1 aggregation:")
    print("  Neighbors: [0, 3]")
    print("  Messages from:")
    print("    - Node 0:", h_0_prime)
    print("    - Node 3:", h_3_prime)

    tensor msg_to_1: float16[2] = h_0_prime + h_3_prime
    tensor agg_1: float16[2] = msg_to_1 / [2.0]

    print("  Aggregated (mean):", agg_1)

    // For node 2: neighbors are [0, 3]
    print("\nNode 2 aggregation:")
    print("  Neighbors: [0, 3]")
    print("  Messages from:")
    print("    - Node 0:", h_0_prime)
    print("    - Node 3:", h_3_prime)

    tensor msg_to_2: float16[2] = h_0_prime + h_3_prime
    tensor agg_2: float16[2] = msg_to_2 / [2.0]

    print("  Aggregated (mean):", agg_2)

    // For node 3: neighbors are [1, 2]
    print("\nNode 3 aggregation:")
    print("  Neighbors: [1, 2]")
    print("  Messages from:")
    print("    - Node 1:", h_1_prime)
    print("    - Node 2:", h_2_prime)

    tensor msg_to_3: float16[2] = h_1_prime + h_2_prime
    tensor agg_3: float16[2] = msg_to_3 / [2.0]

    print("  Aggregated (mean):", agg_3)

    // ========================================
    // Step 3: Combine and Activate
    // ========================================
    print("\n" + "=" * 70)
    print("Step 3: Combine Self + Neighbors and Apply Activation")
    print("=" * 70)

    print("\nFormula: h_i^(new) = σ(h'_i + m_i)")
    print("  Combines own transformed features with neighbor messages")
    print("  σ = ReLU activation function")

    // Combine with self-features
    tensor combined_0: float16[2] = h_0_prime + agg_0
    tensor combined_1: float16[2] = h_1_prime + agg_1
    tensor combined_2: float16[2] = h_2_prime + agg_2
    tensor combined_3: float16[2] = h_3_prime + agg_3

    print("\nBefore activation (h'_i + m_i):")
    print("  Node 0:", combined_0)
    print("  Node 1:", combined_1)
    print("  Node 2:", combined_2)
    print("  Node 3:", combined_3)

    // Apply activation (ReLU)
    tensor new_0: float16[2] = relu(combined_0)
    tensor new_1: float16[2] = relu(combined_1)
    tensor new_2: float16[2] = relu(combined_2)
    tensor new_3: float16[2] = relu(combined_3)

    print("\nAfter ReLU activation:")
    print("  Node 0: h_0^(new) =", new_0)
    print("  Node 1: h_1^(new) =", new_1)
    print("  Node 2: h_2^(new) =", new_2)
    print("  Node 3: h_3^(new) =", new_3)

    // ========================================
    // まとめ
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nMessage Passing Pipeline:")
    print("  1. Transform    → h'_i = W @ h_i")
    print("  2. Aggregate    → m_i = mean({h'_j | j ∈ neighbors(i)})")
    print("  3. Combine      → combined = h'_i + m_i")
    print("  4. Activate     → h_i^(new) = ReLU(combined)")

    print("\nKey Insights:")
    print("  - Each node receives information from its neighbors")
    print("  - Aggregation function (mean) ensures scale invariance")
    print("  - Self-connection (h'_i + m_i) preserves own information")
    print("  - ReLU adds non-linearity for learning complex patterns")

    print("\nMulti-Layer GNN:")
    print("  - Stack multiple message passing layers")
    print("  - Each layer expands the receptive field by 1 hop")
    print("  - K layers → nodes gather information from K-hop neighbors")

    print("\nAggregation Variants:")
    print("  - Mean:  (1/|N|) Σ h'_j  (used here)")
    print("  - Sum:   Σ h'_j")
    print("  - Max:   max({h'_j})")
    print("  - Attention: Σ α_ij * h'_j  (Graph Attention Networks)")

    print("\nProduction Implementation Would Include:")
    print("  - Multiple GNN layers with residual connections")
    print("  - Batch processing for multiple graphs")
    print("  - Edge features and edge message passing")
    print("  - Attention mechanisms for weighted aggregation")
    print("  - Graph pooling for graph-level predictions")

    print("\n" + "=" * 70)
    print("End of Message Passing Demonstration")
    print("=" * 70)
}
