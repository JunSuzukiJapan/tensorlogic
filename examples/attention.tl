// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================
//
// Transformerの中核となるAttentionメカニズムを実装します。
//
// 背景：
//   Attentionは、入力シーケンスの各位置が他の位置をどれだけ「注目」すべきかを
//   学習します。これにより、長距離依存関係を効率的に捉えることができます。
//
// 数式：
//   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
//   where:
//     Q = Query  (何を探しているか)
//     K = Key    (何を提供できるか)
//     V = Value  (実際の情報)
//     d_k = Key の次元数
//
// 処理の流れ：
//   1. Q と K の類似度を計算 (内積)
//   2. sqrt(d_k) でスケーリング (勾配の安定化)
//   3. softmax で正規化 (重みの合計 = 1)
//   4. V の重み付き和を計算
//
// 参考文献：
//   "Attention is All You Need" (Vaswani et al., 2017)
//   https://arxiv.org/abs/1706.03762
// ============================================================================

main {
    print("=" * 70)
    print("Scaled Dot-Product Attention Demonstration")
    print("=" * 70)

    // ========================================
    // 入力テンソルの定義
    // ========================================
    // シーケンス長: 3 (3つの単語/トークン)
    // Key/Query次元: 2
    // Value次元: 2

    // Query: "各位置が何を探しているか"
    // [seq_len, d_k] = [3, 2]
    tensor Q: float16[3, 2] = [[1.0, 0.0],   // Position 0: 1次元目に注目
                                [0.0, 1.0],   // Position 1: 2次元目に注目
                                [1.0, 1.0]]   // Position 2: 両方に注目

    // Key: "各位置が何を提供できるか"
    // [seq_len, d_k] = [3, 2]
    tensor K: float16[3, 2] = [[1.0, 0.0],   // Position 0: 1次元目の情報を持つ
                                [0.0, 1.0],   // Position 1: 2次元目の情報を持つ
                                [0.5, 0.5]]   // Position 2: バランスの取れた情報

    // Value: "実際の情報内容"
    // [seq_len, d_v] = [3, 2]
    tensor V: float16[3, 2] = [[2.0, 0.0],   // Position 0 の値
                                [0.0, 2.0],   // Position 1 の値
                                [1.0, 1.0]]   // Position 2 の値

    print("\nInput Tensors:")
    print("-" * 70)
    print("Query (Q) - What each position is looking for:")
    print("  Position 0:", [Q[0, 0], Q[0, 1]], "→ Looking for dimension 0")
    print("  Position 1:", [Q[1, 0], Q[1, 1]], "→ Looking for dimension 1")
    print("  Position 2:", [Q[2, 0], Q[2, 1]], "→ Looking for both")

    print("\nKey (K) - What each position can provide:")
    print("  Position 0:", [K[0, 0], K[0, 1]], "→ Provides dimension 0")
    print("  Position 1:", [K[1, 0], K[1, 1]], "→ Provides dimension 1")
    print("  Position 2:", [K[2, 0], K[2, 1]], "→ Provides balanced info")

    print("\nValue (V) - Actual information content:")
    print("  Position 0:", [V[0, 0], V[0, 1]])
    print("  Position 1:", [V[1, 0], V[1, 1]])
    print("  Position 2:", [V[2, 0], V[2, 1]])

    // ========================================
    // Step 1: Attention Scores (Q @ K^T)
    // ========================================
    print("\n" + "=" * 70)
    print("Step 1: Compute Attention Scores (Q @ K^T)")
    print("=" * 70)

    tensor K_T: float16[2, 3] = transpose(K)
    tensor scores: float16[3, 3] = Q @ K_T

    print("\nAttention scores matrix [3, 3]:")
    print("  (how much each Query attends to each Key)")
    print("\n  scores[i,j] = similarity between Q[i] and K[j]")
    print("\n         Key0  Key1  Key2")
    print("  Query0", [scores[0, 0], scores[0, 1], scores[0, 2]])
    print("  Query1", [scores[1, 0], scores[1, 1], scores[1, 2]])
    print("  Query2", [scores[2, 0], scores[2, 1], scores[2, 2]])

    print("\nInterpretation:")
    print("  - scores[0,0] = 1.0: Query0 strongly matches Key0")
    print("  - scores[1,1] = 1.0: Query1 strongly matches Key1")
    print("  - scores[2,2] = 0.5: Query2 moderately matches Key2")

    // ========================================
    // Step 2: Scaling by sqrt(d_k)
    // ========================================
    print("\n" + "=" * 70)
    print("Step 2: Scale by sqrt(d_k)")
    print("=" * 70)

    tensor d_k: float16[1] = [2.0]
    tensor sqrt_d_k: float16[1] = sqrt(d_k)
    tensor scale: float16[1] = [1.0] / sqrt_d_k

    print("\nd_k =", d_k)
    print("sqrt(d_k) =", sqrt_d_k)
    print("scale factor = 1/sqrt(d_k) =", scale)

    print("\nWhy scale?")
    print("  - Large d_k → Large dot products → Softmax saturation")
    print("  - Scaling prevents gradients from becoming too small")

    // Extract rows for scaling
    tensor scores_row0: float16[3] = [scores[0, 0], scores[0, 1], scores[0, 2]]
    tensor scores_row1: float16[3] = [scores[1, 0], scores[1, 1], scores[1, 2]]
    tensor scores_row2: float16[3] = [scores[2, 0], scores[2, 1], scores[2, 2]]

    // Scale each row
    tensor scaled_0: float16[3] = scores_row0 * scale
    tensor scaled_1: float16[3] = scores_row1 * scale
    tensor scaled_2: float16[3] = scores_row2 * scale

    print("\nScaled scores:")
    print("  Row 0:", scaled_0)
    print("  Row 1:", scaled_1)
    print("  Row 2:", scaled_2)

    // ========================================
    // Step 3: Apply Softmax
    // ========================================
    print("\n" + "=" * 70)
    print("Step 3: Apply Softmax (normalize attention weights)")
    print("=" * 70)

    tensor weights_0: float16[3] = softmax(scaled_0)
    tensor weights_1: float16[3] = softmax(scaled_1)
    tensor weights_2: float16[3] = softmax(scaled_2)

    print("\nAttention weights (sum = 1.0 for each row):")
    print("  Position 0 attends to:", weights_0)
    print("  Position 1 attends to:", weights_1)
    print("  Position 2 attends to:", weights_2)

    print("\nInterpretation:")
    print("  - Each row shows how much that position attends to others")
    print("  - Larger values = stronger attention")
    print("  - Sum of each row = 1.0 (probability distribution)")

    // ========================================
    // Step 4: Weighted Sum of Values
    // ========================================
    print("\n" + "=" * 70)
    print("Step 4: Compute Weighted Sum of Values")
    print("=" * 70)

    print("\nFor demonstration, we compute output for position 0:")
    print("  output[0] = Σ weights_0[j] * V[j]")

    // Extract value vectors
    tensor V_0: float16[2] = [V[0, 0], V[0, 1]]
    tensor V_1: float16[2] = [V[1, 0], V[1, 1]]
    tensor V_2: float16[2] = [V[2, 0], V[2, 1]]

    // Extract weights for position 0
    tensor w0_0: float16[1] = [weights_0[0]]
    tensor w0_1: float16[1] = [weights_0[1]]
    tensor w0_2: float16[1] = [weights_0[2]]

    print("\nWeights for position 0:")
    print("  w[0] =", w0_0, "→ attention to position 0")
    print("  w[1] =", w0_1, "→ attention to position 1")
    print("  w[2] =", w0_2, "→ attention to position 2")

    // Weighted contributions
    tensor contrib_0: float16[2] = V_0 * w0_0
    tensor contrib_1: float16[2] = V_1 * w0_1
    tensor contrib_2: float16[2] = V_2 * w0_2

    print("\nWeighted value contributions:")
    print("  From position 0:", contrib_0)
    print("  From position 1:", contrib_1)
    print("  From position 2:", contrib_2)

    // Final output for position 0
    tensor output_0: float16[2] = contrib_0 + contrib_1 + contrib_2

    print("\nFinal attention output for position 0:", output_0)
    print("  This is the context-aware representation!")

    // ========================================
    // Summary
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nAttention Mechanism Pipeline:")
    print("  1. Q @ K^T        → Compute similarity scores")
    print("  2. / sqrt(d_k)    → Scale for stability")
    print("  3. softmax(...)   → Normalize to probabilities")
    print("  4. @ V            → Weighted sum of values")

    print("\nKey Insights:")
    print("  - Attention learns which positions are relevant to each other")
    print("  - Softmax ensures all attention weights sum to 1")
    print("  - Output is a weighted combination of all input values")
    print("  - Each position can attend to any other position")

    print("\nIn Practice:")
    print("  - Q, K, V are computed via learned linear projections")
    print("  - Multi-head attention uses multiple Q/K/V sets in parallel")
    print("  - Attention can be masked to prevent looking at future tokens")

    print("\n" + "=" * 70)
    print("End of Attention Demonstration")
    print("=" * 70)
}
