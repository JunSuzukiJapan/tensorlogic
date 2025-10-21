// ============================================================================
// Transformer Block Architecture
// ============================================================================
//
// Transformerブロックの基本構造を実装します。
//
// 背景：
//   Transformerブロックは、Self-Attention層とFeed-Forward層を
//   Residual ConnectionとLayer Normalizationで組み合わせた構造です。
//
// アーキテクチャ：
//   1. Multi-Head Self-Attention
//      - 入力から Q, K, V を生成
//      - Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
//   2. Residual Connection + Layer Norm
//      - LayerNorm(X + Attention(X))
//
//   3. Feed-Forward Network
//      - FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
//
//   4. Residual Connection + Layer Norm
//      - LayerNorm(X + FFN(X))
//
// 数式（完全な実装）：
//   Output = LayerNorm(X + FFN(LayerNorm(X + Attention(X))))
//
// 注意：
//   このデモは簡略化されており、教育目的です。
//   実際の実装では、Q/K/V射影、マルチヘッド、完全な正規化が必要です。
//
// 参考文献：
//   "Attention is All You Need" (Vaswani et al., 2017)
//   https://arxiv.org/abs/1706.03762
// ============================================================================

main {
    print("=" * 70)
    print("Transformer Block Architecture Demonstration")
    print("=" * 70)

    // ========================================
    // 入力データの定義
    // ========================================
    // Input sequence: [seq_len, d_model] = [2, 4]
    // seq_len = 2: 2つのトークン（単語）
    // d_model = 4: 各トークンの特徴次元数

    tensor X: float16[2, 4] = [[1.0, 0.5, 0.0, 0.5],   // Token 0
                                [0.0, 1.0, 0.5, 0.0]]   // Token 1

    print("\nInput Configuration:")
    print("  Sequence Length (seq_len): 2 tokens")
    print("  Model Dimension (d_model): 4 features per token")
    print("\nInput X [2, 4]:")
    print("  Token 0:", [X[0, 0], X[0, 1], X[0, 2], X[0, 3]])
    print("  Token 1:", [X[1, 0], X[1, 1], X[1, 2], X[1, 3]])

    // ========================================
    // Component 1: Self-Attention
    // ========================================
    print("\n" + "=" * 70)
    print("Component 1: Self-Attention Mechanism")
    print("=" * 70)

    print("\nNote: In production, we would compute:")
    print("  Q = X @ W_q  (Query projection)")
    print("  K = X @ W_k  (Key projection)")
    print("  V = X @ W_v  (Value projection)")
    print("  For simplicity, we use X directly as Q, K, V here")

    // Compute attention scores: X @ X^T
    // これは各トークンペア間の類似度を計算
    tensor X_T: float16[4, 2] = transpose(X)
    tensor scores: float16[2, 2] = X @ X_T

    print("\nStep 1.1: Compute Attention Scores (X @ X^T)")
    print("  Scores [2, 2] (similarity between tokens):")
    print("    scores[0,0] =", [scores[0, 0]], "← Token 0 similarity with itself")
    print("    scores[0,1] =", [scores[0, 1]], "← Token 0 similarity with Token 1")
    print("    scores[1,0] =", [scores[1, 0]], "← Token 1 similarity with Token 0")
    print("    scores[1,1] =", [scores[1, 1]], "← Token 1 similarity with itself")

    // Scale by sqrt(d_model)
    tensor d_model: float16[1] = [4.0]
    tensor sqrt_d: float16[1] = sqrt(d_model)
    tensor scale: float16[1] = [1.0] / sqrt_d

    print("\nStep 1.2: Scale by sqrt(d_model)")
    print("  d_model =", d_model)
    print("  sqrt(d_model) =", sqrt_d)
    print("  scale factor = 1/sqrt(d_model) =", scale)
    print("  Why? Prevents softmax saturation for large d_model")

    // Extract rows for scaling
    tensor scores_0: float16[2] = [scores[0, 0], scores[0, 1]]
    tensor scores_1: float16[2] = [scores[1, 0], scores[1, 1]]

    tensor scaled_0: float16[2] = scores_0 * scale
    tensor scaled_1: float16[2] = scores_1 * scale

    print("\n  Scaled scores:")
    print("    Token 0:", scaled_0)
    print("    Token 1:", scaled_1)

    // Apply softmax
    tensor attn_weights_0: float16[2] = softmax(scaled_0)
    tensor attn_weights_1: float16[2] = softmax(scaled_1)

    print("\nStep 1.3: Apply Softmax (normalize to probabilities)")
    print("  Attention weights (sum = 1.0 for each row):")
    print("    Token 0 attends to:", attn_weights_0)
    print("    Token 1 attends to:", attn_weights_1)

    print("\n  Interpretation:")
    print("    - Each row shows how much that token attends to others")
    print("    - Higher values = stronger attention")
    print("    - Sum of each row = 1.0 (probability distribution)")

    // Simplified attention output
    // 実際の実装では: attn_output = attention_weights @ V
    tensor attn_out: float16[2, 4] = X  // Placeholder for demonstration

    print("\nStep 1.4: Compute Attention Output")
    print("  In production: attn_output = attention_weights @ V")
    print("  For this demo: using X as placeholder")
    print("  Attention output [2, 4]:")
    print("    Token 0:", [attn_out[0, 0], attn_out[0, 1], attn_out[0, 2], attn_out[0, 3]])
    print("    Token 1:", [attn_out[1, 0], attn_out[1, 1], attn_out[1, 2], attn_out[1, 3]])

    // ========================================
    // Component 2: Residual Connection + Layer Norm
    // ========================================
    print("\n" + "=" * 70)
    print("Component 2: Residual Connection + Layer Normalization")
    print("=" * 70)

    tensor residual_1: float16[2, 4] = X + attn_out

    print("\nStep 2.1: Add Residual Connection")
    print("  Formula: X + Attention(X)")
    print("  Why? Enables gradient flow, prevents degradation")
    print("\n  After residual [2, 4]:")
    print("    Token 0:", [residual_1[0, 0], residual_1[0, 1], residual_1[0, 2], residual_1[0, 3]])
    print("    Token 1:", [residual_1[1, 0], residual_1[1, 1], residual_1[1, 2], residual_1[1, 3]])

    // Layer Normalization (demonstration)
    tensor pos_0: float16[4] = [residual_1[0, 0], residual_1[0, 1],
                                 residual_1[0, 2], residual_1[0, 3]]

    tensor mean_0: float16[1] = mean(pos_0)

    print("\nStep 2.2: Layer Normalization (simplified)")
    print("  Formula: LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)")
    print("  For Token 0:")
    print("    Features:", pos_0)
    print("    Mean:", mean_0)
    print("\n  Note: Full implementation would:")
    print("    - Compute variance")
    print("    - Normalize: (x - mean) / sqrt(variance + eps)")
    print("    - Apply learnable scale and shift: gamma * norm(x) + beta")

    // ========================================
    // Component 3: Feed-Forward Network
    // ========================================
    print("\n" + "=" * 70)
    print("Component 3: Feed-Forward Network (FFN)")
    print("=" * 70)

    print("\nFFN Architecture:")
    print("  FFN(x) = max(0, x @ W1 + b1) @ W2 + b2")
    print("  Where:")
    print("    - First layer: Linear transformation + ReLU")
    print("    - Second layer: Linear transformation")
    print("    - Typical dimension: d_model → 4*d_model → d_model")

    tensor W1: float16[4, 4] learnable = [[0.5, 0.5, 0.0, 0.0],
                                           [0.0, 0.5, 0.5, 0.0],
                                           [0.0, 0.0, 0.5, 0.5],
                                           [0.5, 0.0, 0.0, 0.5]]

    print("\nStep 3.1: First Linear Layer")
    print("  Weight matrix W1 [4, 4]:")
    print("    (In practice, this would be [d_model, 4*d_model])")

    // Apply FFN to normalized output
    tensor ffn_out: float16[2, 4] = residual_1 @ W1

    print("\n  FFN linear output [2, 4]:")
    print("    Token 0:", [ffn_out[0, 0], ffn_out[0, 1], ffn_out[0, 2], ffn_out[0, 3]])
    print("    Token 1:", [ffn_out[1, 0], ffn_out[1, 1], ffn_out[1, 2], ffn_out[1, 3]])

    // Apply ReLU activation
    tensor ffn_activated: float16[2, 4] = relu(ffn_out)

    print("\nStep 3.2: Apply ReLU Activation")
    print("  Formula: ReLU(x) = max(0, x)")
    print("\n  After ReLU [2, 4]:")
    print("    Token 0:", [ffn_activated[0, 0], ffn_activated[0, 1], ffn_activated[0, 2], ffn_activated[0, 3]])
    print("    Token 1:", [ffn_activated[1, 0], ffn_activated[1, 1], ffn_activated[1, 2], ffn_activated[1, 3]])

    print("\n  Note: In production, we would apply a second linear layer:")
    print("    FFN(x) = ReLU(x @ W1) @ W2")

    // ========================================
    // Component 4: Final Residual Connection
    // ========================================
    print("\n" + "=" * 70)
    print("Component 4: Final Residual Connection")
    print("=" * 70)

    tensor output: float16[2, 4] = residual_1 + ffn_activated

    print("\nStep 4: Add Second Residual Connection")
    print("  Formula: X + FFN(X)")
    print("\n  Final Transformer block output [2, 4]:")
    print("    Token 0:", [output[0, 0], output[0, 1], output[0, 2], output[0, 3]])
    print("    Token 1:", [output[1, 0], output[1, 1], output[1, 2], output[1, 3]])

    // ========================================
    // まとめ
    // ========================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nTransformer Block Pipeline:")
    print("  1. Self-Attention       → Captures token relationships")
    print("  2. Residual + LayerNorm → Stabilizes training")
    print("  3. Feed-Forward Network → Adds non-linearity")
    print("  4. Residual + LayerNorm → Final stabilization")

    print("\nKey Architectural Features:")
    print("  - Residual Connections: Enable gradient flow in deep networks")
    print("  - Layer Normalization: Stabilize activations across layers")
    print("  - Self-Attention: Model long-range dependencies")
    print("  - FFN: Add representation capacity and non-linearity")

    print("\nProduction Implementation Would Include:")
    print("  - Multi-head attention (typically 8-16 heads)")
    print("  - Proper Q/K/V projections with learned weight matrices")
    print("  - Complete layer normalization with learnable parameters")
    print("  - FFN with expansion ratio (e.g., d_model → 4*d_model → d_model)")
    print("  - Dropout for regularization")
    print("  - Positional encoding for sequence order information")

    print("\nTypical Hyperparameters:")
    print("  - d_model: 512 or 768")
    print("  - num_heads: 8 or 12")
    print("  - d_ff: 2048 or 3072 (4 * d_model)")
    print("  - num_layers: 6, 12, or 24")

    print("\n" + "=" * 70)
    print("End of Transformer Block Demonstration")
    print("=" * 70)
}
