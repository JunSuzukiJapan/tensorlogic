// ============================================================================
// Positional Encoding for Transformer
// ============================================================================
//
// Transformerアーキテクチャにおける位置情報の埋め込み方法を実装します。
//
// 背景：
//   Transformerは順序情報を持たないため、入力シーケンスの位置情報を
//   明示的にエンコードする必要があります。
//
// 数式：
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//
//   where:
//     pos      = 位置インデックス (0, 1, 2, ...)
//     i        = 次元インデックス (0, 1, 2, ..., d_model/2)
//     d_model  = モデルの次元数（偶数である必要がある）
//
// 特徴：
//   - 低次元（小さいi）：高周波 → 位置の変化に敏感
//   - 高次元（大きいi）：低周波 → 長期的なパターンを捉える
//   - 相対位置の関係を学習しやすい
//
// 参考文献：
//   "Attention is All You Need" (Vaswani et al., 2017)
//   https://arxiv.org/abs/1706.03762
// ============================================================================

main {
    print("=" * 60)
    print("Transformer Positional Encoding Demonstration")
    print("=" * 60)

    // ========================================
    // パラメータ設定
    // ========================================
    tensor max_len: float16[1] = [10.0]    // 最大シーケンス長
    tensor d_model: float16[1] = [8.0]     // モデル次元（8次元 = 4ペア）

    print("\nConfiguration:")
    print("  Max Sequence Length: 10")
    print("  Model Dimension (d_model): 8")
    print("  Number of sin/cos pairs: 4")

    // ========================================
    // 位置インデックスの定義
    // ========================================
    // デモンストレーションのため、いくつかの位置を手動で定義
    tensor pos_0: float16[1] = [0.0]
    tensor pos_1: float16[1] = [1.0]
    tensor pos_2: float16[1] = [2.0]
    tensor pos_3: float16[1] = [3.0]

    // ========================================
    // 除数項の計算
    // ========================================
    // div_term = 10000^(2i/d_model)
    // 各次元ペアに対して異なる周波数を生成

    tensor div_0: float16[1] = [1.0]      // 10000^(0/8) = 1.0    (最高周波数)
    tensor div_1: float16[1] = [10.0]     // 10000^(2/8) ≈ 10.0
    tensor div_2: float16[1] = [100.0]    // 10000^(4/8) = 100.0
    tensor div_3: float16[1] = [1000.0]   // 10000^(6/8) = 1000.0 (最低周波数)

    print("\nDivision Terms (frequency modulators):")
    print("  dim 0-1 (i=0): div = 1.0    (high frequency)")
    print("  dim 2-3 (i=1): div = 10.0")
    print("  dim 4-5 (i=2): div = 100.0")
    print("  dim 6-7 (i=3): div = 1000.0 (low frequency)")

    // ========================================
    // 位置0のエンコーディング
    // ========================================
    print("\n" + "=" * 60)
    print("Position 0 Encoding")
    print("=" * 60)

    // 次元 0-1 (i=0, 最高周波数)
    tensor angle_0_0: float16[1] = pos_0 / div_0  // 0.0 / 1.0 = 0.0
    tensor pe_0_0: float16[1] = sin(angle_0_0)    // sin(0) = 0.0
    tensor pe_0_1: float16[1] = cos(angle_0_0)    // cos(0) = 1.0

    // 次元 2-3 (i=1)
    tensor angle_0_2: float16[1] = pos_0 / div_1  // 0.0 / 10.0 = 0.0
    tensor pe_0_2: float16[1] = sin(angle_0_2)    // sin(0) = 0.0
    tensor pe_0_3: float16[1] = cos(angle_0_2)    // cos(0) = 1.0

    // 次元 4-5 (i=2)
    tensor angle_0_4: float16[1] = pos_0 / div_2  // 0.0 / 100.0 = 0.0
    tensor pe_0_4: float16[1] = sin(angle_0_4)    // sin(0) = 0.0
    tensor pe_0_5: float16[1] = cos(angle_0_4)    // cos(0) = 1.0

    // 次元 6-7 (i=3, 最低周波数)
    tensor angle_0_6: float16[1] = pos_0 / div_3  // 0.0 / 1000.0 = 0.0
    tensor pe_0_6: float16[1] = sin(angle_0_6)    // sin(0) = 0.0
    tensor pe_0_7: float16[1] = cos(angle_0_6)    // cos(0) = 1.0

    print("  PE[0, 0] (sin, high freq):", pe_0_0)
    print("  PE[0, 1] (cos, high freq):", pe_0_1)
    print("  PE[0, 2] (sin):", pe_0_2)
    print("  PE[0, 3] (cos):", pe_0_3)
    print("  PE[0, 4] (sin):", pe_0_4)
    print("  PE[0, 5] (cos):", pe_0_5)
    print("  PE[0, 6] (sin, low freq):", pe_0_6)
    print("  PE[0, 7] (cos, low freq):", pe_0_7)

    print("\nNote: Position 0 always produces [0, 1, 0, 1, 0, 1, 0, 1]")
    print("      (sin terms = 0, cos terms = 1)")

    // ========================================
    // 位置1のエンコーディング
    // ========================================
    print("\n" + "=" * 60)
    print("Position 1 Encoding")
    print("=" * 60)

    // 次元 0-1 (高周波なので大きく変化)
    tensor angle_1_0: float16[1] = pos_1 / div_0  // 1.0 / 1.0 = 1.0
    tensor pe_1_0: float16[1] = sin(angle_1_0)    // sin(1.0) ≈ 0.841
    tensor pe_1_1: float16[1] = cos(angle_1_0)    // cos(1.0) ≈ 0.540

    // 次元 2-3 (中周波数)
    tensor angle_1_2: float16[1] = pos_1 / div_1  // 1.0 / 10.0 = 0.1
    tensor pe_1_2: float16[1] = sin(angle_1_2)    // sin(0.1) ≈ 0.0998
    tensor pe_1_3: float16[1] = cos(angle_1_2)    // cos(0.1) ≈ 0.995

    print("  PE[1, 0] (sin, high freq):", pe_1_0, "← Large change!")
    print("  PE[1, 1] (cos, high freq):", pe_1_1, "← Large change!")
    print("  PE[1, 2] (sin):", pe_1_2)
    print("  PE[1, 3] (cos):", pe_1_3, "← Small change")

    print("\nObservation:")
    print("  - High frequency dims (0-1) changed significantly")
    print("  - Low frequency dims barely changed from position 0")

    // ========================================
    // 周波数特性のデモンストレーション
    // ========================================
    print("\n" + "=" * 60)
    print("Frequency Characteristics Demonstration")
    print("=" * 60)

    // 位置2での高周波 vs 低周波の比較
    tensor angle_2_0: float16[1] = pos_2 / div_0  // 2.0 / 1.0 = 2.0
    tensor pe_2_0: float16[1] = sin(angle_2_0)    // sin(2.0) ≈ 0.909

    tensor angle_2_6: float16[1] = pos_2 / div_3  // 2.0 / 1000.0 = 0.002
    tensor pe_2_6: float16[1] = sin(angle_2_6)    // sin(0.002) ≈ 0.002

    print("\nPosition 2 comparison:")
    print("  High freq (dim 0):", pe_2_0, "← Oscillates rapidly")
    print("  Low freq (dim 6):", pe_2_6, "← Changes very slowly")

    // 位置3でも確認
    tensor angle_3_0: float16[1] = pos_3 / div_0  // 3.0 / 1.0 = 3.0
    tensor pe_3_0: float16[1] = sin(angle_3_0)    // sin(3.0) ≈ 0.141

    tensor angle_3_6: float16[1] = pos_3 / div_3  // 3.0 / 1000.0 = 0.003
    tensor pe_3_6: float16[1] = sin(angle_3_6)    // sin(0.003) ≈ 0.003

    print("\nPosition 3 comparison:")
    print("  High freq (dim 0):", pe_3_0, "← Different pattern")
    print("  Low freq (dim 6):", pe_3_6, "← Still near zero")

    // ========================================
    // まとめ
    // ========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey Properties:")
    print("  1. Different dimensions encode position at different scales")
    print("  2. High-frequency dims: sensitive to local position changes")
    print("  3. Low-frequency dims: capture long-range positional patterns")
    print("  4. This multi-scale encoding helps Transformer learn")
    print("     both local and global positional relationships")

    print("\nImplementation Notes:")
    print("  - In practice, positional encoding is added to token embeddings")
    print("  - The same encoding can be pre-computed and reused")
    print("  - Total encoding size must match model dimension (d_model)")

    print("\n" + "=" * 60)
    print("End of Positional Encoding Demonstration")
    print("=" * 60)
}
