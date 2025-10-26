# Layer-by-Layer Checkpoint Analysis

## 調査概要

TensorLogicとllama.cppの出力差異の原因を特定するため、22層すべてのチェックポイントで中間出力を分析しました。

## 実行環境

- **入力**: "Hello" (token ID: 15043)
- **モデル**: tinyllama-1.1b-chat-q4_0.gguf
- **設定**: temperature=0 (greedy sampling)
- **TensorLogic**: Buffer pool fix適用後、決定的動作を確認済み

## レイヤーごとの予測トークン

| Layer | Token ID | デコード結果 | 備考 |
|-------|----------|--------------|------|
| 0 | 2354 | "ily" | 初期予測 |
| 5 | 29415 | "isser" | 中間層開始 |
| 10 | 29415 | "isser" | 安定 |
| 15 | 29415 | "isser" | 安定 |
| 20 | 1744 | "SB" | 後期層で変化 |
| 21 | 14106 | " daß" | 最終出力 (ドイツ語) |

## 主要観察事項

### 1. 予測の変遷パターン

- **Layer 0→5**: 大きな変化 ("ily" → "isser")
- **Layer 5→15**: 安定期 (一貫して "isser")
- **Layer 15→20**: 再変化 ("isser" → "SB")
- **Layer 20→21**: 最終調整 ("SB" → " daß")

### 2. llama.cppとの比較

**llama.cpp出力**:
```
Input: "Hello"
Output: "Hello" (同じトークンを繰り返し)
```

**TensorLogic出力**:
```
Input: "Hello"
Output: " daß" (ドイツ語の単語)
```

### 3. 重大な発見

TensorLogicとllama.cppの出力は**完全に異なっています**。これは：

1. **f16精度だけの問題ではない**
   - f16の累積誤差は~0.093程度（前回の調査）
   - この程度の誤差では完全に異なるトークンにはならない

2. **実装に根本的な違いがある可能性**
   - RoPE実装の違い
   - Attention計算の違い
   - GQA (Grouped Query Attention) の展開方法の違い
   - Softmax/正規化の違い

## 次のステップ (Priority 3)

### RoPE実装の検証

1. RoPE周波数計算の確認
2. 位置エンコーディングの適用方法
3. llama.cppとの数値比較

### Attention実装の検証

1. Q/K/V射影の確認
2. スケーリング係数 (0.125 = 1/sqrt(64)) の検証
3. Attention weightの計算

### GQA展開の検証

1. K/Vヘッドの展開方法 (4→32)
2. reshape/broadcast操作の正確性
3. 次元の整合性

## 技術的詳細

### テストスクリプト

- **チェックポイント分析**: `examples/tests/debug_layer_outputs.tl`
- **トークンデコード**: `examples/tests/decode_token.tl`
- **f16精度テスト**: `examples/test_f16_precision.rs`

### 発見された問題

**初期の問題 (解決済み)**:
- バッファプールのゼロクリア不足 → 非決定的動作
- コミット ea69694 で修正完了

**現在の問題**:
- llama.cppとの出力完全不一致
- 原因: 実装の根本的な違い（詳細調査が必要）

## まとめ

1. ✅ **f16精度影響評価**: 完了
   - 累積誤差 ~0.093
   - 小さいが存在する

2. ✅ **中間レイヤー比較**: 完了
   - レイヤーごとの予測変遷を特定
   - llama.cppとの完全不一致を確認

3. ⏳ **実装検証**: 次のステップ
   - RoPE、Attention、GQAの詳細検証が必要
   - 数値レベルでの比較を実施

## 結論

TensorLogicとllama.cppの出力差異は、f16精度だけの問題ではなく、**コア操作の実装に根本的な違いがある**可能性が高い。特にRoPE、Attention、GQAの実装を重点的に検証する必要があります。
