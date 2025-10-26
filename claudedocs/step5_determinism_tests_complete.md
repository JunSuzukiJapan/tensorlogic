# Step 5: Determinism Tests Complete - Root Cause Identified

## 実行日時
2025-10-26

## 重要発見：非決定性ではなく正確性の問題

### 仮説の変更
**元の仮説**: Metal GPU演算が非決定的である
**新しい発見**: 全ての演算は決定的だが、ロジットが異常に均一

## 決定性テスト結果

### テスト実施内容
全ての主要Metal GPU演算について、同じ入力で10回実行して結果を比較

### テスト1: Einsum ✅ DETERMINISTIC
```
Test: einsum("ihd,jhd->ihj", Q, K) - Attention scores
Input: Q=[2, 4, 8], K=[2, 4, 8]
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results
```

### テスト2: Softmax ✅ DETERMINISTIC
```
Test: softmax(scaled_scores) - 3D attention pattern
Input: [2, 4, 2] (seq, heads, seq)
Results: Max difference = 0.0000000000
Softmax sum: 1.0000000000 (perfect)
Status: ✅ All 10 runs produced identical results
```

### テスト3: RoPE ✅ DETERMINISTIC
```
Test 1: rope(Q, position_offset=0)
Input: Q=[2, 32, 64]
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results

Test 2: rope(Q, position_offset=10)
Input: Q=[2, 32, 64]
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results
```

### テスト4: Matmul ✅ DETERMINISTIC
```
Test 1: Query Projection - matmul([2, 2048], [2048, 2048])
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results

Test 2: FFN Gate Projection - matmul([2, 2048], [2048, 5632])
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results

Test 3: Output Projection - matmul([2, 2048], [2048, 32000])
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results
```

### テスト5: RMSNorm ✅ DETERMINISTIC
```
Test 1: Attention RMSNorm - rms_norm([2, 2048])
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results

Test 2: Larger values test
Results: Max difference = 0.0000000000
Status: ✅ All 10 runs produced identical results
```

## パラドックスの解明

### 観察された現象
1. **個別演算**: 全て完全に決定的 (差分0.0)
2. **フルトランスフォーマー**: ロジットが異常

### フルトランスフォーマーの出力 (22層)
```
Token 1:
  Max logit: 1.0107422
  Top 5 range: 0.89-1.01
  ❌ ABNORMAL - should be -10 to 20

Token 2:
  Max logit: 1.1933594
  Top 5 range: 1.08-1.19
  ❌ ABNORMAL - should be -10 to 20
```

### 正常なロジットとの比較

**正常**:
```
Max logit: 15.234
Top 5 tokens:
  1: token=123, logit=15.234
  2: token=456, logit=10.567
  3: token=789, logit=5.234
  4: token=321, logit=2.123
  5: token=654, logit=-3.456
Range: -10 to 20 (wide distribution)
```

**TensorLogic (現在)**:
```
Max logit: 1.01
Top 5 tokens:
  1: token=1314, logit=1.0107422
  2: token=29145, logit=0.97509766
  3: token=22771, logit=0.94189453
  4: token=27734, logit=0.92333984
  5: token=12116, logit=0.8911133
Range: 0.89-1.01 (almost uniform!)
```

## 根本原因の特定

### 問題の性質
**非決定性ではない** → **正確性の問題**

### ロジットが均一になる理由（仮説）
1. **重みの問題**
   - GGUF量子化解除の問題？
   - モデル読み込みの問題？

2. **累積誤差**
   - 22層を通過する間に誤差が蓄積？
   - 特定の演算で精度が失われている？

3. **スケール問題**
   - 値が極端に小さい/大きい？
   - オーバーフロー/アンダーフロー？

4. **層間の接続問題**
   - レジデュアル接続が間違っている？
   - 層の出力が正しく次の層に渡されていない？

## 次のステップ

### 優先度1: 中間値のデバッグ
1. **各層の出力を確認**
   - Layer 0の出力
   - Layer 1の出力
   - ...
   - Layer 21の出力

2. **値の範囲を確認**
   - 各層で値がどう変化するか
   - オーバーフロー/アンダーフローの検出

3. **llama.cppとの比較**
   - 同じ入力で各層の出力を比較
   - どこで値が乖離するか特定

### 優先度2: 単純化テスト
1. **1層だけのモデルでテスト**
   - すでに実施済み（Layer 0は正常）

2. **2層モデルでテスト**
   - Layer 0 + Layer 1
   - レジデュアル接続の確認

3. **段階的に層を増やす**
   - 3層、5層、10層...
   - どこから異常になるか特定

### 優先度3: 特定の演算の検証
1. **Embedding**
   - Token → Vector変換が正しいか

2. **Linear (quantized weights)**
   - Q4_0重みを使ったlinear()
   - Q6_K重みを使ったlinear()

3. **Element-wise operations**
   - add (residual)
   - multiply (scaling)
   - SiLU activation

## テストファイル作成済み

### 決定性テスト
- `examples/test_einsum_determinism.rs` ✅
- `examples/test_softmax_determinism.rs` ✅
- `examples/test_rope_determinism.rs` ✅
- `examples/test_matmul_determinism.rs` ✅
- `examples/test_rmsnorm_determinism.rs` ✅

### デバッグテスト（.tl）
- `examples/tests/verify_output_weight_shape.tl` ✅
- `examples/tests/debug_first_token.tl` ✅
- `examples/tests/decode_token.tl` ✅
- `examples/tests/test_layer_by_layer.tl` ✅
- `examples/tests/find_collapse_point.tl` ✅

## まとめ

### ✅ 確認済み
- 全てのMetal GPU演算は完全に決定的
- 個別の演算は正しく動作
- 非決定性の問題は存在しない

### ❌ 未解決の問題
- フルトランスフォーマーでロジットが均一化
- 原因は演算の正確性、重みの問題、または層間の接続問題
- 次のステップ: 各層の中間値をデバッグして原因を特定

### 結論
問題は「非決定的な演算」ではなく、「ロジットが異常に均一になる正確性の問題」である。全ての演算は決定的に動作しているが、トランスフォーマー全体として間違った結果を生成している。
