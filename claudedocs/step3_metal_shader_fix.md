# Step 3: Metal Shader検証と修正結果

## 実行日時
2025-10-26

## 検証目的
Metal GPU実装とCPU実装の数値一致を確認し、精度問題を特定・修正する

## 発見した問題

### 🔴 **重大な問題: Matmul のf16累積誤差**

**症状**:
- 小さい行列（2x3 @ 3x2）: 完璧に一致 ✅
- 大きい行列（1x2048 @ 2048x2048）:
  - **最大誤差: 2.125**
  - **平均誤差: 0.568**

**原因**:
Metal matmul実装が **f16（half）で累積**していた

#### 問題のコード

**Before (間違い)**:
```metal
half sum = 0.0h;  // f16累積
for (uint k = 0; k < K; k++) {
    sum += a[row * K + k] * b[k * N + col];  // f16で2048回累積
}
c[row * N + col] = sum;
```

**After (修正)**:
```metal
float sum = 0.0f;  // f32累積
for (uint k = 0; k < K; k++) {
    sum += float(a[row * K + k]) * float(b[k * N + col]);  // f32で累積
}
c[row * N + col] = half(sum);  // 最後にf16変換
```

## 修正したファイル

### 1. `/shaders/elementwise.metal`
- `matmul_f16`: Naive実装（小さい行列用）
- **修正箇所**: 151-157行目

### 2. `/shaders/matmul_tiled.metal`
- `matmul_tiled_f16`: 16x16タイル実装（中規模行列用）
- `matmul_tiled_32x32_f16`: 32x32タイル実装（大規模行列用）
- `matmul_tiled_bias_f16`: Bias付きmatmul
- **修正箇所**: 60-108行目、138-176行目、206-243行目

## 修正結果

### Softmax検証 (CPU vs GPU)

```
Test 1: [2, 5] tensor
  Row 0 sum - CPU: 0.999817, GPU: 0.999870
  Maximum difference: 0.000061
  ✅ 一致（f16精度として妥当）

Test 2: [1, 32] tensor
  Maximum difference: 0.000061
  ✅ 一致
```

**結論**: Softmax実装は正しい ✅

### Matmul検証 (CPU vs GPU)

#### 修正前
```
Test 2: [1, 2048] @ [2048, 2048]
  CPU: [85.0625, 2.0957031, ...]
  GPU: [83.0, 2.1875, ...]
  Maximum difference: 2.125000  ❌
  Average difference: 0.568268  ❌
```

#### 修正後
```
Test 2: [1, 2048] @ [2048, 2048]
  CPU: [85.0625, 2.0957031, ...]
  GPU: [85.0625, 2.0976563, ...]
  Maximum difference: 0.0625  ✅
  Average difference: 0.002106  ✅
```

**改善率**:
- 最大誤差: **34倍改善** (2.125 → 0.0625)
- 平均誤差: **280倍改善** (0.568 → 0.002)

## モデル推論での影響

### 修正前
```
Layer 0: Token 2354
Layer 21: Token 18712
```

### 修正後
```
Layer 0: Token 2354 (変わらず)
Layer 21: Token 18712 (変わらず)
```

**結論**: Matmul精度向上は確認できたが、**モデル予測は依然として不正確**

## 残る問題

matmul修正後も Layer 0 の予測が不正確なことから、**他の要因**が存在：

### 疑わしい箇所（優先度順）

1. **🔴 次元転置/形状変換**
   - GGUF → PyTorch形式の転置処理
   - linear() 関数内の転置
   - reshape/broadcast演算

2. **🟡 RoPE実装**
   - Metal GPU vs CPU の数値精度
   - 位置エンコーディングの正確性

3. **🟡 Einsum実装**
   - Attention計算（Q·K, Attn·V）
   - Metal GPU実装の正確性

4. **🟡 GQA (broadcast_to)**
   - 4 KV heads → 32 Q heads の展開
   - Metal GPU実装

5. **🟡 RMS Norm**
   - Metal GPU実装の数値精度

## 検証用テストファイル

### 作成したテスト

1. **[examples/compare_metal_cpu_softmax.rs](../examples/compare_metal_cpu_softmax.rs)**
   - Softmax: Metal GPU vs CPU比較
   - 結果: ✅ 一致

2. **[examples/compare_metal_cpu_matmul.rs](../examples/compare_metal_cpu_matmul.rs)**
   - Matmul: Metal GPU vs CPU比較
   - 結果: ✅ 修正後一致

3. **[examples/test_q4_0_layout.rs](../examples/test_q4_0_layout.rs)**
   - Q4_0 dequantizationのレイアウト検証
   - 結果: Layout B (Grouped) が正しい

4. **[examples/test_gguf_q4_0_values.rs](../examples/test_gguf_q4_0_values.rs)**
   - 実際のGGUF重み読み込み検証
   - 結果: ✅ 正常

## 結論

### ✅ 達成したこと

1. **Softmax**: Metal GPU実装が正しいことを確認
2. **Matmul**: **f16累積誤差を特定・修正** → 精度34倍～280倍向上
3. **GGUF Q4_0**: dequantizationが正しいことを確認

### ❌ 未解決の問題

**モデル予測が依然として不正確**
- Layer 0で既に divergence（Token 2354）
- matmul修正だけでは不十分
- **他の演算または次元処理に問題がある**

## 次のステップ

**Step 1**: llama.cppとの直接比較
- 同じ入力で各演算の中間値を比較
- どの時点で divergence が起きるか特定
- RoPE、Einsum、Broadcast、RMS Normなどを個別検証

または

**追加検証**:
- Linear関数の転置処理を確認
- GGUF次元の reverse処理を検証
- 各演算のMetal実装をCPU実装と1つずつ比較
