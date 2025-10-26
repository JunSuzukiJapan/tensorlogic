# Layer 0 Shape Verification Results

## 実行日時
2025-10-26

## テストスクリプト
`examples/tests/dump_layer0_values.tl`

## 検証結果

### ✅ すべての形状が正しい

```
Step 1: Embedding
  Shape:  [1.0000, 2048.0000]

Step 2: RMS Normalization (attn)
  Shape:  [1.0000, 2048.0000]

Step 3: Q, K, V Projections
  Q shape:  [1.0000, 2048.0000]
  K shape:  [1.0000, 256.0000]
  V shape:  [1.0000, 256.0000]

Step 4: Reshape to heads
  Q_heads shape:  [1.0000, 32.0000, 64.0000]
  K_heads shape:  [1.0000, 4.0000, 64.0000]
  V_heads shape:  [1.0000, 4.0000, 64.0000]

Step 5: Apply RoPE (position=0)
  Q_rope shape:  [1.0000, 32.0000, 64.0000]
  K_rope shape:  [1.0000, 4.0000, 64.0000]

Step 6: GQA expansion (4 -> 32 heads)
  K_final shape:  [1.0000, 32.0000, 64.0000]
  V_final shape:  [1.0000, 32.0000, 64.0000]

Step 7: Attention
  scores shape:  [1.0000, 32.0000, 1.0000]
  scaled shape:  [1.0000, 32.0000, 1.0000]
  attn_weights shape:  [1.0000, 32.0000, 1.0000]
  attn_output shape:  [1.0000, 32.0000, 64.0000]

Step 8: Output projection
  attn_proj shape:  [1.0000, 2048.0000]

Step 9: Residual connection
  h1 shape:  [1.0000, 2048.0000]

Step 10: FFN normalization
  h1_norm shape:  [1.0000, 2048.0000]

Step 11: SwiGLU FFN
  ffn_output shape:  [1.0000, 2048.0000]

Step 12: Final residual
  h2 (layer 0 output) shape:  [1.0000, 2048.0000]
```

## 実装の正しさ検証

### RoPE実装
- ✅ NeoX style
- ✅ rope_base = 10000
- ✅ position = 0でcos(0)=1, sin(0)=0により入力がほぼ保存
- ✅ 形状変換が正しい: [seq, heads, dim] → [seq, heads, dim]

### GQA展開
- ✅ 4 KV heads → 32 Q heads
- ✅ Expansion factor = 8
- ✅ KV head i → Q heads (i*8) to (i*8+7)
- ✅ 実装: reshape → broadcast_to → reshape

### Attention
- ✅ einsum("ihd,jhd->ihj", Q, K) でscores計算
- ✅ スケーリング: scores * 0.125 (= 1/sqrt(64))
- ✅ softmax(scaled, axis=2)
- ✅ einsum("ihj,jhd->ihd", attn_weights, V) で出力

### SwiGLU FFN
- ✅ gate = linear(x, W_gate)
- ✅ up = linear(x, W_up)
- ✅ silu(gate) = gate * sigmoid(gate)
- ✅ intermediate = silu(gate) * up
- ✅ output = linear(intermediate, W_down)

## TensorLogic実装の特徴

### 関数定義が必要
TensorLogicでは、以下の関数は組み込みではなく、ユーザー定義が必要：

```tensorlogic
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result := x * sigmoid(x)
}
```

### Attention実装パターン
```tensorlogic
let scores = einsum("ihd,jhd->ihj", Q_rope, K_final)
let scaled = scores * 0.125
let attn_weights = softmax(scaled, 2)
let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_final)
```

### 演算子
- ✅ `+` : 要素ごとの加算（residual connections）
- ✅ `*` : 要素ごとの乗算、スカラー乗算
- ❌ `add()`, `mul()` : 組み込み関数ではない

## 結論

**形状レベル**: すべての操作で正しい形状が確認できた ✅

**実装レベル**:
- RoPE、GQA、Attention、SwiGLUの実装は数学的に正しい ✅
- TensorLogicのAPIに従って正しく実装されている ✅

**数値レベル**:
- 形状は正しいが、実際の数値値は未検証 ❓
- Layer 0の予測トークンが llama.cpp と異なる（2354 vs 期待値）
- この違いが f16 精度だけでは説明できない

## 次のステップ

1. **小さな入力での手動計算**
   - 1トークン、1ヘッド、小さい次元で手動計算
   - TensorLogicの数値とPython/NumPyの数値を直接比較

2. **Metal GPU実装の検証**
   - RoPE, Attention, SoftmaxのMetal shader実装を確認
   - CPU実装と比較

3. **GGUF重みロードの検証**
   - 重みが正しくデコードされているか確認
   - Q4_0, Q6_K dequantizationの正確性

4. **中間結果のダンプ**
   - TensorLogicで各ステップの実際の数値をダンプ
   - Pythonリファレンスと数値レベルで比較
