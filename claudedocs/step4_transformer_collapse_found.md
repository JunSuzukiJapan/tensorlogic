# Step 4: Transformer Layers Causing Logits Collapse

## 実行日時
2025-10-26

## 重大な発見

### 🎯 **根本原因を特定**

**症状**:
- Embedding → output_norm → linear: **正常な logits** (6.5 ~ 7.25 range, difference 0.7)
- Embedding → 22 Transformer Layers → output_norm → linear: **異常に均一な logits** (0.89 ~ 1.2 range, difference 0.04)

**結論**: **Transformer layersが hidden statesを均一化している**

## 検証結果

### Test 1: Embedding → Final Linear (Transformer をスキップ)

```bash
./target/release/tl run examples/tests/debug_first_token.tl
```

**結果**:
```
Max logit: 7.25
Top 5 tokens:
  1: token=30466, logit=7.25
  2: token=2603, logit=7.140625
  3: token=13269, logit=7.0195313
  4: token=2057, logit=6.5507813
  5: token=1050, logit=6.5195313
```

✅ **Logits are diverse and healthy** (difference: 0.7)

### Test 2: Embedding → 22 Layers → Final Linear (通常のフロー)

```bash
./target/release/tl run examples/archived/old_chat/chat_demo_full_22_layers.tl
```

**結果**:
```
Max logit: 1.0107422
Top 5 tokens:
  1: token=1314, logit=1.0107422
  2: token=29145, logit=0.97509766
  3: token=22771, logit=0.94189453
  4: token=27734, logit=0.92333984
  5: token=12116, logit=0.8911133
```

❌ **Logits are abnormally uniform** (difference: only 0.12)

## 確認済みの正常な箇所

1. ✅ **Q6_K dequantization**: 正しく動作し、多様な値を生成 (-0.24 to 0.20)
2. ✅ **output.weight shape**: 正しい shape [32000, 2048]
3. ✅ **linear() operation**: 正しい出力 shape を生成
4. ✅ **output_norm**: 正常に動作
5. ✅ **Dimension reverse**: GGUF → PyTorch 形式の変換が正しい
6. ✅ **Transpose in linear()**: Weight が正しく transpose されている

## 問題の箇所

❌ **Transformer Layers** - 22層のTransformer処理中にhidden statesが均一化される

### Transformer Layer 実装

```typescript
fn transformer_layer(x, W_q, W_k, W_v, W_o, attn_norm, W_gate, W_up, W_down, ffn_norm) {
    // Pre-attention RMSNorm
    let x_norm1 = rms_norm(x, attn_norm)

    // QKV projections
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)

    // GQA Attention
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)

    // Residual
    let x1 = x + attn_out

    // Pre-FFN RMSNorm
    let x_norm2 = rms_norm(x1, ffn_norm)

    // SwiGLU FFN
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)

    // Final residual
    result := x1 + ffn_out
}
```

実装は標準的なTransformerアーキテクチャに従っており、構造的には正しい。

### 疑わしい演算（優先度順）

1. **🔴 Einsum実装**
   - Attention scores計算: `einsum("ihd,jhd->ihj", Q_rope, K_expanded)`
   - Attention output計算: `einsum("ihj,jhd->ihd", attn_weights, V_expanded)`
   - Metal GPU実装が正しくない可能性

2. **🔴 RoPE実装**
   - `rope(Q_heads)` と `rope(K_heads)` の実装
   - 位置エンコーディングが hidden states を均一化している可能性

3. **🟡 Softmax実装**
   - `softmax(scaled_scores, 2)` が正しい次元で計算されているか
   - Metal GPU実装（Step 3で検証済みだが、3次元tensorでは未検証）

4. **🟡 Broadcast操作**
   - GQA: 4 KV heads → 32 Q heads の展開
   - `broadcast_to(K_with_group, [seq, 4, 8, 64])` が正しいか

5. **🟡 Reshape操作**
   - 頻繁なreshape操作が正しく動作しているか
   - 特に `[seq, 32, 64] → [seq, 2048]` の変換

## Layer-by-Layer 検証結果

### 🎯 重要な発見

**Test 1: Embedding → Final Linear (Transformerスキップ)**
- Logits: 6.5 ~ 7.25 (difference: 0.7) ✅

**Test 2: Layer 0のみ**
- Logits: 7.86 ~ 8.58 (difference: 0.72) ✅

**Test 3: Layer 0 + 1**
- Logits: 7.52 ~ 9.69 (difference: 2.16) ✅

**Test 4: Layer 0 + 1 + 2**
- Logits: 2.65 ~ 3.39 (difference: 0.74) ✅

**Test 5: 22層すべて**
- Logits: 0.89 ~ 1.01 (difference: 0.12) ❌

**結論**:
- **Layer 2までは正常に動作**
- Layer 2以降で徐々に崩壊が始まる
- 問題は**累積的**なもの（単一レイヤーではなく、複数レイヤーを重ねることによる）

### 疑わしい原因（更新）

1. **🔴 f16精度の累積誤差**
   - 3層以降で数値誤差が累積
   - RMS Normの正規化 + Residual接続で誤差が増幅

2. **🔴 RMS Normの過剰な正規化**
   - 各層で値が均一化される傾向
   - 22層を経ると完全に均一化

3. **🟡 Residual接続の問題**
   - 残差接続で値が平均化される可能性

## 次のステップ

### 優先順位 1: Layer 3-10 の検証

1. Layer 3, 5, 7, 10 を順次テスト
2. どのレイヤーで崩壊が始まるかを正確に特定
3. 各レイヤーの hidden states 統計を確認（min, max, mean, std）

### 優先順位 2: RMS Norm 検証

1. 各レイヤーの RMS Norm 出力を確認
2. 正規化後の値の分散を確認
3. 22層を経た後の値の均一性を確認

### 優先順位 3: f32精度でのテスト

1. すべてのテンソルをf32に変更
2. 22層で崩壊が解消されるか確認
3. f16精度が原因かどうかを特定

## 検証済みのテストファイル

1. **[examples/tests/debug_first_token.tl](../examples/tests/debug_first_token.tl)**
   - Transformerをスキップして embedding → final linear を直接テスト
   - 結果: ✅ 正常な logits (6.5 ~ 7.25 range)

2. **[examples/tests/verify_output_weight_shape.tl](../examples/tests/verify_output_weight_shape.tl)**
   - output.weight の shape と Q6_K dequantization を検証
   - 結果: ✅ すべて正常

3. **[examples/compare_metal_cpu_rmsnorm.rs](../examples/compare_metal_cpu_rmsnorm.rs)**
   - RMS Norm: Metal GPU vs CPU比較
   - 結果: ✅ f16 精度内で一致

4. **[examples/compare_metal_cpu_matmul.rs](../examples/compare_metal_cpu_matmul.rs)**
   - Matmul: Metal GPU vs CPU比較
   - 結果: ✅ f32累積に修正後、一致

5. **[examples/compare_metal_cpu_softmax.rs](../examples/compare_metal_cpu_softmax.rs)**
   - Softmax: Metal GPU vs CPU比較
   - 結果: ✅ 2D tensorで一致（3D未検証）

## 🚨 最重要発見：非決定的動作

### Layer 0-10 複数回実行テスト

**実行1**:
```
Layer 2: Max = 3.39, Token = 20358
Layer 3: Max = 3.39, Token = 20358  ← 同じ
Layer 4-10: すべて Token = 20358    ← すべて同じ！
```

**実行2**:
```
Layer 0: Max = 8.58, Token = 5392
Layer 1: Max = 9.57, Token = 25799
Layer 2: Max = 9.27, Token = 25799
Layer 3: Max = 7.98, Token = 28309
Layer 4: Max = 9.02, Token = 7175
...
Layer 10: Max = 10.99, Token = 21673
```

### 🎯 決定的証拠

**同じ入力（BOS token）で実行ごとに異なる結果** → **非決定的動作！**

これは元のユーザー問題「非決定的なトークン生成」の根本原因です。

### 非決定性の原因候補

1. **🔴 Metal GPU演算の並列処理**
   - Einsum, Softmax, RoPE のいずれか
   - 浮動小数点演算の順序依存性
   - GPUスレッドの実行順序が非決定的

2. **🔴 Metal並列リダクション**
   - Softmaxのsum/max計算
   - Einsumの累積計算
   - 順序が異なると浮動小数点の結果が変わる

3. **🟡 初期化されていないメモリ**
   - Metal bufferの初期化漏れ
   - 前回の実行結果が残っている

## 推奨される次のアクション（更新）

### 🔴 最優先：非決定性の原因特定

1. **同じ演算を10回実行して一貫性確認**
   - Einsum: 同じ入力で10回
   - Softmax: 同じ入力で10回
   - RoPE: 同じ入力で10回

2. **CPU実装で検証**
   - すべての演算をCPUで実行
   - 決定的な結果が得られるか

3. **Metal実装の順序固定**
   - 並列リダクションの順序を固定
   - GPU計算の決定性を保証

### 影響度

**CRITICAL**: この非決定性により：
- 同じプロンプトで異なる応答
- デバッグ不可能
- テスト不可能
- 本番環境で使用不可

これらの検証により、非決定性の原因を特定し、修正できる。
