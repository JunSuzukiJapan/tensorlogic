# TinyLlama参照値分析 - デバッグ用

**日付**: 2025-11-10
**目的**: TensorLogicのNaN logits問題をデバッグするための正しい中間値

## 概要

HuggingFace Transformersを使用して、TinyLlama 1.1B Chat modelの正しい中間値を計算しました。
これをTensorLogicの出力と比較することで、どこでNaNが発生しているかを特定できます。

## プロンプトとトークン化

### プロンプト
```
<|system|>
You are a friendly chatbot.</s>
<|user|>
Hello! How are you?</s>
<|assistant|>

```

### トークン化結果

**⚠️ 重要な発見: トークン数の違い**

| 実装 | トークン数 | 詳細 |
|------|------------|------|
| **HuggingFace** | **38** | 正しいトークン化 |
| **TensorLogic** | **39** | 1トークン多い |

**HuggingFace tokens (38個)**:
```
[529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 19780, 13563, 7451, 29889, 2, 29871, 13, 29966, 29989, 1792, 29989, 29958, 13, 10994, 29991, 1128, 526, 366, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13]
```

**トークン分解**:
- `<|system|>`: 529, 29989, 5205, 29989, 29958 (5 tokens)
- `\n`: 13 (1 token)
- `You are a friendly chatbot.`: 3492, 526, 263, 19780, 13563, 7451, 29889 (7 tokens)
- `</s>`: 2 (1 token)
- ` `: 29871 (1 token)
- `\n`: 13 (1 token)
- `<|user|>`: 29966, 29989, 1792, 29989, 29958 (5 tokens)
- `\n`: 13 (1 token)
- `Hello! How are you?`: 10994, 29991, 1128, 526, 366, 29973 (6 tokens)
- `</s>`: 2 (1 token)
- ` `: 29871 (1 token)
- `\n`: 13 (1 token)
- `<|assistant|>`: 29966, 29989, 465, 22137, 29989, 29958 (6 tokens)
- `\n`: 13 (1 token)

**合計: 38 tokens**

### トークン数の違いの影響

**仮説**: TensorLogicのトークナイザーが1トークン多く生成している可能性があります。
- これにより、シーケンス長が39になる
- RoPE (Rotary Position Embedding)の位置が1つずれる
- Attention maskのサイズが合わない可能性がある
- これらがNaNを引き起こしている可能性がある

## 中間値 - 正しい参照値

### 1. Embedding層の出力

**Shape**: `[38, 2048]`

**Sample値** (embedding[0, 0:3]):
```
[-9.441375732421875e-05, -0.0023651123046875, 0.01361083984375]
```

**正常性チェック**:
- ✅ 値は有限 (NaNなし)
- ✅ 範囲は妥当 (-0.003 ~ 0.014)
- ✅ Float16精度で表現可能

### 2. Layer 0 - Attention出力

**Shape**: `[38, 2048]`

**Sample値** (attn_output[0, 0:3]):
```
[-0.0015783309936523438, -0.01169586181640625, -0.00438690185546875]
```

**正常性チェック**:
- ✅ 値は有限 (NaNなし)
- ✅ 範囲は妥当 (-0.012 ~ -0.002)
- ✅ Attentionの出力として妥当な範囲

### 3. Layer 0 - MLP (SwiGLU FFN)出力

**Shape**: `[38, 2048]`

**Sample値** (mlp_output[0, 0:3]):
```
[0.0026760101318359375, 0.00316619873046875, 0.0113525390625]
```

**正常性チェック**:
- ✅ 値は有限 (NaNなし)
- ✅ 範囲は妥当 (0.003 ~ 0.011)
- ✅ SwiGLUの出力として妥当

### 4. Layer 0 - 最終出力 (Residual接続後)

**Shape**: `[38, 2048]`

**Sample値** (layer_0_output[0, 0:3]):
```
[0.001003265380859375, -0.010894775390625, 0.02056884765625]
```

**正常性チェック**:
- ✅ 値は有限 (NaNなし)
- ✅ 範囲は妥当 (-0.011 ~ 0.021)
- ✅ Residual接続の結果として妥当

### 5. Layer 21 (最終層) 出力

**Sample値** (layer_21_output[-1, 0:3]): 最後のトークンの値
```
[-2.796875, -1.1787109375, -1.42578125]
```

**正常性チェック**:
- ✅ 値は有限 (NaNなし)
- ✅ 範囲は妥当 (-2.8 ~ -1.2)
- ✅ 最終層の出力として妥当な範囲

### 6. 最終Logits

**Shape**: `[32000]` (Vocabulary size)

**Sample値** (logits[0:3]):
```
[-6.5546875, -6.8828125, 2.875]
```

**Top 10 Logits** (正しい生成トークン):

| Rank | Token ID | Logit Value | Token | 確率 (softmax後) |
|------|----------|-------------|-------|------------------|
| 1 | 29902 | 19.500000 | `"I"` | ~45% |
| 2 | 10994 | 17.578125 | `"Hello"` | ~14% |
| 3 | 18567 | 16.593750 | `"Hi"` | ~5% |
| 4 | 6816 | 14.359375 | `"Me"` | ~1% |
| 5 | 3421 | 14.210938 | `"My"` | ~0.8% |
| 6 | 18420 | 13.546875 | `"Good"` | ~0.4% |
| 7 | 7058 | 13.539062 | `"That"` | ~0.4% |
| 8 | 5328 | 13.507812 | `"How"` | ~0.4% |
| 9 | 29950 | 13.453125 | `"H"` | ~0.3% |
| 10 | 2887 | 13.359375 | `"As"` | ~0.3% |

**正常性チェック**:
- ✅ 全てのlogitは有限 (NaNなし)
- ✅ Top logitは妥当な値 (19.5)
- ✅ Logit範囲は妥当 (-6.88 ~ 19.5)
- ✅ 最も高い確率のトークンは `"I"` → "I am ..." という返答が期待される

## TensorLogicとの比較

### 現在のTensorLogic出力

```
[SAMPLING DEBUG] Top 10 logits:
  #1: token_id=0 logit=NaN
  #2: token_id=1 logit=NaN
  #3: token_id=2 logit=NaN
  ...
  #10: token_id=9 logit=NaN
```

### 問題点

| 項目 | HuggingFace (正解) | TensorLogic (現状) | 差異 |
|------|--------------------|--------------------|------|
| **トークン数** | 38 | 39 | **+1トークン** |
| **Top logit値** | 19.5 | NaN | **完全に異常** |
| **Top token** | 29902 ("I") | 0 (不正) | **完全に異常** |
| **Logit範囲** | -6.88 ~ 19.5 | 全てNaN | **完全に異常** |

### 問題の重大性

1. **第1トークン**: 全てのlogitがNaN
   - Forward passのどこかでNaNが発生している
   - Embedding → Layer 0 → ... → Layer 21 → Final norm → Logitsのどこか

2. **第2トークン以降**: 全てのlogitが0.0
   - matmul tile loading bugと類似
   - Multi-token forward passで問題発生

3. **トークン数の違い**: 38 vs 39
   - Tokenizer実装の違い
   - RoPE位置の不一致
   - Attention maskサイズの不一致

## デバッグ戦略

### フェーズ1: トークナイザーの検証

1. TensorLogicのtokenizer出力を確認
2. 39番目のトークンが何か特定
3. 必要に応じてtokenizer設定を修正

### フェーズ2: 中間値のトレース (Binary Search方式)

1. **Embedding出力を確認**
   - もしNaN → Embedding層に問題
   - もし正常 → 次へ

2. **Layer 0のAttention出力を確認**
   - もしNaN → Attention計算に問題
   - もし正常 → 次へ

3. **Layer 0のMLPドを確認**
   - もしNaN → MLP計算に問題
   - もし正常 → 次へ

4. **Layer 0の最終出力を確認**
   - もしNaN → Residual接続に問題
   - もし正常 → 次へ

5. **Layer 21出力を確認**
   - もしNaN → Layer 1-21のどこかに問題
   - もし正常 → 次へ

6. **Final norm出力を確認**
   - もしNaN → RMS normに問題
   - もし正常 → 次へ

7. **Logits出力を確認**
   - NaN → 最終linear layerに問題

### フェーズ3: 操作レベルのデバッグ

NaNが発生している層を特定したら、その層内の各操作を確認:

**Attention層の場合**:
1. RMS norm出力
2. Q, K, V projection出力
3. RoPE適用後のQ, K
4. Attention scores (Q @ K^T)
5. Scaled scores
6. Masked scores
7. Softmax出力
8. Attention output (scores @ V)
9. Output projection

**MLP層の場合**:
1. RMS norm出力
2. Gate projection
3. SiLU activation
4. Up projection
5. Element-wise multiplication
6. Down projection

### フェーズ4: Numerical Stabilityの確認

NaNの一般的な原因:
- **Divide by zero**: RMS normでの分母が0
- **Overflow**: Exponentialやsoftmaxでの値が大きすぎる
- **Underflow**: 非常に小さい値の計算
- **Invalid operations**: sqrt(負の数)、log(負の数)

## 次のアクション

1. ✅ **参照値を取得** (完了)
2. ⏳ **TensorLogicでトークナイザー出力を確認** (次)
3. ⏳ **中間値トレースプログラムを作成** (次)
4. ⏳ **Binary searchでNaN発生箇所を特定**
5. ⏳ **特定箇所の詳細デバッグ**
6. ⏳ **修正と検証**

## 参考ファイル

- 参照値JSON: `claudedocs/transformers_reference_values.json`
- Python計算スクリプト: `debug/compute_reference_values.py`
- TensorLogic実装: `examples/chat_demo_22layers.tl`

---

**最終更新**: 2025-11-10
**ステータス**: 参照値取得完了、デバッグ準備完了
