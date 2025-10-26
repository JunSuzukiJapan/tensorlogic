# 実装検証結果

## 実施したテスト

### 1. ✅ f16精度の影響評価

**テスト**: `examples/test_f16_precision.rs`

**結果**:
- 単純な演算: 誤差 ~0.00008
- Softmax計算: 誤差 ~0.08
- 1000回加算: 誤差 0 (完璧)
- 22層シミュレーション:
  - Layer 0: 誤差 0.0008
  - Layer 10: 誤差 0.023
  - Layer 21: 誤差 0.093

**結論**: f16の累積誤差は存在するが、完全に異なる出力を説明するには小さすぎる

### 2. ✅ 中間レイヤーのlogits比較

**テスト**: `examples/tests/debug_layer_outputs.tl`, `examples/tests/decode_token.tl`

**結果**:
| Layer | Token ID | デコード結果 |
|-------|----------|--------------|
| 0 | 2354 | "ily" |
| 5 | 29415 | "isser" |
| 10 | 29415 | "isser" |
| 15 | 29415 | "isser" |
| 20 | 1744 | "SB" |
| 21 | 14106 | " daß" |

**llama.cpp**: "Hello" → "Hello"
**TensorLogic**: "Hello" → " daß"

**結論**: レイヤー0の時点で既に異なるトークンを予測。実装に根本的な違いがある。

### 3. ✅ RoPE実装の検証

**テスト**: `examples/tests/test_rope.rs`, `examples/tests/test_rope_impl.tl`

**検証項目**:
- 周波数計算: `freq = 1.0 / (rope_base^(2*i/head_dim))`
- 回転計算: `out[2i] = in[2i] * cos(θ) - in[2i+1] * sin(θ)`
- Metal shader実装

**結果**:
- ✅ 形状変換が正しい: [1, 32, 64] → [1, 32, 64]
- ✅ RoPE base = 10000 (TinyLlama設定と一致)
- ✅ NeoX style実装
- ✅ position=0でcos(0)=1, sin(0)=0により入力がほぼ保存される

**結論**: RoPE実装は数学的に正しい

### 4. ✅ GQA展開の検証

**テスト**: `examples/tests/test_gqa_expansion.rs`, `examples/tests/test_gqa_impl.tl`

**検証項目**:
- K/Vヘッドの展開: 4 → 32 (expansion factor = 8)
- broadcast_to + reshapeの組み合わせ

**期待される展開**:
```
KV head 0 → Q heads 0-7
KV head 1 → Q heads 8-15
KV head 2 → Q heads 16-23
KV head 3 → Q heads 24-31
```

**実装方法**:
```
Step 1: [1, 4, 64] → [1, 4, 1, 64] (reshape)
Step 2: [1, 4, 1, 64] → [1, 4, 8, 64] (broadcast_to)
Step 3: [1, 4, 8, 64] → [1, 32, 64] (reshape)
```

**結果**:
- ✅ 形状変換が正しい
- ✅ 各KVヘッドが8回複製される
- ✅ 期待されるグループ構造を維持

**結論**: GQA展開は正しく実装されている

### 5. ✅ Softmax検証

**実装**: `src/ops/activations.rs:154-175`, `shaders/softmax.metal`

**特徴**:
- ✅ 数値安定性: `exp(x - max)` を使用
- ✅ テスト済み: `[1, 2, 3]` → `[0.090, 0.245, 0.665]`
- ✅ Metal GPU実装

**結論**: Softmax実装は検証済みで正しい

## 検証できなかった部分

### 数値レベルの比較

**問題**:
- TensorLogicには中間テンソルの値を直接出力する機能が限定的
- Metal GPUで計算された結果をCPUに転送して確認する手段が不足

**影響**:
- 形状は正しいが、実際の数値が正しいかは未確認
- RoPE、GQA、Attentionの計算結果の数値検証ができていない

### Attention計算の詳細

**未検証項目**:
1. スケーリング係数 (`1/sqrt(64) = 0.125`) の適用
2. Attentionスコアの計算: `Q @ K.T`
3. Softmax後のweightsの数値
4. Weighted sum: `weights @ V`

## 現状の理解

### 形状レベル
**すべて正しい**:
- ✅ RoPE: [seq, heads, dim] → [seq, heads, dim]
- ✅ GQA: [seq, 4, 64] → [seq, 32, 64]
- ✅ Attention: [seq, 32, 64] @ [seq, 32, 64].T → [seq, 32, seq]
- ✅ Softmax: [seq, 32, seq] → [seq, 32, seq]

### 数値レベル
**不明**:
- ❓ RoPEの回転計算の数値
- ❓ GQA展開後のK/V値
- ❓ Attention scoresの数値
- ❓ Softmax weightsの数値
- ❓ Weighted sumの結果

## 次のアクション候補

### オプションA: 数値デバッグを強化

**アプローチ**:
1. Metal shaderにprintfデバッグを追加
2. 中間結果をCPUに転送してダンプする機能を実装
3. 小さな入力（1トークン、1ヘッド）で詳細に比較

**メリット**: 根本原因を特定できる
**デメリット**: 時間がかかる、TensorLogicのコア実装を変更する必要がある

### オプションB: llama.cppのコードを参照

**アプローチ**:
1. llama.cppのAttention実装を詳細に読む
2. TensorLogicとの違いを特定
3. 特に注意すべき部分（正規化の順序、スケーリングなど）を確認

**メリット**: 実装の違いを直接確認できる
**デメリット**: llama.cppのコードが複雑

### オプションC: 簡易版を作成して比較

**アプローチ**:
1. Python (numpy/torch) で最小限のTinyLlama実装を作成
2. TensorLogicと同じ入力で実行
3. 中間結果をすべて出力して比較

**メリット**: 完全な制御、デバッグが容易
**デメリット**: 新しいコードを書く必要がある

### オプションD: 異なるアプローチ

**アプローチ**:
1. より大きなコンテキストで実行（複数トークン）
2. 統計的な分析（出力分布の比較など）
3. 既知の問題がないか確認（GGUFローダー、次元の扱いなど）

**メリット**: 別の角度から問題を見る
**デメリット**: 根本原因の特定には至らない可能性

## 推奨

現時点では**オプションB + C**の組み合わせが最も効率的：

1. llama.cppのattention実装を確認（特にGQA部分）
2. 簡単なPythonスクリプトで参照実装を作成
3. TensorLogicとPythonの両方で同じ入力を処理
4. レイヤー0の出力を詳細に比較

これにより、形状は合っているが数値が違う原因を特定できる可能性が高い。
