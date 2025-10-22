# TensorLogic: Added Operations (2025-01-22)

このドキュメントは、2025年1月22日のセッションで追加されたすべてのテンソル操作を記録しています。

## 概要

Rustバックエンドで実装済みだったがインタープリターから利用できなかった操作を調査し、**合計29個の新しい操作**をTensorLogicインタープリターに公開しました。

---

## 1. 基本テンソル操作（20個）

### 1.1 テンソル作成（2個）

#### `zeros(shape: [int]) -> Tensor`
ゼロで埋められたテンソルを作成します。

**使用例:**
```tensorlogic
let z = zeros([3, 4])  // [3, 4]のゼロテンソル
```

**用途:** 初期化、プレースホルダー

---

#### `ones(shape: [int]) -> Tensor`
1で埋められたテンソルを作成します。

**使用例:**
```tensorlogic
let o = ones([2, 3])  // [2, 3]の1テンソル
```

**用途:** 初期化、マスク生成

---

### 1.2 形状操作（4個）

#### `reshape(tensor: Tensor, new_shape: [int]) -> Tensor`
テンソルの形状を変更します（要素数は保持）。

**使用例:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let reshaped = reshape(data, [3, 8])  // [3, 8]に変形
```

**用途:** バッチ処理、次元調整

---

#### `flatten(tensor: Tensor) -> Tensor`
テンソルを1次元に平坦化します。

**使用例:**
```tensorlogic
let matrix = positional_encoding(4, 5)  // [4, 5]
let flat = flatten(matrix)              // [20]
```

**用途:** 全結合層への入力、特徴量ベクトル化

---

#### `transpose(tensor: Tensor) -> Tensor`
2次元テンソルの転置（行と列を入れ替え）。

**使用例:**
```tensorlogic
let matrix = positional_encoding(3, 4)  // [3, 4]
let t = transpose(matrix)               // [4, 3]
```

**用途:** 行列演算、Attention計算

**注意:** パーサーの競合問題を修正済み（grammar.pestから単項演算子定義を削除）

---

#### `permute(tensor: Tensor, dims: [int]) -> Tensor`
次元の順序を並べ替えます。

**使用例:**
```tensorlogic
let tensor = positional_encoding(6, 4)  // [6, 4]
let permuted = permute(tensor, [1, 0])  // [4, 6]
```

**用途:** バッチ・シーケンス次元の入れ替え、チャネル順序変更

---

### 1.3 集約操作（2個）

#### `max(tensor: Tensor) -> float`
テンソル全体の最大値を返します。

**使用例:**
```tensorlogic
let data = positional_encoding(4, 5)
let max_val = max(data)  // スカラー値
```

**用途:** 統計量計算、正規化

**制限:** 現在は全体の最大値のみ。次元指定版は未実装。

---

#### `min(tensor: Tensor) -> float`
テンソル全体の最小値を返します。

**使用例:**
```tensorlogic
let data = positional_encoding(4, 5)
let min_val = min(data)  // スカラー値
```

**用途:** 統計量計算、正規化

**制限:** 現在は全体の最小値のみ。次元指定版は未実装。

---

### 1.4 活性化関数（2個）

#### `gelu(tensor: Tensor) -> Tensor`
GELU (Gaussian Error Linear Unit) 活性化関数。

**使用例:**
```tensorlogic
let x = positional_encoding(3, 4)
let activated = gelu(x)
```

**用途:** Transformerの標準活性化関数（BERT, GPTなど）

**数式:** `GELU(x) = x * Φ(x)` (Φは標準正規分布の累積分布関数)

---

#### `tanh(tensor: Tensor) -> Tensor`
双曲線正接（ハイパボリックタンジェント）活性化関数。

**使用例:**
```tensorlogic
let x = positional_encoding(3, 4)
let activated = tanh(x)
```

**用途:** RNN, LSTM, 古典的ニューラルネット

**範囲:** `[-1, 1]`

---

### 1.5 数学関数（7個）

#### `exp(tensor: Tensor) -> Tensor`
要素ごとの指数関数（e^x）。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 3)
let result = exp(x)
```

**用途:** Softmax計算、確率分布

---

#### `log(tensor: Tensor) -> Tensor`
要素ごとの自然対数（ln(x)）。

**使用例:**
```tensorlogic
let x = exp(positional_encoding(2, 3))
let result = log(x)  // 元の値に戻る
```

**用途:** 対数尤度、情報理論

---

#### `sqrt(tensor: Tensor) -> Tensor`
要素ごとの平方根。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 2)
let result = sqrt(x)
```

**用途:** 正規化、距離計算

---

#### `pow(tensor: Tensor, exponent: float) -> Tensor`
要素ごとのべき乗（x^n）。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 3)
let squared = pow(x, 2)  // x^2
```

**用途:** 多項式演算、エネルギー計算

---

#### `sin(tensor: Tensor) -> Tensor`
要素ごとの正弦関数。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 3)
let result = sin(x)
```

**用途:** 位置エンコーディング、周期関数

---

#### `cos(tensor: Tensor) -> Tensor`
要素ごとの余弦関数。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 3)
let result = cos(x)
```

**用途:** 位置エンコーディング、周期関数

---

#### `tan(tensor: Tensor) -> Tensor`
要素ごとの正接関数。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 3)
let result = tan(x)
```

**用途:** 角度計算、三角関数演算

---

### 1.6 インデックス操作（2個）

#### `gather(tensor: Tensor, dim: int, indices: Tensor) -> Tensor`
指定された次元に沿ってインデックスで要素を収集します。

**使用例:**
```tensorlogic
// 高度な使用例：GNNのメッセージパッシング
let node_features = positional_encoding(10, 8)
let neighbor_indices = /* インデックステンソル */
let gathered = gather(node_features, 0, neighbor_indices)
```

**用途:** GNNのメッセージ収集、動的インデックス

**注意:** インデックステンソルは整数型が必要

---

#### `scatter(tensor: Tensor, dim: int, indices: Tensor, src: Tensor) -> Tensor`
指定されたインデックスに値を配置します。

**使用例:**
```tensorlogic
// 高度な使用例：GNNのメッセージ更新
let output = scatter(base_tensor, 0, indices, updates)
```

**用途:** GNNのメッセージ配置、スパース更新

**注意:** インデックステンソルは整数型が必要

---

## 2. 高度な操作（9個）

### 2.1 マスク操作（3個）

#### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`
Attentionスコアにマスクを適用します。

**使用例:**
```tensorlogic
let scores = positional_encoding(4, 4)  // Attentionスコア
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**用途:**
- Transformer Attention
- パディングトークンの無視
- Causalマスキング

**動作:** マスクが0の位置を-∞に設定

---

#### `padding_mask(lengths: [int], max_len: int) -> Tensor`
シーケンス長からパディングマスクを生成します。

**使用例:**
```tensorlogic
let lengths = [3, 5, 2, 4]  // 各シーケンスの実際の長さ
let mask = padding_mask(lengths, 5)  // [4, 5]のマスク
```

**用途:**
- 可変長シーケンスのバッチ処理
- パディングトークンの無視

**出力:** 実際の長さまでは1、それ以降は0

---

#### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`
複数のマスクを要素ごとに結合します。

**使用例:**
```tensorlogic
let padding = padding_mask([3, 5], 5)
let causal = causal_mask(5)
let combined = combine_masks(padding, causal)
```

**用途:**
- パディングマスク + Causalマスク
- 複数の制約の適用

**動作:** 要素ごとの論理積（AND）

---

### 2.2 ブロードキャスト操作（1個）

#### `broadcast_to(tensor: Tensor, target_shape: [int]) -> Tensor`
テンソルを指定された形状にブロードキャストします。

**使用例:**
```tensorlogic
let small = positional_encoding(1, 4)  // [1, 4]
let broadcasted = broadcast_to(small, [3, 4])  // [3, 4]
```

**用途:**
- バッチ次元の追加
- 要素ごと演算のための形状調整

---

### 2.3 Fused操作（5個）

Fused操作は複数の演算を1つのカーネルで実行し、メモリ転送を削減します。

#### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`
加算とReLUを1操作で実行します。

**使用例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let result = fused_add_relu(a, b)  // max(a + b, 0)
```

**用途:** ResNet残差接続、スキップ接続

**利点:** 中間結果のメモリ割り当てを回避

---

#### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`
乗算とReLUを1操作で実行します。

**使用例:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let result = fused_mul_relu(a, b)  // max(a * b, 0)
```

**用途:** ゲート機構、要素ごとの重み付け

---

#### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`
アフィン変換（scale * x + bias）を1操作で実行します。

**使用例:**
```tensorlogic
let x = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let result = fused_affine(x, scale, bias)
```

**用途:**
- Batch Normalizationの最終ステップ
- Layer Normalizationの変換

---

#### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`
GELU活性化と線形変換を融合します。

**使用例:**
```tensorlogic
let x = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias = zeros([2, 3])
let result = fused_gelu_linear(x, weight, bias)
```

**用途:**
- Transformer FFNの最適化
- BERT/GPTの高速化

**等価:** `gelu(matmul(x, weight) + bias)`

---

## 3. 既存の操作（参考）

以下は今回の追加前から利用可能だった操作です：

### GNN操作（Table 1より）
- `sigmoid(tensor)` - シグモイド活性化
- `sum(tensor, dim, keepdim)` - 合計集約
- `mean(tensor, dim, keepdim)` - 平均集約

### Transformer操作（Table 2より）
- `relu(tensor)` - ReLU活性化
- `matmul(a, b)` - 行列乗算
- `layer_norm(tensor)` - Layer Normalization
- `concat(tensor1, tensor2, dim)` - テンソル結合

### その他
- `argmax(tensor, dim, keepdim)` - 最大値のインデックス
- `argmin(tensor, dim, keepdim)` - 最小値のインデックス
- `unsqueeze(tensor, dim)` - 次元追加
- `squeeze(tensor, dim)` - 次元削除
- `split(tensor, size, dim)` - テンソル分割
- `chunk(tensor, chunks, dim)` - テンソル分割（均等）
- `embedding(table, ids)` - 埋め込みルックアップ
- `positional_encoding(seq_len, d_model)` - 位置エンコーディング
- `softmax(tensor)` - Softmax関数
- `sample(probs)` - 確率分布からサンプリング
- `causal_mask(seq_len)` - Causalマスク生成
- `batch_norm(tensor, gamma, beta, eps)` - Batch Normalization
- `dropout(tensor, p, training)` - Dropout

---

## 4. テスト

すべての新しい操作には包括的なテストが用意されています：

- **基本操作（20個）**: `examples/test_all_20_ops.tl`
- **高度な操作（9個）**: `examples/test_advanced_ops.tl`
- **GNN操作**: `examples/test_gnn.tl`
- **Transformer操作**: `examples/test_transformer_functional.tl`

---

## 5. パーサー修正

### 問題
`transpose`と`permute`が"unary_op"として定義されていたため、関数呼び出し構文と競合していました。

### 解決策
`src/parser/grammar.pest`を修正：
1. `unary_op`定義から`"transpose"`を削除（135行目）
2. 予約語リストから`"transpose"`を削除（320行目）

### 結果
両関数が正常な関数呼び出しとして動作するようになりました。

---

## 6. 実装の詳細

### ファイル
- **メイン実装**: `src/interpreter/mod.rs`
- **行数**: 約200行の新しいコード
- **パーサー修正**: `src/parser/grammar.pest`

### アーキテクチャ
すべての操作はRustバックエンド（`src/ops/`）の既存実装を利用しています。インタープリターレイヤーは単に：

1. TensorLogicの引数を評価
2. Rust関数を呼び出し
3. 結果をTensorLogic値として返却

### エラーハンドリング
- 引数の数チェック
- 型チェック（Tensor, Integer, Float, Array）
- Rustエラーを`RuntimeError`に変換

---

## 7. 利用可能な操作の総数

| カテゴリ | 操作数 |
|---------|--------|
| テンソル作成 | 2 |
| 形状操作 | 4 |
| 集約 | 4 (max, min, sum, mean) |
| 活性化 | 5 (gelu, tanh, sigmoid, relu, softmax) |
| 数学関数 | 7 |
| インデックス | 4 (gather, scatter, argmax, argmin) |
| マスク | 4 (apply_attention_mask, padding_mask, combine_masks, causal_mask) |
| Fused | 5 |
| Transformer | 5 (matmul, layer_norm, concat, embedding, positional_encoding) |
| その他 | 8 (unsqueeze, squeeze, split, chunk, sample, batch_norm, dropout, broadcast_to) |
| **合計** | **48操作** |

---

## 8. 今後の改善点

### 優先度：中
1. `max(tensor, dim, keepdim)` - 次元指定版
2. `min(tensor, dim, keepdim)` - 次元指定版
3. `split()`と`chunk()`の配列返却対応（現在は単一値のみ）

### 優先度：低
4. Metal kernelの最適化（concat, permute, broadcast）
5. Python統合の強化
6. 型システムの完全実装

---

## 9. 参考文献

- **論文**: "Transformers in Tensor Logic" (arXiv:2510.12269)
  - Table 1: GNN操作
  - Table 2: Transformer操作
- **Metal API**: Apple Metal Performance Shaders
- **PyTorch互換性**: 関数シグネチャはPyTorchに準拠

---

## 10. 変更履歴

| 日付 | コミット | 内容 |
|------|---------|------|
| 2025-01-22 | 7ea2e39 | 18個の基本テンソル操作を追加 |
| 2025-01-22 | cb7ff17 | transposeとpermuteのパーサー競合を修正 |
| 2025-01-22 | eef34ab | 9個の高度な操作（マスク、ブロードキャスト、Fused）を追加 |
| 2025-01-22 | 4773acb | GNN操作（sigmoid, sum, mean）を追加 |

---

このドキュメントは今後の開発とメンテナンスの参考資料として保持されます。
