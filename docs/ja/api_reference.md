# TensorLogic API リファレンス

TensorLogicで利用可能なすべての組み込み関数のリファレンスです。

**最終更新**: 2025-01-22
**利用可能な操作数**: 48

---

## 目次

1. [テンソル作成](#1-テンソル作成)
2. [形状操作](#2-形状操作)
3. [数学関数](#3-数学関数)
4. [集約操作](#4-集約操作)
5. [活性化関数](#5-活性化関数)
6. [行列演算](#6-行列演算)
7. [正規化](#7-正規化)
8. [マスク操作](#8-マスク操作)
9. [インデックス操作](#9-インデックス操作)
10. [埋め込み](#10-埋め込み)
11. [サンプリング](#11-サンプリング)
12. [Fused操作](#12-fused操作)
13. [最適化](#13-最適化)
14. [その他](#14-その他)

---

## 1. テンソル作成

### `zeros(shape: [int]) -> Tensor`
ゼロで埋められたテンソルを作成します。

```tensorlogic
let z = zeros([3, 4])  // [3, 4]のゼロテンソル
```

---

### `ones(shape: [int]) -> Tensor`
1で埋められたテンソルを作成します。

```tensorlogic
let o = ones([2, 3])  // [2, 3]の1テンソル
```

---

### `positional_encoding(seq_len: int, d_model: int) -> Tensor`
Transformer用の正弦波位置エンコーディングを生成します。

```tensorlogic
let pos = positional_encoding(100, 512)  // [100, 512]
```

**数式:**
- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

---

## 2. 形状操作

### `reshape(tensor: Tensor, new_shape: [int]) -> Tensor`
テンソルの形状を変更します。

```tensorlogic
let t = positional_encoding(6, 4)     // [6, 4]
let reshaped = reshape(t, [3, 8])     // [3, 8]
```

---

### `flatten(tensor: Tensor) -> Tensor`
テンソルを1次元に平坦化します。

```tensorlogic
let matrix = positional_encoding(4, 5)  // [4, 5]
let flat = flatten(matrix)              // [20]
```

---

### `transpose(tensor: Tensor) -> Tensor`
2次元テンソルの転置（行と列を入れ替え）。

```tensorlogic
let matrix = positional_encoding(3, 4)  // [3, 4]
let t = transpose(matrix)               // [4, 3]
```

---

### `permute(tensor: Tensor, dims: [int]) -> Tensor`
次元の順序を並べ替えます。

```tensorlogic
let tensor = positional_encoding(6, 4)  // [6, 4]
let permuted = permute(tensor, [1, 0])  // [4, 6]
```

---

### `unsqueeze(tensor: Tensor, dim: int) -> Tensor`
指定された位置に次元を追加します。

```tensorlogic
let t = positional_encoding(3, 4)     // [3, 4]
let unsqueezed = unsqueeze(t, 0)      // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor, dim: int) -> Tensor`
サイズ1の次元を削除します。

```tensorlogic
let t = positional_encoding(1, 4)  // [1, 4]
let squeezed = squeeze(t, 0)       // [4]
```

---

### `concat(tensor1: Tensor, tensor2: Tensor, dim: int) -> Tensor`
指定された次元に沿ってテンソルを結合します。

```tensorlogic
let a = positional_encoding(4, 8)
let b = positional_encoding(4, 8)
let result = concat(a, b, 1)  // [4, 16]
```

---

### `split(tensor: Tensor, size: int, dim: int) -> Tensor`
テンソルを指定されたサイズで分割します。

```tensorlogic
let t = positional_encoding(10, 4)
let part = split(t, 5, 0)  // [5, 4]
```

**注意:** 現在は単一の結果のみ返します。

---

### `chunk(tensor: Tensor, chunks: int, dim: int) -> Tensor`
テンソルを均等に分割します。

```tensorlogic
let t = positional_encoding(12, 4)
let part = chunk(t, 3, 0)  // [4, 4]
```

**注意:** 現在は単一の結果のみ返します。

---

### `broadcast_to(tensor: Tensor, target_shape: [int]) -> Tensor`
テンソルを指定された形状にブロードキャストします。

```tensorlogic
let small = positional_encoding(1, 4)
let large = broadcast_to(small, [3, 4])  // [3, 4]
```

---

## 3. 数学関数

### `exp(tensor: Tensor) -> Tensor`
要素ごとの指数関数（e^x）。

```tensorlogic
let x = positional_encoding(2, 3)
let result = exp(x)
```

---

### `log(tensor: Tensor) -> Tensor`
要素ごとの自然対数（ln(x)）。

```tensorlogic
let x = exp(positional_encoding(2, 3))
let result = log(x)
```

---

### `sqrt(tensor: Tensor) -> Tensor`
要素ごとの平方根。

```tensorlogic
let x = positional_encoding(2, 2)
let result = sqrt(x)
```

---

### `pow(tensor: Tensor, exponent: float) -> Tensor`
要素ごとのべき乗（x^n）。

```tensorlogic
let x = positional_encoding(2, 3)
let squared = pow(x, 2)  // x^2
```

---

### `sin(tensor: Tensor) -> Tensor`
要素ごとの正弦関数。

```tensorlogic
let x = positional_encoding(2, 3)
let result = sin(x)
```

---

### `cos(tensor: Tensor) -> Tensor`
要素ごとの余弦関数。

```tensorlogic
let x = positional_encoding(2, 3)
let result = cos(x)
```

---

### `tan(tensor: Tensor) -> Tensor`
要素ごとの正接関数。

```tensorlogic
let x = positional_encoding(2, 3)
let result = tan(x)
```

---

## 4. 集約操作

### `sum(tensor: Tensor) -> float`
### `sum(tensor: Tensor, dim: int, keepdim: bool) -> Tensor`
全体または次元に沿った合計。

```tensorlogic
// 全体の合計
let total = sum(tensor)

// 次元0に沿った合計
let result = sum(tensor, 0, false)  // [n]
let result_keep = sum(tensor, 0, true)  // [1, n]
```

---

### `mean(tensor: Tensor) -> float`
### `mean(tensor: Tensor, dim: int, keepdim: bool) -> Tensor`
全体または次元に沿った平均。

```tensorlogic
// 全体の平均
let avg = mean(tensor)

// 次元0に沿った平均
let result = mean(tensor, 0, false)
```

---

### `max(tensor: Tensor) -> float`
全体の最大値。

```tensorlogic
let max_val = max(tensor)
```

---

### `min(tensor: Tensor) -> float`
全体の最小値。

```tensorlogic
let min_val = min(tensor)
```

---

### `argmax(tensor: Tensor, dim: int, keepdim: bool) -> Tensor`
最大値のインデックス。

```tensorlogic
let indices = argmax(logits, 1, false)
```

---

### `argmin(tensor: Tensor, dim: int, keepdim: bool) -> Tensor`
最小値のインデックス。

```tensorlogic
let indices = argmin(tensor, 0, false)
```

---

## 5. 活性化関数

### `relu(tensor: Tensor) -> Tensor`
ReLU活性化関数（max(0, x)）。

```tensorlogic
let activated = relu(tensor)
```

---

### `gelu(tensor: Tensor) -> Tensor`
GELU活性化関数（Transformer標準）。

```tensorlogic
let activated = gelu(tensor)
```

---

### `tanh(tensor: Tensor) -> Tensor`
双曲線正接活性化関数。

```tensorlogic
let activated = tanh(tensor)
```

---

### `sigmoid(tensor: Tensor) -> Tensor`
シグモイド活性化関数（1 / (1 + e^-x)）。

```tensorlogic
let activated = sigmoid(tensor)
```

---

### `softmax(tensor: Tensor) -> Tensor`
Softmax関数（確率分布に変換）。

```tensorlogic
let logits = positional_encoding(1, 10)
let probs = softmax(logits)
```

---

## 6. 行列演算

### `matmul(a: Tensor, b: Tensor) -> Tensor`
行列乗算。

```tensorlogic
let a = positional_encoding(4, 8)  // [4, 8]
let b = positional_encoding(8, 6)  // [8, 6]
let result = matmul(a, b)          // [4, 6]
```

---

## 7. 正規化

### `layer_norm(tensor: Tensor) -> Tensor`
Layer Normalization（Transformer標準）。

```tensorlogic
let normalized = layer_norm(tensor)
```

---

### `batch_norm(tensor: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor`
Batch Normalization。

```tensorlogic
let gamma = ones([channels])
let beta = zeros([channels])
let normalized = batch_norm(tensor, gamma, beta, 1e-5)
```

---

### `dropout(tensor: Tensor, p: float, training: bool) -> Tensor`
Dropout（ランダムにニューロンを無効化）。

```tensorlogic
let dropped = dropout(tensor, 0.1, true)
```

---

## 8. マスク操作

### `causal_mask(seq_len: int) -> Tensor`
Causalマスクを生成（未来のトークンを隠す）。

```tensorlogic
let mask = causal_mask(10)  // [10, 10]
```

---

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`
Attentionスコアにマスクを適用。

```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked = apply_attention_mask(scores, mask)
```

---

### `padding_mask(lengths: [int], max_len: int) -> Tensor`
シーケンス長からパディングマスクを生成。

```tensorlogic
let lengths = [3, 5, 2, 4]
let mask = padding_mask(lengths, 5)  // [4, 5]
```

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`
複数のマスクを結合。

```tensorlogic
let combined = combine_masks(mask1, mask2)
```

---

## 9. インデックス操作

### `gather(tensor: Tensor, dim: int, indices: Tensor) -> Tensor`
インデックスで要素を収集（高度な使用）。

```tensorlogic
// GNNでの使用例
let gathered = gather(node_features, 0, neighbor_indices)
```

---

### `scatter(tensor: Tensor, dim: int, indices: Tensor, src: Tensor) -> Tensor`
インデックスに値を配置（高度な使用）。

```tensorlogic
let output = scatter(base, 0, indices, updates)
```

---

## 10. 埋め込み

### `embedding(table: Tensor, token_ids: TokenIds) -> Tensor`
トークンIDから埋め込みベクトルを取得。

```tensorlogic
let embedding_table = positional_encoding(1000, 128)
let token_ids = tokenize(tokenizer, "hello", true)
let embedded = embedding(embedding_table, token_ids)
```

---

### `load_tokenizer(path: string) -> Tokenizer`
トークナイザーを読み込み。

```tensorlogic
let tokenizer = load_tokenizer("tokenizer.json")
```

---

### `tokenize(tokenizer: Tokenizer, text: string, add_special: bool) -> TokenIds`
テキストをトークンIDに変換。

```tensorlogic
let ids = tokenize(tokenizer, "Hello world", true)
```

---

### `detokenize(tokenizer: Tokenizer, ids: TokenIds, skip_special: bool) -> string`
トークンIDをテキストに変換。

```tensorlogic
let text = detokenize(tokenizer, token_ids, true)
```

---

## 11. サンプリング

### `temperature(logits: Tensor, temp: float) -> Tensor`
温度スケーリング（サンプリングのランダム性を調整）。

```tensorlogic
let scaled = temperature(logits, 0.8)
```

---

### `top_k(logits: Tensor, k: int) -> Tensor`
Top-Kサンプリング（上位k個のみ残す）。

```tensorlogic
let filtered = top_k(logits, 50)
```

---

### `top_p(logits: Tensor, p: float) -> Tensor`
Nucleus（Top-P）サンプリング。

```tensorlogic
let filtered = top_p(logits, 0.9)
```

---

### `sample(probs: Tensor) -> int`
確率分布からサンプリング。

```tensorlogic
let probs = softmax(logits)
let token_id = sample(probs)
```

---

## 12. Fused操作

Fused操作は複数の演算を1つのカーネルで実行し、性能を最適化します。

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`
加算とReLUを融合。

```tensorlogic
let result = fused_add_relu(a, b)  // max(a + b, 0)
```

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`
乗算とReLUを融合。

```tensorlogic
let result = fused_mul_relu(a, b)  // max(a * b, 0)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`
アフィン変換（scale * x + bias）。

```tensorlogic
let result = fused_affine(x, scale, bias)
```

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`
GELU活性化と線形変換を融合。

```tensorlogic
let result = fused_gelu_linear(x, weight, bias)
```

---

## 13. 最適化

### `load_model(path: string) -> Model`
モデルを読み込み（GGUF, SafeTensors, CoreML対応）。

```tensorlogic
let model = load_model("model.gguf")
```

---

### `save(tensor: Tensor, path: string) -> void`
テンソルをファイルに保存。

```tensorlogic
save(tensor, "output.bin")
```

---

### `load(path: string) -> Tensor`
テンソルをファイルから読み込み。

```tensorlogic
let tensor = load("input.bin")
```

---

## 14. その他

### `print(value: any) -> void`
値を出力。

```tensorlogic
print("Hello")
print(tensor)
```

---

### `input(prompt: string) -> string`
ユーザー入力を受け取る。

```tensorlogic
let text = input("Enter text: ")
```

---

### `env(var_name: string) -> string`
環境変数を取得。

```tensorlogic
let path = env("HOME")
```

---

## 演算子

### 算術演算子
- `+` : 加算
- `-` : 減算
- `*` : 乗算
- `/` : 除算
- `**` : べき乗

### 比較演算子
- `==` : 等しい
- `!=` : 等しくない
- `<` : より小さい
- `>` : より大きい
- `<=` : 以下
- `>=` : 以上

### 論理演算子
- `and` : 論理AND
- `or` : 論理OR
- `not` : 論理NOT

---

## 型

- `Tensor` : テンソル
- `int` : 整数
- `float` : 浮動小数点数
- `string` : 文字列
- `bool` : 真偽値
- `TokenIds` : トークンID列
- `Tokenizer` : トークナイザー
- `Model` : モデル

---

## 関連ドキュメント

- [言語リファレンス](./language_reference.md)
- [モデルローディング](./model_loading.md)
- [GGUF量子化](./gguf_quantization.md)
- [CoreML Neural Engine](./coreml_neural_engine.md)

---

**更新履歴:**
- 2025-01-22: 初版作成（48操作）
