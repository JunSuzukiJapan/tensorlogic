# Candle関数リファレンス

TensorLogicインタープリターから呼び出せるCandle由来の関数一覧です。
すべての関数は`cndl_`接頭辞を持ち、既存の実装とは独立しています。

## 実装状況

✅ = 完全実装・テスト済み
⚠️ = 実装済み（外部ファイル依存）
🚧 = 部分実装

---

## 1. テンソル操作

### ✅ cndl_matmul(a, b) -> tensor
行列積を計算します（Candle実装）。

```tl
a := f32::ones([2, 3])
b := f32::ones([3, 4])
result := cndl_matmul(a, b)  // [2, 4]
```

**引数:**
- `a`: 行列A
- `b`: 行列B

**戻り値:** A @ B の結果

**サポート型:** f16, f32

---

### ✅ cndl_transpose(x, dim0, dim1) -> tensor
2つの次元を入れ替えます。

```tl
x := f32::ones([2, 3, 4])
result := cndl_transpose(x, 0, 2)  // [4, 3, 2]
```

**引数:**
- `x`: 入力テンソル
- `dim0`: 入れ替える次元1
- `dim1`: 入れ替える次元2

**戻り値:** 転置されたテンソル

**サポート型:** f16, f32

---

### ✅ cndl_reshape(x, shape) -> tensor
テンソルの形状を変更します。

```tl
x := f32::ones([12])
new_shape := f32::from_array([3.0, 4.0])
result := cndl_reshape(x, new_shape)  // [3, 4]
```

**引数:**
- `x`: 入力テンソル
- `shape`: 新しい形状（テンソルとして指定）

**戻り値:** 形状変更されたテンソル

**サポート型:** f16, f32

---

## 2. 数学操作

### ✅ cndl_softmax(x, dim) -> tensor
Softmax関数を適用します。

```tl
x := f32::from_array([1.0, 2.0, 3.0, 4.0])
result := cndl_softmax(x, 0)
```

**引数:**
- `x`: 入力テンソル
- `dim`: Softmaxを適用する次元

**戻り値:** Softmax適用後のテンソル

**サポート型:** f16, f32

---

### ✅ cndl_log_softmax(x, dim) -> tensor
Log Softmax関数を適用します。

```tl
x := f32::from_array([1.0, 2.0, 3.0, 4.0])
result := cndl_log_softmax(x, 0)
```

**引数:**
- `x`: 入力テンソル
- `dim`: Log Softmaxを適用する次元

**戻り値:** Log Softmax適用後のテンソル

**サポート型:** f16, f32

---

## 3. アクティベーション関数

### ✅ cndl_gelu(x) -> tensor
GELU (Gaussian Error Linear Unit) 活性化関数を適用します。

```tl
x := f32::from_array([0.0, 1.0, -1.0, 2.0])
result := cndl_gelu(x)
```

**引数:**
- `x`: 入力テンソル

**戻り値:** GELU適用後のテンソル

**サポート型:** f16, f32

---

### ✅ cndl_silu(x) -> tensor
SiLU (Swish) 活性化関数を適用します。

```tl
x := f32::from_array([0.0, 1.0, -1.0, 2.0])
result := cndl_silu(x)
```

**引数:**
- `x`: 入力テンソル

**戻り値:** SiLU適用後のテンソル

**サポート型:** f16, f32

---

### ✅ cndl_relu(x) -> tensor
ReLU (Rectified Linear Unit) 活性化関数を適用します。

```tl
x := f32::from_array([-2.0, -1.0, 0.0, 1.0, 2.0])
result := cndl_relu(x)  // [0.0, 0.0, 0.0, 1.0, 2.0]
```

**引数:**
- `x`: 入力テンソル

**戻り値:** ReLU適用後のテンソル

**サポート型:** f16, f32

---

### ✅ cndl_tanh(x) -> tensor
Tanh (Hyperbolic Tangent) 活性化関数を適用します。

```tl
x := f32::from_array([0.0, 1.0, -1.0])
result := cndl_tanh(x)
```

**引数:**
- `x`: 入力テンソル

**戻り値:** Tanh適用後のテンソル

**サポート型:** f16, f32

---

## 4. 正規化

### ✅ cndl_layer_norm(x, normalized_shape, [weight], [bias], [eps]) -> tensor
Layer Normalizationを適用します。

```tl
x := f32::ones([2, 4])
result := cndl_layer_norm(x, 4)
```

**引数:**
- `x`: 入力テンソル
- `normalized_shape`: 正規化する形状（整数またはテンソル）
- `weight`: (オプション) スケールパラメータ
- `bias`: (オプション) シフトパラメータ
- `eps`: (オプション) 安定化のための小さな値（デフォルト: 1e-5）

**戻り値:** Layer Norm適用後のテンソル

**サポート型:** f16, f32

---

### ✅ cndl_rms_norm(x, [weight], [eps]) -> tensor
RMS Normalizationを適用します。

```tl
x := f32::ones([2, 4])
result := cndl_rms_norm(x)
```

**引数:**
- `x`: 入力テンソル
- `weight`: (オプション) スケールパラメータ
- `eps`: (オプション) 安定化のための小さな値（デフォルト: 1e-5）

**戻り値:** RMS Norm適用後のテンソル

**サポート型:** f16, f32

---

## 5. ニューラルネットワーク操作

### 🚧 cndl_embedding(indices, embeddings) -> tensor
Embedding lookupを実行します。

```tl
embeddings := f32::from_array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]])
result := cndl_embedding(1, embeddings)  // [4.0, 5.0, 6.0]
```

**引数:**
- `indices`: インデックス（整数またはテンソル）
- `embeddings`: Embeddingテーブル

**戻り値:** 選択されたEmbedding

**サポート型:** f16, f32

**注意:** 現在は単一インデックスのみサポート

---

### 🚧 cndl_rope(x, position_ids, [rope_theta]) -> tensor
Rotary Position Embedding (RoPE)を適用します。

```tl
x := f32::ones([2, 8, 64])  // [seq_len, n_heads, head_dim]
result := cndl_rope(x, 0, 10000.0)
```

**引数:**
- `x`: 入力テンソル [seq_len, n_heads, head_dim]
- `position_ids`: 位置オフセット
- `rope_theta`: (オプション) RoPEの周波数パラメータ（デフォルト: 10000.0）

**戻り値:** RoPE適用後のテンソル

**サポート型:** f16, f32

**注意:** 実装は進行中、テストは#[ignore]

---

## 6. モデルの保存と読み込み

### ✅ cndl_save_safetensor(tensor, path, tensor_name) -> void
テンソルをSafetensors形式で保存します。

```tl
weights := f32::from_array([1.0, 2.0, 3.0, 4.0])
cndl_save_safetensor(weights, "weights.safetensors", "layer1.weight")
```

**引数:**
- `tensor`: 保存するテンソル
- `path`: 保存先ファイルパス
- `tensor_name`: テンソル名

**戻り値:** なし

**サポート型:** f16, f32

---

### ✅ cndl_load_safetensor(path, tensor_name) -> tensor
Safetensorsファイルからテンソルを読み込みます。

```tl
loaded := cndl_load_safetensor("weights.safetensors", "layer1.weight")
```

**引数:**
- `path`: ファイルパス
- `tensor_name`: 読み込むテンソル名

**戻り値:** 読み込まれたテンソル

**サポート型:** f16, f32（自動判別）

---

### ✅ cndl_list_safetensors(path) -> void
Safetensorsファイルの内容を一覧表示します。

```tl
cndl_list_safetensors("weights.safetensors")
```

**出力例:**
```
Tensors in weights.safetensors:
  Total: 3 tensors

  - layer1.weight : F32 [768, 3072]
  - layer2.bias : F32 [3072]
  - layer3.weight : F16 [3072, 768]
```

**引数:**
- `path`: ファイルパス

**戻り値:** なし（標準出力に表示）

---

### ⚠️ cndl_load_gguf_tensor(path, tensor_name) -> tensor
GGUFファイルからテンソルを読み込みます。

```tl
embedding := cndl_load_gguf_tensor("model.gguf", "token_embd.weight")
```

**引数:**
- `path`: GGUFファイルパス
- `tensor_name`: 読み込むテンソル名

**戻り値:** 読み込まれたテンソル

**サポート型:** f16, f32（量子化テンソルはf32に変換）

**注意:** GGUFファイルが必要

---

### ⚠️ cndl_list_gguf_tensors(path) -> void
GGUFファイルの内容を一覧表示します。

```tl
cndl_list_gguf_tensors("tinyllama-1.1b-q4_0.gguf")
```

**出力例:**
```
Tensors in tinyllama-1.1b-q4_0.gguf:
  GGUF version: 3
  Total: 201 tensors

  - blk.0.attn_k.weight : Q4_0 [2048, 512]
  - blk.0.attn_norm.weight : F32 [2048]
  - token_embd.weight : Q4_0 [2048, 32000]
```

**引数:**
- `path`: GGUFファイルパス

**戻り値:** なし（標準出力に表示）

**注意:** GGUFファイルが必要

---

## 使用例

### 基本的な使い方

```tl
main {
    // 1. 行列演算
    a := f32::ones([2, 3])
    b := f32::ones([3, 4])
    c := cndl_matmul(a, b)

    // 2. アクティベーション
    x := f32::from_array([-1.0, 0.0, 1.0, 2.0])
    activated := cndl_gelu(x)

    // 3. 正規化
    normalized := cndl_rms_norm(x)

    // 4. モデルの保存
    cndl_save_safetensor(c, "output.safetensors", "result")

    // 5. モデルの読み込み
    loaded := cndl_load_safetensor("output.safetensors", "result")

    print("Result:", loaded)
}
```

### 複雑な例（LLM推論パイプライン）

```tl
main {
    // トークンのEmbedding
    token_ids := f32::from_array([1.0, 5.0, 10.0])
    embeddings := cndl_load_safetensor("model.safetensors", "token_embd.weight")
    x := cndl_embedding(token_ids, embeddings)

    // RoPE適用
    x := cndl_rope(x, 0)

    // Attention層の重み
    q_weight := cndl_load_safetensor("model.safetensors", "attn.q.weight")
    q := cndl_matmul(x, cndl_transpose(q_weight, 0, 1))

    // RMS Norm
    q := cndl_rms_norm(q)

    // GELU活性化
    output := cndl_gelu(q)

    print("Output:", output)
}
```

---

## 既存関数との比較

| 操作 | 既存関数 | Candle関数 |
|------|---------|-----------|
| 行列積 | `matmul()` | `cndl_matmul()` |
| ReLU | `relu()` | `cndl_relu()` |
| GELU | `gelu()` | `cndl_gelu()` |
| Softmax | `softmax()` | `cndl_softmax()` |
| RMS Norm | `rms_norm()` | `cndl_rms_norm()` |
| RoPE | `rope()` | `cndl_rope()` |
| モデル読み込み | `load_model_f16()` | `cndl_load_safetensor()` |

**違い:**
- 既存関数: TensorLogicのネイティブ実装（Metal GPU最適化）
- Candle関数: Candleライブラリを使用（互換性とエコシステム連携）

---

## テスト状況

### ✅ 完全テスト済み
- cndl_matmul (f16/f32)
- cndl_softmax (f16/f32)
- cndl_gelu, cndl_silu, cndl_relu, cndl_tanh (f16/f32)
- cndl_transpose (f16/f32)
- cndl_rms_norm (f16/f32)
- cndl_save_safetensor, cndl_load_safetensor (f16/f32)
- cndl_list_safetensors

### ⚠️ 外部ファイル必要
- cndl_load_gguf_tensor (#[ignore])
- cndl_list_gguf_tensors (#[ignore])

### 🚧 実装進行中
- cndl_rope (#[ignore])
- cndl_embedding (#[ignore])
- cndl_layer_norm (基本実装のみ、weight/biasパラメータ未対応)

---

## パフォーマンス特性

- **GPU加速**: すべての関数はMetal GPU上で実行
- **型変換**: TensorLogic ↔ Candle間で自動変換（若干のオーバーヘッド）
- **メモリ効率**: データコピーが発生（今後の最適化候補）

---

## 今後の拡張予定

1. **量子化サポート**: Q4_0, Q8_0などの量子化テンソル直接操作
2. **バッチ処理**: より効率的なバッチ操作
3. **カスタムカーネル**: Candleの拡張機能活用
4. **HuggingFace統合**: HFモデルの直接ロード

---

## トラブルシューティング

### Safetensorsファイルが読めない
```
Error: Failed to load safetensors file
```
→ ファイルパスを確認、ファイルが破損していないか確認

### 型不一致エラー
```
Error: Expected f32 tensor, got f16
```
→ 既存関数とCandle関数で型を統一してください

### GGUFファイルが見つからない
```
Error: No such file or directory
```
→ #[ignore]テストは実際のGGUFファイルが必要です

---

## 関連ファイル

- 実装: `src/interpreter/builtin_candle.rs`
- テスト: `tests/test_candle_functions.rs`
- デモ: `examples/candle_functions_demo.tl`
- ドキュメント: `docs/candle_functions_reference.md`
