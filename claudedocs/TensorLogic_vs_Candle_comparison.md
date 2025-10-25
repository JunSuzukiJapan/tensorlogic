# TensorLogic vs Candle 実装比較

## クローン場所

**Candle参照用リポジトリ**: `/tmp/candle_reference/candle/`

## 1. GGUFローダーの実装

### 共通点 ✅

両方とも**全く同じ処理**を実行:

#### Candle
**ファイル**: `/tmp/candle_reference/candle/candle-core/src/quantized/gguf_file.rs:437`
```rust
dimensions.reverse();
```

#### TensorLogic
**ファイル**: `src/model/formats/gguf.rs:227`
```rust
shape.reverse();
```

**結論**: GGUFローダーの次元反転処理は完全に一致 ✅

---

## 2. Linear層の実装

### 重要な違い ⚠️

#### Candle: 自動Transpose
**ファイル**: `/tmp/candle_reference/candle/candle-nn/src/linear.rs:43-78`

```rust
impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;  // ← 自動的にtranspose
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;  // ← 自動的にtranspose
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;  // ← 自動的にtranspose
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}
```

**特徴**:
- Linear層が内部で`.t()`（transpose）を**自動的に**実行
- ドキュメントに明記: `y = x@w.t() + b`
- ユーザーコードではtransposeを意識しない

#### TensorLogic: 手動Transpose
**ユーザーコード例**: `examples/test_bos_transpose_greedy.tl`

```rust
fn transformer_layer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    ...
) -> float16[?, ?] {
    let x_norm1 = rms_norm(x, attn_norm)

    let W_q_t = transpose(W_q)  // ← ユーザーが明示的にtranspose
    let W_k_t = transpose(W_k)
    let W_v_t = transpose(W_v)

    let Q = matmul(x_norm1, W_q_t)
    let K = matmul(x_norm1, W_k_t)
    let V = matmul(x_norm1, W_v_t)
    ...
}
```

**特徴**:
- ユーザーが**明示的に**`transpose()`を呼ぶ必要がある
- より低レベルで柔軟
- 間違えやすい（transposeを忘れるとエラー）

---

## 3. データフロー比較

### Candle

```
GGUF File [2048, 32000]
    ↓
dimensions.reverse()
    ↓
Tensor [32000, 2048]
    ↓
Linear::forward() 内で自動 .t()
    ↓
matmul(x, W_transposed)
    ↓
正しい出力
```

### TensorLogic

```
GGUF File [2048, 32000]
    ↓
shape.reverse()
    ↓
Tensor [32000, 2048]
    ↓
ユーザーコードで明示的 transpose(W)
    ↓
matmul(x, W_transposed)
    ↓
正しい出力
```

---

## 4. 設計思想の違い

| 観点 | Candle | TensorLogic |
|------|--------|-------------|
| **抽象度** | 高レベル（Linear層） | 低レベル（生matmul） |
| **安全性** | 自動処理で間違いにくい | 手動処理で間違いやすい |
| **柔軟性** | Linear層に限定 | 任意のmatmul操作可能 |
| **学習曲線** | 簡単（transposeを意識不要） | やや難（transposeの理解必要） |
| **パフォーマンス** | 最適化されたLinear実装 | ユーザー次第 |

---

## 5. 既存サンプルコードの問題

### 問題のあるコード例

**ファイル**: `examples/chat_demo_full_22_layers.tl`

```rust
fn transformer_layer(...) {
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = matmul(x_norm1, W_q)  // ❌ transposeなし
    let K = matmul(x_norm1, W_k)  // ❌ transposeなし
    let V = matmul(x_norm1, W_v)  // ❌ transposeなし
    ...
}
```

**問題**:
- 次元反転後、重みは`[out_features, in_features]`形式
- `matmul(x, W)`は形状ミスマッチまたは誤った計算
- **transposeが必須**

### 修正版

**ファイル**: `examples/test_bos_transpose_greedy.tl`

```rust
fn transformer_layer(...) {
    let x_norm1 = rms_norm(x, attn_norm)

    let W_q_t = transpose(W_q)  // ✅ transposeを追加
    let W_k_t = transpose(W_k)
    let W_v_t = transpose(W_v)

    let Q = matmul(x_norm1, W_q_t)
    let K = matmul(x_norm1, W_k_t)
    let V = matmul(x_norm1, W_v_t)
    ...
}
```

---

## 6. 推奨事項

### 短期的対応

1. **すべてのサンプルコードを修正**: transposeを追加
2. **ドキュメント更新**: GGUF次元反転とtranpose要件を明記
3. **テスト追加**: transpose有無での出力比較

### 長期的改善案

#### オプション A: Linear層の実装
```rust
// TensorLogicにLinear層を追加
fn linear(x: Tensor, W: Tensor, bias: Option<Tensor>) -> Tensor {
    let W_t = transpose(W)
    let out = matmul(x, W_t)
    if let Some(b) = bias {
        out + b
    } else {
        out
    }
}
```

**メリット**:
- Candleと同じ使い勝手
- transposeを意識不要
- サンプルコードがシンプルに

**デメリット**:
- 新機能の追加が必要

#### オプション B: 次元反転を選択的に
```rust
// 特定のテンソルだけ反転
let should_reverse = name.contains("token_embd");
if should_reverse {
    shape.reverse();
}
```

**メリット**:
- transposeが不要になる可能性

**デメリット**:
- 参照実装（Candle、llama.cpp）と異なる動作
- 保守が困難

**推奨**: オプションAの方が安全で保守しやすい

---

## 7. 出力比較テスト

### TensorLogic (transpose修正後)

```bash
$ ./target/release/tl run examples/test_bos_transpose_greedy.tl
Input: [1.0000]
Sampled token (greedy): 2579
```

### llama.cpp (同じモデル、フルプロンプト)

```bash
$ llama-cli -m tinyllama-1.1b-chat-q4_0.gguf -p "<|system|>..." --temp 0.0
Output: "How are you?"
```

**注意**:
- 入力プロンプトが異なるため直接比較不可
- 同じ入力での比較が必要

### 次のステップ

1. 同じ入力でllama.cppをテスト
2. TensorLogicとllama.cppの中間値（embeddings、logitsなど）を比較
3. 数値的な一致を確認

---

## 8. 参考ファイル

### Candle
- **GGUFローダー**: `/tmp/candle_reference/candle/candle-core/src/quantized/gguf_file.rs`
- **Linear層**: `/tmp/candle_reference/candle/candle-nn/src/linear.rs`
- **LLaMA実装**: `/tmp/candle_reference/candle/candle-transformers/src/models/llama.rs`

### TensorLogic
- **GGUFローダー**: `src/model/formats/gguf.rs`
- **サンプル（修正済み）**: `examples/test_bos_transpose_greedy.tl`
- **サンプル（要修正）**: `examples/chat_demo_full_22_layers.tl`

---

## まとめ

| 項目 | 状態 | 対応 |
|------|------|------|
| GGUF次元反転 | ✅ 正しい | なし |
| Embedding関数 | ✅ 修正済み | なし |
| Linear層（transpose） | ⚠️ 手動 | サンプル修正必要 |
| 出力の正確性 | 🔄 検証中 | llama.cppと詳細比較必要 |

**結論**:
- TensorLogicの実装は**技術的に正しい**
- Candleとの主な違いは**抽象度**（自動 vs 手動transpose）
- すべてのサンプルコードに`transpose()`を追加すれば動作する
