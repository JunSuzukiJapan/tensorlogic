# Transformer Tests for TensorLogic

このディレクトリには、Candle の transformer 実装をもとに作成した TensorLogic 用のテストが含まれています。

## 作成したテスト

### 1. test_transformer_simple.tl
**目的**: 基本的な transformer コンポーネントのテスト

**テスト内容**:
- Q, K, V プロジェクション (行列乗算)
- Attention scores 計算 (Q @ K^T)
- Scaled attention (1/sqrt(d_k) でスケーリング)
- Softmax attention weights
- Attention を values に適用
- Layer normalization
- Residual connection
- Feed-forward network (MLP)
- 最終 residual connection

**参照**:
- Candle の `candle-transformers/src/models/llama.rs`
- "Attention is All You Need" paper (Vaswani et al., 2017)

### 2. test_transformer_multihead_attention.tl
**目的**: Multi-head attention メカニズムのテスト

**テスト内容**:
- 複数ヘッドでの attention 計算
- ヘッドごとの独立した Q, K, V プロジェクション
- Scaled dot-product attention
- Output projection

**参照**:
- Candle の `CausalSelfAttention` 実装
- Multi-head attention の理論

### 3. test_transformer_complete_block.tl
**目的**: 完全な transformer ブロックのテスト

**テスト内容**:
- Pre-attention layer normalization
- Self-attention mechanism
- Attention output projection + residual
- Pre-FFN layer normalization
- Feed-forward network (2層 MLP)
- Final residual connection

**パターン**:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**参照**:
- Candle の `Block` 実装
- Pre-LayerNorm transformer architecture

### 4. test_transformer_sampling.tl
**目的**: テキスト生成のサンプリングメカニズムのテスト

**テスト内容**:
- Greedy sampling (argmax)
- Temperature scaling
- Top-K sampling concept
- Top-P (nucleus) sampling concept
- Softmax mathematical properties
- Probability distribution properties

**参照**:
- Candle の `generation_tests.rs`
- LogitsProcessor 実装

## Candle からの知見

### 1. RoPE (Rotary Position Embedding)
Candle の実装:
```rust
fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
    let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
    let cos = cache.cos.narrow(0, index_pos, seq_len)?;
    let sin = cache.sin.narrow(0, index_pos, seq_len)?;
    candle_nn::rotary_emb::rope(x, &cos, &sin)
}
```

事前計算された cos/sin テーブルを使用して効率化。

### 2. KV Cache
Candle の実装:
```rust
if cache.use_kv_cache {
    if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
        k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
        v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
    }
    cache.kvs[block_idx] = Some((k.clone(), v.clone()))
}
```

推論時の効率化のため、過去の key/value をキャッシュ。

### 3. SwiGLU Feed-Forward
Candle の実装:
```rust
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
    self.c_proj.forward(&x)
}
```

SiLU (Swish) activation を使った gated linear unit。

### 4. Attention Masking
Candle の実装:
```rust
let att = if seq_len == 1 {
    att  // No mask needed for single token
} else {
    let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
    masked_fill(&att, &mask, f32::NEG_INFINITY)?
};
```

Causal masking for autoregressive generation。

### 5. Multi-head Attention の構造
```
Input [batch, seq_len, hidden_size]
  ↓
Q, K, V projections → [batch, seq_len, hidden_size]
  ↓
Reshape → [batch, seq_len, num_heads, head_dim]
  ↓
Transpose → [batch, num_heads, seq_len, head_dim]
  ↓
Attention per head
  ↓
Transpose back → [batch, seq_len, num_heads, head_dim]
  ↓
Reshape → [batch, seq_len, hidden_size]
  ↓
Output projection
```

## 現在の問題

テストの実行中に以下の問題が発生しています:

1. **文字列繰り返し演算子**: `"=" * 80` が動作しない
   - 解決策: 直接文字列を記述

2. **配列内の文字列混在**: `[value1, value2, "..."]` が動作しない
   - 解決策: 文字列を除外

3. **`learnable` キーワード**: 直接値を指定する場合は使用不可
   - 解決策: `learnable` を削除

4. **行列初期化**: 大きな行列の直接初期化で異常な値が発生
   - 現在調査中: TensorLogic の実装に問題がある可能性

## 将来の作業

TensorLogic の構文が安定したら:

1. 行列初期化の問題を解決
2. すべてのテストを実行して動作確認
3. より高度なテストを追加:
   - Grouped Query Attention (GQA)
   - Flash Attention integration
   - Mixed precision training
   - Gradient checkpointing

## 参考資料

### Candle
- Repository: https://github.com/huggingface/candle
- Transformers: `candle-transformers/src/models/llama.rs`
- Generation: `candle-transformers/tests/generation_tests.rs`

### 論文
- "Attention is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "GLU Variants Improve Transformer" (Shazeer, 2020)

### TensorLogic
- Existing tests: `tests/test_transformer_paper_equations.tl`
- Rust tests: `tests/test_attention_math.rs`
