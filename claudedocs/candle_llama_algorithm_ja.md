# Candle LLaMA 実装 - 完全アルゴリズム解析

**ソース**: `/tmp/candle/candle-transformers/src/models/llama.rs`
**日付**: 2025-11-08
**目的**: TensorLogic transformerデバッグのためのリファレンス実装

---

## 目次

1. [概要](#概要)
2. [キャッシュ構造](#キャッシュ構造)
3. [RoPE (回転位置埋め込み)](#rope-回転位置埋め込み)
4. [アテンション機構](#アテンション機構)
5. [MLP/FFN層](#mlpffn層)
6. [Transformerブロック](#transformerブロック)
7. [モデルのフォワードパス](#モデルのフォワードパス)
8. [重要な形状変換](#重要な形状変換)
9. [TensorLogicとの主な違い](#tensorlogicとの主な違い)

---

## 概要

CandleのLLaMA実装は、以下の標準的なtransformerアーキテクチャに従っています：
- **Grouped Query Attention (GQA)**: Q headsよりも少ないKV headsを使用する効率的なアテンション
- **RoPE**: QとKに適用される回転位置埋め込み
- **SwiGLU**: ゲート付きFFN活性化関数
- **RMSNorm**: アテンションとFFNの前の事前正規化
- **KVキャッシング**: 効率的な自己回帰生成

**モデルパラメータ (TinyLlama 1.1B)**:
- 層数: 22
- 隠れ層サイズ: 2048
- Q heads: 32
- KV heads: 4
- Head次元: 64
- 語彙サイズ: 32000
- 最大位置埋め込み: 2048

---

## キャッシュ構造

**145-216行目** - キャッシュの初期化と管理

```rust
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,      // 異なるシーケンス長のための因果マスク
    pub use_kv_cache: bool,             // KVキャッシングの有効/無効
    kvs: Vec<Option<(Tensor, Tensor)>>, // 各層の(K, V)ペア
    cos: Tensor,                        // RoPE用の事前計算されたcos
    sin: Tensor,                        // RoPE用の事前計算されたsin
    device: Device,
}
```

### RoPE周波数計算 (155-161行目)

```rust
fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;  // 64
    (0..head_dim)
        .step_by(2)  // 0, 2, 4, ..., 62
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}
```

**出力**: `[1/10000^(0/64), 1/10000^(2/64), ..., 1/10000^(62/64)]` → 32個の周波数

### キャッシュ初期化 (164-216行目)

```rust
pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
    let theta = calculate_default_inv_freq(config);  // [32] 周波数
    let theta = Tensor::new(theta, device)?;

    // 位置インデックスの作成: [0, 1, 2, ..., max_pos-1]
    let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((config.max_position_embeddings, 1))?  // [2048, 1]
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;  // [2048, 32]

    // すべての位置のcosとsinを事前計算
    let cos = idx_theta.cos()?.to_dtype(dtype)?;  // [2048, 32]
    let sin = idx_theta.sin()?.to_dtype(dtype)?;  // [2048, 32]

    Ok(Self {
        masks: HashMap::new(),
        use_kv_cache,
        kvs: vec![None; config.num_hidden_layers],  // 22層
        device: device.clone(),
        cos,
        sin,
    })
}
```

**重要なポイント**:
- `cos`と`sin`は、すべての位置 [0, max_pos) に対して事前計算される
- 形状: `[max_position_embeddings=2048, head_dim/2=32]`
- 再計算を避けるためキャッシュに保存される

### 因果マスク生成 (218-229行目)

```rust
fn mask(&mut self, t: usize) -> Result<Tensor> {
    if let Some(mask) = self.masks.get(&t) {
        Ok(mask.clone())  // キャッシュされたマスクを再利用
    } else {
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }
}
```

**例** (t=4):
```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

---

## RoPE (回転位置埋め込み)

**264-270行目** - QまたはKに回転埋め込みを適用

```rust
fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
    let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;  // [1, num_heads, seq_len, head_dim]

    // 現在の位置範囲のcos/sinを抽出
    let cos = cache.cos.narrow(0, index_pos, seq_len)?;  // [seq_len, 32]
    let sin = cache.sin.narrow(0, index_pos, seq_len)?;  // [seq_len, 32]

    candle_nn::rotary_emb::rope(x, &cos, &sin)
}
```

**パラメータ**:
- `x`: QueryまたはKeyテンソル `[batch, num_heads, seq_len, head_dim]`
- `index_pos`: シーケンス内の開始位置（prefillは0、decodeは累積）
- `seq_len`: 現在のシーケンス長（prefillは35、decodeは1）

**動作**:
- **Prefill**: `index_pos=0, seq_len=35` → `cos[0:35]`, `sin[0:35]`を使用
- **Decode**: `index_pos=35, seq_len=1` → `cos[35:36]`, `sin[35:36]`を使用

---

## アテンション機構

**272-358行目** - GQAとKVキャッシングを使用した因果的自己アテンション

### 完全なフォワードパス

```rust
fn forward(
    &self,
    x: &Tensor,           // [batch=1, seq_len, hidden=2048]
    index_pos: usize,     // シーケンス内の位置
    block_idx: usize,     // 層インデックス (0-21)
    cache: &mut Cache,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_size) = x.dims3()?;

    // 1. 線形射影
    let q = self.q_proj.forward(x)?;  // [1, seq_len, 2048]
    let k = self.k_proj.forward(x)?;  // [1, seq_len, 256]
    let v = self.v_proj.forward(x)?;  // [1, seq_len, 256]

    // 2. マルチヘッド形式にリシェイプ
    let q = q
        .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?  // [1, seq_len, 32, 64]
        .transpose(1, 2)?         // [1, 32, seq_len, 64]
        .contiguous()?;

    let k = k
        .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?  // [1, seq_len, 4, 64]
        .transpose(1, 2)?         // [1, 4, seq_len, 64]
        .contiguous()?;

    let mut v = v
        .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?  // [1, seq_len, 4, 64]
        .transpose(1, 2)?;        // [1, 4, seq_len, 64]

    // 3. QとKにRoPEを適用
    let q = self.apply_rotary_emb(&q, index_pos, cache)?;  // [1, 32, seq_len, 64]
    let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;  // [1, 4, seq_len, 64]

    // 4. KVキャッシュ管理
    if cache.use_kv_cache {
        if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
            // キャッシュされたK, Vに新しいものを追加
            k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;  // [1, 4, cache_len+seq_len, 64]
            v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;  // [1, 4, cache_len+seq_len, 64]

            // オプション: max_position_embeddingsを超える場合は切り詰める
            let k_seq_len = k.dims()[2];
            if k_seq_len > self.max_position_embeddings {
                k = k.narrow(D::Minus1, k_seq_len - self.max_position_embeddings, self.max_position_embeddings)?.contiguous()?
            }
            // (Vも同様)
        }
        // 新しいK, Vでキャッシュを更新
        cache.kvs[block_idx] = Some((k.clone(), v.clone()))
    }

    // 5. GQAのためのKVの繰り返し (4 KV heads → 32 Q heads)
    let k = self.repeat_kv(k)?;  // [1, 32, total_len, 64]
    let v = self.repeat_kv(v)?;  // [1, 32, total_len, 64]

    // 6. スケール付きドット積アテンション
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;  // [1, 32, seq_len, total_len]

    // 7. 因果マスクの適用（seq_len > 1の場合のみ、つまりprefill時）
    let att = if seq_len == 1 {
        att  // 単一トークンの場合はマスク不要（decode フェーズ）
    } else {
        let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
        masked_fill(&att, &mask, f32::NEG_INFINITY)?  // 未来の位置をマスク
    };

    // 8. Softmaxと重み付き和
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let y = att.matmul(&v.contiguous()?)?;  // [1, 32, seq_len, 64]

    // 9. [batch, seq_len, hidden]にリシェイプ
    let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;  // [1, seq_len, 2048]

    // 10. 出力射影
    let y = self.o_proj.forward(&y)?;
    Ok(y)
}
```

### GQAのためのKVの繰り返し (360-362行目)

```rust
fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
    // 各KV headを8回繰り返す: 4 heads → 32 heads
    crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
}
```

**入力**: `[1, 4, seq_len, 64]`
**出力**: `[1, 32, seq_len, 64]` (各KV headが8回繰り返される)

---

## MLP/FFN層

**406-410行目** - SwiGLU活性化関数

```rust
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // SwiGLU: silu(W_gate × x) ⊙ (W_up × x)
    let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
    self.c_proj.forward(&x)
}
```

**形状の流れ**:
- 入力: `[batch, seq_len, hidden=2048]`
- `c_fc1` (W_gate): `[2048, 5632]` → `[batch, seq_len, 5632]`
- `c_fc2` (W_up): `[2048, 5632]` → `[batch, seq_len, 5632]`
- 要素ごと: `silu(gate) * up` → `[batch, seq_len, 5632]`
- `c_proj` (W_down): `[5632, 2048]` → `[batch, seq_len, 2048]`

---

## Transformerブロック

**438-455行目** - 残差接続を伴う完全なtransformer層

```rust
fn forward(
    &self,
    x: &Tensor,
    index_pos: usize,
    block_idx: usize,
    cache: &mut Cache,
) -> Result<Tensor> {
    // 事前正規化と残差接続を伴うアテンションブロック
    let residual = x;
    let x = self.rms_1.forward(x)?;  // 事前正規化
    let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;

    // 事前正規化と残差接続を伴うFFNブロック
    let residual = &x;
    let x = self.rms_2.forward(&x)?;  // 事前正規化
    self.mlp.forward(&x)? + residual
}
```

**パターン**: Pre-LayerNormアーキテクチャ
```
x → RMSNorm → Attention → + residual →
    RMSNorm → FFN → + residual → output
```

---

## モデルのフォワードパス

**505-515行目** - 🚨 **重要**: logitsのためのトークン抽出

```rust
pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
    let (_b_sz, seq_len) = x.dims2()?;  // xはトークンID: [batch, seq_len]

    // 1. 埋め込み
    let mut x = self.wte.forward(x)?;  // [batch, seq_len, hidden]

    // 2. すべてのtransformerブロックを通過
    for (block_idx, block) in self.blocks.iter().enumerate() {
        x = block.forward(&x, index_pos, block_idx, cache)?;
    }
    // x: [batch, seq_len, hidden]

    // 3. 最終正規化
    let x = self.ln_f.forward(&x)?;  // [batch, seq_len, hidden]

    // 🚨 重要: 最後のトークン位置のみを抽出
    let x = x.i((.., seq_len - 1, ..))?.contiguous()?;  // [batch, hidden]

    // 4. 語彙への射影
    let logits = self.lm_head.forward(&x)?;  // [batch, vocab_size]
    logits.to_dtype(DType::F32)
}
```

### 🚨 最後のトークン抽出が重要な理由

**Prefillフェーズ** (seq_len = 35):
- 抽出なし: `x`は`[1, 35, 2048]` → logitsは`[1, 35, 32000]` ❌
- 抽出あり: `x`は`[1, 2048]` → logitsは`[1, 32000]` ✓

**Decodeフェーズ** (seq_len = 1):
- 抽出なし: `x`は`[1, 1, 2048]` → 抽出 → `[1, 2048]` ✓
- 抽出あり: 同じ結果

**根拠**:
- Prefill中、全シーケンスを処理するが、**次トークン予測**のためのlogitsのみが必要
- 最後の位置は、因果的アテンションを通じて以前のすべての位置からのコンテキストを含む
- 他のすべての位置は生成のために無視される

---

## 重要な形状変換

### Prefillフェーズ (index_pos=0, seq_len=35)

```
トークンID:        [1, 35]
    ↓ 埋め込み
埋め込み:          [1, 35, 2048]
    ↓ 22x transformerブロック
隠れ状態:          [1, 35, 2048]
    ↓ RMSNorm
正規化:            [1, 35, 2048]
    ↓ 🚨 最後のトークンを抽出: x.i((.., seq_len-1, ..)
最後のトークン:    [1, 2048]
    ↓ linear(output.weight)
Logits:            [1, 32000] ✓
```

### Decodeフェーズ (index_pos=35, seq_len=1)

```
トークンID:        [1, 1]
    ↓ 埋め込み
埋め込み:          [1, 1, 2048]
    ↓ 22x transformerブロック (KVキャッシュあり)
隠れ状態:          [1, 1, 2048]
    ↓ RMSNorm
正規化:            [1, 1, 2048]
    ↓ 🚨 最後のトークンを抽出: x.i((.., 0, ..)
単一トークン:      [1, 2048]
    ↓ linear(output.weight)
Logits:            [1, 32000] ✓
```

### アテンションの形状の流れ (Prefill)

```
入力 x:            [1, 35, 2048]
    ↓ q_proj
Q:                 [1, 35, 2048] → reshape → [1, 35, 32, 64] → transpose → [1, 32, 35, 64]
    ↓ k_proj
K:                 [1, 35, 256]  → reshape → [1, 35, 4, 64]  → transpose → [1, 4, 35, 64]
    ↓ v_proj
V:                 [1, 35, 256]  → reshape → [1, 35, 4, 64]  → transpose → [1, 4, 35, 64]
    ↓ apply_rotary_emb
Q_rope:            [1, 32, 35, 64]
K_rope:            [1, 4, 35, 64]
    ↓ repeat_kv (K, V)
K_expanded:        [1, 32, 35, 64]
V_expanded:        [1, 32, 35, 64]
    ↓ Q @ K^T / sqrt(64)
アテンションスコア: [1, 32, 35, 35]
    ↓ 因果マスク + softmax
アテンション重み:   [1, 32, 35, 35]
    ↓ @ V
アテンション出力:   [1, 32, 35, 64]
    ↓ transpose + reshape
出力:              [1, 35, 2048]
    ↓ o_proj
最終:              [1, 35, 2048]
```

---

## TensorLogicとの主な違い

### 1. 🚨 最後のトークン抽出 (重要なバグ)

**Candle** (正しい):
```rust
let x = x.i((.., seq_len - 1, ..))?.contiguous()?;  // [1, 2048]
let logits = self.lm_head.forward(&x)?;             // [1, 32000]
```

**TensorLogic** (現在 - バグあり):
```tl
let final_norm = rms_norm(h, output_norm)  // h: [35, 2048]
let logits = linear(final_norm, tok_embd)  // logits: [35, 32000] ❌
```

**必要な修正**:
```tl
let final_norm = rms_norm(h, output_norm)      // [35, 2048]
let last_token = slice_last(final_norm, 0)    // [2048] ← 最後の行を抽出
let logits = linear(last_token, tok_embd)      // [32000] ✓
```

### 2. RoPE位置トラッキング

**Candle**: `index_pos`パラメータが絶対位置を追跡
- Prefill: `index_pos=0` (位置0から開始)
- Decodeイテレーション1: `index_pos=35` (プロンプト長)
- Decodeイテレーション2: `index_pos=36`
- など

**TensorLogic**: `position`パラメータを使用
- Candleの`index_pos`の動作に一致させるべき

### 3. KVキャッシュの形状

**Candle**: head次元を持つ完全なK, Vテンソルをキャッシュ
- K: `[batch=1, num_kv_heads=4, total_seq_len, head_dim=64]`
- V: `[batch=1, num_kv_heads=4, total_seq_len, head_dim=64]`

**TensorLogic**: フラット化されたK, Vをキャッシュ
- K: `[total_seq_len, num_kv_heads * head_dim = 256]`
- V: `[total_seq_len, 256]`

一貫して処理される場合、どちらも有効です。

### 4. 因果マスクの動作

**Candle**: `seq_len > 1`の場合のみマスクを適用
```rust
let att = if seq_len == 1 {
    att  // decodeではマスク不要
} else {
    masked_fill(&att, &mask, f32::NEG_INFINITY)
};
```

**TensorLogic**: 同じパターンに従うべき（組み込みの`attention_with_cache`が処理する可能性あり）

### 5. 重み共有

**Candle** (519-523行目):
```rust
let lm_head = if cfg.tie_word_embeddings {
    Linear::from_weights(wte.embeddings().clone(), None)
} else {
    linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
};
```

**TensorLogic**: 出力射影に`tok_embd`重みを使用
```tl
let logits = linear(final_norm, tok_embd)  // 重み共有
```

`tie_word_embeddings=true`のTinyLlamaでは、どちらも同等です。

---

## 重要な実装要件のまとめ

1. ✅ **RoPEのcos/sinを事前計算** キャッシュ初期化時にすべての位置について
2. ✅ **絶対位置を追跡** decodeイテレーション全体で(`index_pos`)
3. 🚨 **最後のトークンを抽出** 最終射影前: `x.i((.., seq_len-1, ..))`
4. ✅ **KVキャッシュに追加** decode中、置き換えではなく
5. ✅ **KV headsの繰り返し** GQAのため (4 → 32 heads)
6. ✅ **因果マスクを適用** prefill中のみ (seq_len > 1)
7. ✅ **重み共有を使用** 出力射影のため

---

## 検証チェックリスト

- [ ] RoPE周波数が正しく計算されている
- [ ] max_position_embeddingsに対してcos/sinが事前計算されている
- [ ] index_posがdecodeイテレーション全体で正しく増加している
- [ ] KVキャッシュが新しいトークンを追加し、置き換えていない
- [ ] GQAのrepeat_kvが4個のKV headsを32個に拡張している
- [ ] 因果マスクがprefill中のみ適用されている
- [ ] **🚨 logits射影前に最後のトークンが抽出されている**
- [ ] 出力形状が`[seq_len, vocab_size]`ではなく`[vocab_size]`である

---

**ドキュメント終了**
