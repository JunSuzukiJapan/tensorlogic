# TensorLogic最適化戦略 - Candle知見に基づく緻密な計画

**策定日**: 2025-11-11
**基盤**: Candle transformer実装の深層分析
**目標**: 22層TinyLlama 1.1Bの高速・正確な推論

---

## 目次

1. [現状分析とボトルネック](#現状分析とボトルネック)
2. [Phase 1: RoPE最適化](#phase-1-rope最適化)
3. [Phase 2: RMSNorm数値安定性](#phase-2-rmsnorm数値安定性)
4. [Phase 3: KVキャッシュ管理](#phase-3-kvキャッシュ管理)
5. [Phase 4: Attention最適化](#phase-4-attention最適化)
6. [Phase 5: メモリレイアウト最適化](#phase-5-メモリレイアウト最適化)
7. [Phase 6: Metal Shader最適化](#phase-6-metal-shader最適化)
8. [実装優先順位と工数見積もり](#実装優先順位と工数見積もり)

---

## 現状分析とボトルネック

### 既存の実装状況

**✅ 実装済み**:
- RoPE基本実装（`rope()`, `rope_candle()`）
- Metal shader（f16/f32対応）
- タイル化matmul
- 基本的なtensor操作

**⚠️ 改善の余地**:
1. **RoPE**: 毎回計算 vs Candleの事前計算+キャッシング
2. **型変換**: F16のまま計算 vs Candleの内部F32計算
3. **KVキャッシュ**: 実装の詳細不明
4. **Attention**: seq_len==1での最適化なし
5. **メモリレイアウト**: 連続性チェックのみ、最適化余地あり

### パフォーマンスボトルネック（推定）

1. **計算ボトルネック** (60%):
   - RoPE周波数計算の重複（各forward呼び出し）
   - F16精度による累積誤差
   - 非最適なAttentionパス

2. **メモリボトルネック** (30%):
   - GPU↔CPU転送
   - 非連続テンソルのコピー
   - KVキャッシュの非効率な管理

3. **同期ボトルネック** (10%):
   - 過剰なcommand buffer flush
   - 不要なGPU-CPU同期

---

## Phase 1: RoPE最適化

### 目標

- [ ] cos/sinの事前計算とキャッシング実装
- [ ] 計算量削減: O(seq_len × head_dim) → O(1) (forward呼び出しごと)
- [ ] メモリ使用量: +2 × max_pos × head_dim/2 要素（TinyLlama: ~256KB）

### アルゴリズム詳細

#### Step 1: RoPE周波数計算（初期化時のみ）

```rust
/// RoPE Cache構造体
pub struct RopeCache {
    cos: Tensor<f16>,  // [max_position_embeddings, head_dim/2]
    sin: Tensor<f16>,  // [max_position_embeddings, head_dim/2]
    max_positions: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl RopeCache {
    /// 初期化時にすべての位置についてcos/sinを事前計算
    pub fn new(
        device: &MetalDevice,
        max_positions: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> TensorResult<Self> {
        // 1. 周波数計算
        let freqs = Self::calculate_freqs(head_dim, rope_theta);

        // 2. 位置インデックス × 周波数の行列を作成
        let positions: Vec<f32> = (0..max_positions)
            .map(|i| i as f32)
            .collect();

        // 3. cos/sin値を計算
        let mut cos_data = Vec::with_capacity(max_positions * freqs.len());
        let mut sin_data = Vec::with_capacity(max_positions * freqs.len());

        for pos in &positions {
            for &freq in &freqs {
                let theta = pos * freq;
                cos_data.push(f16::from_f32(theta.cos()));
                sin_data.push(f16::from_f32(theta.sin()));
            }
        }

        // 4. GPUテンソルとして保存
        let cos = Tensor::from_vec_gpu(
            device,
            cos_data,
            vec![max_positions, freqs.len()],
        )?;

        let sin = Tensor::from_vec_gpu(
            device,
            sin_data,
            vec![max_positions, freqs.len()],
        )?;

        Ok(Self {
            cos,
            sin,
            max_positions,
            head_dim,
            rope_theta,
        })
    }

    /// 周波数計算（Candleアルゴリズム）
    fn calculate_freqs(head_dim: usize, rope_theta: f32) -> Vec<f32> {
        (0..head_dim)
            .step_by(2)  // 次元ペアごと
            .map(|i| {
                let exponent = i as f32 / head_dim as f32;
                1.0 / rope_theta.powf(exponent)
            })
            .collect()
    }

    /// 指定位置範囲のcos/sinを抽出
    pub fn get_cos_sin(
        &self,
        start_pos: usize,
        seq_len: usize,
    ) -> TensorResult<(Tensor<f16>, Tensor<f16>)> {
        if start_pos + seq_len > self.max_positions {
            return Err(TensorError::InvalidOperation(
                format!(
                    "Position range [{}, {}) exceeds max_positions {}",
                    start_pos,
                    start_pos + seq_len,
                    self.max_positions
                )
            ));
        }

        // narrowでスライス取得（GPUメモリ上で効率的）
        let cos_slice = self.cos.narrow(0, start_pos, seq_len)?;
        let sin_slice = self.sin.narrow(0, start_pos, seq_len)?;

        Ok((cos_slice, sin_slice))
    }
}
```

#### Step 2: RoPE適用（forward時）

```rust
/// Attention層でのRoPE適用
pub fn apply_rope_with_cache(
    q: &Tensor<f16>,          // [batch, n_heads, seq_len, head_dim]
    k: &Tensor<f16>,          // [batch, n_heads, seq_len, head_dim]
    rope_cache: &RopeCache,
    position_offset: usize,   // KVキャッシュ内の開始位置
) -> TensorResult<(Tensor<f16>, Tensor<f16>)> {
    let seq_len = q.dims()[2];

    // 1. 事前計算されたcos/sinを取得（GPU上でのnarrow操作のみ）
    let (cos, sin) = rope_cache.get_cos_sin(position_offset, seq_len)?;

    // 2. RoPE適用（既存のrope_candleカーネル使用）
    let q_rope = q.rope_candle(&cos, &sin)?;
    let k_rope = k.rope_candle(&cos, &sin)?;

    Ok((q_rope, k_rope))
}
```

### Metal Shader最適化

既存の`rope_candle`カーネルを最適化:

```metal
/// Optimized RoPE kernel with precomputed cos/sin
kernel void rope_candle_optimized_f16(
    device const half* input [[buffer(0)]],   // [seq_len, n_heads, head_dim]
    device const half* cos [[buffer(1)]],     // [seq_len, head_dim/2]
    device const half* sin [[buffer(2)]],     // [seq_len, head_dim/2]
    device half* output [[buffer(3)]],
    constant uint4& params [[buffer(4)]],     // [seq_len, n_heads, head_dim, head_dim/2]
    uint gid [[thread_position_in_grid]]
) {
    const uint seq_len = params.x;
    const uint n_heads = params.y;
    const uint head_dim = params.z;
    const uint half_dim = params.w;

    const uint total_elements = seq_len * n_heads * head_dim;
    if (gid >= total_elements) return;

    // 3D indexing: [seq_len, n_heads, head_dim]
    const uint pos = gid / (n_heads * head_dim);
    const uint head = (gid / head_dim) % n_heads;
    const uint dim = gid % head_dim;

    // Dimension pair index
    const uint dim_pair = dim / 2;
    const bool is_first = (dim % 2 == 0);

    // Index into cos/sin: [pos, dim_pair]
    const uint cs_idx = pos * half_dim + dim_pair;

    // Load cos/sin values (shared across all heads)
    const half c = cos[cs_idx];
    const half s = sin[cs_idx];

    // Load input values for this dimension pair
    const uint pair_base = gid - (dim % 2);
    const half x0 = input[pair_base];
    const half x1 = input[pair_base + 1];

    // Apply 2D rotation
    // y0 = x0 * cos - x1 * sin
    // y1 = x0 * sin + x1 * cos
    half result;
    if (is_first) {
        result = x0 * c - x1 * s;
    } else {
        result = x0 * s + x1 * c;
    }

    output[gid] = result;
}
```

### パフォーマンス予測

**現状** (rope()メソッド):
- 各forward呼び出しで周波数計算: ~32回のpow()演算
- 各トークン・各ヘッド・各次元でsin/cos計算
- 計算量: O(seq_len × n_heads × head_dim)

**最適化後** (rope_candle()メソッド + キャッシング):
- 初期化時のみ周波数計算: 1回のみ
- forward時はnarrow + kernel起動のみ
- 計算量: O(seq_len × n_heads × head_dim)（カーネル内は同じだが、事前計算により定数項が大幅削減）

**期待される高速化**: 1.5-2.0x（RoPE処理のみ）

---

## Phase 2: RMSNorm数値安定性

### 目標

- [ ] F16入力でも内部F32計算で数値安定性確保
- [ ] Candleパターンの適用
- [ ] 連続テンソルの場合の専用カーネル使用

### アルゴリズム詳細

#### Step 1: 数値安定な RMSNorm実装

```rust
/// 数値安定なRMSNorm（Candleスタイル）
pub fn rms_norm_stable<T: FloatType>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    eps: f32,
) -> TensorResult<Tensor<T>> {
    let is_f16 = std::mem::size_of::<T>() == 2;

    if is_f16 {
        // F16入力の場合、内部F32で計算
        rms_norm_f16_to_f32_impl(input, weight, eps)
    } else {
        // F32入力の場合、そのまま計算
        rms_norm_f32_impl(input, weight, eps)
    }
}

/// F16 → F32 → F16 パス（数値安定）
fn rms_norm_f16_to_f32_impl(
    input: &Tensor<f16>,
    weight: &Tensor<f16>,
    eps: f32,
) -> TensorResult<Tensor<f16>> {
    // 連続性チェック
    if input.is_contiguous() {
        // 専用カーネル使用（F16→F32変換組み込み）
        rms_norm_f16_optimized_kernel(input, weight, eps)
    } else {
        // フォールバック: 連続化してから処理
        let contiguous = input.contiguous()?;
        rms_norm_f16_optimized_kernel(&contiguous, weight, eps)
    }
}
```

#### Step 2: Metal Shader実装

```metal
/// Numerically stable RMS Norm for f16 inputs
/// Internal computation in f32 for stability
kernel void rms_norm_stable_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint batch_seq_idx = gid.y * grid_size.x + gid.x;
    const uint offset = batch_seq_idx * hidden_size;

    // Step 1: Compute mean of squares in f32
    float mean_sq = 0.0f;
    for (uint i = 0; i < hidden_size; i++) {
        float val = float(input[offset + i]);  // Convert to f32
        mean_sq += val * val;
    }
    mean_sq /= float(hidden_size);

    // Step 2: Compute RMS
    float rms = sqrt(mean_sq + eps);

    // Step 3: Normalize and scale
    for (uint i = 0; i < hidden_size; i++) {
        float val = float(input[offset + i]);
        float normalized = val / rms;
        float scaled = normalized * float(weight[i]);
        output[offset + i] = half(scaled);  // Convert back to f16
    }
}
```

### 最適化ポイント

1. **型変換戦略**:
   - 入力: F16
   - 内部計算: F32（mean_sq, rms計算）
   - 出力: F16

2. **メモリアクセスパターン**:
   - 連続メモリアクセス（coalesced）
   - ループ内でF16→F32変換
   - 最小限のメモリ往復

3. **数値精度**:
   - `mean_sq`をF32で累積→精度向上
   - `sqrt(mean_sq + eps)`もF32→安定性向上

---

## Phase 3: KVキャッシュ管理

### 目標

- [ ] Candleスタイルの追加方式KVキャッシュ実装
- [ ] 層別キャッシュ管理
- [ ] max_position_embeddings制限の実装

### アルゴリズム詳細

#### Step 1: KVキャッシュ構造体

```rust
/// KVキャッシュ（Candleスタイル）
#[derive(Debug, Clone)]
pub struct KVCache {
    /// 各層のKVペア: kvs[layer_idx] = Some((K, V))
    /// K, V shape: [batch=1, num_kv_heads, total_seq_len, head_dim]
    kvs: Vec<Option<(Tensor<f16>, Tensor<f16>)>>,

    /// キャッシング有効/無効
    use_kv_cache: bool,

    /// 最大位置埋め込み
    max_position_embeddings: usize,

    /// デバイス
    device: MetalDevice,
}

impl KVCache {
    /// 新しいKVキャッシュを作成
    pub fn new(
        num_layers: usize,
        use_kv_cache: bool,
        max_position_embeddings: usize,
        device: MetalDevice,
    ) -> Self {
        Self {
            kvs: vec![None; num_layers],
            use_kv_cache,
            max_position_embeddings,
            device,
        }
    }

    /// KVキャッシュの取得と更新
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: Tensor<f16>,  // [batch, num_kv_heads, seq_len, head_dim]
        new_v: Tensor<f16>,
    ) -> TensorResult<(Tensor<f16>, Tensor<f16>)> {
        if !self.use_kv_cache {
            // キャッシング無効の場合、そのまま返す
            return Ok((new_k, new_v));
        }

        let (k, v) = if let Some((cached_k, cached_v)) = &self.kvs[layer_idx] {
            // キャッシュ存在: 連結
            let k = Tensor::cat(&[cached_k, &new_k], 2)?;  // seq_len軸で連結
            let v = Tensor::cat(&[cached_v, &new_v], 2)?;

            // 長さ制限チェック
            let k = Self::trim_if_needed(k, self.max_position_embeddings)?;
            let v = Self::trim_if_needed(v, self.max_position_embeddings)?;

            (k, v)
        } else {
            // キャッシュなし: 新規
            (new_k, new_v)
        };

        // キャッシュ更新
        self.kvs[layer_idx] = Some((k.clone(), v.clone()));

        Ok((k, v))
    }

    /// 最大長を超える場合の切り詰め
    fn trim_if_needed(
        tensor: Tensor<f16>,
        max_len: usize,
    ) -> TensorResult<Tensor<f16>> {
        let seq_len = tensor.dims()[2];  // [batch, heads, seq_len, head_dim]

        if seq_len > max_len {
            // 古い部分を削除、最新のmax_len分のみ保持
            let start = seq_len - max_len;
            tensor.narrow(2, start, max_len)
        } else {
            Ok(tensor)
        }
    }

    /// キャッシュクリア
    pub fn clear(&mut self) {
        for kv in &mut self.kvs {
            *kv = None;
        }
    }

    /// 現在のシーケンス長を取得
    pub fn current_seq_len(&self, layer_idx: usize) -> usize {
        self.kvs[layer_idx]
            .as_ref()
            .map(|(k, _v)| k.dims()[2])
            .unwrap_or(0)
    }
}
```

#### Step 2: Attention層での使用

```rust
/// CausalSelfAttention with KV cache
pub fn forward_with_cache(
    &self,
    x: &Tensor<f16>,           // [batch, seq_len, hidden_size]
    layer_idx: usize,
    kv_cache: &mut KVCache,
    rope_cache: &RopeCache,
) -> TensorResult<Tensor<f16>> {
    let (batch, seq_len, _hidden) = x.dims3()?;

    // 1. Q, K, V projections
    let q = self.q_proj.forward(x)?;  // [batch, seq_len, q_size]
    let k = self.k_proj.forward(x)?;  // [batch, seq_len, kv_size]
    let v = self.v_proj.forward(x)?;  // [batch, seq_len, kv_size]

    // 2. Reshape to multi-head
    let q = q.reshape(&[batch, seq_len, self.num_q_heads, self.head_dim])?
             .transpose(1, 2)?;  // [batch, num_q_heads, seq_len, head_dim]

    let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
             .transpose(1, 2)?;  // [batch, num_kv_heads, seq_len, head_dim]

    let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
             .transpose(1, 2)?;  // [batch, num_kv_heads, seq_len, head_dim]

    // 3. Apply RoPE
    let position_offset = kv_cache.current_seq_len(layer_idx);
    let (q_rope, k_rope) = apply_rope_with_cache(&q, &k, rope_cache, position_offset)?;

    // 4. Update KV cache
    let (k_full, v_full) = kv_cache.update(layer_idx, k_rope, v)?;
    // k_full, v_full: [batch, num_kv_heads, total_seq_len, head_dim]

    // 5. Repeat KV for GQA
    let k_repeated = repeat_kv(k_full, self.num_q_heads / self.num_kv_heads)?;
    let v_repeated = repeat_kv(v_full, self.num_q_heads / self.num_kv_heads)?;
    // k_repeated, v_repeated: [batch, num_q_heads, total_seq_len, head_dim]

    // 6. Scaled dot-product attention
    let total_seq_len = k_repeated.dims()[2];
    let attn = compute_attention(
        &q_rope,
        &k_repeated,
        &v_repeated,
        seq_len,
        total_seq_len,
        self.head_dim,
    )?;

    // 7. Output projection
    let output = attn.transpose(1, 2)?
                     .reshape(&[batch, seq_len, self.hidden_size])?;
    self.o_proj.forward(&output)
}
```

#### Step 3: GQA用のKV繰り返し

```rust
/// Repeat KV heads for Grouped Query Attention
/// Input:  [batch, num_kv_heads, seq_len, head_dim]
/// Output: [batch, num_q_heads, seq_len, head_dim]
fn repeat_kv(
    tensor: Tensor<f16>,
    n_rep: usize,  // num_q_heads / num_kv_heads
) -> TensorResult<Tensor<f16>> {
    if n_rep == 1 {
        return Ok(tensor);  // No repetition needed
    }

    let dims = tensor.dims();
    let batch = dims[0];
    let num_kv_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    // Reshape: [batch, num_kv_heads, 1, seq_len, head_dim]
    let expanded = tensor.reshape(&[batch, num_kv_heads, 1, seq_len, head_dim])?;

    // Broadcast: [batch, num_kv_heads, n_rep, seq_len, head_dim]
    let broadcasted = expanded.broadcast_to(&[batch, num_kv_heads, n_rep, seq_len, head_dim])?;

    // Flatten: [batch, num_q_heads, seq_len, head_dim]
    broadcasted.reshape(&[batch, num_kv_heads * n_rep, seq_len, head_dim])
}
```

---

## Phase 4: Attention最適化

### 目標

- [ ] seq_len == 1の場合の因果マスク省略
- [ ] F32での内部計算
- [ ] 効率的なsoftmax実装

### アルゴリズム詳細

#### Step 1: 条件付きAttention計算

```rust
/// Scaled dot-product attention with optional causal masking
fn compute_attention(
    q: &Tensor<f16>,      // [batch, num_heads, seq_len, head_dim]
    k: &Tensor<f16>,      // [batch, num_heads, total_seq_len, head_dim]
    v: &Tensor<f16>,      // [batch, num_heads, total_seq_len, head_dim]
    seq_len: usize,       // クエリのシーケンス長
    total_seq_len: usize, // キー/バリューのシーケンス長（キャッシュ含む）
    head_dim: usize,
) -> TensorResult<Tensor<f16>> {
    // 1. F32に変換（数値安定性のため）
    let q_f32 = q.to_f32()?;
    let k_f32 = k.to_f32()?;
    let v_f32 = v.to_f32()?;

    // 2. Attention scores: Q @ K^T / sqrt(head_dim)
    let scale = (head_dim as f32).sqrt();
    let scores = q_f32.matmul(&k_f32.transpose(-2, -1)?)?
                      .div_scalar(scale)?;
    // scores: [batch, num_heads, seq_len, total_seq_len]

    // 3. Apply causal mask (only if seq_len > 1)
    let masked_scores = if seq_len == 1 {
        // Decode phase: no masking needed
        // Single query can attend to all previous keys
        scores
    } else {
        // Prefill phase: apply causal mask
        // Prevent attending to future positions
        apply_causal_mask(scores, seq_len)?
    };

    // 4. Softmax
    let attn_weights = softmax_last_dim(&masked_scores)?;

    // 5. Weighted sum: weights @ V
    let output = attn_weights.matmul(&v_f32)?;

    // 6. Convert back to f16
    output.to_f16()
}
```

#### Step 2: 因果マスク生成

```rust
/// Generate and apply causal mask
fn apply_causal_mask(
    scores: Tensor<f32>,  // [batch, num_heads, seq_len, total_seq_len]
    seq_len: usize,
) -> TensorResult<Tensor<f32>> {
    let total_seq_len = scores.dims()[3];

    // Causal mask: 上三角行列（対角線含まず）
    // mask[i, j] = 1 if j > i else 0
    let mut mask_data = vec![0f32; seq_len * total_seq_len];
    for i in 0..seq_len {
        for j in 0..total_seq_len {
            // i番目のクエリは、i + (total_seq_len - seq_len)までのキーに attend可能
            let current_pos = i + (total_seq_len - seq_len);
            if j > current_pos {
                mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let device = scores.device();
    let mask = Tensor::from_vec_gpu(
        device,
        mask_data,
        vec![seq_len, total_seq_len],
    )?;

    // Broadcast mask to match scores shape
    let mask_broadcasted = mask.broadcast_to(scores.dims())?;

    // Add mask to scores
    scores.add(&mask_broadcasted)
}
```

#### Step 3: 高速Softmax実装

```metal
/// Numerically stable softmax on last dimension
kernel void softmax_last_dim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& last_dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one row
    const uint row_start = gid * last_dim;

    // Step 1: Find max (for numerical stability)
    float max_val = -INFINITY;
    for (uint i = 0; i < last_dim; i++) {
        max_val = max(max_val, input[row_start + i]);
    }

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint i = 0; i < last_dim; i++) {
        float exp_val = exp(input[row_start + i] - max_val);
        output[row_start + i] = exp_val;
        sum += exp_val;
    }

    // Step 3: Normalize
    for (uint i = 0; i < last_dim; i++) {
        output[row_start + i] /= sum;
    }
}
```

---

## Phase 5: メモリレイアウト最適化

### 目標

- [ ] 連続メモリアクセスの保証
- [ ] 不要なコピーの削減
- [ ] contiguous()呼び出しの最小化

### 戦略

#### 1. 早期連続化

```rust
/// Ensure contiguous layout before expensive operations
fn optimize_layout<T: FloatType>(tensor: &Tensor<T>) -> TensorResult<Tensor<T>> {
    if tensor.is_contiguous() {
        // Zero-copy: just borrow
        Ok(tensor.shallow_clone())
    } else {
        // Copy to contiguous layout
        tensor.contiguous()
    }
}

/// Apply to critical paths
pub fn forward_optimized(&self, x: &Tensor<f16>) -> TensorResult<Tensor<f16>> {
    // Ensure contiguous at entry point
    let x_contig = optimize_layout(x)?;

    // All subsequent operations on contiguous tensor
    let h = self.attention.forward(&x_contig)?;
    let output = self.mlp.forward(&h)?;

    Ok(output)
}
```

#### 2. Reshape最適化

```rust
/// Efficient reshape without copy when possible
impl<T: FloatType> Tensor<T> {
    pub fn reshape_efficient(&self, new_shape: &[usize]) -> TensorResult<Self> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.numel() {
            return Err(TensorError::InvalidOperation(
                format!("Cannot reshape {} elements to {:?}", self.numel(), new_shape)
            ));
        }

        // Check if reshape can be done without copy
        if self.can_reshape_without_copy(new_shape) {
            // Zero-copy reshape (just update metadata)
            Ok(self.reshape_metadata_only(new_shape))
        } else {
            // Need to make contiguous first
            let contiguous = self.contiguous()?;
            Ok(contiguous.reshape_metadata_only(new_shape))
        }
    }

    fn can_reshape_without_copy(&self, _new_shape: &[usize]) -> bool {
        // Simple check: tensor must be contiguous
        self.is_contiguous()
    }
}
```

---

## Phase 6: Metal Shader最適化

### 目標

- [ ] Threadgroup memory活用
- [ ] Coalesced memory access
- [ ] レジスタ使用量最適化

### 戦略

#### 1. Matmul最適化（既存を活用）

現在の`matmul_tiled_f16`は既に最適化済み:
- 16×16タイル
- Threadgroup memory
- F32累積

**追加最適化**: Bias fusionとActivation fusion活用

```tl
// Instead of separate operations:
let h = linear(x, w)       // Matmul
let h = add(h, b)          // Bias addition
let h = silu(h)            // Activation

// Use fused kernel:
let h = linear_bias_silu(x, w, b)  // Single kernel launch
```

#### 2. 最適化されたRoPEカーネル（Phase 1参照）

#### 3. 最適化されたRMSNormカーネル（Phase 2参照）

---

## 実装優先順位と工数見積もり

### Phase 1: RoPE最適化 ⭐⭐⭐ [優先度: 最高]

**工数**: 3-4日
**期待効果**: 1.5-2.0x 高速化（RoPE処理）

**実装タスク**:
- [ ] Day 1-2: `RopeCache`構造体実装
  - 周波数計算関数
  - cos/sin事前計算
  - get_cos_sin()メソッド
- [ ] Day 2-3: `apply_rope_with_cache()`実装
  - Attention層への統合
  - position_offset管理
- [ ] Day 3-4: Metal shader最適化
  - rope_candle_optimized_f16カーネル
  - テストとベンチマーク

**検証方法**:
```bash
# Before optimization
TL_PERF=1 ./target/release/tl run:examples/chat_demo_22layers.tl

# After optimization
TL_PERF=1 ./target/release/tl run:examples/chat_demo_22layers_optimized.tl

# Compare RoPE execution time
```

### Phase 2: RMSNorm数値安定性 ⭐⭐ [優先度: 高]

**工数**: 2-3日
**期待効果**: 精度向上、わずかな高速化

**実装タスク**:
- [ ] Day 1: `rms_norm_stable()`関数実装
  - F16→F32→F16パス
  - 連続性チェック
- [ ] Day 2: Metal shader実装
  - rms_norm_stable_f16カーネル
  - F32内部計算
- [ ] Day 3: テストと検証
  - 数値精度テスト
  - Candleとの比較

### Phase 3: KVキャッシュ管理 ⭐⭐⭐ [優先度: 最高]

**工数**: 4-5日
**期待効果**: 正確性向上、Decode phase高速化

**実装タスク**:
- [ ] Day 1-2: `KVCache`構造体実装
  - 層別管理
  - update()メソッド
  - trim_if_needed()
- [ ] Day 2-3: Attention層統合
  - forward_with_cache()
  - position_offset管理
- [ ] Day 3-4: repeat_kv()実装
  - GQA対応
  - 効率的なbroadcast
- [ ] Day 4-5: テストとデバッグ
  - Prefill/Decodeテスト
  - 長文生成テスト

### Phase 4: Attention最適化 ⭐⭐ [優先度: 中]

**工数**: 3-4日
**期待効果**: 1.2-1.5x 高速化（Attention処理）

**実装タスク**:
- [ ] Day 1-2: compute_attention()実装
  - F16→F32変換
  - 条件付きマスク
- [ ] Day 2-3: apply_causal_mask()実装
  - 効率的なマスク生成
  - キャッシュ機構
- [ ] Day 3-4: Softmax最適化
  - Metal shader実装
  - 数値安定性確保

### Phase 5: メモリレイアウト最適化 ⭐ [優先度: 低]

**工数**: 2-3日
**期待効果**: メモリ使用量削減、わずかな高速化

**実装タスク**:
- [ ] Day 1: optimize_layout()実装
- [ ] Day 2: reshape_efficient()実装
- [ ] Day 3: プロファイリングと調整

### Phase 6: Metal Shader最適化 ⭐ [優先度: 低]

**工数**: 継続的（他のPhaseに組み込み）
**期待効果**: 全体的なパフォーマンス向上

---

## 推奨実装順序

### Sprint 1 (Week 1): 基盤整備
1. **Phase 1**: RoPE最適化（4日）
2. **Phase 3**: KVキャッシュ基本実装（3日）

**目標**: Prefill/Decodeの正確性確保

### Sprint 2 (Week 2): 精度とパフォーマンス
3. **Phase 2**: RMSNorm数値安定性（3日）
4. **Phase 3**: KVキャッシュ完成（2日）
5. **Phase 4**: Attention最適化（2日）

**目標**: Candleレベルの精度達成

### Sprint 3 (Week 3): 仕上げ
6. **Phase 4**: Attention最適化完成（2日）
7. **Phase 5**: メモリレイアウト最適化（2日）
8. 総合テストとベンチマーク（3日）

**目標**: 実用レベルのパフォーマンス

---

## 成功指標

### 定量的指標

1. **速度**:
   - Prefill: < 0.5秒 (35トークン)
   - Decode: < 0.05秒/トークン
   - 全体: 50トークン生成 < 3秒

2. **精度**:
   - Candleとのlogits誤差: < 0.01
   - 22層すべてで安定動作

3. **メモリ**:
   - GPU使用量: < 2GB
   - キャッシュ効率: > 90%

### 定性的指標

- [ ] 正確な文章生成（Candleと同等）
- [ ] 安定した長文生成（> 200トークン）
- [ ] デバッグしやすいコード構造
- [ ] 拡張可能なアーキテクチャ

---

## リスクと対策

### リスク 1: 実装の複雑性

**対策**:
- 段階的実装（Phase毎）
- 各Phaseでテスト・検証
- Candle実装を参考に

### リスク 2: パフォーマンス目標未達

**対策**:
- プロファイリングツール活用（`TL_PERF=1`）
- ボトルネック特定と集中最適化
- Metal Performance Shaders検討

### リスク 3: 数値精度問題

**対策**:
- F32内部計算の徹底
- Candleとの比較テスト
- 段階的デバッグ

---

## 結論

この最適化戦略は、Candleの実装から学んだベストプラクティスをTensorLogicに適用するものです。段階的な実装により、リスクを最小化しながら、高速で正確な推論を実現します。

**最重要ポイント**:
1. RoPE事前計算とキャッシング
2. KVキャッシュの追加方式管理
3. F16→F32→F16の数値安定パス
4. 条件付きAttention最適化

これらを実装することで、TensorLogicはCandleと同等の品質と性能を達成できます。

---

**ドキュメント終了**
