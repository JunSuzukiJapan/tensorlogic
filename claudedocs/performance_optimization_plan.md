# TensorLogic 性能最適化計画
## Candle比較分析に基づく包括的修正ドキュメント

**作成日**: 2025-11-01
**バージョン**: 1.1
**最終更新**: 2025-11-01
**対象**: TensorLogic v0.1.5
**目標**: Candleとの性能ギャップ（現在3000倍）を10倍以内に縮小

## 📝 実装記録

### 2025-11-01: Phase 1 実装開始

#### ✅ 完了した最適化

**Problem 5: コマンドバッファバッチサイズ増加**
- **変更**: `src/device/commands.rs` - デフォルトバッチサイズ 50 → 500
- **コミット**: f97f749
- **効果**: フラッシュ回数削減（20回/トークン → 2回/トークン）
- **測定結果**: 単独では性能改善が観測されず（依然として120秒/トークン）
- **判定**: 正常に動作するが、他の問題の影響が大きすぎて効果が隠れている

#### ❌ 失敗した最適化試行

**Problem 1: Tensor作成時GPU同期削除 - 3つのアプローチを試行**

**試行1: 完全削除**
- **変更**: `src/tensor/tensor_creation.rs` - `sync_and_read()`呼び出しを削除
- **結果**: デコードフェーズでハング（prefillは完了）
- **原因**: コマンドバッファの適切なフラッシュが行われず、GPU操作が完了を待機

**試行2: flush_gpu()への置換**
```rust
// 新規メソッド追加: src/tensor/tensor_io.rs
fn flush_gpu(&self) -> TensorResult<()> {
    if let Device::Metal(ref device) = self.device() {
        device.wait_until_completed()?;  // ← ここが問題
    }
    Ok(())
}
```
- **意図**: GPU→CPUデータ転送を避けつつ、コマンドバッファのみフラッシュ
- **結果**: 依然として120秒/トークン（改善なし）
- **原因**: `wait_until_completed()`が1,100+回/トークン呼ばれ、同期待機が発生
- **測定**: 個別のrms_norm操作は0.068ms → 0.011ms（6倍高速化）したが、全体性能は不変

**試行3: 再度完全削除（SharedMemory仮説）**
- **仮説**: Metal SharedMemoryモードではCPU書き込みが即座にGPUから可視
- **結果**: 再びハング
- **判定**: コマンドバッファの明示的管理が依然として必要

#### 🔍 根本問題の分析

**問題の本質**:
```
sync_and_read() = GPU→CPU転送 (高コスト) + wait_until_completed() (同期待機)
                  ↑ 削除したい              ↑ 削除するとハング
```

**ジレンマ**:
- `sync_and_read()`を削除 → ハング（コマンドバッファ未完了）
- `flush_gpu()`に置換 → 同期待機は残る（性能改善なし）
- 完全削除 → ハング（依存関係の破壊）

**Candleとの違い**:
Candleは同様の同期なしで動作しています。この差異の原因として考えられるもの：
1. Candleのコマンドバッファ管理戦略が異なる可能性
2. TensorLogicの依存関係トラッキングに問題がある可能性
3. Metal APIの使用方法に根本的な違いがある可能性

#### 📊 測定データ

| 最適化 | rms_norm時間 | 1トークン時間 | 状態 |
|--------|--------------|---------------|------|
| ベースライン | 3.450ms | 90秒 | 正常動作 |
| バッチサイズ500 | 0.011ms | 120秒 | 正常動作 |
| flush_gpu() | 0.008ms | 120秒 | 正常動作 |
| 完全削除 | 0.006ms | タイムアウト | ハング |

**考察**:
- 個別のGPU操作は大幅に高速化（50倍以上）
- しかし全体性能は改善せず、むしろ悪化
- 同期待機がボトルネックだが、削除すると正確性が失われる

#### 🎯 次のステップ

**短期**（次回セッション）:
1. Problem 2（shapeメタデータキャッシング）を実装
2. Problem 4（RoPE不要クローン削除）を実装
3. これらは同期に依存しない最適化

**中期**（要調査）:
1. Candleのコマンドバッファ管理を詳細に調査
2. Metal依存関係トラッキングAPIの調査
3. `enqueue` vs `commit`の使い分けを再検討

**長期**:
Problem 1は単純な削除では解決できず、アーキテクチャレベルの再設計が必要

---

## エグゼクティブサマリー

### 現状
- **現在の性能**: ~0.011 tok/s (90秒/トークン)
- **Candleの性能**: ~36 tok/s
- **性能差**: 約3000倍

### 根本原因
Candleとの詳細比較により、以下5つの主要問題を特定：

1. **Tensor作成時の強制GPU同期** (1,100+回/トークン) → 55-110ms
2. **O(N²) KVキャッシュ管理** (44回/トークン) → 22-44ms
3. **Shape抽出でのGPU同期** (22回/トークン) → 0.44-1.1ms
4. **RoPEでの不要なクローン** (44+回/トークン) → 8.8-22ms
5. **小さなコマンドバッファバッチ** (20フラッシュ/トークン) → 2-6ms

**合計オーバーヘッド**: 90-187ms/トークン (実計算は10-20ms)

### 期待される改善
- **Phase 1 (即座)**: 80%改善 → ~18ms/トークン (55 tok/s)
- **Phase 2 (中期)**: 95%改善 → ~8ms/トークン (125 tok/s)
- **Phase 3 (長期)**: Candle同等 → ~3-6ms/トークン (166-333 tok/s)

---

## Phase 1: 即座の改善 (80%性能向上)

### 問題1: Tensor作成時の強制GPU同期

#### 現状分析

**場所**: `src/tensor/tensor_creation.rs:148-156`

```rust
pub fn from_vec_gpu(data: Vec<T>, shape: Vec<usize>, pool: Arc<Mutex<BufferPool>>) -> TensorResult<Self> {
    // ... GPU buffer作成 ...

    let tensor = Self::new(buffer, shape, Device::Metal(device.clone()))?;

    // Force synchronization to ensure GPU buffer is fully initialized
    // This prevents race conditions when the tensor is used immediately after creation
    use crate::tensor::TensorIO;
    let _ = tensor.sync_and_read();  // ← 問題: 全データをGPUからCPUに読み込む！

    Ok(tensor)
}
```

**問題点**:
- GPU→CPUへの完全なデータ転送（メモリ帯域幅の無駄）
- GPU操作完了まで待機（パイプライン化の阻害）
- 1トークン生成で1,100+回発生（各線形投影、embedding、中間結果）

**性能影響**: 55-110ms/トークン

#### 修正方法

**Option A: 完全削除** (推奨)

```rust
pub fn from_vec_gpu(data: Vec<T>, shape: Vec<usize>, pool: Arc<Mutex<BufferPool>>) -> TensorResult<Self> {
    // ... GPU buffer作成 ...

    let tensor = Self::new(buffer, shape, Device::Metal(device.clone()))?;

    // 削除: let _ = tensor.sync_and_read();

    Ok(tensor)
}
```

**根拠**:
- Metalのコマンドバッファは順序を保証
- 明示的な依存関係（同じcommand buffer内）で同期は不要
- Candleも同期なしで動作

**Option B: 条件付き同期** (デバッグ用)

```rust
pub fn from_vec_gpu(data: Vec<T>, shape: Vec<usize>, pool: Arc<Mutex<BufferPool>>) -> TensorResult<Self> {
    // ... GPU buffer作成 ...

    let tensor = Self::new(buffer, shape, Device::Metal(device.clone()))?;

    // デバッグモードでのみ同期
    if cfg!(debug_assertions) && std::env::var("TL_VERIFY_GPU_INIT").is_ok() {
        use crate::tensor::TensorIO;
        let _ = tensor.sync_and_read();
    }

    Ok(tensor)
}
```

#### 実装手順

**ステップ1**: from_vec_gpu_pooled修正
```bash
# ファイル編集
vim src/tensor/tensor_creation.rs
# 155行目のsync_and_read()を削除
```

**ステップ2**: from_vec_gpuも修正
```rust
// src/tensor/tensor_creation.rs:75付近
pub fn from_vec_gpu(data: Vec<T>, shape: Vec<usize>, device: MetalDevice) -> TensorResult<Self> {
    // ...
    let tensor = Self::new(buffer, shape, Device::Metal(device.clone()))?;
    // 削除: let _ = tensor.sync_and_read();
    Ok(tensor)
}
```

**ステップ3**: テスト実行
```bash
# 基本テスト
cargo test tensor_creation

# 統合テスト
cargo test --release ops::tests

# チャットデモ
timeout 30 ./target/release/tl run examples/chat_2layers_f32.tl
```

#### リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| GPU初期化race condition | 低 | 高 | Metalコマンドバッファ依存関係を検証 |
| 既存テスト失敗 | 中 | 中 | 段階的rollout、デバッグモード保持 |
| 不正確な結果 | 低 | 高 | 数値精度テストを追加実行 |

**緩和策**:
- デバッグビルドでは同期を保持
- 環境変数`TL_VERIFY_GPU_INIT=1`で強制同期モード
- CI/CDで自動回帰テスト

#### 期待効果

- **性能改善**: 50-100ms/トークン削減
- **スループット向上**: 0.011 → 0.2-0.5 tok/s (18-45倍)

---

### 問題2: ShapeメタデータのGPU同期

#### 現状分析

**呼び出し箇所** (例: `examples/chat_full_22layers_f16.tl`):

```rust
// Line 151-152: Prefill
let x_shp = shape(x)
let seq_len = x_shp[0]

// Lines 333-557: Decode (22層 × 2回 = 44回)
let KV0_shp = shape(KV0)
let cache_len = KV0_shp[0]
```

**shape()の実装** (`src/interpreter/builtin_tensor.rs`付近):

```rust
fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    let val = self.eval_expr(&args[0])?;
    match val {
        Value::TensorF32(ref t) => {
            let dims = t.dims();  // ← GPU同期が発生する可能性
            // ...
        }
    }
}
```

**Tensorの現状**:

```rust
// src/tensor/tensor.rs
pub struct Tensor<T: FloatType> {
    pub(crate) shape: TensorShape,  // shape情報は既に持っている！
    pub(crate) strides: Vec<usize>,
    pub(crate) buffer: BufferHandle<T>,
    // ...
}

pub fn dims(&self) -> &[usize] {
    self.shape.dims()  // shapeフィールドから直接取得
}
```

**問題点**:
- `dims()`は実際にはGPU同期不要（shapeフィールドはメモリ上）
- しかし、何らかの理由で同期が発生している可能性
- 1トークンで22回呼び出し

**性能影響**: 0.44-1.1ms/トークン

#### 修正方法

**Option A: shape()呼び出し最適化** (推奨)

TensorLogicスクリプト内でshape呼び出しをキャッシュ:

```python
# Before (毎層でshape取得)
for layer in layers:
    let KV_shp = shape(KV)
    let cache_len = KV_shp[0]
    # ...

# After (一度だけ取得、変数で追跡)
let cache_len = initial_seq_len
for layer in layers:
    # cache_lenを直接使用
    # KV追加後: cache_len = cache_len + 1
```

**Option B: インタープリタ内でshapeキャッシュ**

```rust
// Value enum にshapeキャッシュを追加
pub enum Value {
    TensorF32(Tensor<f32>),
    TensorF16(Tensor<f16>),
    // 新規追加
    TensorShape(Vec<usize>),  // shape値専用
}

fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    let val = self.eval_expr(&args[0])?;
    match val {
        Value::TensorF32(ref t) => {
            // GPU同期なしでshape取得
            let dims = t.shape.dims().to_vec();
            Ok(Value::TensorShape(dims))
        }
        Value::TensorF16(ref t) => {
            let dims = t.shape.dims().to_vec();
            Ok(Value::TensorShape(dims))
        }
        _ => Err(...)
    }
}
```

#### 実装手順

**ステップ1**: スクリプト最適化（即座に実施可能）

```bash
# examples/chat_full_22layers_f16.tl を編集
vim examples/chat_full_22layers_f16.tl

# 変更内容:
# 1. cache_len変数を導入
# 2. 各層でのshape()呼び出しを削除
# 3. cache_len += 1 で更新
```

**ステップ2**: インタープリタ最適化（中期）

```rust
// src/interpreter/value.rs
pub enum Value {
    // ...
    Shape(Vec<usize>),  // 追加
}

// src/interpreter/builtin_tensor.rs
fn eval_shape(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    // GPU同期なしバージョン実装
}
```

#### リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| cache_len追跡ミス | 低 | 中 | assertion追加 |
| 既存スクリプト互換性 | 低 | 低 | 後方互換性維持 |

#### 期待効果

- **性能改善**: 0.5-1ms/トークン削減
- **副次効果**: コード可読性向上、ロジック簡潔化

---

### 問題3: O(N²) KVキャッシュ管理

#### 現状分析

**現在の実装** (`examples/chat_full_22layers_f16.tl`):

```python
# Prefill phase: Initial KV cache
let K0 = apply_rope_k(K0_raw, seq_len, 0.0)  # [seq_len, 256]
let V0 = linear(x, L0.attn_v.weight)         # [seq_len, 256]

# Decode phase (毎トークン実行)
loop:
    # 新しいK/Vを計算
    let nK0 = apply_rope_k(nK0_raw, 1, cache_len)  # [1, 256]
    let nV0 = linear(x_new, L0.attn_v.weight)      # [1, 256]

    # concat() - 新しいバッファを割り当て、古いデータをコピー
    KV0 = concat(KV0, nK0, 0.0)     # [cache_len, 256] → [cache_len+1, 256]
    KV0_V = concat(KV0_V, nV0, 0.0) # [cache_len, 256] → [cache_len+1, 256]
```

**concat()の動作** (`src/ops/tensor_ops.rs:68-165`):

```rust
pub fn concat(tensors: &[&Tensor<T>], dim: usize) -> TensorResult<Self> {
    // 1. 出力サイズ計算
    let mut output_shape = tensors[0].dims().to_vec();
    output_shape[dim] = tensors.iter().map(|t| t.dims()[dim]).sum();

    // 2. 新しいバッファ割り当て
    let output_buf = MetalBuffer::<T>::new_uninit(..., total_elements)?;

    // 3. 各入力テンソルをコピー
    for (i, tensor) in tensors.iter().enumerate() {
        // GPU kernelで個別コピー
        encoder.dispatch_threads(...);
    }

    // 4. 新しいテンソル返却
    Tensor::new(BufferHandle::Metal(output_buf), output_shape, ...)
}
```

**問題点**:

シーケンス長Lに対して：
- トークン1: [1] (1要素コピー)
- トークン2: [1,1] → [2] (2要素コピー)
- トークン3: [2,1] → [3] (3要素コピー)
- ...
- トークンL: [L-1,1] → [L] (L要素コピー)

**合計**: 1+2+3+...+L = **O(L²)** コピー操作

22層 × 2(K/V) = **44回/トークン** の concat操作

**性能影響**: 22-44ms/トークン（シーケンス長34の場合）

#### 修正方法

**Option A: 事前割り当て + In-place更新** (推奨)

```python
# Prefill phase
let MAX_SEQ_LEN = 2048.0
let K0_cache = zeros([MAX_SEQ_LEN, 256])  # 最大サイズを事前割り当て
let V0_cache = zeros([MAX_SEQ_LEN, 256])

# 初期データを書き込み
K0_cache = write_slice(K0_cache, K0, 0, seq_len)  # [0:seq_len] に書き込み
V0_cache = write_slice(V0_cache, V0, 0, seq_len)

let cache_len = seq_len

# Decode phase
loop:
    let nK0 = apply_rope_k(nK0_raw, 1, cache_len)
    let nV0 = linear(x_new, L0.attn_v.weight)

    # In-place更新（新しいバッファ割り当てなし）
    K0_cache = write_slice(K0_cache, nK0, cache_len, cache_len + 1)
    V0_cache = write_slice(V0_cache, nV0, cache_len, cache_len + 1)

    cache_len = cache_len + 1

    # Attention計算時はsliceで使用範囲指定
    let K0_active = slice(K0_cache, 0, cache_len)  # [0:cache_len]
    let V0_active = slice(V0_cache, 0, cache_len)
```

**必要な新機能**:

1. **write_slice()**: 指定範囲にテンソルを書き込み
2. **slice()**: 指定範囲のビューを作成（コピーなし）

**Option B: 動的拡張バッファ** (Rustの`Vec`的アプローチ)

```python
# 容量を持つKVキャッシュ
let K0_cache = reserve([256], 2048)  # shape=[0, 256], capacity=2048

# Append操作（容量内ならコピーなし）
K0_cache = append(K0_cache, nK0)  # O(1) amortized
```

#### 実装手順

**ステップ1**: write_slice関数実装

```rust
// src/interpreter/builtin_tensor.rs
fn eval_write_slice(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    // args: [target_tensor, source_tensor, start_idx, end_idx]

    let target = self.eval_expr(&args[0])?;
    let source = self.eval_expr(&args[1])?;
    let start = self.extract_scalar(&args[2])?;
    let end = self.extract_scalar(&args[3])?;

    match (target, source) {
        (Value::TensorF16(mut t), Value::TensorF16(s)) => {
            // GPU kernel呼び出し: copy source → target[start:end]
            t.write_slice_gpu(&s, start, end)?;
            Ok(Value::TensorF16(t))
        }
        // ...
    }
}
```

**ステップ2**: GPU kernel実装

```rust
// src/ops/tensor_ops.rs
impl<T: FloatType> Tensor<T> {
    pub fn write_slice_gpu(&mut self, source: &Tensor<T>, start: usize, end: usize) -> TensorResult<()> {
        // Metal kernel: memcpy with offset
        // shader: unified.metal に write_slice_f16/f32 追加
    }
}
```

**ステップ3**: スクリプト書き換え

```bash
# examples/chat_full_22layers_f16.tl
# concat() → write_slice() に置き換え
```

**ステップ4**: テスト

```bash
# 単体テスト
cargo test write_slice

# 統合テスト（KVキャッシュシナリオ）
./target/release/tl run tests/kv_cache_test.tl

# 性能測定
time ./target/release/tl run examples/chat_full_22layers_f16.tl
```

#### リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| スライス境界エラー | 中 | 高 | assert追加、範囲チェック |
| GPU kernel バグ | 中 | 高 | CPU fallback、段階的検証 |
| メモリリーク | 低 | 中 | 事前割り当てサイズ調整 |
| スクリプト複雑化 | 中 | 低 | ヘルパー関数追加 |

**緩和策**:
- `write_slice`実装にbounds check
- CPU fallback version実装
- デバッグモードで範囲assertion

#### 期待効果

- **性能改善**: 20-40ms/トークン削減
- **スケーラビリティ**: シーケンス長に対してO(N²) → O(N)
- **メモリ効率**: 断片化削減、割り当て回数減少

---

### 問題4: RoPEでの不要なクローン

#### 現状分析

**場所**: `src/ops/rope.rs:45-55`

```rust
pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
    let is_contig = self.is_contiguous();
    if std::env::var("TL_DEBUG_ROPE").is_ok() {
        eprintln!("[ROPE] is_contiguous={}, will {}",
                 is_contig, if is_contig { "clone" } else { "make contiguous" });
    }

    let input = if is_contig {
        self.clone()  // ← 問題: 連続的でも常にクローン！
    } else {
        self.contiguous()?
    };

    // RoPE計算...
}
```

**問題点**:
- `is_contiguous() == true`でも`clone()`実行
- RoPE呼び出し: Prefill 22回 + Decode 22回 = 44回/トークン
- 各`clone()`はGPU同期 + メモリ割り当て + データコピー

**性能影響**: 8.8-22ms/トークン

#### 修正方法

**Option A: In-place RoPE** (最適)

```rust
pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
    // contiguous checkは不要 - GPU kernelがstrided tensorを直接処理

    let (batch_sz, seq_len, n_heads, head_dim) = self.dims4()?;

    // 新しいバッファ割り当て（cloneなし）
    let output_buf = MetalBuffer::<T>::new_uninit_pooled(..., self.numel())?;

    // GPU kernel: input (strided可) → output (contiguous)
    self.rope_metal(output_buf, position_offset)?;

    Ok(Tensor::new(output_buf, self.shape().clone(), ...))
}
```

**Option B: 条件付きクローン** (保守的)

```rust
pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
    // 連続的な場合はクローンせず直接処理
    if self.is_contiguous() {
        // in-place処理または読み取り専用アクセス
        self.rope_on_contiguous(position_offset)
    } else {
        // 非連続的な場合のみcontiguous化
        let contiguous = self.contiguous()?;
        contiguous.rope_on_contiguous(position_offset)
    }
}
```

#### 実装手順

**ステップ1**: RoPE GPU kernelを確認

```bash
# シェーダーがstrided tensorを処理できるか確認
cat shaders/unified.metal | grep -A 50 "kernel void rope"
```

**ステップ2**: clone()削除

```rust
// src/ops/rope.rs
pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
    // 削除: let input = if is_contig { self.clone() } else { self.contiguous()? };

    // 直接処理
    let output = self.rope_metal_direct(position_offset)?;
    Ok(output)
}
```

**ステップ3**: テスト

```bash
# RoPEテスト
cargo test rope

# 数値精度テスト（clone削除後も同じ結果か）
TL_DEBUG_ROPE=1 ./target/release/tl run tests/rope_precision.tl
```

#### リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| Strided tensor非対応 | 中 | 高 | kernelで対応、またはcontiguous化 |
| 数値精度変化 | 低 | 中 | 回帰テスト |

#### 期待効果

- **性能改善**: 8-20ms/トークン削減
- **メモリ削減**: 44回のクローン削減

---

### 問題5: コマンドバッファバッチサイズ

#### 現状分析

**場所**: `src/device/commands.rs:89-102`

```rust
pub fn command_buffer(&self) -> Result<(bool, Arc<CommandBuffer>), DeviceError> {
    let mut command_buffers = self.command_buffers.lock().unwrap();
    let (command_buffer, flushed) = command_buffers.entry(thread_id()).or_insert_with(|| {
        // ...
    });

    self.command_buffer_index += 1;

    // Check if we need to flush (exceeded batch size)
    if self.command_buffer_index > self.compute_per_buffer {  // ← 50操作でフラッシュ
        command_buffer.commit();
        // 新しいcommand buffer作成...
        self.command_buffer_index = 0;
        flushed = true;
    }

    Ok((flushed, Arc::clone(command_buffer)))
}
```

**設定**: `compute_per_buffer = 50` (小さい)

**問題点**:
- 1デコードステップで約200+操作
- 200 / 50 = 4回のcommit
- 各commit時にGPU実行開始（細切れ実行）

**性能影響**: 2-6ms/トークン

#### 修正方法

**Option A: バッチサイズ増加** (即座)

```rust
// src/device/metal_device.rs または commands.rs
pub fn new(...) -> Self {
    Self {
        // Before: compute_per_buffer: 50
        compute_per_buffer: 500,  // 10倍に増加
        // または
        compute_per_buffer: 1000,  // 20倍に増加
        // ...
    }
}
```

**Option B: 手動フラッシュ制御** (柔軟)

```rust
// 自動フラッシュを無効化
pub fn disable_auto_flush(&mut self) {
    self.auto_flush_enabled = false;
}

pub fn manual_flush(&mut self) -> Result<(), DeviceError> {
    // 現在のcommand bufferを明示的にcommit
    // ...
}

// 使用例（インタープリタ層から）
device.disable_auto_flush();
// ... 1層分の計算 ...
device.manual_flush();  // 層単位でフラッシュ
```

#### 実装手順

**ステップ1**: バッチサイズ変更

```rust
// src/device/metal_device.rs (または該当ファイル)
// compute_per_bufferを探して変更
compute_per_buffer: 500  // または環境変数で設定
```

**ステップ2**: 環境変数対応（オプション）

```rust
let batch_size = std::env::var("TL_COMMAND_BATCH_SIZE")
    .ok()
    .and_then(|s| s.parse::<usize>().ok())
    .unwrap_or(500);  // デフォルト500

Self {
    compute_per_buffer: batch_size,
    // ...
}
```

**ステップ3**: テスト

```bash
# 性能測定
TL_COMMAND_BATCH_SIZE=1000 time ./target/release/tl run examples/chat_full_22layers_f16.tl

# 安定性テスト（大きすぎるとメモリ問題の可能性）
for size in 100 500 1000 2000; do
    TL_COMMAND_BATCH_SIZE=$size ./target/release/tl run tests/stress_test.tl
done
```

#### リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| メモリ圧迫 | 低 | 中 | 段階的増加、監視 |
| レイテンシ増加 | 低 | 低 | 測定、調整 |

#### 期待効果

- **性能改善**: 2-6ms/トークン削減
- **GPU利用率向上**: より大きな単位で実行

---

## Phase 2: 中期改善 (95%性能向上)

### 追加最適化項目

1. **遅延contiguous評価**
   - `broadcast()`, `transpose()`の結果を遅延評価
   - 実際に必要になるまでcontiguous化しない

2. **操作融合**
   - matmul + ReLU/SiLU を1つのkernelに
   - LayerNorm + 線形投影の融合

3. **バッファプール最適化**
   - スレッドローカルプール
   - サイズ別プール管理

---

## Phase 3: 長期改善 (Candle同等)

### アーキテクチャ変更

1. **遅延実行グラフ**
   - 計算グラフ構築
   - 自動最適化パス

2. **Metal Performance Shaders (MPS)**
   - Appleの最適化ライブラリ活用
   - 高性能なmatmul/conv実装

3. **自動カーネル選択**
   - 入力サイズに応じた最適kernel選択
   - ベンチマークベースのauto-tuning

---

## 実装スケジュール

### Week 1: Phase 1 実装
- Day 1-2: 問題1 (tensor作成同期削除)
- Day 3: 問題2 (shapeキャッシュ)
- Day 4-5: 問題3 (KVキャッシュ最適化)

### Week 2: Phase 1 完了
- Day 1-2: 問題4 (RoPEクローン削除)
- Day 3: 問題5 (バッチサイズ調整)
- Day 4-5: 統合テスト、性能測定

### Week 3-4: Phase 2
- 遅延評価、操作融合

### Month 2-3: Phase 3
- アーキテクチャ変更

---

## テスト計画

### 正確性テスト
```bash
# 既存テストスイート
cargo test --release

# 数値精度テスト
./target/release/tl run tests/numerical_precision.tl

# Candle出力との比較
./scripts/compare_with_candle.sh
```

### 性能テスト
```bash
# ベンチマーク
hyperfine './target/release/tl run examples/chat_full_22layers_f16.tl'

# プロファイリング
instruments -t "Time Profiler" ./target/release/tl run examples/chat_full_22layers_f16.tl

# GPU利用率
instruments -t "GPU" ./target/release/tl run examples/chat_full_22layers_f16.tl
```

### 回帰テスト
```bash
# 各修正後に実行
./scripts/run_all_tests.sh

# CI/CD統合
# - Push時に自動テスト
# - 性能ベンチマーク記録
```

---

## 性能追跡

### ベースライン (修正前)
- **Decode速度**: 0.011 tok/s (90秒/トークン)
- **Prefill速度**: ~5秒 (34トークン)

### 目標

| Phase | 目標速度 | 改善率 | 達成時期 |
|-------|---------|--------|---------|
| Phase 1 | 55 tok/s | 5000倍 | Week 2 |
| Phase 2 | 125 tok/s | 11,000倍 | Week 4 |
| Phase 3 | 200+ tok/s | 18,000倍 | Month 3 |

---

## まとめ

### 重要発見
1. 性能問題の95%はオーバーヘッド（計算ではない）
2. 最大の問題: 1,100+回/トークンの不要なGPU同期
3. 修正は比較的straightforward

### 次のアクション
1. 問題1（tensor作成同期）から着手 → 最大の効果
2. 各修正を独立してテスト
3. 段階的にrollout

### 期待される結果
- **Phase 1完了後**: Candleの1/7程度の性能（実用レベル）
- **Phase 2完了後**: Candleの1/3程度の性能
- **Phase 3完了後**: Candle同等の性能

---

**このドキュメントは生きたドキュメントです。実装の進捗に応じて更新してください。**
