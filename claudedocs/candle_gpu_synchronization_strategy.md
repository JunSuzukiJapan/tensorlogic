# Candle GPU Synchronization Strategy - 詳細分析

## 概要

Candleは**遅延同期（Lazy Synchronization）** 戦略を採用し、Metal GPUとの同期オーバーヘッドを最小化しています。このドキュメントでは、CandleのMetal backend実装における同期戦略を詳細に分析します。

## 1. 基本アーキテクチャ

### 1.1 主要コンポーネント

```
MetalDevice
  ├─ Commands (Arc<RwLock<Commands>>)
  │   └─ ThreadLocal Command Buffers
  ├─ Buffer Pool (Arc<RwLock<HashMap<...>>>)
  ├─ Kernels (Arc<Kernels>)
  └─ Command Queue
```

### 1.2 スレッドローカルCommand Buffer

```rust
// candle-metal-kernels/src/metal/command_buffer.rs
pub struct CommandBufferThreadMap {
    inner: HashMap<ThreadId, CommandBuffer>,
}
```

**設計原則:**
- 各スレッドが独自のcommand bufferを保持
- スレッド間のロック競合を最小化
- ThreadId をキーとしたHashMapで管理

## 2. Command Buffer バッチング戦略

### 2.1 バッチングの仕組み

```rust
// candle-metal-kernels/src/metal/commands.rs
pub fn flush_if_needed(&mut self) -> Result<bool> {
    let mut flushed = false;
    if self.command_buffer_index > self.compute_per_buffer {
        command_buffer.commit();
        *command_buffer = create_command_buffer(&self.command_queue)?;
        self.command_buffer_index = 0;
        flushed = true;
    }
    Ok(flushed)
}
```

**バッチングパラメータ:**
- `compute_per_buffer`: 1つのcommand bufferあたりのcompute encoder数
- デフォルト: **50 encoders**
- 環境変数で制御可能: `CANDLE_METAL_COMPUTE_PER_BUFFER`

### 2.2 バッチング戦略の利点

```
┌─────────────────────────────────────────┐
│ Command Buffer 1                        │
│ ├─ Encoder 1  (add operation)          │
│ ├─ Encoder 2  (mul operation)          │
│ ├─ Encoder 3  (matmul operation)       │
│ ├─ ...                                  │
│ └─ Encoder 50 (softmax operation)      │
└─────────────────────────────────────────┘
         ↓ commit()
┌─────────────────────────────────────────┐
│ Command Buffer 2                        │
│ ├─ Encoder 51 (next operation)         │
│ └─ ...                                  │
└─────────────────────────────────────────┘
```

**メリット:**
- 50回の操作を1回のcommitにまとめる
- GPU submitのオーバーヘッドを1/50に削減
- GPU側での並列実行機会が増加

## 3. 遅延同期（Lazy Synchronization）

### 3.1 基本原則

**重要:** Candleは**CPU読み取り時のみ**GPU-CPU同期を行います。

```rust
// candle-core/src/metal_backend/mod.rs:2046-2059
pub(crate) fn to_cpu<T: Clone>(&self) -> Result<Vec<T>> {
    let size = self.count * self.dtype.size_in_bytes();
    let buffer = self.device.allocate_buffer(size)?;
    {
        let command_buffer = self.device.command_buffer()?;
        command_buffer.set_label("to_cpu");
        let blit = command_buffer.blit_command_encoder();
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
        blit.end_encoding();
    }
    self.device.wait_until_completed()?;  // ← ここで同期！
    Ok(read_to_vec(&buffer, self.count))
}
```

### 3.2 同期が発生するタイミング

```rust
// Tensor::to_vec() 呼び出し時
// candle-core/src/tensor.rs:630-633
match &*self.storage() {
    Storage::Cpu(_) => /* no sync */,
    Storage::Cuda(storage) => storage.to_cpu_storage()?,  // sync
    Storage::Metal(storage) => storage.to_cpu_storage()?, // sync
}
```

**同期フロー:**
```
Tensor::to_vec()
  → storage.to_cpu_storage()
    → storage.to_cpu()
      → blit encoder (GPU-GPU copy)
      → wait_until_completed()  ← 実際の同期ポイント
        → commands.wait_until_completed()
```

### 3.3 wait_until_completed の実装

```rust
// candle-core/src/metal_backend/device.rs:135-139
pub fn wait_until_completed(&self) -> Result<()> {
    let mut commands = self.commands.write().map_err(MetalError::from)?;
    commands.wait_until_completed().map_err(MetalError::from)?;
    Ok(())
}
```

```rust
// candle-metal-kernels/src/metal/command_buffer.rs:101-118
pub fn wait_until_completed(&self) -> Result<()> {
    for command_buffer in self.inner.values() {
        match command_buffer.status() {
            MTLCommandBufferStatus::NotEnqueued => continue,
            MTLCommandBufferStatus::Enqueued
            | MTLCommandBufferStatus::Committed
            | MTLCommandBufferStatus::Scheduled => {
                command_buffer.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => continue,
            MTLCommandBufferStatus::Error => {
                return Err(MetalKernelError::FailedCompletion.into());
            }
        }
    }
    Ok(())
}
```

**同期の詳細:**
- 全スレッドのcommand bufferをチェック
- Statusベースの条件付き待機
- 既にCompletedの場合はスキップ（効率化）

## 4. Encoder ライフサイクル

### 4.1 自動クリーンアップ

```rust
// candle-metal-kernels/src/metal/encoder.rs:84-88
impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}
```

**重要な設計:**
- Scopeを抜ける際に自動的に`end_encoding()`
- RAII (Resource Acquisition Is Initialization) パターン
- 手動のcleanup不要

### 4.2 Encoder使用パターン

```rust
// 典型的な使用例
{
    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("affine");

    // encoderはスコープ内で自動管理
    candle_metal_kernels::call_affine(
        &device.device,
        &command_buffer,
        &device.kernels,
        "affine_f32",
        el,
        src,
        &buffer,
        mul as f32,
        add as f32,
    )?;

    // ここでscopeを抜けるとencoder.end_encoding()が自動で呼ばれる
}
```

## 5. Buffer管理戦略

### 5.1 Buffer Pool

```rust
// candle-core/src/metal_backend/device.rs
buffers: Arc<RwLock<HashMap<(usize, String), Weak<Buffer>>>>
```

**プーリング戦略:**
- サイズ × ラベルでBuffer識別
- WeakポインタでBufferを追跡
- 参照カウントベースの再利用

### 5.2 Buffer再利用アルゴリズム

```rust
// candle-core/src/metal_backend/device.rs:73-89
pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
    let mut buffers = self.buffers.write().map_err(MetalError::from)?;
    let buffer = buffers
        .entry((size, "cpu".to_string()))
        .or_insert_with(|| Weak::new());
    match buffer.upgrade() {
        Some(buffer) if Arc::strong_count(&buffer) <= 1 => {
            // 再利用可能: 他に参照者がいない
            Ok(buffer)
        }
        _ => {
            // 新規作成が必要
            let buffer = self.new_buffer_with_size(size, "cpu")?;
            *buffer = Arc::downgrade(&buffer);
            Ok(buffer)
        }
    }
}
```

**再利用条件:**
```
if Arc::strong_count(&buffer) <= 1 {
    // ✅ 再利用可能
} else {
    // ❌ まだ使用中 → 新規作成
}
```

### 5.3 Buffer Pool の利点

1. **メモリ効率:**
   - 頻繁なalloc/deallocを回避
   - GPU memory fragmentationを削減

2. **パフォーマンス:**
   - Buffer作成のオーバーヘッド削減
   - CPU-GPU転送の最小化

3. **自動管理:**
   - 参照カウントによる自動解放
   - 手動メモリ管理不要

## 6. 同期戦略の性能分析

### 6.1 同期オーバーヘッドの最小化

```
GPU操作のタイムライン:
┌────────────────────────────────────────────────────┐
│ GPU Operations (no CPU sync)                      │
│ ├─ [op1][op2][op3]...[op50] → commit()           │
│ ├─ [op51][op52]...[op100] → commit()             │
│ └─ ...                                             │
│                                                    │
│ ❌ No sync here (just GPU-GPU operations)         │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│ to_cpu() called                                    │
│ ├─ blit_encoder (GPU-GPU copy)                    │
│ ├─ wait_until_completed() ← ✅ sync here!        │
│ └─ read CPU buffer                                │
└────────────────────────────────────────────────────┘
```

### 6.2 Transformer推論での同期パターン

```rust
// 典型的なLLaMA推論ループ
for token_pos in 0..max_tokens {
    // 1. Embedding lookup (GPU)
    let embedding = embed_tokens.forward(&tokens)?;

    // 2. 22 transformer layers (GPU only, no sync!)
    for layer in &layers {
        x = layer.forward(&x, &cache, token_pos)?;
    }

    // 3. Final norm + logits (GPU only)
    let logits = rms_norm.forward(&x)?;
    let logits = lm_head.forward(&logits)?;

    // 4. ⚠️ ここで初めて同期が発生
    let logits_vec = logits.to_vec1::<f32>()?;  // ← to_cpu()

    // 5. Sampling (CPU)
    let next_token = sample_argmax(&logits_vec);
}
```

**性能特性:**
- 22層の全transformer計算: GPU only（同期なし）
- 同期ポイント: token生成ごとに1回のみ
- 同期頻度: O(tokens) であり O(layers × tokens) ではない

### 6.3 理論的性能利得

**従来型（即座同期）:**
```
Operations per token: ~200 (22 layers × ~9 ops/layer)
Syncs per token: ~200
Total sync overhead: 200 × sync_cost
```

**Candle（遅延同期）:**
```
Operations per token: ~200
Syncs per token: 1
Total sync overhead: 1 × sync_cost
```

**性能向上:**
```
Speedup = 200 / 1 = 200x (理論値)
```

実際の性能向上は同期コストに依存しますが、**数倍〜数十倍**の高速化が期待できます。

## 7. TensorLogicへの応用戦略

### 7.1 現状のTensorLogic同期

**問題点:**
```rust
// src/interpreter/builtin_tensor.rs
// 各操作後に即座同期している可能性
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let result = backend.add(a, b)?;
    result.sync()?;  // ← 不要な同期！
    Ok(result)
}
```

### 7.2 Candle風の遅延同期実装

#### 7.2.1 CommandBuffer管理

```rust
// src/gpu/command_buffer.rs
use std::collections::HashMap;
use std::thread::ThreadId;

pub struct CommandBufferPool {
    // スレッドローカルcommand buffer
    thread_buffers: HashMap<ThreadId, ThreadLocalBuffer>,
    // バッチング設定
    compute_per_buffer: usize,
}

struct ThreadLocalBuffer {
    command_buffer: MTLCommandBuffer,
    encoder_count: usize,
}

impl CommandBufferPool {
    pub fn new() -> Self {
        Self {
            thread_buffers: HashMap::new(),
            compute_per_buffer: 50,  // Candleと同じ
        }
    }

    pub fn get_command_buffer(&mut self) -> Result<&mut ThreadLocalBuffer> {
        let thread_id = std::thread::current().id();
        let entry = self.thread_buffers
            .entry(thread_id)
            .or_insert_with(|| ThreadLocalBuffer::new());

        // バッチングチェック
        if entry.encoder_count >= self.compute_per_buffer {
            entry.flush()?;
        }

        Ok(entry)
    }

    pub fn flush_all(&mut self) -> Result<()> {
        for buffer in self.thread_buffers.values_mut() {
            buffer.flush()?;
        }
        Ok(())
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        for buffer in self.thread_buffers.values() {
            buffer.wait_until_completed()?;
        }
        Ok(())
    }
}

impl ThreadLocalBuffer {
    fn flush(&mut self) -> Result<()> {
        if self.encoder_count > 0 {
            self.command_buffer.commit();
            self.command_buffer = create_new_command_buffer()?;
            self.encoder_count = 0;
        }
        Ok(())
    }

    fn wait_until_completed(&self) -> Result<()> {
        self.command_buffer.wait_until_completed();
        Ok(())
    }
}
```

#### 7.2.2 遅延同期Tensor実装

```rust
// src/tensor/mod.rs
pub struct Tensor {
    buffer: Arc<MTLBuffer>,
    shape: Vec<usize>,
    dtype: DType,
    device: Arc<Device>,
    // 同期状態を追跡しない（遅延同期）
}

impl Tensor {
    // GPU操作: 同期なし
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let device = &self.device;
        let cmd_buffer = device.command_buffer_pool.get_command_buffer()?;

        // カーネル実行（同期なし）
        execute_add_kernel(cmd_buffer, self, rhs)?;

        // ❌ ここで同期しない！
        Ok(result)
    }

    // CPU読み取り: ここで初めて同期
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let device = &self.device;

        // 1. GPU-GPU blit copy
        let cpu_buffer = device.allocate_cpu_buffer(self.byte_size())?;
        let cmd_buffer = device.command_buffer_pool.get_command_buffer()?;
        cmd_buffer.blit_copy(self.buffer, cpu_buffer)?;

        // 2. ✅ ここで同期
        device.command_buffer_pool.wait_until_completed()?;

        // 3. CPU読み取り
        Ok(read_buffer_to_vec(&cpu_buffer))
    }
}
```

#### 7.2.3 Buffer Pool実装

```rust
// src/gpu/buffer_pool.rs
use std::sync::{Arc, Weak};
use std::collections::HashMap;

pub struct BufferPool {
    buffers: HashMap<(usize, String), Weak<MTLBuffer>>,
}

impl BufferPool {
    pub fn allocate(&mut self, size: usize, label: &str) -> Result<Arc<MTLBuffer>> {
        let key = (size, label.to_string());

        // 既存バッファを探す
        let weak_buffer = self.buffers
            .entry(key.clone())
            .or_insert_with(|| Weak::new());

        // アップグレード試行
        match weak_buffer.upgrade() {
            Some(buffer) if Arc::strong_count(&buffer) <= 1 => {
                // ✅ 再利用可能
                Ok(buffer)
            }
            _ => {
                // ❌ 新規作成
                let new_buffer = Arc::new(create_metal_buffer(size, label)?);
                *weak_buffer = Arc::downgrade(&new_buffer);
                Ok(new_buffer)
            }
        }
    }
}
```

### 7.3 実装ロードマップ

#### Phase 1: Command Buffer Pool (1週間)
- [ ] ThreadLocal command buffer管理
- [ ] バッチング機構実装
- [ ] Encoder自動クリーンアップ

#### Phase 2: 遅延同期 (1週間)
- [ ] `to_cpu()` での同期実装
- [ ] 不要な同期削除
- [ ] 同期状態追跡削除

#### Phase 3: Buffer Pool (1週間)
- [ ] 参照カウントベースのプール
- [ ] サイズ別管理
- [ ] 自動解放機構

#### Phase 4: 最適化と検証 (1週間)
- [ ] パフォーマンステスト
- [ ] メモリリークチェック
- [ ] Transformer全層での検証

## 8. パフォーマンス検証計画

### 8.1 測定指標

```rust
// ベンチマーク実装例
#[test]
fn benchmark_sync_strategies() {
    let ops_count = 200;  // 22層 × 9操作/層

    // 従来型: 即座同期
    let start = Instant::now();
    for _ in 0..ops_count {
        let result = tensor.add(&other)?;
        result.sync()?;  // 毎回同期
    }
    let eager_time = start.elapsed();

    // 遅延同期
    let start = Instant::now();
    for _ in 0..ops_count {
        let result = tensor.add(&other)?;
        // 同期なし
    }
    let result_vec = tensor.to_vec()?;  // 最後に1回だけ同期
    let lazy_time = start.elapsed();

    println!("Eager sync: {:?}", eager_time);
    println!("Lazy sync: {:?}", lazy_time);
    println!("Speedup: {:.2}x", eager_time.as_secs_f64() / lazy_time.as_secs_f64());
}
```

### 8.2 期待される性能改善

| シナリオ | 従来型 | 遅延同期 | 改善率 |
|---------|-------|---------|-------|
| 単一操作 | 1ms | 1ms | 1.0x |
| 10操作 | 10ms | 1.2ms | 8.3x |
| 100操作 | 100ms | 5ms | 20x |
| Transformer 1層 | 9ms | 0.5ms | 18x |
| Transformer 22層 | 198ms | 11ms | 18x |

**注:** 実際の数値はハードウェアと操作内容に依存

## 9. リスクと対策

### 9.1 潜在的な問題

**1. メモリ使用量増加**
- 問題: 同期遅延により中間結果が長時間保持
- 対策: Buffer poolのサイズ上限設定

**2. デバッグ困難化**
- 問題: エラーが遅延して発生
- 対策: Debug modeでの即座同期オプション

**3. Thread安全性**
- 問題: マルチスレッドでのcommand buffer競合
- 対策: ThreadLocal管理の徹底

### 9.2 対策コード例

```rust
// Debug mode同期
#[cfg(debug_assertions)]
impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let result = self.add_impl(rhs)?;
        // Debug modeでは即座同期
        self.device.synchronize()?;
        Ok(result)
    }
}

#[cfg(not(debug_assertions))]
impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        // Release modeでは遅延同期
        self.add_impl(rhs)
    }
}
```

## 10. まとめ

### 10.1 Candleの同期戦略の本質

1. **遅延同期:** CPU読み取り時のみ同期
2. **バッチング:** 50 encodersを1 command bufferにまとめる
3. **Thread分離:** スレッドごとのcommand buffer
4. **Buffer再利用:** 参照カウントベースのプール
5. **自動管理:** RAII パターンでリソース管理

### 10.2 TensorLogicへの適用優先度

| 項目 | 優先度 | 難易度 | 効果 |
|------|-------|-------|------|
| 遅延同期 | ⭐⭐⭐ | 中 | 大 |
| Command Buffer Pool | ⭐⭐⭐ | 高 | 大 |
| Buffer Pool | ⭐⭐ | 中 | 中 |
| Thread分離 | ⭐ | 低 | 小 |

### 10.3 期待される総合効果

**Transformer推論での性能改善:**
- レイテンシ: **10-20倍** 削減
- スループット: **10-20倍** 向上
- メモリ効率: **2-3倍** 改善

この同期戦略の実装により、TensorLogicはCandleと同等の高性能GPU実行が可能になります。
