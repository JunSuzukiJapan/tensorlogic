# テンソルロジック: f16 + Neural Engine + Metal 仕様

## 設計方針

### コア原則
1. **f16オンリー**: すべての浮動小数点演算はf16 (half precision)
2. **GPU最優先**: 可能な限りMetal/Neural Engineで実行
3. **ゼロコピー**: Neural Engine ↔ Metal間でデータ変換なし
4. **CPU最小化**: 条件分岐などの制御フローのみCPU使用

### アーキテクチャ階層

```
┌─────────────────────────────────────────┐
│  TensorLogic Language Layer             │
│  (Parser, AST, Type System)             │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Execution Engine (CPU - Control Only)  │
│  - Rule evaluation                      │
│  - Conditional branching                │
│  - Memory orchestration                 │
└─────────────────────────────────────────┘
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
┌──────────────┐      ┌──────────────┐
│ Neural Engine│      │    Metal     │
│  (CoreML)    │←────→│   (f16 ops)  │
│              │ f16  │              │
│ - Inference  │ zero │ - Basic ops  │
│ - MatMul     │ copy │ - Custom     │
│ - Conv       │      │   kernels    │
└──────────────┘      └──────────────┘
```

## 1. コアデータ型 (f16専用)

### 1.1 テンソル型

```rust
use half::f16;

#[repr(C)]
pub struct Tensor {
    // メタデータ (CPU)
    shape: Vec<usize>,
    strides: Vec<usize>,

    // データ本体 (GPU)
    buffer: BufferHandle,

    // デバイス情報
    device: Device,

    // 自動微分用
    grad: Option<Box<Tensor>>,
    requires_grad: bool,
}

pub enum BufferHandle {
    Metal(MetalBuffer),
    NeuralEngine(MLMultiArray), // CoreML
}

pub enum Device {
    Metal(MetalDevice),
    NeuralEngine,
    CPU, // 制御フローのみ
}
```

### 1.2 Metal バッファ

```rust
pub struct MetalBuffer {
    buffer: metal::Buffer,      // MTLBuffer
    length: usize,              // 要素数
    device: metal::Device,      // MTLDevice
}

impl MetalBuffer {
    // f16データを直接Metalバッファに書き込み
    pub fn from_f16_slice(device: &metal::Device, data: &[f16]) -> Self;

    // Metalバッファからf16データを読み出し
    pub fn to_f16_vec(&self) -> Vec<f16>;

    // ゼロコピーでNeuralEngineバッファに変換
    pub fn as_mlmultiarray(&self, shape: &[usize]) -> MLMultiArray;
}
```

### 1.3 Neural Engine バッファ

```rust
use coreml::MLMultiArray;

pub struct NeuralEngineBuffer {
    array: MLMultiArray,  // MLMultiArray (Float16)
}

impl NeuralEngineBuffer {
    // f16配列から直接作成
    pub fn from_f16_slice(data: &[f16], shape: &[usize]) -> Self;

    // ゼロコピーでMetalバッファに変換
    pub fn as_metal_buffer(&self, device: &metal::Device) -> MetalBuffer;
}
```

## 2. コア関数仕様

### 2.1 テンソル生成 (Metal実装)

```rust
impl Tensor {
    // すべてMetal GPUで生成
    pub fn zeros_metal(device: &MetalDevice, shape: Vec<usize>) -> Self {
        // Metal kernelでゼロ埋め
    }

    pub fn ones_metal(device: &MetalDevice, shape: Vec<usize>) -> Self {
        // Metal kernelで1埋め
    }

    pub fn rand_metal(device: &MetalDevice, shape: Vec<usize>) -> Self {
        // Metal kernelでランダム生成 (f16)
    }

    pub fn from_f16_vec(device: &MetalDevice, data: Vec<f16>, shape: Vec<usize>) -> Self {
        // f16ベクタを直接Metalバッファにコピー
    }
}
```

### 2.2 基本演算 (Metal Shaders)

各演算はMetalシェーダーで実装し、f16で計算。

#### Metal Shader例: 要素ごと加算

```metal
// add.metal
#include <metal_stdlib>
using namespace metal;

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}
```

#### Rust側インターフェース

```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        match (&self.buffer, &other.buffer) {
            (BufferHandle::Metal(a), BufferHandle::Metal(b)) => {
                // Metalカーネル実行
                metal_add_f16(a, b)
            },
            _ => self.to_metal()?.add(&other.to_metal()?),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError>;
}
```

### 2.3 行列演算 (Neural Engine優先)

行列積などの複雑な演算はNeural Engineを優先使用。

```rust
impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        // Neural Engineで実行
        self.to_neural_engine()?.matmul_ne(&other.to_neural_engine()?)
    }
}

// Neural Engine実装
fn matmul_ne(a: &NeuralEngineBuffer, b: &NeuralEngineBuffer) -> Result<Tensor, TensorError> {
    // CoreMLモデルとして行列積を実行
    let model = create_matmul_coreml_model(a.shape(), b.shape());
    let result = model.predict(&[a.array, b.array])?;

    // 結果をf16バッファとして保持（変換なし）
    Ok(Tensor::from_neural_engine(result))
}
```

### 2.4 アインシュタイン和 (ハイブリッド実装)

```rust
pub fn einsum(spec: &str, tensors: &[&Tensor]) -> Result<Tensor, TensorError> {
    // 仕様を解析 (CPU)
    let plan = parse_einsum_spec(spec)?;

    // 実行戦略を選択 (CPU)
    match classify_einsum_operation(&plan) {
        EinsumOp::MatMul => {
            // Neural Engineで実行
            execute_matmul_ne(tensors, &plan)
        },
        EinsumOp::Transpose => {
            // Metalで実行
            execute_transpose_metal(tensors[0], &plan)
        },
        EinsumOp::ElementWise => {
            // Metalで実行
            execute_elementwise_metal(tensors, &plan)
        },
        EinsumOp::Complex => {
            // Metalカスタムカーネルで実行
            execute_custom_einsum_metal(tensors, &plan)
        },
    }
}
```

### 2.5 活性化関数 (Metal Shaders)

```metal
// activation.metal
kernel void sigmoid_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    half x = input[index];
    output[index] = half(1.0) / (half(1.0) + exp(-x));
}

kernel void relu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    half x = input[index];
    output[index] = max(x, half(0.0));
}
```

```rust
impl Tensor {
    pub fn sigmoid(&self) -> Tensor {
        execute_metal_kernel("sigmoid_f16", &[self.buffer])
    }

    pub fn relu(&self) -> Tensor {
        execute_metal_kernel("relu_f16", &[self.buffer])
    }

    pub fn tanh(&self) -> Tensor {
        execute_metal_kernel("tanh_f16", &[self.buffer])
    }
}
```

### 2.6 集約演算 (Metal MPS使用)

Metal Performance Shadersを活用。

```rust
use metal_performance_shaders as mps;

impl Tensor {
    pub fn sum(&self, dim: Option<usize>) -> Tensor {
        match dim {
            None => {
                // 全要素の合計 (MPS)
                mps::reduction_sum(&self.buffer)
            },
            Some(d) => {
                // 特定次元の合計 (カスタムMetal kernel)
                metal_reduce_sum_dim(&self.buffer, d)
            },
        }
    }

    pub fn mean(&self, dim: Option<usize>) -> Tensor;
    pub fn max(&self, dim: Option<usize>) -> Tensor;
    pub fn min(&self, dim: Option<usize>) -> Tensor;
}
```

## 3. 自動微分 (Neural Engine + Metal)

### 3.1 計算グラフ

```rust
pub struct ComputationGraph {
    nodes: Vec<Node>,

    // デバイス情報
    metal_device: MetalDevice,
    use_neural_engine: bool,
}

pub struct Node {
    id: NodeId,
    operation: Operation,
    inputs: Vec<NodeId>,

    // f16バッファ
    output: BufferHandle,
    gradient: Option<BufferHandle>,
}

pub enum Operation {
    // Neural Engine ops
    MatMul,
    Conv2D,

    // Metal ops
    Add, Sub, Mul, Div,
    Sigmoid, ReLU, Tanh,

    // Custom ops
    Einsum(String),
}
```

### 3.2 後向き伝播

```rust
impl ComputationGraph {
    pub fn backward(&mut self, output_grad: Tensor) -> Result<(), TensorError> {
        // トポロジカルソート (CPU)
        let sorted_nodes = self.topological_sort();

        // 逆順に勾配計算 (GPU)
        for node_id in sorted_nodes.iter().rev() {
            let node = &self.nodes[node_id];

            match node.operation {
                Operation::MatMul => {
                    // Neural Engineで勾配計算
                    self.backward_matmul_ne(node)?;
                },
                Operation::Add | Operation::Mul => {
                    // Metalで勾配計算
                    self.backward_elementwise_metal(node)?;
                },
                _ => {
                    // カスタムMetal kernelで勾配計算
                    self.backward_custom_metal(node)?;
                },
            }
        }

        Ok(())
    }
}
```

## 4. デバイス間のゼロコピー変換

### 4.1 共有メモリ戦略

```rust
impl Tensor {
    // Metal → Neural Engine (ゼロコピー)
    pub fn to_neural_engine(&self) -> Result<Tensor, TensorError> {
        match &self.buffer {
            BufferHandle::Metal(metal_buf) => {
                // MTLBufferをMLMultiArrayでラップ（データコピーなし）
                let ml_array = wrap_metal_buffer_as_mlmultiarray(
                    metal_buf,
                    &self.shape,
                )?;

                Ok(Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    buffer: BufferHandle::NeuralEngine(ml_array),
                    device: Device::NeuralEngine,
                    grad: None,
                    requires_grad: self.requires_grad,
                })
            },
            BufferHandle::NeuralEngine(_) => Ok(self.clone()),
            BufferHandle::CPU(_) => {
                // CPUからの場合のみコピーが発生
                self.to_metal()?.to_neural_engine()
            },
        }
    }

    // Neural Engine → Metal (ゼロコピー)
    pub fn to_metal(&self) -> Result<Tensor, TensorError> {
        match &self.buffer {
            BufferHandle::NeuralEngine(ml_array) => {
                // MLMultiArrayの内部MTLBufferを直接参照
                let metal_buf = extract_metal_buffer_from_mlmultiarray(ml_array)?;

                Ok(Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    buffer: BufferHandle::Metal(metal_buf),
                    device: Device::Metal(get_default_metal_device()),
                    grad: self.grad.clone(),
                    requires_grad: self.requires_grad,
                })
            },
            BufferHandle::Metal(_) => Ok(self.clone()),
            BufferHandle::CPU(cpu_data) => {
                // CPUからMetalへコピー
                let metal_buf = MetalBuffer::from_f16_slice(
                    &get_default_metal_device(),
                    cpu_data,
                );

                Ok(Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    buffer: BufferHandle::Metal(metal_buf),
                    device: Device::Metal(get_default_metal_device()),
                    grad: None,
                    requires_grad: self.requires_grad,
                })
            },
        }
    }
}
```

## 5. 最適化戦略

### 5.1 演算の自動配置

```rust
pub struct ExecutionPlanner {
    metal_device: MetalDevice,
    neural_engine_available: bool,
}

impl ExecutionPlanner {
    pub fn select_device(&self, operation: &Operation) -> Device {
        match operation {
            // Neural Engineが得意な演算
            Operation::MatMul if self.neural_engine_available => Device::NeuralEngine,
            Operation::Conv2D if self.neural_engine_available => Device::NeuralEngine,

            // Metalで実行
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
                Device::Metal(self.metal_device.clone())
            },

            // カスタム判定
            _ => self.benchmark_and_select(operation),
        }
    }
}
```

### 5.2 バッファプール

```rust
pub struct BufferPool {
    metal_buffers: Vec<MetalBuffer>,
    neural_engine_buffers: Vec<MLMultiArray>,
}

impl BufferPool {
    // 再利用可能なバッファを取得
    pub fn acquire_metal(&mut self, size: usize) -> MetalBuffer {
        self.metal_buffers
            .iter()
            .position(|b| b.length >= size)
            .map(|i| self.metal_buffers.swap_remove(i))
            .unwrap_or_else(|| MetalBuffer::allocate(size))
    }

    // バッファを返却
    pub fn release_metal(&mut self, buffer: MetalBuffer) {
        self.metal_buffers.push(buffer);
    }
}
```

## 6. エラー処理

```rust
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Metal error: {0}")]
    MetalError(String),

    #[error("Neural Engine error: {0}")]
    NeuralEngineError(String),

    #[error("Device conversion error: {0}")]
    DeviceConversionError(String),

    #[error("f16 precision overflow")]
    PrecisionOverflow,
}
```

## 7. 実装優先順序

### Phase 1: Metal基盤 (完了 ✅)
- [x] Metal device初期化
- [x] f16バッファ管理
- [x] 基本演算shaders (add, sub, mul, div)
- [x] テンソル型とShape管理
- [x] CPUフォールバック実装

### Phase 2: Metal GPU高速化 (完了 ✅)
- [x] Metal compute shaders
- [x] KernelExecutor実装
- [x] Element-wise演算のGPU実装
- [x] パイプラインキャッシング
- [x] スレッドグループ最適化

### Phase 3: 高度な演算 (完了 ✅)
- [x] MatMul実装 (2D GPU kernel)
- [x] 活性化関数 (ReLU, GELU, Softmax)
- [x] Broadcasting (broadcast_to, broadcast_with)
- [x] 集約演算 (sum, mean, max, min, sum_dim, mean_dim)
- [ ] Einsum実装 (保留 - 基本演算で十分)
- [ ] GPU kernels for reductions (Phase 5最適化で実装)

### Phase 4: Neural Engine統合 (未着手)
- [ ] CoreML統合
- [ ] MLMultiArray ↔ Metal ゼロコピー変換
- [ ] Neural Engine推論実行

### Phase 4: 自動微分 (4週間)
- [ ] 計算グラフ構築
- [ ] 後向き伝播 (Metal)
- [ ] Neural Engine勾配計算

### Phase 5: 最適化 (2週間)
- [ ] バッファプール
- [ ] 演算融合
- [ ] デバイス自動配置

合計: 約13週間
