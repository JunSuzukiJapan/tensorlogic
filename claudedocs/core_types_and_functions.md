# テンソルロジック: コア型と関数

## 1. コアデータ型

### 1.1 テンソル型
```rust
pub struct Tensor {
    data: Vec<f32>,           // 実データ (CPU)
    shape: Vec<usize>,        // 形状 [dim1, dim2, ...]
    dtype: DataType,          // データ型
    device: Device,           // CPU/GPU
    grad: Option<Box<Tensor>>, // 勾配 (自動微分用)
}

pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Complex64,
}

pub enum Device {
    CPU,
    CUDA(usize), // デバイスID
}
```

### 1.2 論理型
```rust
pub struct Atom {
    predicate: String,
    args: Vec<Term>,
}

pub enum Term {
    Variable(String),
    Constant(String),
    TensorExpr(Box<TensorExpr>),
}

pub struct Rule {
    head: Atom,
    body: Vec<Atom>,
}
```

### 1.3 型システム
```rust
pub enum Type {
    Tensor {
        dtype: DataType,
        shape: Shape,
    },
    Entity,
    Relation { arity: usize },
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
}

pub enum Shape {
    Known(Vec<usize>),
    Unknown(Vec<Option<usize>>),
    Variable(String),
}
```

## 2. コア関数

### 2.1 テンソル生成
```rust
impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self;
    pub fn zeros(shape: Vec<usize>) -> Self;
    pub fn ones(shape: Vec<usize>) -> Self;
    pub fn rand(shape: Vec<usize>) -> Self;
    pub fn randn(shape: Vec<usize>) -> Self;
    pub fn from_scalar(value: f32) -> Self;
}
```

### 2.2 基本演算
```rust
impl Tensor {
    // 要素ごと演算
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError>;

    // 行列演算
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError>;
    pub fn dot(&self, other: &Tensor) -> Result<Tensor, TensorError>;

    // 形状操作
    pub fn transpose(&self) -> Tensor;
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, TensorError>;
    pub fn squeeze(&self) -> Tensor;
    pub fn unsqueeze(&self, dim: usize) -> Tensor;
}
```

### 2.3 アインシュタイン和
```rust
pub fn einsum(spec: &str, tensors: &[&Tensor]) -> Result<Tensor, TensorError>;

// 例:
// "ij,jk->ik" : 行列積
// "ij->ji"    : 転置
// "ii->i"     : 対角要素抽出
```

### 2.4 集約演算
```rust
impl Tensor {
    pub fn sum(&self, dim: Option<usize>) -> Tensor;
    pub fn mean(&self, dim: Option<usize>) -> Tensor;
    pub fn max(&self, dim: Option<usize>) -> Tensor;
    pub fn min(&self, dim: Option<usize>) -> Tensor;
}
```

### 2.5 活性化関数
```rust
pub fn sigmoid(x: &Tensor) -> Tensor;
pub fn relu(x: &Tensor) -> Tensor;
pub fn tanh(x: &Tensor) -> Tensor;
pub fn softmax(x: &Tensor, dim: usize) -> Tensor;
pub fn step(x: &Tensor) -> Tensor; // Heaviside step function
```

### 2.6 論理演算
```rust
// テンソルジョイン
pub fn tensor_join(u: &Tensor, v: &Tensor, common_indices: &[usize])
    -> Result<Tensor, TensorError>;

// テンソル射影
pub fn tensor_project(t: &Tensor, keep_indices: &[usize])
    -> Result<Tensor, TensorError>;
```

### 2.7 埋め込み
```rust
pub struct Embedding {
    vectors: Tensor,  // shape: [num_entities, embedding_dim]
    entity_map: HashMap<String, usize>,
}

impl Embedding {
    pub fn new(num_entities: usize, embedding_dim: usize) -> Self;
    pub fn lookup(&self, entity: &str) -> Result<Tensor, EmbeddingError>;
    pub fn random_init(&mut self);
    pub fn xavier_init(&mut self);
}
```

### 2.8 自動微分
```rust
impl Tensor {
    pub fn backward(&mut self);
    pub fn zero_grad(&mut self);
    pub fn requires_grad(&mut self, requires: bool);
}

pub struct ComputationGraph {
    pub fn forward(&mut self, inputs: Vec<Tensor>) -> Tensor;
    pub fn backward(&mut self, grad: Tensor);
}
```

### 2.9 最適化
```rust
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [Tensor]);
    fn zero_grad(&mut self);
}

pub struct SGD {
    learning_rate: f32,
}

pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}
```

### 2.10 推論エンジン
```rust
pub struct InferenceEngine {
    pub fn forward_chain(&mut self, rules: &[Rule], facts: &[Atom])
        -> Vec<Atom>;
    pub fn backward_chain(&mut self, query: &Atom, rules: &[Rule], facts: &[Atom])
        -> bool;
}
```

## 3. GPU対応関数

```rust
impl Tensor {
    pub fn to(&self, device: Device) -> Result<Tensor, TensorError>;
    pub fn cuda(&self) -> Result<Tensor, TensorError>;
    pub fn cpu(&self) -> Result<Tensor, TensorError>;
}

pub struct CudaTensor {
    device_ptr: *mut f32,
    shape: Vec<usize>,
    device_id: usize,
}

impl CudaTensor {
    pub fn allocate(shape: Vec<usize>, device_id: usize) -> Self;
    pub fn copy_to_device(data: &[f32]) -> Self;
    pub fn copy_to_host(&self) -> Vec<f32>;
}
```

## 4. エラー型

```rust
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid dimension: {dim}")]
    InvalidDimension { dim: usize },

    #[error("Device error: {msg}")]
    DeviceError { msg: String },
}
```

## 5. 優先実装順序

1. **Phase 1: 基本テンソル型** (2週間)
   - Tensor構造体
   - DataType, Device列挙型
   - 基本的なメモリ管理

2. **Phase 2: 基本演算** (3週間)
   - 要素ごと演算 (+, -, *, /)
   - 行列積 (matmul)
   - 形状操作 (transpose, reshape)

3. **Phase 3: アインシュタイン和** (2週間)
   - einsum関数の実装
   - インデックス解析

4. **Phase 4: 論理型と推論** (3週間)
   - Atom, Term, Rule型
   - 前向き/後向きチェーン

5. **Phase 5: 埋め込み** (2週間)
   - Embedding構造体
   - ルックアップ機能

6. **Phase 6: 自動微分** (4週間)
   - 計算グラフ
   - backward実装

7. **Phase 7: GPU対応** (3週間)
   - CUDA統合
   - メモリ転送

合計: 約19週間
