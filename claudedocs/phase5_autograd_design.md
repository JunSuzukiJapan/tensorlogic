# Phase 5: Autograd (自動微分) 詳細設計

## 概要

Phase 5では、TensorLogicに自動微分機能を追加し、ニューラルネットワークの学習を可能にします。PyTorchやTensorFlowと同様の動的計算グラフを構築し、Metal GPUを活用した高速な勾配計算を実装します。

## 設計目標

1. **動的計算グラフ**: 実行時にグラフを構築（PyTorchスタイル）
2. **Metal GPU加速**: 勾配計算もMetal GPUで実行
3. **f16精度維持**: 勾配もf16で計算・保存
4. **メモリ効率**: 中間値の適切な管理と解放
5. **API簡潔性**: `tensor.backward()`で自動微分実行

## アーキテクチャ

### 計算グラフ構造

```rust
// 計算グラフのノード
pub struct GradNode {
    id: NodeId,
    operation: Operation,
    inputs: Vec<NodeId>,
    grad_fn: Option<Box<dyn GradientFunction>>,
}

// 勾配計算関数トレイト
pub trait GradientFunction: Send + Sync {
    fn backward(
        &self,
        grad_output: &Tensor,
        inputs: &[&Tensor],
    ) -> TensorResult<Vec<Tensor>>;
}

// Tensorに追加するフィールド
pub struct Tensor {
    buffer: BufferHandle,
    shape: TensorShape,
    device: Device,

    // Autograd関連 (新規)
    requires_grad: bool,
    grad: Option<Box<Tensor>>,
    grad_node: Option<Arc<GradNode>>,
    version: u64,  // 勾配累積バージョン管理
}
```

### 計算グラフ管理

```rust
// グローバル計算グラフ（スレッドローカル）
thread_local! {
    static COMPUTATION_GRAPH: RefCell<ComputationGraph> = RefCell::new(ComputationGraph::new());
}

pub struct ComputationGraph {
    nodes: HashMap<NodeId, GradNode>,
    next_id: NodeId,
    enabled: bool,  // no_grad()コンテキスト用
}

impl ComputationGraph {
    pub fn add_node(&mut self, operation: Operation, inputs: Vec<NodeId>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let grad_fn = operation.create_grad_fn();

        self.nodes.insert(id, GradNode {
            id,
            operation,
            inputs,
            grad_fn,
        });

        id
    }

    pub fn backward(&self, node_id: NodeId, grad: Tensor) -> TensorResult<HashMap<NodeId, Tensor>> {
        // トポロジカルソートで逆順にノードを処理
        let topo_order = self.topological_sort(node_id)?;
        let mut gradients: HashMap<NodeId, Tensor> = HashMap::new();

        gradients.insert(node_id, grad);

        for &current_id in topo_order.iter().rev() {
            let node = &self.nodes[&current_id];
            let grad_output = &gradients[&current_id];

            if let Some(ref grad_fn) = node.grad_fn {
                // 入力テンソルの参照を取得（実装時に詳細化）
                let input_tensors: Vec<&Tensor> = vec![];  // TODO: 実装
                let input_grads = grad_fn.backward(grad_output, &input_tensors)?;

                // 各入力ノードに勾配を累積
                for (input_id, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
                    gradients.entry(*input_id)
                        .and_modify(|existing_grad| {
                            *existing_grad = existing_grad.add(input_grad).unwrap();
                        })
                        .or_insert_with(|| input_grad.clone());
                }
            }
        }

        Ok(gradients)
    }

    fn topological_sort(&self, start: NodeId) -> TensorResult<Vec<NodeId>> {
        // DFSベースのトポロジカルソート実装
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        self.dfs(start, &mut visited, &mut order)?;

        Ok(order)
    }

    fn dfs(&self, node_id: NodeId, visited: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) -> TensorResult<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.dfs(input_id, visited, order)?;
            }
        }

        order.push(node_id);
        Ok(())
    }
}
```

### 演算ごとの勾配関数

#### 加算 (Add)

```rust
struct AddBackward;

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * 1 = grad_output
        // ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * 1 = grad_output

        let grad_a = grad_output.clone();
        let grad_b = grad_output.clone();

        // ブロードキャストされている場合は次元を縮約
        let grad_a = reduce_grad_for_broadcast(&grad_a, inputs[0].shape())?;
        let grad_b = reduce_grad_for_broadcast(&grad_b, inputs[1].shape())?;

        Ok(vec![grad_a, grad_b])
    }
}

fn reduce_grad_for_broadcast(grad: &Tensor, original_shape: &TensorShape) -> TensorResult<Tensor> {
    if grad.shape() == original_shape {
        return Ok(grad.clone());
    }

    // ブロードキャストされた次元を合計で縮約
    let mut result = grad.clone();

    let grad_dims = grad.shape().dims();
    let orig_dims = original_shape.dims();

    let rank_diff = grad_dims.len() - orig_dims.len();

    // 先頭の次元を縮約（元の形状より高次元の場合）
    for i in 0..rank_diff {
        result = result.sum_dim(0, false)?;
    }

    // サイズ1だった次元を縮約
    for (i, (&grad_size, &orig_size)) in grad_dims[rank_diff..].iter().zip(orig_dims.iter()).enumerate() {
        if orig_size == 1 && grad_size > 1 {
            result = result.sum_dim(i, true)?;
        }
    }

    Ok(result)
}
```

#### 乗算 (Mul)

```rust
struct MulBackward {
    a_shape: TensorShape,
    b_shape: TensorShape,
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // c = a * b
        // ∂L/∂a = ∂L/∂c * ∂c/∂a = grad_output * b
        // ∂L/∂b = ∂L/∂c * ∂c/∂b = grad_output * a

        let a = inputs[0];
        let b = inputs[1];

        let grad_a = grad_output.mul(b)?;
        let grad_b = grad_output.mul(a)?;

        let grad_a = reduce_grad_for_broadcast(&grad_a, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast(&grad_b, &self.b_shape)?;

        Ok(vec![grad_a, grad_b])
    }
}
```

#### 行列積 (MatMul)

```rust
struct MatMulBackward;

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // C = A @ B  (A: [M, K], B: [K, N], C: [M, N])
        // ∂L/∂A = ∂L/∂C @ B^T  ([M, N] @ [N, K] = [M, K])
        // ∂L/∂B = A^T @ ∂L/∂C  ([K, M] @ [M, N] = [K, N])

        let a = inputs[0];
        let b = inputs[1];

        // B^T を計算（einsum使用）
        let b_t = Tensor::einsum("ij->ji", &[b])?;
        let grad_a = grad_output.matmul(&b_t)?;

        // A^T を計算
        let a_t = Tensor::einsum("ij->ji", &[a])?;
        let grad_b = a_t.matmul(grad_output)?;

        Ok(vec![grad_a, grad_b])
    }
}
```

#### ReLU

```rust
struct ReLUBackward {
    input_data: Vec<f16>,  // forward時の入力を保存
}

impl GradientFunction for ReLUBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // ReLU: y = max(0, x)
        // ∂y/∂x = 1 if x > 0, else 0

        let grad_output_data = grad_output.to_vec();

        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(self.input_data.iter())
            .map(|(&grad_out, &input_val)| {
                if input_val > f16::ZERO {
                    grad_out
                } else {
                    f16::ZERO
                }
            })
            .collect();

        Tensor::from_vec(grad_input_data, grad_output.shape().dims().to_vec())
    }
}
```

#### GELU

```rust
struct GELUBackward {
    input_data: Vec<f16>,
}

impl GradientFunction for GELUBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // GELU: y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        // ∂y/∂x は複雑なので数値微分または解析解を実装

        let grad_output_data = grad_output.to_vec();
        let sqrt_2_over_pi = f16::from_f32((2.0_f32 / std::f32::consts::PI).sqrt());

        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(self.input_data.iter())
            .map(|(&grad_out, &x)| {
                let x_f32 = x.to_f32();
                let x3 = x_f32 * x_f32 * x_f32;
                let inner = sqrt_2_over_pi.to_f32() * (x_f32 + 0.044715 * x3);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;

                // ∂GELU/∂x = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * derivative_of_inner
                let derivative_of_inner = sqrt_2_over_pi.to_f32() * (1.0 + 3.0 * 0.044715 * x_f32 * x_f32);
                let gelu_derivative = 0.5 * (1.0 + tanh_val) + 0.5 * x_f32 * sech2 * derivative_of_inner;

                f16::from_f32(grad_out.to_f32() * gelu_derivative)
            })
            .collect();

        Tensor::from_vec(grad_input_data, grad_output.shape().dims().to_vec())
    }
}
```

#### Softmax

```rust
struct SoftmaxBackward {
    output_data: Vec<f16>,  // forward時の出力を保存
    shape: TensorShape,
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // Softmax: y_i = exp(x_i) / Σ_j exp(x_j)
        // ∂y_i/∂x_j = y_i * (δ_ij - y_j)
        // ∂L/∂x_i = Σ_j (∂L/∂y_j * ∂y_j/∂x_i) = Σ_j (grad_output_j * y_j * (δ_ij - y_i))
        //         = grad_output_i * y_i - y_i * Σ_j (grad_output_j * y_j)

        let grad_output_data = grad_output.to_vec();
        let y = &self.output_data;

        // Σ_j (grad_output_j * y_j)
        let sum_grad_y: f16 = grad_output_data
            .iter()
            .zip(y.iter())
            .map(|(&g, &yi)| g * yi)
            .fold(f16::ZERO, |acc, x| acc + x);

        let grad_input_data: Vec<f16> = grad_output_data
            .iter()
            .zip(y.iter())
            .map(|(&g_i, &y_i)| {
                g_i * y_i - y_i * sum_grad_y
            })
            .collect();

        Tensor::from_vec(grad_input_data, self.shape.dims().to_vec())
    }
}
```

### Tensor API拡張

```rust
impl Tensor {
    /// 勾配計算を有効化
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// 勾配を取得
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// 勾配をクリア
    pub fn zero_grad(&mut self) {
        self.grad = None;
        self.version += 1;
    }

    /// 逆伝播実行
    pub fn backward(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string()
            ));
        }

        // 勾配の初期値（スカラーの場合は1.0、それ以外はエラー）
        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward() can only be called on scalar tensors. Use backward(grad) for non-scalar tensors.".to_string()
            ));
        }

        let grad = Tensor::from_vec(vec![f16::ONE], vec![1])?;
        self.backward_with_grad(grad)
    }

    /// 勾配を指定して逆伝播実行
    pub fn backward_with_grad(&mut self, grad: Tensor) -> TensorResult<()> {
        if let Some(node_id) = self.grad_node.as_ref().map(|n| n.id) {
            COMPUTATION_GRAPH.with(|graph_cell| {
                let graph = graph_cell.borrow();
                let gradients = graph.backward(node_id, grad)?;

                // 各テンソルに勾配を設定（実装時に詳細化）
                // TODO: グローバルなテンソル管理が必要

                Ok(())
            })
        } else {
            // 葉ノードの場合は自身に勾配を設定
            self.grad = Some(Box::new(grad));
            Ok(())
        }
    }

    /// 勾配計算を一時的に無効化
    pub fn no_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        COMPUTATION_GRAPH.with(|graph_cell| {
            let prev_enabled = graph_cell.borrow().enabled;
            graph_cell.borrow_mut().enabled = false;

            let result = f();

            graph_cell.borrow_mut().enabled = prev_enabled;
            result
        })
    }
}
```

### 演算の拡張（Add例）

```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> TensorResult<Self> {
        // 既存のadd実装を呼び出し
        let result = self.add_impl(other)?;

        // Autograd有効時は計算グラフに追加
        if self.requires_grad || other.requires_grad {
            COMPUTATION_GRAPH.with(|graph_cell| {
                let mut graph = graph_cell.borrow_mut();

                if graph.enabled {
                    let self_node_id = self.grad_node.as_ref().map(|n| n.id).unwrap_or(graph.next_id);
                    let other_node_id = other.grad_node.as_ref().map(|n| n.id).unwrap_or(graph.next_id + 1);

                    let node_id = graph.add_node(
                        Operation::Add,
                        vec![self_node_id, other_node_id]
                    );

                    result.grad_node = Some(Arc::new(GradNode {
                        id: node_id,
                        operation: Operation::Add,
                        inputs: vec![self_node_id, other_node_id],
                        grad_fn: Some(Box::new(AddBackward)),
                    }));

                    result.requires_grad = true;
                }
            });
        }

        Ok(result)
    }
}
```

## Metal GPU勾配計算

勾配計算もMetal GPUで実行するため、各演算の勾配計算用カーネルを実装します。

### ReLU勾配カーネル例

```metal
// shaders/gradients.metal

kernel void relu_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    half grad_out = grad_output[idx];
    half in_val = input[idx];

    grad_input[idx] = (in_val > 0.0h) ? grad_out : 0.0h;
}

kernel void matmul_backward_a_f16(
    device const half* grad_output [[buffer(0)]],  // [M, N]
    device const half* b [[buffer(1)]],             // [K, N]
    device half* grad_a [[buffer(2)]],              // [M, K]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.x;
    uint k = gid.y;

    if (m >= M || k >= K) return;

    half sum = 0.0h;
    for (uint n = 0; n < N; n++) {
        sum += grad_output[m * N + n] * b[k * N + n];  // B^T: [K, N] stored as [k, n]
    }

    grad_a[m * K + k] = sum;
}
```

## メモリ管理戦略

### 中間値の保存

勾配計算に必要な中間値（ReLUのinput、Softmaxのoutputなど）を効率的に管理します。

```rust
pub struct GradientCache {
    tensors: HashMap<NodeId, Tensor>,
}

impl GradientCache {
    pub fn save(&mut self, node_id: NodeId, tensor: Tensor) {
        self.tensors.insert(node_id, tensor);
    }

    pub fn get(&self, node_id: NodeId) -> Option<&Tensor> {
        self.tensors.get(&node_id)
    }

    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}
```

### メモリ解放戦略

- **Eager Release**: backward()完了後、計算グラフと中間値を即座に解放
- **Retain Graph**: 複数回backward()を呼ぶ場合のオプション（将来実装）

## テスト戦略

### 単体テスト

各演算の勾配関数を個別にテスト：

```rust
#[test]
fn test_add_backward() {
    let a = Tensor::from_vec(vec![f16::from_f32(2.0), f16::from_f32(3.0)], vec![2])
        .unwrap()
        .requires_grad(true);

    let b = Tensor::from_vec(vec![f16::from_f32(4.0), f16::from_f32(5.0)], vec![2])
        .unwrap()
        .requires_grad(true);

    let c = a.add(&b).unwrap();
    let loss = c.sum().unwrap();

    loss.backward().unwrap();

    // ∂loss/∂a = [1.0, 1.0]
    assert_eq!(a.grad().unwrap().to_vec(), vec![f16::ONE, f16::ONE]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![f16::ONE, f16::ONE]);
}

#[test]
fn test_matmul_backward() {
    let a = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0),
             f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    ).unwrap().requires_grad(true);

    let b = Tensor::from_vec(
        vec![f16::from_f32(5.0), f16::from_f32(6.0),
             f16::from_f32(7.0), f16::from_f32(8.0)],
        vec![2, 2]
    ).unwrap().requires_grad(true);

    let c = a.matmul(&b).unwrap();
    let loss = c.sum().unwrap();

    loss.backward().unwrap();

    // 勾配の値を検証（手計算またはPyTorchと比較）
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}
```

### 統合テスト

実際のニューラルネットワーク層をテスト：

```rust
#[test]
fn test_simple_neural_network() {
    // 簡単な2層ニューラルネットワーク
    // y = ReLU(x @ W1) @ W2

    let x = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![1, 2]
    ).unwrap();

    let w1 = Tensor::from_vec(
        vec![f16::from_f32(0.5), f16::from_f32(0.3),
             f16::from_f32(0.2), f16::from_f32(0.8)],
        vec![2, 2]
    ).unwrap().requires_grad(true);

    let w2 = Tensor::from_vec(
        vec![f16::from_f32(0.6), f16::from_f32(0.4)],
        vec![2, 1]
    ).unwrap().requires_grad(true);

    // Forward pass
    let h = x.matmul(&w1).unwrap().relu().unwrap();
    let y = h.matmul(&w2).unwrap();

    // Backward pass
    y.backward().unwrap();

    // 勾配が計算されていることを確認
    assert!(w1.grad().is_some());
    assert!(w2.grad().is_some());
}
```

### 数値微分との比較

自動微分の正確性を数値微分で検証：

```rust
fn numerical_gradient(
    f: impl Fn(&Tensor) -> TensorResult<Tensor>,
    x: &Tensor,
    epsilon: f32,
) -> Tensor {
    let x_data = x.to_vec();
    let mut grad_data = vec![f16::ZERO; x_data.len()];

    for i in 0..x_data.len() {
        let mut x_plus = x_data.clone();
        let mut x_minus = x_data.clone();

        x_plus[i] = f16::from_f32(x_plus[i].to_f32() + epsilon);
        x_minus[i] = f16::from_f32(x_minus[i].to_f32() - epsilon);

        let x_plus_tensor = Tensor::from_vec(x_plus, x.shape().dims().to_vec()).unwrap();
        let x_minus_tensor = Tensor::from_vec(x_minus, x.shape().dims().to_vec()).unwrap();

        let f_plus = f(&x_plus_tensor).unwrap().to_vec()[0].to_f32();
        let f_minus = f(&x_minus_tensor).unwrap().to_vec()[0].to_f32();

        grad_data[i] = f16::from_f32((f_plus - f_minus) / (2.0 * epsilon));
    }

    Tensor::from_vec(grad_data, x.shape().dims().to_vec()).unwrap()
}

#[test]
fn test_gradient_numerical_check() {
    let x = Tensor::from_vec(
        vec![f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![2]
    ).unwrap().requires_grad(true);

    let y = x.mul(&x).unwrap().sum().unwrap();  // y = x1^2 + x2^2
    y.backward().unwrap();

    let analytical_grad = x.grad().unwrap().to_vec();

    let numerical_grad = numerical_gradient(
        |x_val| x_val.mul(x_val).unwrap().sum(),
        &x,
        1e-4
    ).to_vec();

    for (a, n) in analytical_grad.iter().zip(numerical_grad.iter()) {
        assert!((a.to_f32() - n.to_f32()).abs() < 1e-2);  // f16精度を考慮
    }
}
```

## 実装順序

1. **基本構造** (1-2日)
   - `GradNode`, `ComputationGraph`, `GradientFunction`トレイト
   - Tensorへの`requires_grad`, `grad`, `grad_node`フィールド追加
   - `backward()`, `zero_grad()`, `no_grad()`メソッド

2. **基本演算の勾配** (2-3日)
   - Add, Sub, Mul, Divの勾配関数
   - ブロードキャスト対応の勾配縮約実装
   - 単体テスト

3. **高度な演算の勾配** (2-3日)
   - MatMulの勾配関数
   - ReLU, GELU, Softmaxの勾配関数
   - 中間値保存機構
   - 単体テスト

4. **Metal GPU勾配計算** (2-3日)
   - gradients.metalシェーダー実装
   - ReLU, MatMul勾配のGPUカーネル
   - パフォーマンステスト

5. **統合とテスト** (1-2日)
   - 複雑な計算グラフのテスト
   - 数値微分との比較検証
   - ニューラルネットワーク層のテスト

6. **最適化** (1-2日)
   - メモリ管理の最適化
   - 計算グラフの効率化
   - ベンチマーク

**合計: 9-15日**

## API使用例

```rust
use tensorlogic::prelude::*;

// 簡単な線形回帰
fn linear_regression_example() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // データ: y = 2x + 3 + noise
    let x = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![3, 1]
    )?;

    let y_target = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(5.0), f16::from_f32(7.0), f16::from_f32(9.0)],
        vec![3, 1]
    )?;

    // パラメータ（学習対象）
    let mut w = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(0.1)],
        vec![1, 1]
    )?.requires_grad(true);

    let mut b = Tensor::from_vec_gpu(
        &device,
        vec![f16::from_f32(0.0)],
        vec![1]
    )?.requires_grad(true);

    // 学習ループ
    let lr = 0.01;
    for epoch in 0..100 {
        // Forward pass
        let y_pred = x.matmul(&w)?.add(&b)?;

        // Loss: MSE
        let diff = y_pred.sub(&y_target)?;
        let loss = diff.mul(&diff)?.mean()?;

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.to_vec()[0].to_f32());
        }

        // Backward pass
        loss.backward()?;

        // パラメータ更新（SGD）
        Tensor::no_grad(|| {
            let w_grad = w.grad().unwrap();
            let b_grad = b.grad().unwrap();

            w = w.sub(&w_grad.mul_scalar(lr)?)?;
            b = b.sub(&b_grad.mul_scalar(lr)?)?;
        });

        // 勾配クリア
        w.zero_grad();
        b.zero_grad();
    }

    println!("Final w: {}, b: {}", w.to_vec()[0].to_f32(), b.to_vec()[0].to_f32());
    // Expected: w ≈ 2.0, b ≈ 3.0

    Ok(())
}
```

## 将来の拡張

- **高階微分**: `backward(create_graph=true)`で2階以上の微分
- **カスタム勾配**: ユーザー定義の勾配関数
- **Checkpointing**: メモリ削減のための再計算戦略
- **並列化**: 計算グラフの並列実行
- **JITコンパイル**: 計算グラフの最適化とコンパイル

## まとめ

Phase 5では、動的計算グラフによる自動微分を実装し、TensorLogicをディープラーニングフレームワークとして完成させます。Metal GPUを活用した高速な勾配計算により、Apple Silicon上で効率的な学習が可能になります。

実装は段階的に進め、各ステップで十分なテストを行うことで、正確性と信頼性を確保します。
