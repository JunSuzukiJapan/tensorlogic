# Phase 8.6完全実装: 二階微分サポート設計

## 概要

Phase 8.6基本実装を拡張し、完全な二階微分計算を可能にする。これによりL-BFGS、Newton's method等の高度な最適化手法が使用可能になる。

## 現状の制限

**Phase 8.6基本実装の問題点:**
- `backward_create_graph()`で勾配に`requires_grad=true`を設定可能
- しかし勾配自体はプレーンテンソル（演算結果ではない）
- 勾配の勾配を計算しようとすると失敗する

**原因:**
```rust
// 現在の勾配設定（backward_impl内）
tensor.set_grad(gradient);  // gradientはプレーンなTensor
```

勾配は`add()`, `mul()`等の演算結果ではなく、直接設定されるため、計算グラフが存在しない。

## 設計目標

1. **勾配計算のOperation化**: 勾配の計算自体を演算として記録
2. **Hessian計算**: 二階微分行列の計算
3. **HVP (Hessian-Vector Product)**: メモリ効率的なHessian演算
4. **L-BFGS対応**: 準ニュートン法の実装基盤

## アーキテクチャ

```
┌──────────────────────────────────────────┐
│   Second-Order Derivatives System        │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Gradient as Operations            │ │
│  │  - GradientOp (新しいOperation)    │ │
│  │  - Gradient computation記録       │ │
│  │  - Backward pass自体が演算        │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Hessian Computation               │ │
│  │  - hessian()                       │ │
│  │  - hessian_diagonal()              │ │
│  │  - hessian_vector_product()        │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Advanced Optimizers               │ │
│  │  - L-BFGS                          │ │
│  │  - Newton's method                 │ │
│  │  - Trust region methods            │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

## 実装戦略

### Strategy 1: Gradient Operations (PyTorch方式)

勾配計算自体を演算として記録する。

**実装アプローチ:**

1. **GradientOp**を新しいOperationとして追加
2. 勾配計算時に`GradientOp`ノードを作成
3. 二回目のbackward時に`GradientOp`の勾配を計算

```rust
// 新しいOperation
pub enum Operation {
    // 既存の演算
    Add,
    Mul,
    // ...

    // 新規: 勾配演算
    Gradient {
        original_op: Box<Operation>,
        input_index: usize,  // どの入力に対する勾配か
    },
}

// GradientBackward（GradientOpの逆伝播）
pub struct GradientBackward {
    original_op: Operation,
    original_inputs: Vec<Tensor>,
    input_index: usize,
}

impl GradientFunction for GradientBackward {
    fn backward(&self, grad_output: &Tensor) -> TensorResult<Vec<Tensor>> {
        // 二階微分の計算
        // d/dx (dL/dx) = d²L/dx²

        // 元の演算のHessianを計算
        self.compute_hessian(grad_output)
    }
}
```

**修正箇所:**

`src/tensor/tensor.rs`の`backward_impl()`:

```rust
fn backward_impl(&mut self, grad: Tensor, create_graph: bool) -> TensorResult<()> {
    // ... 既存のbackward計算 ...

    for (tensor_node_id, gradient) in gradients {
        if let Some(mut tensor) = AutogradContext::get_tensor(tensor_node_id) {
            let final_gradient = if create_graph {
                // 勾配を演算として記録
                let grad_op = Operation::Gradient {
                    original_op: Box::new(/* 元の演算 */),
                    input_index: /* インデックス */,
                };

                let grad_fn = Box::new(GradientBackward::new(
                    /* 必要な情報 */
                ));

                let grad_node_id = AutogradContext::add_node(
                    grad_op,
                    vec![self.grad_node.unwrap()],  // 元のテンソルに依存
                    Some(grad_fn),
                );

                let mut grad_tensor = gradient;
                grad_tensor.set_grad_node(grad_node_id);
                grad_tensor.set_requires_grad(true);

                grad_tensor
            } else {
                gradient
            };

            tensor.set_grad(final_gradient);
            AutogradContext::register_tensor(tensor_node_id, tensor);
        }
    }
}
```

### Strategy 2: Explicit Hessian APIs

明示的なHessian計算APIを提供する。

```rust
impl Tensor {
    /// Compute full Hessian matrix
    pub fn hessian(&self, output: &Tensor) -> TensorResult<Tensor> {
        // output: スカラー
        // self: [N] パラメータ
        // 戻り値: [N, N] Hessian行列

        let n = self.numel();
        let mut hessian = vec![f16::ZERO; n * n];

        for i in 0..n {
            // i番目の要素に対する一階微分を計算
            let grad_i = self.compute_gradient_for_element(output, i)?;

            // grad_iに対してさらに微分（二階微分）
            for j in 0..n {
                let hess_ij = grad_i.compute_gradient_for_element_j(j)?;
                hessian[i * n + j] = hess_ij;
            }
        }

        Tensor::from_vec(hessian, vec![n, n])
    }

    /// Compute Hessian diagonal only (more efficient)
    pub fn hessian_diagonal(&self, output: &Tensor) -> TensorResult<Tensor> {
        // 対角要素のみ計算: ∂²L/∂xᵢ²
        let n = self.numel();
        let mut diag = vec![f16::ZERO; n];

        for i in 0..n {
            let grad_i = self.compute_gradient_for_element(output, i)?;
            let hess_ii = grad_i.compute_gradient_for_element(i)?;
            diag[i] = hess_ii;
        }

        Tensor::from_vec(diag, vec![n])
    }

    /// Compute Hessian-vector product (most efficient)
    pub fn hessian_vector_product(
        &self,
        output: &Tensor,
        vector: &Tensor,
    ) -> TensorResult<Tensor> {
        // H·v を H を明示的に計算せずに求める
        // Forward-over-reverse AD を使用

        // 1. 一階微分を計算（create_graph=true）
        let grad = self.compute_gradient(output, true)?;

        // 2. grad とvectorの内積を計算
        let grad_v_product = grad.dot(vector)?;

        // 3. この内積に対して微分（これがHessian-vector product）
        grad_v_product.backward()?;
        let hvp = self.grad()?;

        Ok(hvp)
    }
}
```

### Strategy 3: Functional API (JAX方式)

関数型のgrad() APIを提供。

```rust
/// Compute gradient of a function
pub fn grad<F>(f: F) -> impl Fn(&Tensor) -> TensorResult<Tensor>
where
    F: Fn(&Tensor) -> TensorResult<Tensor>,
{
    move |x: &Tensor| {
        let mut x_copy = x.clone();
        x_copy.set_requires_grad(true);

        let y = f(&x_copy)?;
        y.backward()?;

        // 重要: レジストリから取得
        x_copy = AutogradContext::get_tensor(x_copy.grad_node().unwrap()).unwrap();
        x_copy.grad()
    }
}

/// Compute second derivative
pub fn grad2<F>(f: F) -> impl Fn(&Tensor) -> TensorResult<Tensor>
where
    F: Fn(&Tensor) -> TensorResult<Tensor> + Clone,
{
    let f_clone = f.clone();
    grad(move |x| grad(&f_clone)(x))
}

/// Compute Jacobian-vector product
pub fn jvp<F>(
    f: F,
    x: &Tensor,
    v: &Tensor,
) -> TensorResult<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> TensorResult<Tensor>,
{
    // (f(x), df/dx · v)
    // Forward-mode AD
    unimplemented!("Requires forward-mode AD")
}

/// Compute vector-Jacobian product
pub fn vjp<F>(
    f: F,
    x: &Tensor,
    v: &Tensor,
) -> TensorResult<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> TensorResult<Tensor>,
{
    // (f(x), v^T · df/dx)
    // Reverse-mode AD (already implemented via backward)

    let y = f(x)?;
    y.backward_with_grad(v.clone())?;

    let x_grad = x.grad()?;
    Ok((y, x_grad))
}
```

## 実装計画

### Phase 8.6.1: GradientOp基盤

1. **Operation::Gradientの追加**
   - 新しいOperation種別
   - GradientBackward実装

2. **backward_impl修正**
   - create_graph=true時に勾配を演算として記録
   - GradientOpノード作成

3. **テスト**
   - 簡単な二階微分: f(x)=x², f''(x)=2
   - チェーンルール: f(x)=(x²)², f''(x)=12x²

### Phase 8.6.2: Hessian APIs

1. **hessian_diagonal()実装**
   - 最も効率的（対角のみ）
   - L-BFGSのpreconditioning用

2. **hessian_vector_product()実装**
   - Hessian行列を作らずにH·vを計算
   - メモリ効率が良い

3. **テスト**
   - 二次形式: f(x) = x^T A x
   - Rosenbrock関数

### Phase 8.6.3: Full Hessian

1. **hessian()実装**
   - 完全なHessian行列
   - 小規模問題用

2. **対称性検証**
   - H[i,j] = H[j,i]の確認

3. **Newton's method実装例**
   - 二階微分を使った最適化

### Phase 8.6.4: Functional API

1. **grad()関数**
   - JAX風の関数型API
   - 合成可能

2. **grad2(), grad3()等**
   - 高階微分

3. **テスト**
   - 関数合成のテスト

## 使用例

### Newton's Method

```rust
// f(x) = x² を最小化
let mut x = Tensor::from_vec(vec![f16::from_f32(10.0)], vec![1])?;
x.set_requires_grad(true);

for iter in 0..10 {
    // f(x) = x²
    let f = x.mul(&x)?;

    // 一階微分: f'(x) = 2x
    f.backward_create_graph()?;
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let grad = x.grad().unwrap();

    // 二階微分: f''(x) = 2
    grad.backward()?;
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let hess = x.grad().unwrap();

    // Newton update: x_new = x - f'(x) / f''(x)
    let update = grad.div(&hess)?;
    x.sub_(&update)?;

    x.zero_grad();
    AutogradContext::register_tensor(x.grad_node().unwrap(), x.clone());
}

// xは0に収束（最小値）
```

### L-BFGS用のHessian対角

```rust
let loss = compute_loss(&params, &data)?;

// Hessian対角を計算（preconditioner用）
let hess_diag = params.hessian_diagonal(&loss)?;

// L-BFGSでpreconditioningに使用
let preconditioner = hess_diag.add_scalar(1e-6)?.reciprocal()?;
```

### Hessian-Vector Product

```rust
let loss = compute_loss(&params, &data)?;
let direction = compute_search_direction()?;

// H·dを計算（Hessianを作らずに）
let hv = params.hessian_vector_product(&loss, &direction)?;

// Conjugate gradient等で使用
```

### Functional API

```rust
// f(x) = x³
let f = |x: &Tensor| {
    let x2 = x.mul(x)?;
    x2.mul(x)
};

// f'(x) = 3x²
let df = grad(f);

// f''(x) = 6x
let d2f = grad2(f);

let x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
let first_deriv = df(&x)?;   // 3*4 = 12
let second_deriv = d2f(&x)?;  // 6*2 = 12
```

## テスト戦略

### 単体テスト

1. **簡単な関数の二階微分**
   - f(x) = x²: f''(x) = 2
   - f(x) = x³: f''(x) = 6x
   - f(x) = sin(x): f''(x) = -sin(x)

2. **Hessian行列**
   - f(x,y) = x² + y²: H = diag([2, 2])
   - f(x,y) = xy: H = [[0,1],[1,0]]

3. **数値勾配との比較**
   - Phase 8.5のGradientCheckerで検証

### 統合テスト

1. **Newton's method収束**
   - 二次関数は1ステップで収束
   - 非二次関数の収束速度

2. **L-BFGS準備**
   - Hessian対角の計算
   - HVPの計算

### パフォーマンステスト

1. **メモリ使用量**
   - Full Hessian vs HVP vs 対角
   - N=100, 1000, 10000で測定

2. **計算時間**
   - 二階微分の追加コスト
   - GPU高速化の効果

## 期待される結果

| 手法 | メモリ | 計算時間 | 用途 |
|------|--------|---------|------|
| Full Hessian | O(N²) | O(N²) | 小規模問題 |
| Hessian対角 | O(N) | O(N) | Preconditioning |
| HVP | O(N) | O(N) | 反復法 |
| 二階微分（要素単位） | O(1) | O(1) | ポイントワイズ |

## 実装ノート

- **計算グラフのメモリ**: create_graph=trueは2倍のメモリを使用
- **数値安定性**: 二階微分は数値誤差に敏感
- **f16の限界**: 二階微分ではf32も検討
- **GPU対応**: Hessian計算もGPUで実行

## 制限事項

- **Forward-mode AD未実装**: JVPには必要
- **スパースHessian**: 大規模問題では必要
- **三階微分以上**: 理論的には可能だが実用性低い

## 将来の拡張

1. **Forward-mode AD**: JVP効率化
2. **Sparse Hessian**: スパース性の活用
3. **Hessian Free Optimization**: CG + HVP
4. **Natural Gradient**: Fisher情報行列

## 参考文献

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Numerical Optimization (Nocedal & Wright)](https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf)
