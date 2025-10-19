# Phase 8.6: Higher-Order Derivatives Design

## Overview

Implement support for computing second-order and higher-order derivatives (Hessian, gradients of gradients) for advanced optimization algorithms and numerical analysis.

## Design Goals

1. **Second-Order Derivatives**: Compute Hessian matrix for Newton's method and second-order optimization
2. **Gradient of Gradients**: Enable computing derivatives of derivatives systematically
3. **Efficient Computation**: Leverage existing autograd infrastructure without massive overhead
4. **API Simplicity**: Clean interface for higher-order differentiation
5. **Selective Computation**: Compute only needed higher-order derivatives

## Mathematical Foundation

### Second Derivative (Hessian)

For scalar function f: ℝⁿ → ℝ, the Hessian matrix H is:

```
H[i,j] = ∂²f/∂xᵢ∂xⱼ
```

**Symmetry property**: H[i,j] = H[j,i] (Schwarz's theorem)

### Gradient of Gradient

For a computation graph with loss L and parameters θ:
- First-order: ∂L/∂θ (gradient)
- Second-order: ∂²L/∂θ² (Hessian diagonal)
- Second-order: ∂(∂L/∂θᵢ)/∂θⱼ (Hessian off-diagonal)

### Higher-Order Derivatives

For f: ℝ → ℝ:
- f'(x) = df/dx (first derivative)
- f''(x) = d²f/dx² (second derivative)
- f⁽ⁿ⁾(x) = dⁿf/dxⁿ (nth derivative)

## Architecture

```
┌──────────────────────────────────────────┐
│      Higher-Order Derivatives            │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Hessian Computation               │ │
│  │  - Full Hessian matrix             │ │
│  │  - Hessian-vector product (HVP)    │ │
│  │  - Hessian diagonal                │ │
│  │  └────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Gradient-of-Gradient              │ │
│  │  - create_graph=true support       │ │
│  │  - Nested backward() calls         │ │
│  │  - Gradient retention              │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Optimization Support              │ │
│  │  - Newton's method (full Hessian)  │ │
│  │  - L-BFGS (HVP)                    │ │
│  │  - Trust region methods            │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

## Implementation Strategy

### Approach 1: create_graph Flag (PyTorch-style)

Enable computation graph creation during backward pass:

```rust
impl Tensor {
    /// Backward pass with computation graph creation
    pub fn backward_with_graph(&self, create_graph: bool) -> TensorResult<()> {
        if create_graph {
            // Keep computation graph for higher-order derivatives
            // Allow gradients to have their own gradients
        } else {
            // Standard backward (current implementation)
        }
    }
}
```

**Usage**:
```rust
// First-order gradient
let x = Tensor::from_vec(vec![2.0], vec![1])?;
x.set_requires_grad(true);

let y = x.mul(&x)?; // y = x²
y.backward()?; // dy/dx = 2x = 4

// Second-order gradient
let grad1 = x.grad().unwrap();
grad1.set_requires_grad(true); // Enable gradient of gradient

let z = grad1.mul(&x)?; // Create new computation
z.backward_with_graph(true)?; // Compute second derivative
let grad2 = x.grad()?; // d²y/dx² = 2
```

### Approach 2: Explicit Hessian Computation

Provide dedicated API for Hessian computation:

```rust
impl Tensor {
    /// Compute Hessian matrix for scalar output
    pub fn hessian(&self, output: &Tensor) -> TensorResult<Tensor> {
        // Compute full Hessian matrix H[i,j] = ∂²output/∂xᵢ∂xⱼ
    }

    /// Compute Hessian-vector product (more efficient)
    pub fn hessian_vector_product(
        &self,
        output: &Tensor,
        vector: &Tensor,
    ) -> TensorResult<Tensor> {
        // Compute H·v without forming full Hessian
    }

    /// Compute Hessian diagonal only
    pub fn hessian_diagonal(&self, output: &Tensor) -> TensorResult<Tensor> {
        // Compute diag(H) = [∂²f/∂x₁², ∂²f/∂x₂², ...]
    }
}
```

**Usage**:
```rust
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
x.set_requires_grad(true);

// f(x) = sum(x²) = x₁² + x₂² + x₃²
let y = x.mul(&x)?.sum()?;

// Compute full Hessian (3x3 matrix)
let hessian = x.hessian(&y)?;
// Expected: diag([2, 2, 2]) with zeros off-diagonal

// Compute Hessian diagonal only (more efficient)
let hess_diag = x.hessian_diagonal(&y)?;
// Expected: [2, 2, 2]
```

### Approach 3: grad() Function (JAX-style)

Functional approach with composable grad():

```rust
/// Compute gradient of a function
pub fn grad<F>(f: F) -> impl Fn(&Tensor) -> TensorResult<Tensor>
where
    F: Fn(&Tensor) -> TensorResult<Tensor>,
{
    move |x: &Tensor| {
        x.set_requires_grad(true);
        let y = f(x)?;
        y.backward()?;
        x.grad()
    }
}

/// Second-order: grad(grad(f))
pub fn grad2<F>(f: F) -> impl Fn(&Tensor) -> TensorResult<Tensor>
where
    F: Fn(&Tensor) -> TensorResult<Tensor>,
{
    grad(grad(f))
}
```

**Usage**:
```rust
let f = |x: &Tensor| x.mul(x); // f(x) = x²

// First derivative: f'(x) = 2x
let df = grad(f);
let x = Tensor::from_vec(vec![3.0], vec![1])?;
let grad1 = df(&x)?; // = 6.0

// Second derivative: f''(x) = 2
let d2f = grad2(f);
let grad2 = d2f(&x)?; // = 2.0
```

## Data Structures

### Option 1: Extend AutogradContext

```rust
pub struct AutogradContext {
    graph: ComputationGraph,
    tensor_registry: HashMap<usize, Tensor>,

    // New fields for higher-order derivatives
    create_graph: bool,
    gradient_mode: GradientMode,
}

pub enum GradientMode {
    /// Standard gradient computation (no graph retention)
    Standard,

    /// Create graph for gradients (enables higher-order derivatives)
    CreateGraph,

    /// Compute Hessian
    Hessian,
}
```

### Option 2: Separate Hessian Computer

```rust
/// Hessian computation engine
pub struct HessianComputer {
    /// Configuration
    config: HessianConfig,

    /// Cache for intermediate results
    cache: HashMap<usize, Tensor>,
}

pub struct HessianConfig {
    /// Compute full Hessian or diagonal only
    pub mode: HessianMode,

    /// Use numerical approximation if analytical fails
    pub fallback_to_numerical: bool,

    /// Epsilon for numerical Hessian
    pub epsilon: f32,
}

pub enum HessianMode {
    /// Full Hessian matrix
    Full,

    /// Diagonal only (∂²f/∂xᵢ²)
    Diagonal,

    /// Hessian-vector product
    VectorProduct,
}
```

## Implementation Plan

### Phase 8.6.1: create_graph Support

1. **Extend AutogradContext**:
   - Add `create_graph` flag
   - Modify `backward()` to retain computation graph if flag is set
   - Enable gradients to have their own gradients

2. **Gradient Retention**:
   - Store gradient tensors in registry with `requires_grad=true`
   - Allow backward() to be called on gradients

3. **Testing**:
   - Simple second derivative: f(x) = x², f''(x) = 2
   - Chain rule: f(x) = (x²)², f'(x) = 4x³, f''(x) = 12x²

### Phase 8.6.2: Hessian Computation

1. **Hessian Diagonal**:
   - Most efficient: only compute ∂²f/∂xᵢ²
   - Use forward-over-reverse automatic differentiation
   - Implementation via repeated backward() calls

2. **Full Hessian Matrix**:
   - Compute all ∂²f/∂xᵢ∂xⱼ
   - Exploit symmetry: H[i,j] = H[j,i]
   - Only compute upper triangle

3. **Hessian-Vector Product (HVP)**:
   - Compute H·v without forming full H
   - More efficient for large dimensions
   - Use reverse-over-reverse or forward-over-reverse

### Phase 8.6.3: Functional API

1. **grad() Function**:
   - Functional gradient computation
   - Composable: grad(grad(f)) for second-order

2. **Higher-Order Helpers**:
   - `jacobian()`: Compute Jacobian matrix
   - `hessian()`: Compute full Hessian
   - `grad_and_value()`: Compute gradient and function value together

## Usage Examples

### Example 1: Newton's Method

```rust
// Minimize f(x) = x² using Newton's method
let x = Tensor::from_vec(vec![10.0], vec![1])?;
x.set_requires_grad(true);

for iter in 0..10 {
    // Compute f(x) = x²
    let f = x.mul(&x)?;

    // First derivative: f'(x) = 2x
    f.backward_with_graph(true)?;
    let grad1 = x.grad()?;

    // Second derivative: f''(x) = 2
    grad1.backward()?;
    let grad2 = x.grad()?;

    // Newton update: x_new = x - f'(x) / f''(x)
    let update = grad1.div(&grad2)?;
    x.sub_(&update)?;

    x.zero_grad()?;
}

// x should converge to 0
```

### Example 2: Hessian Diagonal for L-BFGS

```rust
let x = Tensor::randn(vec![100])?;
x.set_requires_grad(true);

// Loss function
let loss = compute_loss(&x)?;

// Compute Hessian diagonal for preconditioning
let hess_diag = x.hessian_diagonal(&loss)?;

// Use in L-BFGS optimization
let preconditioner = hess_diag.reciprocal()?;
```

### Example 3: Functional Second Derivative

```rust
// Define function f(x) = x³
let f = |x: &Tensor| {
    let x2 = x.mul(x)?;
    x2.mul(x)
};

// First derivative: f'(x) = 3x²
let df = grad(f);

// Second derivative: f''(x) = 6x
let d2f = grad(df);

let x = Tensor::from_vec(vec![2.0], vec![1])?;
let second_deriv = d2f(&x)?; // = 12.0
```

### Example 4: Hessian Matrix

```rust
// f(x,y) = x² + xy + y²
let x = Tensor::from_vec(vec![1.0, 2.0], vec![2])?;
x.set_requires_grad(true);

let f = |x: &Tensor| {
    let x_vec = x.to_vec();
    let x1 = x_vec[0];
    let x2 = x_vec[1];

    // f = x1² + x1*x2 + x2²
    let term1 = Tensor::from_vec(vec![x1 * x1], vec![1])?;
    let term2 = Tensor::from_vec(vec![x1 * x2], vec![1])?;
    let term3 = Tensor::from_vec(vec![x2 * x2], vec![1])?;

    term1.add(&term2)?.add(&term3)
};

// Compute Hessian
let y = f(&x)?;
let hessian = x.hessian(&y)?;

// Expected Hessian:
// H = [[2, 1],
//      [1, 2]]
```

## Testing Strategy

### Unit Tests

1. **Simple Functions**:
   - f(x) = x²: f''(x) = 2
   - f(x) = x³: f''(x) = 6x
   - f(x) = exp(x): f''(x) = exp(x)
   - f(x) = sin(x): f''(x) = -sin(x)

2. **Multivariate Functions**:
   - f(x,y) = x² + y²: Hessian = diag([2, 2])
   - f(x,y) = xy: Hessian = [[0, 1], [1, 0]]
   - f(x,y) = x² + xy + y²: Hessian = [[2, 1], [1, 2]]

3. **Chain Rule**:
   - f(g(x)) where f and g are known functions
   - Verify: (f∘g)''(x) = f''(g(x))·g'(x)² + f'(g(x))·g''(x)

### Integration Tests

1. **Newton's Method**:
   - Optimize quadratic function
   - Verify convergence in 1-2 iterations

2. **Gradient Checking**:
   - Use Phase 8.5 gradient checker to validate second derivatives
   - Compare analytical Hessian with numerical approximation

3. **Performance Tests**:
   - HVP vs full Hessian computation
   - Hessian diagonal vs full Hessian

## Implementation Priorities

**Priority 1 (MVP)**:
- create_graph flag in backward()
- Simple second derivative for scalar functions
- Basic tests with f(x) = x², x³

**Priority 2 (Useful)**:
- Hessian diagonal computation
- Hessian-vector product
- Integration with optimization algorithms

**Priority 3 (Complete)**:
- Full Hessian matrix computation
- Functional grad() API
- JAX-style composable derivatives

**Priority 4 (Advanced)**:
- Higher-order derivatives (3rd, 4th, ...)
- Automatic differentiation of gradients in parallel
- GPU acceleration for Hessian computation

## Expected Results

| Operation | Input | Expected Output |
|-----------|-------|-----------------|
| f''(x) for f(x)=x² | x=3 | 2.0 |
| f''(x) for f(x)=x³ | x=2 | 12.0 |
| Hessian diag(x²+y²) | x=[1,2] | [2.0, 2.0] |
| Hessian of xy | x=[1,1] | [[0,1],[1,0]] |
| HVP for x² | x=3, v=1 | 2.0 |

## Implementation Notes

- **Graph Retention**: create_graph requires keeping the computation graph after backward()
- **Memory Overhead**: Higher-order derivatives increase memory usage significantly
- **Numerical Stability**: Second derivatives are more sensitive to numerical errors
- **f16 Precision**: May need f32 for accurate higher-order derivatives
- **Optimization**: Hessian-vector product is much more efficient than full Hessian

## Future Enhancements

1. **Automatic Hessian Optimization**: Choose best method based on problem size
2. **Sparse Hessian**: Exploit sparsity patterns in Hessian matrix
3. **Parallel Hessian Computation**: Compute columns in parallel
4. **Mixed Precision**: Use f32 for Hessian, f16 for forward pass
5. **Lazy Evaluation**: Compute Hessian elements on-demand

## References

- [Automatic Differentiation - Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Computing Higher-Order Derivatives - JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#higher-order-derivatives)
- [PyTorch Autograd - create_graph](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
- [Hessian-Free Optimization](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)
