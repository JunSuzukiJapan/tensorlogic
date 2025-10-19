# Phase 8.5: Gradient Checking Design

## Overview

Implement gradient checking to validate automatic differentiation by comparing analytical gradients with numerical gradients computed via finite differences.

## Design Goals

1. **Validation**: Verify correctness of analytical gradient implementations
2. **Debugging**: Identify gradient computation errors during development
3. **Numerical Stability**: Detect numerical precision issues
4. **User-Friendly API**: Simple interface for gradient verification
5. **Configurable Tolerance**: Adjustable error thresholds

## Mathematical Foundation

### Numerical Gradient (Finite Differences)

**Central difference formula** (most accurate):
```
∂f/∂x ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

**Forward difference** (simpler but less accurate):
```
∂f/∂x ≈ [f(x + ε) - f(x)] / ε
```

**Optimal epsilon**: ε ≈ 1e-4 for f16 precision

### Error Metrics

**Relative error**:
```
relative_error = |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)
```

**Absolute error**:
```
absolute_error = |analytical - numerical|
```

## Architecture

```
┌──────────────────────────────────────────┐
│      GradientChecker (Singleton)         │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Numerical Gradient Computation    │ │
│  │  - Central difference              │ │
│  │  - Forward difference              │ │
│  │  - Element-wise computation        │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Gradient Comparison               │ │
│  │  - Relative error                  │ │
│  │  - Absolute error                  │ │
│  │  - Tolerance checking              │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Report Generation                 │ │
│  │  - Per-element errors              │ │
│  │  - Statistics (max, mean, median)  │ │
│  │  - Pass/fail status                │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

## Data Structures

```rust
/// Gradient checking configuration
#[derive(Debug, Clone)]
pub struct GradCheckConfig {
    /// Epsilon for finite differences
    pub epsilon: f32,

    /// Relative error tolerance
    pub relative_tolerance: f32,

    /// Absolute error tolerance
    pub absolute_tolerance: f32,

    /// Use central difference (vs forward difference)
    pub use_central_difference: bool,

    /// Print detailed error report
    pub verbose: bool,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            relative_tolerance: 1e-3,
            absolute_tolerance: 1e-5,
            use_central_difference: true,
            verbose: false,
        }
    }
}

/// Gradient checking result for a single tensor
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Tensor name/identifier
    pub name: String,

    /// Maximum relative error
    pub max_relative_error: f32,

    /// Mean relative error
    pub mean_relative_error: f32,

    /// Maximum absolute error
    pub max_absolute_error: f32,

    /// Mean absolute error
    pub mean_absolute_error: f32,

    /// Number of elements checked
    pub num_elements: usize,

    /// Number of elements that passed
    pub num_passed: usize,

    /// Whether all gradients passed tolerance
    pub passed: bool,

    /// Per-element errors (if verbose)
    pub element_errors: Option<Vec<(usize, f32, f32)>>, // (index, analytical, numerical)
}

/// Gradient checker
pub struct GradientChecker {
    config: GradCheckConfig,
}
```

## Implementation

### 1. Numerical Gradient Computation

```rust
impl GradientChecker {
    /// Compute numerical gradient using finite differences
    pub fn compute_numerical_gradient<F>(
        &self,
        f: F,
        input: &Tensor,
    ) -> TensorResult<Tensor>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        let input_data = input.to_vec();
        let num_elements = input_data.len();
        let mut numerical_grad = vec![f16::ZERO; num_elements];

        for i in 0..num_elements {
            if self.config.use_central_difference {
                // Central difference: [f(x+ε) - f(x-ε)] / 2ε
                let grad = self.central_difference(&f, input, i)?;
                numerical_grad[i] = grad;
            } else {
                // Forward difference: [f(x+ε) - f(x)] / ε
                let grad = self.forward_difference(&f, input, i)?;
                numerical_grad[i] = grad;
            }
        }

        Tensor::from_vec(numerical_grad, input.dims().to_vec())
    }

    fn central_difference<F>(
        &self,
        f: &F,
        input: &Tensor,
        index: usize,
    ) -> TensorResult<f16>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        let mut input_plus = input.to_vec();
        let mut input_minus = input.to_vec();

        let epsilon = f16::from_f32(self.config.epsilon);

        // x + ε
        input_plus[index] += epsilon;
        let x_plus = Tensor::from_vec(input_plus, input.dims().to_vec())?;
        let f_plus = f(&x_plus)?;

        // x - ε
        input_minus[index] -= epsilon;
        let x_minus = Tensor::from_vec(input_minus, input.dims().to_vec())?;
        let f_minus = f(&x_minus)?;

        // [f(x+ε) - f(x-ε)] / 2ε
        let diff = f_plus.to_vec()[0] - f_minus.to_vec()[0];
        let grad = diff / (f16::from_f32(2.0) * epsilon);

        Ok(grad)
    }

    fn forward_difference<F>(
        &self,
        f: &F,
        input: &Tensor,
        index: usize,
    ) -> TensorResult<f16>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        let mut input_plus = input.to_vec();
        let epsilon = f16::from_f32(self.config.epsilon);

        // f(x)
        let f_x = f(input)?;
        let f_x_val = f_x.to_vec()[0];

        // x + ε
        input_plus[index] += epsilon;
        let x_plus = Tensor::from_vec(input_plus, input.dims().to_vec())?;
        let f_plus = f(&x_plus)?;
        let f_plus_val = f_plus.to_vec()[0];

        // [f(x+ε) - f(x)] / ε
        let grad = (f_plus_val - f_x_val) / epsilon;

        Ok(grad)
    }
}
```

### 2. Gradient Comparison

```rust
impl GradientChecker {
    /// Check gradient correctness
    pub fn check_gradient<F>(
        &self,
        f: F,
        input: &Tensor,
        analytical_grad: &Tensor,
        name: &str,
    ) -> TensorResult<GradCheckResult>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        // Compute numerical gradient
        let numerical_grad = self.compute_numerical_gradient(f, input)?;

        // Compare gradients
        let analytical_data = analytical_grad.to_vec();
        let numerical_data = numerical_grad.to_vec();

        let mut max_relative_error = 0.0f32;
        let mut sum_relative_error = 0.0f32;
        let mut max_absolute_error = 0.0f32;
        let mut sum_absolute_error = 0.0f32;
        let mut num_passed = 0;
        let mut element_errors = Vec::new();

        for (i, (&analytical, &numerical)) in analytical_data.iter()
            .zip(numerical_data.iter())
            .enumerate()
        {
            let analytical_f32 = analytical.to_f32();
            let numerical_f32 = numerical.to_f32();

            // Compute errors
            let absolute_error = (analytical_f32 - numerical_f32).abs();
            let denominator = analytical_f32.abs()
                .max(numerical_f32.abs())
                .max(1e-8);
            let relative_error = absolute_error / denominator;

            // Update statistics
            max_relative_error = max_relative_error.max(relative_error);
            sum_relative_error += relative_error;
            max_absolute_error = max_absolute_error.max(absolute_error);
            sum_absolute_error += absolute_error;

            // Check tolerance
            let passed = relative_error <= self.config.relative_tolerance
                || absolute_error <= self.config.absolute_tolerance;

            if passed {
                num_passed += 1;
            }

            // Store element errors if verbose
            if self.config.verbose || !passed {
                element_errors.push((i, analytical_f32, numerical_f32));
            }
        }

        let num_elements = analytical_data.len();
        let mean_relative_error = sum_relative_error / num_elements as f32;
        let mean_absolute_error = sum_absolute_error / num_elements as f32;
        let passed = num_passed == num_elements;

        Ok(GradCheckResult {
            name: name.to_string(),
            max_relative_error,
            mean_relative_error,
            max_absolute_error,
            mean_absolute_error,
            num_elements,
            num_passed,
            passed,
            element_errors: if self.config.verbose || !passed {
                Some(element_errors)
            } else {
                None
            },
        })
    }
}
```

### 3. User API

```rust
impl Tensor {
    /// Check gradient correctness using numerical differentiation
    ///
    /// # Arguments
    /// * `f` - Function to differentiate: f(x) → scalar output
    /// * `name` - Name for the gradient check (for reporting)
    ///
    /// # Example
    /// ```
    /// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
    /// x.set_requires_grad(true);
    ///
    /// // Forward pass
    /// let y = x.mul(&x)?; // y = x²
    ///
    /// // Backward pass
    /// y.backward()?;
    ///
    /// // Check gradient
    /// let result = x.check_gradient(
    ///     |x| x.mul(x),
    ///     "x_squared",
    /// )?;
    ///
    /// assert!(result.passed);
    /// ```
    pub fn check_gradient<F>(
        &self,
        f: F,
        name: &str,
    ) -> TensorResult<GradCheckResult>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        // Get analytical gradient
        let analytical_grad = self.grad()
            .ok_or_else(|| TensorError::InvalidOperation(
                "No gradient computed. Call backward() first.".to_string()
            ))?;

        // Use default config
        let checker = GradientChecker::new();
        checker.check_gradient(f, self, analytical_grad, name)
    }

    /// Check gradient with custom configuration
    pub fn check_gradient_with_config<F>(
        &self,
        f: F,
        name: &str,
        config: GradCheckConfig,
    ) -> TensorResult<GradCheckResult>
    where
        F: Fn(&Tensor) -> TensorResult<Tensor>,
    {
        let analytical_grad = self.grad()
            .ok_or_else(|| TensorError::InvalidOperation(
                "No gradient computed. Call backward() first.".to_string()
            ))?;

        let checker = GradientChecker::with_config(config);
        checker.check_gradient(f, self, analytical_grad, name)
    }
}
```

## Usage Examples

### Example 1: Simple Function

```rust
// f(x) = x²
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
x.set_requires_grad(true);

let y = x.mul(&x)?;
y.backward()?;

let result = x.check_gradient(|x| x.mul(x), "x_squared")?;
assert!(result.passed);
println!("Max relative error: {}", result.max_relative_error);
```

### Example 2: Neural Network Layer

```rust
let x = Tensor::randn(vec![10, 5])?;
x.set_requires_grad(true);

let w = Tensor::randn(vec![5, 3])?;
w.set_requires_grad(true);

// Forward: y = relu(x @ w)
let y = x.matmul(&w)?.relu()?;
y.backward()?;

// Check gradients
let x_result = x.check_gradient(
    |x| x.matmul(&w)?.relu(),
    "x_gradient",
)?;

let w_result = w.check_gradient(
    |w| x.matmul(w)?.relu(),
    "w_gradient",
)?;

assert!(x_result.passed);
assert!(w_result.passed);
```

### Example 3: Custom Config

```rust
let config = GradCheckConfig {
    epsilon: 1e-3,
    relative_tolerance: 1e-2,
    absolute_tolerance: 1e-4,
    use_central_difference: true,
    verbose: true,
};

let result = x.check_gradient_with_config(
    |x| x.mul(x),
    "custom_check",
    config,
)?;

if !result.passed {
    println!("Gradient check failed!");
    if let Some(errors) = result.element_errors {
        for (i, analytical, numerical) in errors {
            println!("  Element {}: analytical={}, numerical={}",
                i, analytical, numerical);
        }
    }
}
```

## Testing Strategy

### Unit Tests
1. **Simple functions**: x², x³, exp(x), sin(x)
2. **Binary operations**: add, mul, sub, div
3. **Activations**: relu, gelu, softmax
4. **Reductions**: sum, mean
5. **Matrix operations**: matmul

### Integration Tests
1. **Multi-layer networks**: Linear layers with activations
2. **Complex compositions**: Multiple operations chained
3. **Different devices**: CPU, Metal, Neural Engine

### Precision Tests
1. **Epsilon sensitivity**: Test different epsilon values
2. **Tolerance levels**: Verify appropriate thresholds
3. **f16 limitations**: Document precision boundaries

## Expected Results

| Operation | Expected Max Relative Error |
|-----------|----------------------------|
| x² | < 1e-3 |
| x³ | < 1e-3 |
| add | < 1e-4 |
| mul | < 1e-3 |
| matmul | < 1e-2 |
| relu | < 1e-3 |
| gelu | < 1e-2 |
| softmax | < 1e-2 |

## Implementation Notes

- Use f16 for all computations (consistency with TensorLogic)
- Central difference is more accurate but 2x slower
- For large tensors, consider sampling random elements
- Gradient checking should only be used during development/testing

## Future Enhancements

1. **Randomized checking**: Sample random subset of elements
2. **Parallel computation**: Check multiple elements concurrently
3. **Higher-order derivatives**: Verify second derivatives
4. **Automatic testing**: Integrate with test framework
5. **Visual reports**: Generate plots of error distributions

## References

- [Gradient Checking - CS231n](http://cs231n.github.io/neural-networks-3/#gradcheck)
- [Numerical Differentiation - Wikipedia](https://en.wikipedia.org/wiki/Numerical_differentiation)
- [Testing Automatic Differentiation - JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#numerical-gradient-checking)
