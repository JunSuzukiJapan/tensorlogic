//! Gradient checking utilities for validating automatic differentiation

use crate::tensor::Tensor;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorAutograd};
use crate::error::TensorResult;
use half::f16;

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
            epsilon: 1e-2, // f16 has limited precision, use larger epsilon
            relative_tolerance: 1e-2,
            absolute_tolerance: 1e-3,
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

/// Gradient checker for validating automatic differentiation
pub struct GradientChecker {
    config: GradCheckConfig,
}

impl GradientChecker {
    /// Create a new gradient checker with default configuration
    pub fn new() -> Self {
        Self {
            config: GradCheckConfig::default(),
        }
    }

    /// Create a gradient checker with custom configuration
    pub fn with_config(config: GradCheckConfig) -> Self {
        Self { config }
    }

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
        let x_plus = Tensor::from_vec(input_plus.clone(), input.dims().to_vec())?;
        let f_plus = f(&x_plus)?;

        // For single-element output, use direct indexing; otherwise sum
        let f_plus_val = if f_plus.numel() == 1 {
            f_plus.to_vec()[0]
        } else {
            f_plus.sum()?
        };

        // x - ε
        input_minus[index] -= epsilon;
        let x_minus = Tensor::from_vec(input_minus.clone(), input.dims().to_vec())?;
        let f_minus = f(&x_minus)?;

        let f_minus_val = if f_minus.numel() == 1 {
            f_minus.to_vec()[0]
        } else {
            f_minus.sum()?
        };

        // [f(x+ε) - f(x-ε)] / 2ε
        let diff = f_plus_val - f_minus_val;
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
        let f_x_val = if f_x.numel() == 1 {
            f_x.to_vec()[0]
        } else {
            f_x.sum()?
        };

        // x + ε
        input_plus[index] += epsilon;
        let x_plus = Tensor::from_vec(input_plus.clone(), input.dims().to_vec())?;
        let f_plus = f(&x_plus)?;
        let f_plus_val = if f_plus.numel() == 1 {
            f_plus.to_vec()[0]
        } else {
            f_plus.sum()?
        };

        // [f(x+ε) - f(x)] / ε
        let grad = (f_plus_val - f_x_val) / epsilon;

        Ok(grad)
    }

    /// Check gradient correctness by comparing analytical and numerical gradients
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

            // Store element errors if verbose or failed
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

impl Default for GradientChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GradCheckConfig::default();
        assert_eq!(config.epsilon, 1e-2); // f16 precision
        assert_eq!(config.relative_tolerance, 1e-2);
        assert_eq!(config.absolute_tolerance, 1e-3);
        assert!(config.use_central_difference);
        assert!(!config.verbose);
    }

    #[test]
    fn test_checker_creation() {
        let _checker = GradientChecker::new();
        assert!(true);
    }

    #[test]
    fn test_numerical_gradient_simple() {
        let checker = GradientChecker::new();

        // f(x) = x², so f'(x) = 2x
        let x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1]).unwrap();

        let numerical_grad = checker.compute_numerical_gradient(
            |x: &Tensor| {
                // Use tensor mul operation
                x.mul(x)
            },
            &x,
        ).unwrap();

        let grad_val = numerical_grad.to_vec()[0].to_f32();

        // f'(2) = 2*2 = 4
        let expected = 4.0f32;
        let error = (grad_val - expected).abs();

        assert!(error < 0.5, "Numerical gradient error too large: {} vs {}", grad_val, expected);
    }

    #[test]
    fn test_central_vs_forward_difference() {
        // Central difference should be more accurate
        let x = Tensor::from_vec(vec![f16::from_f32(3.0)], vec![1]).unwrap();

        // f(x) = x³, f'(x) = 3x²
        // We'll use x * x * x via tensor operations
        let f = |x: &Tensor| {
            let x2 = x.mul(x)?;
            x2.mul(x)
        };

        let central_config = GradCheckConfig {
            use_central_difference: true,
            ..Default::default()
        };
        let central_checker = GradientChecker::with_config(central_config);
        let central_grad = central_checker.compute_numerical_gradient(&f, &x).unwrap();

        let forward_config = GradCheckConfig {
            use_central_difference: false,
            ..Default::default()
        };
        let forward_checker = GradientChecker::with_config(forward_config);
        let forward_grad = forward_checker.compute_numerical_gradient(&f, &x).unwrap();

        // f'(3) = 3*3² = 27
        let expected = 27.0f32;
        let central_error = (central_grad.to_vec()[0].to_f32() - expected).abs();
        let forward_error = (forward_grad.to_vec()[0].to_f32() - expected).abs();

        // Both should be reasonably close
        assert!(central_error < 5.0,
            "Central difference error {} should be reasonable",
            central_error);
        assert!(forward_error < 5.0,
            "Forward difference error {} should be reasonable",
            forward_error);
    }
}
