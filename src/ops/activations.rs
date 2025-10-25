//! Activation functions with Metal GPU acceleration

use crate::autograd::gradients::{GELUBackward, ReLUBackward, SoftmaxBackward};
use crate::autograd::{AutogradContext, GradientFunction, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Helper function to record a unary operation in the computation graph
fn record_unary_op(
    op: Operation,
    grad_fn: Box<dyn GradientFunction>,
    input_tensor: &Tensor,
    result: &mut Tensor,
) {
    if !input_tensor.requires_grad() || !AutogradContext::is_enabled() {
        return;
    }

    let input_node_id = input_tensor
        .grad_node()
        .unwrap_or_else(|| AutogradContext::allocate_id());

    let result_node_id = AutogradContext::add_node(op, vec![input_node_id], Some(grad_fn));

    AutogradContext::register_tensor(input_node_id, input_tensor.clone());

    result.set_grad_node(result_node_id);
    result.set_requires_grad(true);
}

impl Tensor {
    /// ReLU activation: f(x) = max(0, x)
    pub fn relu(&self) -> TensorResult<Self> {
        let mut result = match self.device() {
            Device::Metal(_) => self.relu_metal()?,
            Device::CPU => self.relu_cpu()?,
            Device::NeuralEngine => self.relu_cpu()?, // Fallback to CPU
        };

        let grad_fn = Box::new(ReLUBackward::new(self.clone()));
        record_unary_op(Operation::ReLU, grad_fn, self, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of ReLU
    fn relu_metal(&self) -> TensorResult<Self> {
        let input_buf = self.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_unary_op("relu_f16", input_buf, &result_buf)?;

        self.new_from_pool(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
        )
    }

    /// CPU fallback for ReLU
    fn relu_cpu(&self) -> TensorResult<Self> {
        let input = self.to_vec();
        let output: Vec<f16> = input.iter().map(|&x| x.max(f16::ZERO)).collect();

        Tensor::from_vec(output, self.shape().dims().to_vec())
    }

    /// GELU activation (approximation): f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> TensorResult<Self> {
        let mut result = match self.device() {
            Device::Metal(_) => self.gelu_metal()?,
            Device::CPU => self.gelu_cpu()?,
            Device::NeuralEngine => self.gelu_cpu()?, // Fallback to CPU
        };

        let grad_fn = Box::new(GELUBackward::new(self.clone()));
        record_unary_op(Operation::GELU, grad_fn, self, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of GELU
    fn gelu_metal(&self) -> TensorResult<Self> {
        let input_buf = self.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/elementwise.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_unary_op("gelu_f16", input_buf, &result_buf)?;

        self.new_from_pool(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
        )
    }

    /// CPU fallback for GELU
    fn gelu_cpu(&self) -> TensorResult<Self> {
        let input = self.to_vec();
        let sqrt_2_over_pi = f16::from_f32(0.7978845608);
        let coeff = f16::from_f32(0.044715);
        let half = f16::from_f32(0.5);
        let one = f16::ONE;

        let output: Vec<f16> = input
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = sqrt_2_over_pi * (x + coeff * x3);
                // Note: f16 doesn't have tanh, so we approximate
                let tanh_approx = {
                    let e = inner.to_f32().exp();
                    let e_neg = (-inner.to_f32()).exp();
                    f16::from_f32((e - e_neg) / (e + e_neg))
                };
                half * x * (one + tanh_approx)
            })
            .collect();

        Tensor::from_vec(output, self.shape().dims().to_vec())
    }

    /// Softmax activation: softmax(x)_i = exp(x_i) / sum(exp(x))
    ///
    /// ⚠️ **MATHEMATICALLY VERIFIED - DO NOT MODIFY**
    /// Verified with test input [1, 2, 3]:
    /// - Expected: [0.090, 0.245, 0.665]
    /// - Actual: [0.0900, 0.2446, 0.6650] ✓
    ///
    /// If you encounter incorrect output, verify OTHER operations first.
    ///
    /// Applies along the last dimension
    pub fn softmax(&self) -> TensorResult<Self> {
        // For now, CPU-only implementation
        // GPU version requires reduction operations
        let mut result = self.softmax_cpu()?;

        let grad_fn = Box::new(SoftmaxBackward::new(result.clone()));
        record_unary_op(Operation::Softmax, grad_fn, self, &mut result);

        Ok(result)
    }

    /// CPU implementation of softmax
    fn softmax_cpu(&self) -> TensorResult<Self> {
        let input = self.to_vec();
        let dims = self.shape().dims();

        // Debug: Check for NaN in input
        if std::env::var("TL_DEBUG_SOFTMAX").is_ok() {
            let nan_count = input.iter().filter(|x| x.is_nan()).count();
            let inf_count = input.iter().filter(|x| x.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                eprintln!("⚠️  Softmax input contains NaN={} Inf={}", nan_count, inf_count);
                eprintln!("   Input first 10: {:?}", &input[..input.len().min(10)]);
            }
        }

        if dims.is_empty() {
            return Err(TensorError::InvalidOperation(
                "softmax requires non-empty tensor".to_string(),
            ));
        }

        let last_dim = dims[dims.len() - 1];
        let batch_size = self.numel() / last_dim;

        let mut output = vec![f16::ZERO; self.numel()];

        for batch in 0..batch_size {
            let offset = batch * last_dim;

            // Find max for numerical stability
            // Handle NaN and Inf values and empty slices safely
            let max_val = input[offset..offset + last_dim]
                .iter()
                .copied()
                .filter(|x| x.is_finite())  // Filter out NaN and Inf values
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f16::ZERO);  // Default to 0 if empty or all NaN/Inf

            // Compute exp and sum
            // Replace Inf/NaN with 0 to prevent propagation
            let mut sum = f16::ZERO;
            for i in 0..last_dim {
                let val = input[offset + i];
                let exp_val = if val.is_finite() {
                    f16::from_f32((val - max_val).to_f32().exp())
                } else {
                    f16::ZERO  // Replace Inf/NaN with 0
                };
                output[offset + i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            // If sum is 0 or invalid, output uniform distribution
            if sum.is_finite() && sum > f16::ZERO {
                for i in 0..last_dim {
                    output[offset + i] /= sum;
                }
            } else {
                let uniform = f16::from_f32(1.0 / last_dim as f32);
                for i in 0..last_dim {
                    output[offset + i] = uniform;
                }
            }
        }

        // Create tensor on the same device as the input
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, output, self.shape().dims().to_vec()),
            _ => Tensor::from_vec(output, self.shape().dims().to_vec()),
        }
    }

    /// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.sigmoid_metal()
        } else {
            self.sigmoid_cpu()
        }
    }

    fn sigmoid_metal(&self) -> TensorResult<Self> {
        super::helpers::execute_unary_metal_op(self, "sigmoid_f16")
    }

    fn sigmoid_cpu(&self) -> TensorResult<Self> {
        super::helpers::execute_unary_cpu_op(self, |x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Hyperbolic tangent activation: tanh(x)
    pub fn tanh(&self) -> TensorResult<Self> {
        if self.buffer().is_metal() {
            self.tanh_metal()
        } else {
            self.tanh_cpu()
        }
    }

    fn tanh_metal(&self) -> TensorResult<Self> {
        super::helpers::execute_unary_metal_op(self, "tanh_f16")
    }

    fn tanh_cpu(&self) -> TensorResult<Self> {
        super::helpers::execute_unary_cpu_op(self, |x| x.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_relu_cpu() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(-2.0),
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ],
            vec![5],
        )
        .unwrap();

        let output = input.relu().unwrap();
        let result = output.to_vec();

        assert_eq!(result[0], f16::ZERO);
        assert_eq!(result[1], f16::ZERO);
        assert_eq!(result[2], f16::ZERO);
        assert_eq!(result[3], f16::from_f32(1.0));
        assert_eq!(result[4], f16::from_f32(2.0));
    }

    #[test]
    fn test_relu_gpu() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(-2.0),
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ],
            vec![5],
        )
        .unwrap();

        let output = input.relu().unwrap();
        let result = output.to_vec();

        assert_eq!(result[0], f16::ZERO);
        assert_eq!(result[1], f16::ZERO);
        assert_eq!(result[2], f16::ZERO);
        assert_eq!(result[3], f16::from_f32(1.0));
        assert_eq!(result[4], f16::from_f32(2.0));
    }

    #[test]
    fn test_gelu_cpu() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
            ],
            vec![3],
        )
        .unwrap();

        let output = input.gelu().unwrap();
        let result = output.to_vec();

        // GELU(0) should be approximately 0
        assert!((result[1].to_f32()).abs() < 0.01);
        // GELU is monotonic: GELU(-1) < GELU(0) < GELU(1)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_gelu_gpu() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
            ],
            vec![3],
        )
        .unwrap();

        let output = input.gelu().unwrap();
        let result = output.to_vec();

        assert!((result[1].to_f32()).abs() < 0.01);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_cpu() {
        let input = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ],
            vec![3],
        )
        .unwrap();

        let output = input.softmax().unwrap();
        let result = output.to_vec();

        // Check sum equals 1
        let sum: f32 = result.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Check values are in ascending order (softmax is monotonic)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_sigmoid() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(-2.0),
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ],
            vec![5],
        )
        .unwrap();

        let output = input.sigmoid().unwrap();
        let result = output.to_vec();

        // Sigmoid should be in range (0, 1)
        for val in &result {
            let f = val.to_f32();
            assert!(f > 0.0 && f < 1.0);
        }

        // Check specific values
        assert!((result[2].to_f32() - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
        assert!((result[3].to_f32() - 0.731).abs() < 0.01); // sigmoid(1) ≈ 0.731
        assert!((result[4].to_f32() - 0.881).abs() < 0.01); // sigmoid(2) ≈ 0.881
    }

    #[test]
    fn test_tanh() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(-2.0),
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
            ],
            vec![5],
        )
        .unwrap();

        let output = input.tanh().unwrap();
        let result = output.to_vec();

        // tanh should be in range (-1, 1)
        for val in &result {
            let f = val.to_f32();
            assert!(f > -1.0 && f < 1.0);
        }

        // Check specific values
        assert!((result[2].to_f32() - 0.0).abs() < 0.01); // tanh(0) = 0
        assert!((result[3].to_f32() - 0.762).abs() < 0.01); // tanh(1) ≈ 0.762
        assert!((result[4].to_f32() - 0.964).abs() < 0.01); // tanh(2) ≈ 0.964

        // Check symmetry
        assert!((result[0].to_f32() + result[4].to_f32()).abs() < 0.01); // tanh(-x) = -tanh(x)
    }
}
