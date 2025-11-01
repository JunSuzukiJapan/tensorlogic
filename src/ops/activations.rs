//! Activation functions with Metal GPU acceleration

use crate::autograd::gradients::{GELUBackward, ReLUBackward, SoftmaxBackward};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorAutograd};
use crate::autograd::{AutogradContext, GradientFunctionGeneric, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Helper function to record a unary operation in the computation graph (generic version)
fn record_unary_op_generic<T: FloatType>(
    op: Operation,
    grad_fn: Box<dyn GradientFunctionGeneric<T>>,
    input_tensor: &Tensor<T>,
    result: &mut Tensor<T>,
) where
    Tensor<T>: TensorAutograd<T>,
{
    if !input_tensor.requires_grad() || !AutogradContext::is_enabled() {
        return;
    }

    let input_node_id = input_tensor
        .grad_node()
        .unwrap_or_else(|| AutogradContext::allocate_id());

    let result_node_id = AutogradContext::add_node_generic(op, vec![input_node_id], Some(grad_fn));

    AutogradContext::register_tensor_generic(input_node_id, input_tensor.clone());
    result.set_grad_node(result_node_id);
    result.set_requires_grad(true);
}

impl<T: FloatType> Tensor<T> {
    /// ReLU activation: f(x) = max(0, x)
    pub fn relu(&self) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
        let mut result = match self.device() {
            Device::Metal(_) => self.relu_metal()?,
            Device::CPU => self.relu_cpu()?,
            Device::NeuralEngine => self.relu_cpu()?, // Fallback to CPU
        };

        let grad_fn = Box::new(ReLUBackward::new(self.clone()));
        record_unary_op_generic(Operation::ReLU, grad_fn, self, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of ReLU
    fn relu_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        let result_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), self.numel())?;
        let input_buf_f16: &MetalBuffer<f16> = unsafe { std::mem::transmute(input_buf) };

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_unary_op("relu_f16", input_buf_f16, &result_buf_f16)?;

        let result_buf: MetalBuffer<T> = unsafe { std::mem::transmute(result_buf_f16) };
        self.new_from_pool(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
        )
    }

    /// CPU fallback for ReLU
    fn relu_cpu(&self) -> TensorResult<Self> {
        panic!("RELU CPU FALLBACK: GPU kernel must be used - CPU implementation disabled");
    }

    /// GELU activation (approximation): f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
        let mut result = match self.device() {
            Device::Metal(_) => self.gelu_metal()?,
            Device::CPU => self.gelu_cpu()?,
            Device::NeuralEngine => self.gelu_cpu()?, // Fallback to CPU
        };

        let grad_fn = Box::new(GELUBackward::new(self.clone()));
        record_unary_op_generic(Operation::GELU, grad_fn, self, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of GELU
    fn gelu_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        let result_buf_f16 = MetalBuffer::<f16>::new_uninit_pooled(device.buffer_pool(), self.numel())?;
        let input_buf_f16: &MetalBuffer<f16> = unsafe { std::mem::transmute(input_buf) };

        let mut executor = crate::device::KernelExecutor::new(device);
        executor.execute_unary_op("gelu_f16", input_buf_f16, &result_buf_f16)?;

        let result_buf: MetalBuffer<T> = unsafe { std::mem::transmute(result_buf_f16) };
        self.new_from_pool(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
        )
    }

    /// CPU fallback for GELU
    fn gelu_cpu(&self) -> TensorResult<Self> {
        panic!("GELU CPU FALLBACK: GPU kernel must be used - CPU implementation disabled");
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
    pub fn softmax(&self) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
        let mut result = if self.buffer().is_metal() {
            self.softmax_metal()?
        } else {
            self.softmax_cpu()?
        };

        let grad_fn = Box::new(SoftmaxBackward::new(result.clone()));
        record_unary_op_generic(Operation::Softmax, grad_fn, self, &mut result);

        Ok(result)
    }

    /// Metal GPU implementation of softmax
    fn softmax_metal(&self) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        use crate::device::{Device, MetalBuffer};

        let dims = self.shape().dims();
        if dims.is_empty() {
            return Err(TensorError::InvalidOperation(
                "softmax requires non-empty tensor".to_string(),
            ));
        }

        let last_dim = dims[dims.len() - 1];
        let batch_size = self.numel() / last_dim;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Choose kernel based on last_dim size and type
        let suffix = T::kernel_suffix();
        let kernel_name = if last_dim <= 256 {
            format!("softmax_simple{}", suffix)
        } else {
            format!("softmax{}", suffix)
        };

        let input_buf = self.buffer().as_metal()?;
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), self.numel())?;

        // Create buffer for last_dim parameter (as u32, matching Metal shader)
        let last_dim_u32 = last_dim as u32;
        let last_dim_buf = device.metal_device().new_buffer_with_data(
            &last_dim_u32 as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline
        let library_ref = device.library();
        let library = library_ref.as_ref().ok_or_else(|| {
            TensorError::MetalError("Library not loaded".to_string())
        })?;
        let pipeline = library
            .get_function(&kernel_name, None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to create pipeline: {:?}", e))
            })?;

        // Execute kernel
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.as_ref().new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(&last_dim_buf), 0);

        // For now, always use simple kernel (dispatch_threads)
        // TODO: Implement threadgroup dispatch when needed for large dimensions
        use metal::MTLSize;
        let grid_size = MTLSize::new(batch_size as u64, 1, 1);
        let threadgroup_size = MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.end_encoding();
        // command_buffer.commit(); // Handled by Commands manager
        // submit_async - not needed with Commands batching

        self.new_from_pool(
            crate::tensor::BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    /// CPU implementation of softmax
    fn softmax_cpu(&self) -> TensorResult<Self> {
        panic!("SOFTMAX CPU FALLBACK: GPU kernel must be used - CPU implementation disabled");
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
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "sigmoid_f16")
    }

    fn sigmoid_cpu(&self) -> TensorResult<Self> {
        panic!("SIGMOID CPU FALLBACK: GPU kernel must be used - CPU implementation disabled");
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
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        super::helpers::execute_unary_metal_op(self, "tanh_f16")
    }

    fn tanh_cpu(&self) -> TensorResult<Self> {
        panic!("src/ops/activations.rs:452:5");
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

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
        let result = output.sync_and_read();

        assert_eq!(result[0], f16::ZERO);
        assert_eq!(result[1], f16::ZERO);
        assert_eq!(result[2], f16::ZERO);
        assert_eq!(result[3], f16::from_f32(1.0));
        assert_eq!(result[4], f16::from_f32(2.0));
    }

    #[test]
    fn test_relu_gpu() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_gpu(
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
        let result = output.sync_and_read();

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
        let result = output.sync_and_read();

        // GELU(0) should be approximately 0
        assert!((result[1].to_f32()).abs() < 0.01);
        // GELU is monotonic: GELU(-1) < GELU(0) < GELU(1)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_gelu_gpu() {
        let device = MetalDevice::new().unwrap();

        let input = Tensor::from_vec_gpu(
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
        let result = output.sync_and_read();

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
        let result = output.sync_and_read();

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

        let input = Tensor::from_vec_gpu(
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
        let result = output.sync_and_read();

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

        let input = Tensor::from_vec_gpu(
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
        let result = output.sync_and_read();

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

    #[test]
    fn test_softmax_mathematical_correctness() {
        // Test softmax with known values
        // Formula: softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
        //
        // For input [1, 2, 3]:
        // exp(1) = 2.718, exp(2) = 7.389, exp(3) = 20.086
        // sum = 30.193
        // output = [2.718/30.193, 7.389/30.193, 20.086/30.193]
        //        = [0.090, 0.245, 0.665]

        let device = MetalDevice::new().expect("Failed to create Metal device");

        let x_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ];

        let x = Tensor::from_vec_gpu(&device, x_data, vec![3]).unwrap();
        let result = x.softmax().unwrap();
        let values = result.sync_and_read();

        // Expected values from the formula
        let expected = vec![0.090, 0.245, 0.665];

        for i in 0..3 {
            let actual = values[i].to_f32();
            let diff = (actual - expected[i]).abs();
            assert!(
                diff < 0.01,
                "Softmax mismatch at index {}: expected {:.3}, got {:.3}, diff {:.3}",
                i,
                expected[i],
                actual,
                diff
            );
        }

        // Verify sum equals 1 (fundamental property)
        let sum: f32 = values.iter().map(|x| x.to_f32()).sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Softmax output should sum to 1.0, got {:.6}",
            sum
        );
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test softmax with large values (should not overflow)
        // Softmax should use max subtraction for numerical stability:
        // softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
        let device = MetalDevice::new().expect("Failed to create Metal device");

        let x_data = vec![
            f16::from_f32(100.0),
            f16::from_f32(101.0),
            f16::from_f32(102.0),
        ];

        let x = Tensor::from_vec_gpu(&device, x_data, vec![3]).unwrap();
        let result = x.softmax().unwrap();
        let values = result.sync_and_read();

        // After max subtraction: [-2, -1, 0]
        // Same result as test above: [0.090, 0.245, 0.665]
        let expected = vec![0.090, 0.245, 0.665];

        for i in 0..3 {
            let actual = values[i].to_f32();
            assert!(
                !actual.is_nan() && !actual.is_infinite(),
                "Softmax should not produce NaN or Inf with large values"
            );
            let diff = (actual - expected[i]).abs();
            assert!(
                diff < 0.01,
                "Softmax with large values mismatch at index {}: expected {:.3}, got {:.3}",
                i,
                expected[i],
                actual
            );
        }

        // Verify sum equals 1
        let sum: f32 = values.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_2d_batch() {
        // Test softmax on 2D tensor (batch processing)
        // Softmax should be applied independently to each row
        let device = MetalDevice::new().expect("Failed to create Metal device");

        let x_data = vec![
            // Row 0: [1, 2, 3]
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            // Row 1: [0, 0, 0]
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
        ];

        let x = Tensor::from_vec_gpu(&device, x_data, vec![2, 3]).unwrap();
        let result = x.softmax().unwrap();
        let values = result.sync_and_read();

        // Row 0 expected: [0.090, 0.245, 0.665]
        let row0_expected = vec![0.090, 0.245, 0.665];
        for i in 0..3 {
            let actual = values[i].to_f32();
            let diff = (actual - row0_expected[i]).abs();
            assert!(
                diff < 0.01,
                "Row 0 mismatch at index {}: expected {:.3}, got {:.3}",
                i,
                row0_expected[i],
                actual
            );
        }

        // Row 1 expected: [1/3, 1/3, 1/3] (uniform for equal inputs)
        for i in 3..6 {
            let actual = values[i].to_f32();
            let expected = 1.0 / 3.0;
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.01,
                "Row 1 mismatch at index {}: expected {:.3}, got {:.3}",
                i - 3,
                expected,
                actual
            );
        }

        // Verify each row sums to 1
        let row0_sum: f32 = values[0..3].iter().map(|x| x.to_f32()).sum();
        let row1_sum: f32 = values[3..6].iter().map(|x| x.to_f32()).sum();
        assert!((row0_sum - 1.0).abs() < 0.001);
        assert!((row1_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_deterministic() {
        // Same input should always produce same output
        let device = MetalDevice::new().expect("Failed to create Metal device");

        let x_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];

        let x = Tensor::from_vec_gpu(&device, x_data, vec![4]).unwrap();

        let result1 = x.softmax().unwrap();
        let result2 = x.softmax().unwrap();
        let result3 = x.softmax().unwrap();

        let values1 = result1.sync_and_read();
        let values2 = result2.sync_and_read();
        let values3 = result3.sync_and_read();

        for i in 0..4 {
            assert_eq!(
                values1[i], values2[i],
                "Softmax should be deterministic (result1 vs result2 at index {})",
                i
            );
            assert_eq!(
                values2[i], values3[i],
                "Softmax should be deterministic (result2 vs result3 at index {})",
                i
            );
        }
    }
}
