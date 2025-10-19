//! Fused operations for improved performance
//!
//! These operations combine multiple operations into single GPU kernels,
//! reducing memory access overhead and kernel launch overhead.

use crate::device::{Device, MetalBuffer, NeuralEngineOps};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Activation function types for fused operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None = 0,
    ReLU = 1,
    GELU = 2,
}

impl Tensor {
    /// Fused add + relu: relu(self + other)
    ///
    /// This is more efficient than calling add() and relu() separately
    /// as it avoids allocating an intermediate buffer.
    pub fn fused_add_relu(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.fused_add_relu_metal(other),
            Device::CPU => self.fused_add_relu_cpu(other),
            Device::NeuralEngine => self.fused_add_relu_neural_engine(other),
        }
    }

    /// Metal GPU implementation of fused add + relu
    fn fused_add_relu_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load shaders if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/fused_ops.metal");
            device.load_library(shader_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::new_uninit(device.metal_device(), self.numel())?;

        // Execute kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("fused_add_relu_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf.buffer), 0);
        encoder.set_buffer(1, Some(&b_buf.buffer), 0);
        encoder.set_buffer(2, Some(&result_buf.buffer), 0);

        let grid_size = metal::MTLSize::new(self.numel() as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(self.numel() as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU implementation of fused add + relu
    fn fused_add_relu_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a_data = self.buffer().to_cpu_vec();
        let b_data = other.buffer().to_cpu_vec();

        let result: Vec<f16> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                if sum > f16::ZERO { sum } else { f16::ZERO }
            })
            .collect();

        Tensor::from_vec(result, self.dims().to_vec())
    }

    /// Fused multiply + relu: relu(self * other)
    pub fn fused_mul_relu(&self, other: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(other.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.fused_mul_relu_metal(other),
            Device::CPU => self.fused_mul_relu_cpu(other),
            Device::NeuralEngine => self.fused_mul_relu_neural_engine(other),
        }
    }

    /// Metal GPU implementation of fused mul + relu
    fn fused_mul_relu_metal(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/fused_ops.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit(device.metal_device(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("fused_mul_relu_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf.buffer), 0);
        encoder.set_buffer(1, Some(&b_buf.buffer), 0);
        encoder.set_buffer(2, Some(&result_buf.buffer), 0);

        let grid_size = metal::MTLSize::new(self.numel() as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(self.numel() as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU implementation of fused mul + relu
    fn fused_mul_relu_cpu(&self, other: &Tensor) -> TensorResult<Self> {
        let a_data = self.buffer().to_cpu_vec();
        let b_data = other.buffer().to_cpu_vec();

        let result: Vec<f16> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| {
                let product = a * b;
                if product > f16::ZERO { product } else { f16::ZERO }
            })
            .collect();

        Tensor::from_vec(result, self.dims().to_vec())
    }

    /// Fused affine transformation: self * scale + bias
    ///
    /// Used in batch normalization and similar operations.
    pub fn fused_affine(&self, scale: &Tensor, bias: &Tensor) -> TensorResult<Self> {
        if !self.shape().is_same(scale.shape()) || !self.shape().is_same(bias.shape()) {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: scale.dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) => self.fused_affine_metal(scale, bias),
            Device::CPU => self.fused_affine_cpu(scale, bias),
            Device::NeuralEngine => self.fused_affine_neural_engine(scale, bias),
        }
    }

    /// Metal GPU implementation of fused affine
    fn fused_affine_metal(&self, scale: &Tensor, bias: &Tensor) -> TensorResult<Self> {
        let x_buf = self.buffer().as_metal()?;
        let scale_buf = scale.buffer().as_metal()?;
        let bias_buf = bias.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/fused_ops.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit(device.metal_device(), self.numel())?;

        let mut executor = crate::device::KernelExecutor::new(device.clone());
        let pipeline = executor.get_or_compile_pipeline("fused_affine_f16")?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf.buffer), 0);
        encoder.set_buffer(1, Some(&scale_buf.buffer), 0);
        encoder.set_buffer(2, Some(&bias_buf.buffer), 0);
        encoder.set_buffer(3, Some(&result_buf.buffer), 0);

        let grid_size = metal::MTLSize::new(self.numel() as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(self.numel() as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// CPU implementation of fused affine
    fn fused_affine_cpu(&self, scale: &Tensor, bias: &Tensor) -> TensorResult<Self> {
        let x_data = self.buffer().to_cpu_vec();
        let scale_data = scale.buffer().to_cpu_vec();
        let bias_data = bias.buffer().to_cpu_vec();

        let result: Vec<f16> = x_data
            .iter()
            .zip(scale_data.iter())
            .zip(bias_data.iter())
            .map(|((&x, &s), &b)| x * s + b)
            .collect();

        Tensor::from_vec(result, self.dims().to_vec())
    }

    /// Neural Engine implementation of fused add + relu
    fn fused_add_relu_neural_engine(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_neural_engine()?;
        let b_buf = other.buffer().as_neural_engine()?;

        let result_buf = NeuralEngineOps::fused_add_relu(a_buf, b_buf)?;

        Tensor::new(
            BufferHandle::NeuralEngine(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// Neural Engine implementation of fused mul + relu
    fn fused_mul_relu_neural_engine(&self, other: &Tensor) -> TensorResult<Self> {
        let a_buf = self.buffer().as_neural_engine()?;
        let b_buf = other.buffer().as_neural_engine()?;

        let result_buf = NeuralEngineOps::fused_mul_relu(a_buf, b_buf)?;

        Tensor::new(
            BufferHandle::NeuralEngine(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }

    /// Neural Engine implementation of fused affine
    fn fused_affine_neural_engine(&self, scale: &Tensor, bias: &Tensor) -> TensorResult<Self> {
        let x_buf = self.buffer().as_neural_engine()?;
        let scale_buf = scale.buffer().as_neural_engine()?;
        let bias_buf = bias.buffer().as_neural_engine()?;

        let result_buf = NeuralEngineOps::fused_affine(x_buf, scale_buf, bias_buf)?;

        Tensor::new(
            BufferHandle::NeuralEngine(result_buf),
            self.shape().clone(),
            self.device().clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_fused_add_relu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(3.0), f16::from_f32(-4.0)],
            vec![4],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(-1.0), f16::from_f32(3.0), f16::from_f32(-2.0), f16::from_f32(5.0)],
            vec![4],
        )
        .unwrap();

        let result = a.fused_add_relu(&b).unwrap();
        let result_data = result.to_vec();

        // Expected: max(1-1, 0) = 0, max(-2+3, 0) = 1, max(3-2, 0) = 1, max(-4+5, 0) = 1
        assert_eq!(result_data[0], f16::from_f32(0.0));
        assert_eq!(result_data[1], f16::from_f32(1.0));
        assert_eq!(result_data[2], f16::from_f32(1.0));
        assert_eq!(result_data[3], f16::from_f32(1.0));
    }

    #[test]
    fn test_fused_mul_relu() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(-3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(0.5), f16::from_f32(2.0), f16::from_f32(-1.0)],
            vec![3],
        )
        .unwrap();

        let result = a.fused_mul_relu(&b).unwrap();
        let result_data = result.to_vec();

        // Expected: max(2*0.5, 0) = 1, max(-3*2, 0) = 0, max(4*-1, 0) = 0
        assert_eq!(result_data[0], f16::from_f32(1.0));
        assert_eq!(result_data[1], f16::from_f32(0.0));
        assert_eq!(result_data[2], f16::from_f32(0.0));
    }

    #[test]
    fn test_fused_affine() {
        let device = get_test_device();

        let x = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let scale = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![3],
        )
        .unwrap();

        let bias = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let result = x.fused_affine(&scale, &bias).unwrap();
        let result_data = result.to_vec();

        // Expected: 1*2+1=3, 2*3+2=8, 3*4+3=15
        assert_eq!(result_data[0], f16::from_f32(3.0));
        assert_eq!(result_data[1], f16::from_f32(8.0));
        assert_eq!(result_data[2], f16::from_f32(15.0));
    }

    #[test]
    fn test_fused_vs_unfused() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(-1.0)],
            vec![3],
        )
        .unwrap();

        // Fused version
        let fused_result = a.fused_add_relu(&b).unwrap();

        // Unfused version
        let unfused_result = a.add(&b).unwrap().relu().unwrap();

        // Results should be identical
        assert_eq!(fused_result.to_vec(), unfused_result.to_vec());
    }
}
