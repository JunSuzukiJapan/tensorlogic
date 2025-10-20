//! Matrix multiplication operations with Metal GPU acceleration

use crate::autograd::gradients::MatMulBackward;
use crate::autograd::{AutogradContext, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

impl Tensor {
    /// Matrix multiplication: self @ other
    ///
    /// # Arguments
    /// - self: shape [M, K]
    /// - other: shape [K, N]
    ///
    /// # Returns
    /// - result: shape [M, N]
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Self> {
        // Validate shapes for matrix multiplication
        if self.shape().rank() != 2 || other.shape().rank() != 2 {
            return Err(TensorError::InvalidOperation(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let m = self.shape().dims()[0];
        let k = self.shape().dims()[1];
        let k2 = other.shape().dims()[0];
        let n = other.shape().dims()[1];

        if k != k2 {
            return Err(TensorError::InvalidOperation(format!(
                "matmul shape mismatch: [{}, {}] @ [{}, {}]",
                m, k, k2, n
            )));
        }

        // Check if both tensors are on the same device
        if self.device() != other.device() {
            return Err(TensorError::DeviceConversionError(
                "matmul requires tensors on same device".to_string(),
            ));
        }

        let mut result = match self.device() {
            Device::Metal(_) => self.matmul_metal(other, m, k, n)?,
            Device::CPU => self.matmul_cpu(other, m, k, n)?,
            Device::NeuralEngine => {
                return Err(TensorError::InvalidOperation(
                    "matmul not yet supported on Neural Engine".to_string(),
                ))
            }
        };

        // Record in computation graph
        if (self.requires_grad() || other.requires_grad()) && AutogradContext::is_enabled() {
            let self_node_id = self.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());
            let other_node_id =
                other.grad_node().unwrap_or_else(|| AutogradContext::allocate_id());

            let grad_fn = Box::new(MatMulBackward::new(self.clone(), other.clone()));

            let result_node_id = AutogradContext::add_node(
                Operation::MatMul,
                vec![self_node_id, other_node_id],
                Some(grad_fn),
            );

            AutogradContext::register_tensor(self_node_id, self.clone());
            AutogradContext::register_tensor(other_node_id, other.clone());

            result.set_grad_node(result_node_id);
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    /// Metal GPU implementation of matmul with adaptive tiling
    ///
    /// Uses threadgroup memory tiling for improved performance (1.5-2x speedup).
    /// Automatically selects optimal tile size based on matrix dimensions.
    fn matmul_metal(&self, other: &Tensor, m: usize, k: usize, n: usize) -> TensorResult<Self> {
        let a_buf = self.buffer().as_metal()?;
        let b_buf = other.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        // Load shaders if not already loaded
        if device.library().is_none() {
            // Load both elementwise and tiled matmul shaders
            let elementwise_source = include_str!("../../shaders/elementwise.metal");
            let tiled_source = include_str!("../../shaders/matmul_tiled.metal");

            // Combine shader sources
            let combined_source = format!("{}\n\n{}", elementwise_source, tiled_source);
            device.load_library(&combined_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), m * n)?;

        // Create buffers for dimensions
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        // Execute matmul kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // Select optimal kernel based on matrix size
        // Use tiled version for larger matrices (better cache utilization)
        let (kernel_name, tile_size) = if m >= 256 && n >= 256 && k >= 256 {
            // For large matrices (>=256x256), use 32x32 tiles
            ("matmul_tiled_32x32_f16", 32)
        } else if m >= 128 && n >= 128 && k >= 128 {
            // For medium matrices (128-256), use 16x16 tiles
            ("matmul_tiled_f16", 16)
        } else {
            // For small matrices (<128), use naive implementation (less overhead)
            ("matmul_f16", 16)
        };

        // Get pipeline
        let pipeline = executor.get_or_compile_pipeline(kernel_name)?;

        // Create command buffer and encoder
        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(b_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

        // Configure thread groups for 2D grid
        let threadgroup_size = metal::MTLSize {
            width: tile_size,
            height: tile_size,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (n as u64 + tile_size - 1) / tile_size,
            height: (m as u64 + tile_size - 1) / tile_size,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Create result tensor
        let result_shape = crate::tensor::TensorShape::new(vec![m, n]);
        Tensor::new(
            BufferHandle::Metal(result_buf),
            result_shape,
            self.device().clone(),
        )
    }

    /// CPU fallback for matmul
    fn matmul_cpu(&self, other: &Tensor, m: usize, k: usize, n: usize) -> TensorResult<Self> {
        let a = self.to_vec();
        let b = other.to_vec();

        let mut c = vec![f16::ZERO; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = f16::ZERO;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        Tensor::from_vec(c, vec![m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_matmul_cpu() {
        // 2x3 @ 3x2 = 2x2
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![3, 2],
        )
        .unwrap();

        let c = a.matmul(&b).unwrap();
        let result = c.to_vec();

        // Expected: [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        //         = [[22, 28], [49, 64]]
        assert_eq!(result[0], f16::from_f32(22.0));
        assert_eq!(result[1], f16::from_f32(28.0));
        assert_eq!(result[2], f16::from_f32(49.0));
        assert_eq!(result[3], f16::from_f32(64.0));
    }

    #[test]
    fn test_matmul_gpu() {
        let device = MetalDevice::new().unwrap();

        let a = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![3, 2],
        )
        .unwrap();

        let c = a.matmul(&b).unwrap();
        let result = c.to_vec();

        assert_eq!(result[0], f16::from_f32(22.0));
        assert_eq!(result[1], f16::from_f32(28.0));
        assert_eq!(result[2], f16::from_f32(49.0));
        assert_eq!(result[3], f16::from_f32(64.0));
    }

    #[test]
    fn test_matmul_shape_error() {
        let a = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            vec![2, 1],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            vec![2, 1],
        )
        .unwrap();

        // Shape mismatch: [2, 1] @ [2, 1] (should be [2, 1] @ [1, N])
        assert!(a.matmul(&b).is_err());
    }
}
