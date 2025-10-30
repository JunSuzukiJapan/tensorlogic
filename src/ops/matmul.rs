//! Matrix multiplication operations with Metal GPU acceleration

use crate::autograd::gradients::MatMulBackward;
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorAutograd};
use crate::autograd::{AutogradContext, Operation};
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Matrix multiplication: self @ other
    ///
    /// # Arguments
    /// - self: shape [M, K]
    /// - other: shape [K, N]
    ///
    /// # Returns
    /// - result: shape [M, N]
    pub fn matmul(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
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

            let result_node_id = AutogradContext::add_node_generic(
                Operation::MatMul,
                vec![self_node_id, other_node_id],
                Some(grad_fn),
            );

            AutogradContext::register_tensor_generic(self_node_id, self.clone());
            AutogradContext::register_tensor_generic(other_node_id, other.clone());

            result.set_grad_node(result_node_id);
            result.set_requires_grad(true);
        }

        Ok(result)
    }

    /// Metal GPU implementation of matmul with adaptive tiling
    ///
    /// Uses threadgroup memory tiling for improved performance (1.5-2x speedup).
    /// Automatically selects optimal tile size based on matrix dimensions.
    fn matmul_metal(&self, other: &Tensor<T>, m: usize, k: usize, n: usize) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

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
            let elementwise_source = include_str!("../../shaders/unified.metal");
            let tiled_source = include_str!("../../shaders/unified.metal");

            // Combine shader sources
            let combined_source = format!("{}\n\n{}", elementwise_source, tiled_source);
            device.load_library(&combined_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), m * n)?;

        // Create buffers for dimensions
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        // Execute matmul kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // Select optimal kernel based on matrix size and type
        // Optimized for transformer inference patterns
        let suffix = T::kernel_suffix();

        // DEBUG: Panic if f16 kernel selected (remove after f32 verification)
        if suffix == "_f16" {
            panic!("src/ops/matmul.rs:135: f16 kernel selected");
        }

        // Dynamic tile size selection optimized for transformer workloads
        // Key insights:
        // 1. Batch size 1 inference (m=1): use tiled kernels for large K/N
        // 2. Large K dimension (common in transformers): prefer larger tiles
        // 3. Square-ish matrices: use largest tiles for best cache reuse
        let (kernel_name, tile_size) = if m * n * k > 256 * 256 * 256 {
            // Very large matrices: 32x32 tiles for maximum cache reuse
            (format!("matmul_tiled_32x32{}", suffix), 32)
        } else if k >= 512 && (m >= 32 || n >= 32) {
            // Transformer pattern: long K dimension with small batch
            // Example: [1, 2048] @ [2048, 2048] or [10, 2048] @ [2048, 2048]
            (format!("matmul_tiled_32x32{}", suffix), 32)
        } else if m >= 64 && n >= 64 && k >= 64 {
            // Medium matrices: 16x16 tiles
            (format!("matmul_tiled{}", suffix), 16)
        } else if k >= 256 {
            // Small batch, large K: still use tiled version
            (format!("matmul_tiled{}", suffix), 16)
        } else {
            // Small matrices: naive implementation has less overhead
            (format!("matmul{}", suffix), 16)
        };

        // Get pipeline
        let pipeline = executor.get_or_compile_pipeline(&kernel_name)?;

        // Create command buffer and encoder (Commands API - candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.as_ref().new_compute_command_encoder();

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

        // Note: wait_until_completed() is NOT called here (matches candle pattern).
        // Commands manager handles batching and will commit when batch size is exceeded.

        // Create result tensor
        let result_shape = crate::tensor::TensorShape::new(vec![m, n]);
        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            result_shape,
        )
    }

    /// CPU fallback for matmul
    fn matmul_cpu(&self, other: &Tensor<T>, m: usize, k: usize, n: usize) -> TensorResult<Self> {
        panic!("src/ops/matmul.rs:194:5");
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let a = self.to_vec();
        let b = other.to_vec();
        let a_f16: Vec<f16> = unsafe { std::mem::transmute(a) };
        let b_f16: Vec<f16> = unsafe { std::mem::transmute(b) };

        let mut c = vec![f16::ZERO; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = f16::ZERO;
                for p in 0..k {
                    sum += a_f16[i * k + p] * b_f16[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        let c_t: Vec<T> = unsafe { std::mem::transmute(c) };
        Tensor::from_vec(c_t, vec![m, n])
    }

    /// Fused transpose-matmul: self @ other.T
    ///
    /// Optimized for linear layers where weight is [out_features, in_features]
    /// Eliminates separate transpose operation for 20-30% speedup
    ///
    /// # Arguments
    /// - self: shape [M, K]
    /// - other: shape [N, K] (will be transposed to [K, N])
    ///
    /// # Returns
    /// - result: shape [M, N]
    pub fn matmul_transposed_b(&self, other: &Tensor<T>) -> TensorResult<Self>
    where
        Tensor<T>: TensorAutograd<T>,
    {
        // Validate shapes
        if self.shape().rank() != 2 || other.shape().rank() != 2 {
            return Err(TensorError::InvalidOperation(
                "matmul_transposed_b requires 2D tensors".to_string(),
            ));
        }

        let m = self.shape().dims()[0];
        let k = self.shape().dims()[1];
        let n = other.shape().dims()[0];  // number of rows in other
        let k2 = other.shape().dims()[1]; // should match k

        if k != k2 {
            return Err(TensorError::InvalidOperation(format!(
                "matmul_transposed_b shape mismatch: [{}, {}] @ [{}, {}].T",
                m, k, n, k2
            )));
        }

        // Check device compatibility
        if self.device() != other.device() {
            return Err(TensorError::DeviceConversionError(
                "matmul_transposed_b requires tensors on same device".to_string(),
            ));
        }

        let result = match self.device() {
            Device::Metal(_) => self.matmul_transposed_b_metal(other, m, k, n)?,
            Device::CPU => {
                // Fallback: transpose + matmul
                let other_t = other.transpose()?;
                self.matmul(&other_t)?
            }
            Device::NeuralEngine => {
                return Err(TensorError::InvalidOperation(
                    "matmul_transposed_b not yet supported on Neural Engine".to_string(),
                ))
            }
        };

        // Note: Autograd support could be added here if needed
        Ok(result)
    }

    /// Metal GPU implementation of fused transpose-matmul
    fn matmul_transposed_b_metal(&self, other: &Tensor<T>, m: usize, k: usize, n: usize) -> TensorResult<Self> {
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
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), m * n)?;

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // Select optimal kernel based on matrix size (same logic as regular matmul)
        let suffix = T::kernel_suffix();
        let (kernel_name, tile_size) = if m * n * k > 256 * 256 * 256 {
            (format!("matmul_transposed_b_tiled_32x32{}", suffix), 32)
        } else if k >= 512 && (m >= 32 || n >= 32) {
            // Transformer pattern: optimized for batch inference
            (format!("matmul_transposed_b_tiled_32x32{}", suffix), 32)
        } else if m >= 64 && n >= 64 && k >= 64 {
            (format!("matmul_transposed_b_tiled{}", suffix), 16)
        } else if k >= 256 {
            (format!("matmul_transposed_b_tiled{}", suffix), 16)
        } else {
            // For very small matrices, still use tiled version
            // (no naive version available for fused kernel)
            (format!("matmul_transposed_b_tiled{}", suffix), 16)
        };

        let pipeline = executor.get_or_compile_pipeline(&kernel_name)?;

        // Commands API (candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.as_ref().new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(b_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

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

        // Note: wait_until_completed() is NOT called here (matches candle pattern).

        let result_shape = crate::tensor::TensorShape::new(vec![m, n]);
        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            result_shape,
        )
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

        let a = Tensor::from_vec_gpu(
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

        let b = Tensor::from_vec_gpu(
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
