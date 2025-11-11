//! Rotary Position Embedding (RoPE) with Metal GPU acceleration

use crate::device::{Device, EncoderProvider};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation};
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;
use crate::device::MetalBuffer;
use crate::tensor::BufferHandle;
use metal::MTLSize;

impl<T: FloatType> Tensor<T> {
    /// Apply Rotary Position Embedding (RoPE) with precomputed cos/sin arrays (Candle-style)
    /// Input: [seq_len, n_heads, head_dim]
    /// cos: [max_seq_len, head_dim] precomputed cosine values
    /// sin: [max_seq_len, head_dim] precomputed sine values
    /// Returns: Same shape with RoPE applied
    pub fn rope_candle(&self, cos: &Self, sin: &Self) -> TensorResult<Self> {
        use crate::tensor::TensorTransform;

        let dims = self.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidOperation(
                format!("rope_candle requires 3D tensor [seq_len, n_heads, head_dim], got {}D", dims.len())
            ));
        }

        let seq_len = dims[0];
        let n_heads = dims[1];
        let head_dim = dims[2];

        // Validate cos/sin shapes [max_seq_len, head_dim]
        let cos_dims = cos.dims();
        let sin_dims = sin.dims();

        if cos_dims.len() != 2 || sin_dims.len() != 2 {
            return Err(TensorError::InvalidOperation(
                format!("cos/sin must be 2D [max_seq_len, head_dim], got cos: {}D, sin: {}D",
                    cos_dims.len(), sin_dims.len())
            ));
        }

        if cos_dims[1] != head_dim || sin_dims[1] != head_dim {
            return Err(TensorError::InvalidOperation(
                format!("cos/sin last dim must be head_dim = {}, got cos: {}, sin: {}",
                    head_dim, cos_dims[1], sin_dims[1])
            ));
        }

        if cos_dims[0] < seq_len || sin_dims[0] < seq_len {
            return Err(TensorError::InvalidOperation(
                format!("cos/sin first dim must be >= seq_len = {}, got cos: {}, sin: {}",
                    seq_len, cos_dims[0], sin_dims[0])
            ));
        }

        // Ensure contiguous
        let is_contig = self.is_contiguous();
        if is_contig {
            self.rope_candle_metal(seq_len, n_heads, head_dim, cos, sin)
        } else {
            let contiguous = self.contiguous()?;
            contiguous.rope_candle_metal(seq_len, n_heads, head_dim, cos, sin)
        }
    }

    /// Apply Rotary Position Embedding (RoPE) to the tensor
    /// Input: [..., seq_len, n_heads, head_dim]
    /// position_offset: Starting position index for the sequence (for KV cache)
    /// Returns: Same shape with RoPE applied
    pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
        use crate::tensor::TensorTransform;

        if std::env::var("TL_DEBUG_ROPE").is_ok() {
            eprintln!("[ROPE] Called with shape={:?}, strides={:?}, pos_offset={}",
                     self.dims(), self.strides, position_offset);
        }

        let dims = self.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidOperation(
                format!("RoPE requires at least 3D tensor (got {}D)", dims.len())
            ));
        }

        let head_dim = dims[dims.len() - 1];
        let n_heads = dims[dims.len() - 2];
        let seq_len = dims[dims.len() - 3];

        if head_dim % 2 != 0 {
            return Err(TensorError::InvalidOperation(
                format!("RoPE requires even head_dim (got {})", head_dim)
            ));
        }

        // CRITICAL: Ensure tensor is contiguous before GPU operations
        // reshape() only changes metadata (strides), but rope kernel expects
        // contiguous memory layout (linear indexing)
        let is_contig = self.is_contiguous();
        if std::env::var("TL_DEBUG_ROPE").is_ok() {
            eprintln!("[ROPE] is_contiguous={}, will {}",
                     is_contig, if is_contig { "use directly (no clone)" } else { "make contiguous" });
        }

        // OPTIMIZATION: No clone needed when contiguous - GPU kernel only reads input
        if is_contig {
            if std::env::var("TL_DEBUG_ROPE").is_ok() {
                eprintln!("[ROPE] Calling rope_metal on contiguous tensor...");
            }
            self.rope_metal(seq_len, n_heads, head_dim, position_offset)
        } else {
            if std::env::var("TL_DEBUG_ROPE").is_ok() {
                eprintln!("[ROPE] Making contiguous then calling rope_metal...");
            }
            let contiguous = self.contiguous()?;
            contiguous.rope_metal(seq_len, n_heads, head_dim, position_offset)
        }
    }

    /// Metal GPU implementation of Candle-style RoPE with precomputed cos/sin
    fn rope_candle_metal(&self, seq_len: usize, n_heads: usize, head_dim: usize, cos: &Self, sin: &Self) -> TensorResult<Self> {
        let input_buf = self.buffer().as_metal()?;
        let cos_buf = cos.buffer().as_metal()?;
        let sin_buf = sin.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "RoPE requires Metal device".to_string(),
                ))
            }
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, self.numel())?;

        // Create params buffer: [seq_len, n_heads, head_dim, half_dim]
        let half_dim = head_dim / 2;
        let params: [u32; 4] = [seq_len as u32, n_heads as u32, head_dim as u32, half_dim as u32];
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                params.as_ptr() as *const u8,
                std::mem::size_of::<[u32; 4]>()
            )
        };

        let params_buf = device
            .metal_device()
            .new_buffer_with_data(
                params_bytes.as_ptr() as *const std::ffi::c_void,
                params_bytes.len() as u64,
                metal::MTLResourceOptions::CPUCacheModeDefaultCache,
            );

        // Get kernel function
        let library = device.library()
            .ok_or_else(|| TensorError::MetalError("No shader library loaded".to_string()))?;

        let suffix = T::kernel_suffix();
        let kernel_name = format!("rope_candle{}", suffix);
        let function = library
            .get_function(&kernel_name, None)
            .map_err(|e| TensorError::MetalError(format!("Kernel '{}' not found: {}", kernel_name, e)))?;

        // Create pipeline
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {}", e)))?;

        // Execute kernel
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(cos_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(sin_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        // Calculate thread group sizes
        let total_elements = self.numel();
        let max_threads = pipeline.max_total_threads_per_threadgroup().min(256) as usize;
        let threadgroup_size = MTLSize {
            width: max_threads as u64,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: ((total_elements + max_threads - 1) / max_threads) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Return result tensor
        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    /// Metal GPU implementation of RoPE
    fn rope_metal(&self, seq_len: usize, n_heads: usize, head_dim: usize, position_offset: usize) -> TensorResult<Self> {
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
                    "RoPE requires Metal device".to_string(),
                ))
            }
        };

        // Load rope shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, self.numel())?;

        // Create params buffer: [seq_len, n_heads, head_dim, rope_base, position_offset]
        const ROPE_BASE: u32 = 10000;
        let params: [u32; 5] = [seq_len as u32, n_heads as u32, head_dim as u32, ROPE_BASE, position_offset as u32];
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                params.as_ptr() as *const u8,
                std::mem::size_of::<[u32; 5]>()
            )
        };

        let params_buf = device
            .metal_device()
            .new_buffer_with_data(
                params_bytes.as_ptr() as *const std::ffi::c_void,
                params_bytes.len() as u64,
                metal::MTLResourceOptions::CPUCacheModeDefaultCache,
            );

        // Get kernel function - select based on type
        let library = device.library()
            .ok_or_else(|| TensorError::MetalError("No shader library loaded".to_string()))?;

        let suffix = T::kernel_suffix();
        let kernel_name = format!("rope{}", suffix);
        let function = library
            .get_function(&kernel_name, None)
            .map_err(|e| TensorError::MetalError(format!("Kernel '{}' not found: {}", kernel_name, e)))?;

        // Create pipeline
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {}", e)))?;

        // Execute kernel
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        // Calculate thread group sizes
        let total_elements = self.numel();
        let max_threads = pipeline.max_total_threads_per_threadgroup().min(256) as usize;
        let threadgroup_size = MTLSize {
            width: max_threads as u64,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: ((total_elements + max_threads - 1) / max_threads) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // command_buffer.commit(); // Handled by Commands manager
        // submit_async - not needed with Commands batching

        // Return result tensor
        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;
    use crate::device::MetalDevice;
    use crate::tensor::TensorIO;
    use half::f16;

    #[test]
    fn test_rope_shape_preservation() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Test with various shapes: [seq_len, n_heads, head_dim]
        let test_cases = vec![
            (1, 4, 64),   // Single token, 4 heads, 64 dim
            (2, 8, 32),   // 2 tokens, 8 heads, 32 dim
            (4, 16, 128), // 4 tokens, 16 heads, 128 dim
        ];

        for (seq_len, n_heads, head_dim) in test_cases {
            let shape = vec![seq_len, n_heads, head_dim];
            let data: Vec<f16> = (0..(seq_len * n_heads * head_dim))
                .map(|i| f16::from_f32(i as f32 * 0.01))
                .collect();

            let input = Tensor::from_vec_gpu(&device, data, shape.clone()).unwrap();
            let result = input.rope(0).unwrap();

            assert_eq!(
                result.dims(),
                shape,
                "RoPE should preserve shape: expected {:?}, got {:?}",
                shape,
                result.dims()
            );
        }
    }

    #[test]
    fn test_rope_odd_head_dim_error() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // RoPE requires even head_dim (must operate on pairs)
        let odd_head_dims = vec![63, 127, 31];

        for head_dim in odd_head_dims {
            let shape = vec![1, 4, head_dim];
            let data: Vec<f16> = (0..(1 * 4 * head_dim))
                .map(|i| f16::from_f32(i as f32 * 0.01))
                .collect();

            let input = Tensor::from_vec_gpu(&device, data, shape).unwrap();
            let result = input.rope(0);

            assert!(
                result.is_err(),
                "RoPE with odd head_dim={} should fail",
                head_dim
            );
        }
    }

    #[test]
    fn test_rope_rotation_per_pair_magnitude() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // RoPE applies 2D rotations to dimension pairs
        // Each pair's magnitude should be preserved: sqrt(x0^2 + x1^2) remains constant
        let seq_len = 1;
        let n_heads = 1;
        let head_dim = 64;

        // Create simple test data with known magnitudes for each pair
        let mut data: Vec<f16> = Vec::new();
        for pair_idx in 0..(head_dim / 2) {
            // Each pair: [3.0, 4.0] has magnitude 5.0
            data.push(f16::from_f32(3.0));
            data.push(f16::from_f32(4.0));
        }

        let input = Tensor::from_vec_gpu(&device, data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();
        let result = input.rope(0).unwrap();

        let result_data = result.sync_and_read();

        // Check that each dimension pair preserves its magnitude
        // Pair magnitude: sqrt(x0^2 + x1^2)
        let expected_pair_magnitude = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt(); // = 5.0

        for pair_idx in 0..(head_dim / 2) {
            let idx0 = pair_idx * 2;
            let idx1 = pair_idx * 2 + 1;

            let x0 = f16::to_f32(result_data[idx0]);
            let x1 = f16::to_f32(result_data[idx1]);
            let actual_magnitude = (x0 * x0 + x1 * x1).sqrt();

            let magnitude_error = (actual_magnitude - expected_pair_magnitude).abs();
            let relative_error = magnitude_error / expected_pair_magnitude;

            assert!(
                relative_error < 0.02,
                "Pair {} magnitude should be preserved. Expected: {:.6}, Got: {:.6} (x0={:.6}, x1={:.6}), Relative error: {:.6}",
                pair_idx, expected_pair_magnitude, actual_magnitude, x0, x1, relative_error
            );
        }
    }

    #[test]
    fn test_rope_position_offset_effect() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Different position offsets should produce different results
        let seq_len = 1;
        let n_heads = 4;
        let head_dim = 64;

        let data: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 13) as f32 * 0.3))
            .collect();

        let input = Tensor::from_vec_gpu(&device, data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();

        let result_pos0 = input.rope(0).unwrap();
        let result_pos5 = input.rope(5).unwrap();
        let result_pos10 = input.rope(10).unwrap();

        let data_pos0 = result_pos0.sync_and_read();
        let data_pos5 = result_pos5.sync_and_read();
        let data_pos10 = result_pos10.sync_and_read();

        // Results should be different for different positions
        let mut diff_0_5 = 0;
        let mut diff_5_10 = 0;
        for i in 0..data_pos0.len() {
            if (f16::to_f32(data_pos0[i]) - f16::to_f32(data_pos5[i])).abs() > 0.001 {
                diff_0_5 += 1;
            }
            if (f16::to_f32(data_pos5[i]) - f16::to_f32(data_pos10[i])).abs() > 0.001 {
                diff_5_10 += 1;
            }
        }

        assert!(
            diff_0_5 > 10,
            "RoPE with different position offsets (0 vs 5) should produce significantly different results"
        );
        assert!(
            diff_5_10 > 10,
            "RoPE with different position offsets (5 vs 10) should produce significantly different results"
        );
    }

    #[test]
    fn test_rope_deterministic() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Same input and position should always produce same output
        let seq_len = 2;
        let n_heads = 4;
        let head_dim = 64;

        let data: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 11) as f32 * 0.2 + 0.5))
            .collect();

        let input = Tensor::from_vec_gpu(&device, data, vec![seq_len, n_heads, head_dim]).unwrap();

        let result1 = input.rope(5).unwrap();
        let result2 = input.rope(5).unwrap();
        let result3 = input.rope(5).unwrap();

        let data1 = result1.sync_and_read();
        let data2 = result2.sync_and_read();
        let data3 = result3.sync_and_read();

        for i in 0..data1.len() {
            assert_eq!(
                data1[i], data2[i],
                "RoPE should be deterministic (result1 vs result2 at index {})",
                i
            );
            assert_eq!(
                data2[i], data3[i],
                "RoPE should be deterministic (result2 vs result3 at index {})",
                i
            );
        }
    }

    #[test]
    fn test_rope_rotation_formula() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Test the 2D rotation formula for first dimension pair
        // For head_dim=64, dim_pair=0, position=0, rope_base=10000:
        // freq = 1.0 / (10000^(0/64)) = 1.0
        // theta = 0 * 1.0 = 0
        // cos(0) = 1, sin(0) = 0
        // So: [x0, x1] -> [x0*1 - x1*0, x0*0 + x1*1] = [x0, x1]

        let seq_len = 1;
        let n_heads = 1;
        let head_dim = 64;

        let data: Vec<f16> = vec![f16::from_f32(3.0), f16::from_f32(4.0)]
            .into_iter()
            .chain((2..head_dim).map(|i| f16::from_f32(i as f32)))
            .collect();

        let input = Tensor::from_vec_gpu(&device, data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();
        let result = input.rope(0).unwrap(); // position_offset = 0

        let result_data = result.sync_and_read();

        // At position=0, theta=0 for first pair, so rotation is identity
        // First pair should remain approximately unchanged
        let x0_in = f16::to_f32(data[0]);
        let x1_in = f16::to_f32(data[1]);
        let x0_out = f16::to_f32(result_data[0]);
        let x1_out = f16::to_f32(result_data[1]);

        assert!(
            (x0_in - x0_out).abs() < 0.01,
            "At position=0, first element should remain ~unchanged: {} -> {}",
            x0_in, x0_out
        );
        assert!(
            (x1_in - x1_out).abs() < 0.01,
            "At position=0, second element should remain ~unchanged: {} -> {}",
            x1_in, x1_out
        );
    }

    #[test]
    fn test_rope_minimum_3d_tensor() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // RoPE requires at least 3D tensor
        let invalid_shapes = vec![
            vec![64],           // 1D
            vec![4, 64],        // 2D
        ];

        for shape in invalid_shapes {
            let numel: usize = shape.iter().product();
            let data: Vec<f16> = (0..numel).map(|i| f16::from_f32(i as f32)).collect();

            let input = Tensor::from_vec_gpu(&device, data, shape.clone()).unwrap();
            let result = input.rope(0);

            assert!(
                result.is_err(),
                "RoPE should fail on {:?}D tensor (shape: {:?})",
                shape.len(),
                shape
            );
        }
    }

    #[test]
    fn test_rope_batch_dimension() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // RoPE should work with batch dimensions: [batch, seq_len, n_heads, head_dim]
        let batch_size = 2;
        let seq_len = 3;
        let n_heads = 4;
        let head_dim = 64;

        let shape = vec![batch_size, seq_len, n_heads, head_dim];
        let numel: usize = shape.iter().product();
        let data: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 17) as f32 * 0.1)).collect();

        let input = Tensor::from_vec_gpu(&device, data, shape.clone()).unwrap();
        let result = input.rope(0).unwrap();

        assert_eq!(
            result.dims(),
            shape,
            "RoPE should preserve shape with batch dimension"
        );
    }

    #[test]
    fn test_rope_f32_rotation_magnitude() {
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Test RoPE with f32 precision - verify magnitude preservation
        let seq_len = 1;
        let n_heads = 1;
        let head_dim = 64;

        // Create test data with known magnitudes for each pair
        let mut data: Vec<f32> = Vec::new();
        for _pair_idx in 0..(head_dim / 2) {
            // Each pair: [3.0, 4.0] has magnitude 5.0
            data.push(3.0);
            data.push(4.0);
        }

        let input = Tensor::from_vec_gpu(&device, data.clone(), vec![seq_len, n_heads, head_dim]).unwrap();
        let result = input.rope(0).unwrap();

        let result_data = result.sync_and_read();

        // Check that each dimension pair preserves its magnitude
        let expected_pair_magnitude = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt(); // = 5.0

        for pair_idx in 0..(head_dim / 2) {
            let idx0 = pair_idx * 2;
            let idx1 = pair_idx * 2 + 1;

            let x0 = result_data[idx0];
            let x1 = result_data[idx1];
            let actual_magnitude = (x0 * x0 + x1 * x1).sqrt();

            let magnitude_error = (actual_magnitude - expected_pair_magnitude).abs();
            let relative_error = magnitude_error / expected_pair_magnitude;

            assert!(
                relative_error < 0.001, // f32 should be more precise than f16
                "Pair {} magnitude should be preserved. Expected: {:.6}, Got: {:.6} (x0={:.6}, x1={:.6}), Relative error: {:.6}",
                pair_idx, expected_pair_magnitude, actual_magnitude, x0, x1, relative_error
            );
        }
    }
}
