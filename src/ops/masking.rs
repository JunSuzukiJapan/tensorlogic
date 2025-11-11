//! Masking operations for attention mechanisms

use crate::tensor::Tensor;
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO};
use crate::TensorResult;
use crate::error::TensorError;
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Apply attention mask to attention scores
    ///
    /// Replaces masked positions with a large negative value (-10000.0)
    /// so they become ~0 after softmax
    ///
    /// # Arguments
    /// * `mask` - Boolean-like tensor where 0.0 means mask (ignore), 1.0 means keep
    ///
    /// # Shape
    /// - self: [batch, seq_len, seq_len] or [seq_len, seq_len]
    /// - mask: same shape as self
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Causal mask for autoregressive models
    /// let scores = Tensor::from_vec(vec![f16::from_f32(1.0), f16::from_f32(2.0),
    ///                                     f16::from_f32(3.0), f16::from_f32(4.0)], vec![2, 2])?;
    /// let mask = Tensor::from_vec(vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE], vec![2, 2])?;
    /// let masked = scores.apply_attention_mask(&mask)?;
    /// // masked[0,1] will be -10000.0 (masked out)
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_attention_mask(&self, mask: &Tensor<T>) -> TensorResult<Self> {
        // Verify shapes match
        if self.dims() != mask.dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: mask.dims().to_vec(),
            });
        }

        // Use GPU implementation if available, otherwise fallback to CPU
        if self.buffer().is_metal() {
            self.apply_attention_mask_metal(mask)
        } else {
            self.apply_attention_mask_cpu(mask)
        }
    }

    /// Metal GPU implementation of apply_attention_mask
    fn apply_attention_mask_metal(&self, mask: &Tensor<T>) -> TensorResult<Self> {
        use crate::device::{Device, MetalBuffer};

        let size = self.numel();

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Get input buffers
        let scores_buf = self.buffer().as_metal()?;
        let mask_buf = mask.buffer().as_metal()?;

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, size)?;

        // Choose kernel based on type
        let suffix = T::kernel_suffix();
        let kernel_name = format!("apply_attention_mask{}", suffix);

        // Get kernel function
        let library = device.library()
            .ok_or_else(|| TensorError::MetalError("No shader library loaded".to_string()))?;
        let function = library
            .get_function(&kernel_name, None)
            .map_err(|e| TensorError::MetalError(format!("Kernel '{}' not found: {}", kernel_name, e)))?;

        // Create pipeline
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {}", e)))?;

        // Create size parameter buffer
        let size_u32 = size as u32;
        let size_bytes = unsafe {
            std::slice::from_raw_parts(
                &size_u32 as *const u32 as *const u8,
                std::mem::size_of::<u32>()
            )
        };
        let size_buf = device
            .metal_device()
            .new_buffer_with_data(
                size_bytes.as_ptr() as *const std::ffi::c_void,
                size_bytes.len() as u64,
                metal::MTLResourceOptions::CPUCacheModeDefaultCache,
            );

        // Execute kernel
        let command_queue = device.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(scores_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(mask_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(&size_buf), 0);

        // Calculate thread group sizes
        let max_threads = pipeline.max_total_threads_per_threadgroup().min(256);
        let threadgroup_size = metal::MTLSize {
            width: max_threads,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: ((size as u64 + max_threads - 1) / max_threads),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // NOTE: No commit here - the Commands batching system handles this automatically
        // Commands will commit when batch size is reached or when data is read to CPU
        // This matches Candle's lazy batching strategy

        // Create result tensor
        self.new_from_pool(
            crate::tensor::BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    /// CPU fallback for apply_attention_mask
    fn apply_attention_mask_cpu(&self, mask: &Tensor<T>) -> TensorResult<Self> {
        // For CPU, we need to implement the logic
        let self_data = self.sync_and_read();
        let mask_data = mask.sync_and_read();

        let result_data: Vec<T> = self_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&val, &mask_val)| {
                // mask_val: 1=keep, 0=mask
                if T::to_f32(mask_val) == 0.0 {
                    T::from_f32(-10000.0) // Large negative value for masked positions
                } else {
                    val
                }
            })
            .collect();

        Tensor::from_vec(result_data, self.dims().to_vec())
    }

    /// Create a causal mask for autoregressive attention
    ///
    /// Returns a lower triangular matrix of 1s and 0s
    /// Used in decoder self-attention to prevent attending to future positions
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Tensor of shape [seq_len, seq_len] where:
    /// - Upper triangle (including diagonal) = 1.0 (allow attention)
    /// - Lower triangle = 0.0 (mask out)
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mask = Tensor::<f16>::causal_mask(3)?;
    /// // [[1, 0, 0],
    /// //  [1, 1, 0],
    /// //  [1, 1, 1]]
    /// # Ok(())
    /// # }
    /// ```
    pub fn causal_mask(seq_len: usize) -> TensorResult<Tensor> {
        let mut data = Vec::with_capacity(seq_len * seq_len);

        for i in 0..seq_len {
            for j in 0..seq_len {
                // Allow attention to positions <= current position
                if j <= i {
                    data.push(f16::ONE);
                } else {
                    data.push(f16::ZERO);
                }
            }
        }

        Tensor::from_vec(data, vec![seq_len, seq_len])
    }

    /// Create a padding mask
    ///
    /// Masks out padding tokens in a batch of sequences
    ///
    /// # Arguments
    /// * `lengths` - Actual length of each sequence in the batch
    /// * `max_len` - Maximum sequence length (padded length)
    ///
    /// # Returns
    /// Tensor of shape [batch_size, max_len] where:
    /// - 1.0 for real tokens
    /// - 0.0 for padding tokens
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mask = Tensor::<f16>::padding_mask(&[2, 3], 4)?;
    /// // [[1, 1, 0, 0],  // first sequence has length 2
    /// //  [1, 1, 1, 0]]  // second sequence has length 3
    /// # Ok(())
    /// # }
    /// ```
    pub fn padding_mask(lengths: &[usize], max_len: usize) -> TensorResult<Tensor> {
        let batch_size = lengths.len();
        let mut data = Vec::with_capacity(batch_size * max_len);

        for &len in lengths {
            for pos in 0..max_len {
                if pos < len {
                    data.push(f16::ONE);
                } else {
                    data.push(f16::ZERO);
                }
            }
        }

        Tensor::from_vec(data, vec![batch_size, max_len])
    }

    /// Combine multiple masks (logical AND)
    ///
    /// Useful for combining causal mask + padding mask
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let causal = Tensor::<f16>::causal_mask(4)?;
    /// // Create matching 4x4 padding mask (not 2x4)
    /// let padding_data = vec![f16::ONE; 16];  // All ones for simplicity
    /// let padding = Tensor::from_vec(padding_data, vec![4, 4])?;
    /// let combined = causal.combine_masks(&padding)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn combine_masks(&self, other: &Tensor<T>) -> TensorResult<Self> {
        if self.dims() != other.dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.dims().to_vec(),
                actual: other.dims().to_vec(),
            });
        }

        let self_data = self.sync_and_read();
        let other_data = other.sync_and_read();
        let self_f16: Vec<f16> = unsafe { std::mem::transmute(self_data) };
        let other_f16: Vec<f16> = unsafe { std::mem::transmute(other_data) };

        let result_data: Vec<f16> = self_f16
            .iter()
            .zip(other_f16.iter())
            .map(|(&a, &b)| {
                // Logical AND: both must be non-zero
                if a != f16::ZERO && b != f16::ZERO {
                    f16::ONE
                } else {
                    f16::ZERO
                }
            })
            .collect();

        let result_t: Vec<T> = unsafe { std::mem::transmute(result_data) };
        Tensor::from_vec(result_t, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_attention_mask_f16() {
        let scores = Tensor::from_vec(
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0),
                f16::from_f32(3.0), f16::from_f32(4.0),
            ],
            vec![2, 2],
        ).unwrap();

        let mask = Tensor::from_vec(
            vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let result = scores.apply_attention_mask(&mask).unwrap();
        let data = result.sync_and_read();

        assert_eq!(data[0], f16::from_f32(1.0));
        assert_eq!(data[1], f16::from_f32(-10000.0)); // masked
        assert_eq!(data[2], f16::from_f32(3.0));
        assert_eq!(data[3], f16::from_f32(4.0));
    }

    #[test]
    fn test_apply_attention_mask_f32_gpu() {
        use crate::device::MetalDevice;

        let device = MetalDevice::new().expect("Failed to create Metal device");

        let scores = Tensor::from_vec_gpu(
            &device,
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();

        let mask = Tensor::from_vec_gpu(
            &device,
            vec![1.0f32, 0.0, 1.0, 1.0],
            vec![2, 2],
        ).unwrap();

        // Debug: verify input
        let scores_data = scores.sync_and_read();
        let mask_data = mask.sync_and_read();
        println!("Scores: {:?}", scores_data);
        println!("Mask: {:?}", mask_data);

        let result = scores.apply_attention_mask(&mask).unwrap();
        let data = result.sync_and_read();

        println!("Result: {:?}", data);

        assert_eq!(data[0], 1.0, "Expected scores[0]=1.0 to be kept");
        assert_eq!(data[1], -10000.0, "Expected scores[1]=2.0 to be masked"); // masked
        assert_eq!(data[2], 3.0, "Expected scores[2]=3.0 to be kept");
        assert_eq!(data[3], 4.0, "Expected scores[3]=4.0 to be kept");
    }

    #[test]
    fn test_causal_mask() {
        let mask = Tensor::<f16>::causal_mask(3).unwrap();
        let data = mask.sync_and_read();

        // Expected: [[1, 0, 0],
        //            [1, 1, 0],
        //            [1, 1, 1]]
        assert_eq!(data[0], f16::ONE);  // [0,0]
        assert_eq!(data[1], f16::ZERO); // [0,1]
        assert_eq!(data[2], f16::ZERO); // [0,2]

        assert_eq!(data[3], f16::ONE);  // [1,0]
        assert_eq!(data[4], f16::ONE);  // [1,1]
        assert_eq!(data[5], f16::ZERO); // [1,2]

        assert_eq!(data[6], f16::ONE);  // [2,0]
        assert_eq!(data[7], f16::ONE);  // [2,1]
        assert_eq!(data[8], f16::ONE);  // [2,2]
    }

    #[test]
    fn test_padding_mask() {
        let mask = Tensor::<f16>::padding_mask(&[2, 3], 4).unwrap();
        let data = mask.sync_and_read();

        // Expected: [[1, 1, 0, 0],
        //            [1, 1, 1, 0]]

        // First sequence (length 2)
        assert_eq!(data[0], f16::ONE);
        assert_eq!(data[1], f16::ONE);
        assert_eq!(data[2], f16::ZERO);
        assert_eq!(data[3], f16::ZERO);

        // Second sequence (length 3)
        assert_eq!(data[4], f16::ONE);
        assert_eq!(data[5], f16::ONE);
        assert_eq!(data[6], f16::ONE);
        assert_eq!(data[7], f16::ZERO);
    }

    #[test]
    fn test_combine_masks() {
        let mask1 = Tensor::from_vec(
            vec![f16::ONE, f16::ZERO, f16::ONE, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let mask2 = Tensor::from_vec(
            vec![f16::ONE, f16::ONE, f16::ZERO, f16::ONE],
            vec![2, 2],
        ).unwrap();

        let combined = mask1.combine_masks(&mask2).unwrap();
        let data = combined.sync_and_read();

        // Logical AND
        assert_eq!(data[0], f16::ONE);  // 1 & 1 = 1
        assert_eq!(data[1], f16::ZERO); // 0 & 1 = 0
        assert_eq!(data[2], f16::ZERO); // 1 & 0 = 0
        assert_eq!(data[3], f16::ONE);  // 1 & 1 = 1
    }
}
