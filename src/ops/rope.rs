//! Rotary Position Embedding (RoPE) with Metal GPU acceleration

use crate::device::Device;
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;
use crate::device::MetalBuffer;
use crate::tensor::BufferHandle;
use metal::MTLSize;

impl Tensor {
    /// Apply Rotary Position Embedding (RoPE) to the tensor
    /// Input: [..., seq_len, n_heads, head_dim]
    /// position_offset: Starting position index for the sequence (for KV cache)
    /// Returns: Same shape with RoPE applied
    pub fn rope(&self, position_offset: usize) -> TensorResult<Self> {
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

        self.rope_metal(seq_len, n_heads, head_dim, position_offset)
    }

    /// Metal GPU implementation of RoPE
    fn rope_metal(&self, seq_len: usize, n_heads: usize, head_dim: usize, position_offset: usize) -> TensorResult<Self> {
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
            let shader_source = include_str!("../../shaders/rope.metal");
            device.load_library(shader_source)?;
        }

        // Create output buffer
        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), self.numel())?;

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

        // Get kernel function
        let library = device.library()
            .ok_or_else(|| TensorError::MetalError("No shader library loaded".to_string()))?;

        let function = library
            .get_function("rope_f16", None)
            .map_err(|e| TensorError::MetalError(format!("Kernel 'rope_f16' not found: {}", e)))?;

        // Create pipeline
        let pipeline = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {}", e)))?;

        // Execute kernel
        let command_queue = device.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

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

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Return result tensor
        self.new_from_pool(
            BufferHandle::Metal(result_buf),
            self.shape().clone(),
        )
    }
}
