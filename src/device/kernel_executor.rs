//! Metal kernel execution engine

use crate::device::{MetalBuffer, MetalDevice};
use crate::error::{TensorError, TensorResult};
use metal::{ComputePipelineState, MTLSize};
use std::sync::{Arc, Mutex, OnceLock};

/// Metal kernel executor
pub struct KernelExecutor {
    device: MetalDevice,
    pipelines: std::collections::HashMap<String, Arc<ComputePipelineState>>,
}

impl KernelExecutor {
    /// Create a new kernel executor
    pub fn new(device: MetalDevice) -> Self {
        Self {
            device,
            pipelines: std::collections::HashMap::new(),
        }
    }

    /// Load kernel library from Metal source
    pub fn load_kernels(&mut self, source: &str) -> TensorResult<()> {
        self.device.load_library(source)?;
        Ok(())
    }

    /// Get or compile a compute pipeline for a kernel function
    pub fn get_or_compile_pipeline(&mut self, kernel_name: &str) -> TensorResult<Arc<ComputePipelineState>> {
        // Check if already compiled
        if let Some(pipeline) = self.pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }

        // Get library
        let library = self
            .device
            .library()
            .ok_or_else(|| TensorError::MetalError("No shader library loaded".to_string()))?;

        // Get kernel function
        let function = library
            .get_function(kernel_name, None)
            .map_err(|e| TensorError::MetalError(format!("Kernel '{}' not found: {}", kernel_name, e)))?;

        // Create pipeline state
        let pipeline = self
            .device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {}", e)))?;

        let pipeline = Arc::new(pipeline);
        self.pipelines.insert(kernel_name.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    /// Execute a kernel with given buffers
    pub fn execute(
        &mut self,
        kernel_name: &str,
        buffers: &[&MetalBuffer<half::f16>],
        grid_size: usize,
    ) -> TensorResult<()> {
        // Get pipeline
        let pipeline = self.get_or_compile_pipeline(kernel_name)?;

        // Create command buffer
        let command_queue = self.device.command_queue();
        let command_buffer = command_queue.new_command_buffer();

        // Create compute encoder
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline
        encoder.set_compute_pipeline_state(&pipeline);

        // Set buffers
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer.metal_buffer()), 0);
        }

        // Calculate thread group sizes
        let max_threads = pipeline.max_total_threads_per_threadgroup().min(256) as usize;
        let threadgroup_size = MTLSize {
            width: max_threads as u64,
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: ((grid_size + max_threads - 1) / max_threads) as u64,
            height: 1,
            depth: 1,
        };

        // Dispatch
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Execute element-wise binary operation
    pub fn execute_binary_op(
        &mut self,
        kernel_name: &str,
        a: &MetalBuffer<half::f16>,
        b: &MetalBuffer<half::f16>,
        result: &MetalBuffer<half::f16>,
    ) -> TensorResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![b.len(), result.len()],
            });
        }

        self.execute(kernel_name, &[a, b, result], a.len())
    }

    /// Execute element-wise unary operation
    pub fn execute_unary_op(
        &mut self,
        kernel_name: &str,
        a: &MetalBuffer<half::f16>,
        result: &MetalBuffer<half::f16>,
    ) -> TensorResult<()> {
        if a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.len()],
                actual: vec![result.len()],
            });
        }

        self.execute(kernel_name, &[a, result], a.len())
    }
}

/// Global kernel executor (lazy initialized, thread-safe)
static KERNEL_EXECUTOR: OnceLock<Mutex<Option<KernelExecutor>>> = OnceLock::new();

/// Initialize global kernel executor
fn init_kernel_executor() -> Mutex<Option<KernelExecutor>> {
    let result = (|| -> TensorResult<KernelExecutor> {
        let mut device = MetalDevice::new()?;

        // Load built-in kernels
        let shader_source = include_str!("../../shaders/unified.metal");
        device.load_library(shader_source)?;

        Ok(KernelExecutor::new(device))
    })();

    Mutex::new(result.ok())
}

/// Get or initialize global kernel executor
pub fn get_kernel_executor() -> TensorResult<&'static Mutex<Option<KernelExecutor>>> {
    let executor = KERNEL_EXECUTOR.get_or_init(init_kernel_executor);

    // Check if initialization was successful
    let guard = executor.lock().unwrap();
    if guard.is_none() {
        return Err(TensorError::MetalError("Failed to initialize kernel executor".to_string()));
    }
    drop(guard);

    Ok(executor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_kernel_executor_creation() {
        let mut device = MetalDevice::new().unwrap();

        let shader_source = include_str!("../../shaders/unified.metal");
        device.load_library(shader_source).unwrap();

        let _executor = KernelExecutor::new(device);
    }

    #[test]
    fn test_add_kernel() {
        let device = MetalDevice::new().unwrap();

        // Create test buffers
        let a_data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let b_data = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];

        let a = MetalBuffer::<f16>::from_slice(device.metal_device(), &a_data).unwrap();
        let b = MetalBuffer::<f16>::from_slice(device.metal_device(), &b_data).unwrap();
        let result = MetalBuffer::new_uninit_pooled(device.buffer_pool(), 3).unwrap();

        // Use global executor
        let executor = get_kernel_executor().unwrap();
        let mut guard = executor.lock().unwrap();
        let executor = guard.as_mut().unwrap();
        executor.execute_binary_op("add_f16", &a, &b, &result).unwrap();

        // Verify result
        let result_data = result.to_vec();
        assert_eq!(result_data[0], f16::from_f32(5.0));
        assert_eq!(result_data[1], f16::from_f32(7.0));
        assert_eq!(result_data[2], f16::from_f32(9.0));
    }

    #[test]
    fn test_mul_kernel() {
        let device = MetalDevice::new().unwrap();

        let a_data = vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let b_data = vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)];

        let a = MetalBuffer::<f16>::from_slice(device.metal_device(), &a_data).unwrap();
        let b = MetalBuffer::<f16>::from_slice(device.metal_device(), &b_data).unwrap();
        let result = MetalBuffer::new_uninit_pooled(device.buffer_pool(), 3).unwrap();

        // Use global executor
        let executor = get_kernel_executor().unwrap();
        let mut guard = executor.lock().unwrap();
        let executor = guard.as_mut().unwrap();
        executor.execute_binary_op("mul_f16", &a, &b, &result).unwrap();

        let result_data = result.to_vec();
        assert_eq!(result_data[0], f16::from_f32(10.0));
        assert_eq!(result_data[1], f16::from_f32(18.0));
        assert_eq!(result_data[2], f16::from_f32(28.0));
    }
}
