//! Common helper functions for operation implementations
//! Reduces code duplication across elementwise and activation operations

use crate::device::{Device, KernelExecutor, MetalBuffer, EncoderProvider};
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

/// Execute a unary operation using Metal GPU kernel
///
/// # Arguments
/// * `tensor` - Input tensor (must be on Metal device)
/// * `kernel_name` - Name of the Metal kernel function (e.g., "exp_f16")
///
/// # Returns
/// New tensor with the operation applied
pub(crate) fn execute_unary_metal_op<T: FloatType>(
    tensor: &Tensor<T>,
    kernel_name: &str,
) -> TensorResult<Tensor<T>> {
    // Currently only f16 is supported for Metal operations
    if false {
        return Err(TensorError::InvalidOperation(
            "Metal operations currently only support f16".to_string()
        ));
    }
    let input_buf = tensor.buffer().as_metal()?;
    let mut device = match tensor.device() {
        Device::Metal(dev) => dev.clone(),
        _ => {
            return Err(TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            ))
        }
    };

    // Load shader library if not already loaded
    if device.library().is_none() {
        let shader_source = include_str!("../../shaders/unified.metal");
        device.load_library(shader_source)?;
    }

    // Create output buffer from pool
    let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, tensor.numel())?;

    // Execute kernel using new EncoderProvider pattern
    // Get or compile pipeline
    let mut executor = KernelExecutor::new(device.clone());
    let pipeline = executor.get_or_compile_pipeline(kernel_name)?;

    // Create command buffer and encoder (EncoderProvider pattern)
    let (_flushed, command_buffer) = device.command_buffer()?;
    let encoder = command_buffer.encoder();

    // Set pipeline and buffers (Metal API is type-erased, kernel_name specifies f16/f32)
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(result_buf.metal_buffer()), 0);

    // Configure thread groups
    let grid_size = metal::MTLSize::new(tensor.numel() as u64, 1, 1);
    let thread_group_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    // Note: wait_until_completed() is NOT called here (matches candle pattern).
    // Commands manager handles batching and will commit when batch size is exceeded.

    // Return new tensor (no transmute needed, result_buf is already MetalBuffer<T>)
    Tensor::new(
        BufferHandle::Metal(result_buf),
        tensor.shape().clone(),
        tensor.device().clone(),
    )
}

/// Execute a unary operation using CPU
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `op` - Closure that takes f32 and returns f32 (e.g., |x| x.exp())
///
/// # Returns
/// New tensor with the operation applied
pub(crate) fn execute_unary_cpu_op<T: FloatType, F>(tensor: &Tensor<T>, op: F) -> TensorResult<Tensor<T>>
where
    F: Fn(f32) -> f32,
{
    // Currently only f16 is supported
    if false {
        return Err(TensorError::InvalidOperation(
            "CPU operations currently only support f16".to_string()
        ));
    }

    let input = tensor.sync_and_read();
    // Safety: We checked T::is_f16() above
    let input_f16: Vec<f16> = unsafe { std::mem::transmute(input) };
    let result: Vec<f16> = input_f16.iter().map(|&x| f16::from_f32(op(x.to_f32()))).collect();
    let result_t: Vec<T> = unsafe { std::mem::transmute(result) };

    match tensor.device() {
        Device::Metal(dev) => Tensor::from_vec_gpu(dev, result_t, tensor.dims().to_vec()),
        _ => Tensor::from_vec(result_t, tensor.dims().to_vec()),
    }
}

/// Execute a binary operation using Metal GPU kernel
///
/// # Arguments
/// * `tensor` - Input tensor (must be on Metal device)
/// * `scalar` - Scalar parameter tensor
/// * `kernel_name` - Name of the Metal kernel function (e.g., "pow_f16")
///
/// # Returns
/// New tensor with the operation applied
pub(crate) fn execute_binary_metal_op<T: FloatType>(
    tensor: &Tensor<T>,
    scalar: &Tensor<T>,
    kernel_name: &str,
) -> TensorResult<Tensor<T>> {
    // Currently only f16 is supported for Metal operations
    if false {
        return Err(TensorError::InvalidOperation(
            "Metal operations currently only support f16".to_string()
        ));
    }
    let input_buf = tensor.buffer().as_metal()?;
    let scalar_buf = scalar.buffer().as_metal()?;
    let mut device = match tensor.device() {
        Device::Metal(dev) => dev.clone(),
        _ => {
            return Err(TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            ))
        }
    };

    // Load shader library if not already loaded
    if device.library().is_none() {
        let shader_source = include_str!("../../shaders/unified.metal");
        device.load_library(shader_source)?;
    }

    // Create output buffer from pool
    let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, tensor.numel())?;

    // Execute kernel using new EncoderProvider pattern
    // Get or compile pipeline
    let mut executor = KernelExecutor::new(device.clone());
    let pipeline = executor.get_or_compile_pipeline(kernel_name)?;

    // Create command buffer and encoder (EncoderProvider pattern)
    let (_flushed, command_buffer) = device.command_buffer()?;
    let encoder = command_buffer.encoder();

    // Set pipeline and buffers (Metal API is type-erased, kernel_name specifies f16/f32)
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(scalar_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);

    // Configure thread groups
    let grid_size = metal::MTLSize::new(tensor.numel() as u64, 1, 1);
    let thread_group_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    // Note: wait_until_completed() is NOT called here (matches candle pattern).
    // Commands manager handles batching and will commit when batch size is exceeded.

    // Return new tensor (no transmute needed, result_buf is already MetalBuffer<T>)
    Tensor::new(
        BufferHandle::Metal(result_buf),
        tensor.shape().clone(),
        tensor.device().clone(),
    )
}

/// Execute a binary operation using CPU
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `scalar_value` - Scalar value as f32
/// * `op` - Closure that takes two f32s and returns f32 (e.g., |x, y| x.powf(y))
///
/// # Returns
/// New tensor with the operation applied
pub(crate) fn execute_binary_cpu_op<T: FloatType, F>(
    tensor: &Tensor<T>,
    scalar_value: f32,
    op: F,
) -> TensorResult<Tensor<T>>
where
    F: Fn(f32, f32) -> f32,
{
    let input = tensor.sync_and_read();
    let result: Vec<T> = input
        .iter()
        .map(|&x| T::from_f32(op(x.to_f32(), scalar_value)))
        .collect();

    match tensor.device() {
        Device::Metal(dev) => Tensor::from_vec_gpu(dev, result, tensor.dims().to_vec()),
        _ => Tensor::from_vec(result, tensor.dims().to_vec()),
    }
}

/// Execute clamp operation using Metal GPU kernel
///
/// # Arguments
/// * `tensor` - Input tensor (must be on Metal device)
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
/// * `kernel_name` - Name of the Metal kernel function (e.g., "clamp_f16")
///
/// # Returns
/// New tensor with values clamped to [min_val, max_val]
pub(crate) fn execute_clamp_metal_op<T: FloatType>(
    tensor: &Tensor<T>,
    min_val: f32,
    max_val: f32,
    kernel_name: &str,
) -> TensorResult<Tensor<T>> {
    eprintln!("[DEBUG clamp] kernel={}, min={}, max={}, numel={}", kernel_name, min_val, max_val, tensor.numel());
    let input_buf = tensor.buffer().as_metal()?;
    let mut device = match tensor.device() {
        Device::Metal(dev) => dev.clone(),
        _ => {
            return Err(TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            ))
        }
    };

    // Load shader library if not already loaded
    if device.library().is_none() {
        let shader_source = include_str!("../../shaders/unified.metal");
        device.load_library(shader_source)?;
    }

    // Create output buffer from pool
    let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, tensor.numel())?;

    // Execute kernel
    let mut executor = KernelExecutor::new(device.clone());
    let pipeline = executor.get_or_compile_pipeline(kernel_name)?;

    // Create command buffer and encoder
    let (_flushed, command_buffer) = device.command_buffer()?;
    let encoder = command_buffer.encoder();

    // Set pipeline and buffers
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(result_buf.metal_buffer()), 0);
    encoder.set_bytes(2, std::mem::size_of::<f32>() as u64, &min_val as *const f32 as *const _);
    encoder.set_bytes(3, std::mem::size_of::<f32>() as u64, &max_val as *const f32 as *const _);

    // Configure thread groups
    let grid_size = metal::MTLSize::new(tensor.numel() as u64, 1, 1);
    let thread_group_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();

    // Return new tensor
    Tensor::new(
        BufferHandle::Metal(result_buf),
        tensor.shape().clone(),
        tensor.device().clone(),
    )
}
