//! Common helper functions for operation implementations
//! Reduces code duplication across elementwise and activation operations

use crate::device::{Device, KernelExecutor, MetalBuffer};
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
pub(crate) fn execute_unary_metal_op(
    tensor: &Tensor,
    kernel_name: &str,
) -> TensorResult<Tensor> {
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
        let shader_source = include_str!("../../shaders/elementwise.metal");
        device.load_library(shader_source)?;
    }

    // Create output buffer from pool
    let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), tensor.numel())?;

    // Execute kernel
    let mut executor = KernelExecutor::new(device);
    executor.execute_unary_op(kernel_name, input_buf, &result_buf)?;

    // Return new tensor
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
pub(crate) fn execute_unary_cpu_op<F>(tensor: &Tensor, op: F) -> TensorResult<Tensor>
where
    F: Fn(f32) -> f32,
{
    let input = tensor.to_vec();
    let result: Vec<f16> = input.iter().map(|&x| f16::from_f32(op(x.to_f32()))).collect();

    match tensor.device() {
        Device::Metal(dev) => Tensor::from_vec_metal(dev, result, tensor.dims().to_vec()),
        _ => Tensor::from_vec(result, tensor.dims().to_vec()),
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
pub(crate) fn execute_binary_metal_op(
    tensor: &Tensor,
    scalar: &Tensor,
    kernel_name: &str,
) -> TensorResult<Tensor> {
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
        let shader_source = include_str!("../../shaders/elementwise.metal");
        device.load_library(shader_source)?;
    }

    // Create output buffer from pool
    let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), tensor.numel())?;

    // Execute kernel (note: binary ops use a different executor pattern)
    let library_ref = device.library();
    let library = library_ref
        .as_ref()
        .ok_or_else(|| TensorError::MetalError("Library not loaded".to_string()))?;
    let pipeline = library
        .get_function(kernel_name, None)
        .map_err(|e| TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e)))?;

    let pipeline_state = device.metal_device()
        .new_compute_pipeline_state_with_function(&pipeline)
        .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {:?}", e)))?;

    let command_queue = device.command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
    encoder.set_buffer(1, Some(scalar_buf.metal_buffer()), 0);
    encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);

    let grid_size = metal::MTLSize::new(tensor.numel() as u64, 1, 1);
    let thread_group_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Return new tensor
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
pub(crate) fn execute_binary_cpu_op<F>(
    tensor: &Tensor,
    scalar_value: f32,
    op: F,
) -> TensorResult<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let input = tensor.to_vec();
    let result: Vec<f16> = input
        .iter()
        .map(|&x| f16::from_f32(op(x.to_f32(), scalar_value)))
        .collect();

    match tensor.device() {
        Device::Metal(dev) => Tensor::from_vec_metal(dev, result, tensor.dims().to_vec()),
        _ => Tensor::from_vec(result, tensor.dims().to_vec()),
    }
}
