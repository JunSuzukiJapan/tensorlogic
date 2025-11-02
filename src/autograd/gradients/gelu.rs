use crate::tensor::FloatType;
use crate::autograd::GradientFunctionGeneric;
use std::marker::PhantomData;
use super::prelude::*;
use crate::device::{Device, MetalBuffer, EncoderProvider};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};

/// GELU演算の勾配関数
///
/// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// ∂GELU/∂x = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * derivative_of_inner
pub struct GELUBackward<T: FloatType> {
    input: Tensor<T>,
    _phantom: PhantomData<T>,
}

impl<T: FloatType> GELUBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self {
            input,
            _phantom: PhantomData,
        }
    }
}

impl<T: FloatType> GradientFunctionGeneric<T> for GELUBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> TensorResult<Vec<Tensor<T>>> {
        // Use GPU if both tensors are on Metal
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };

        Ok(vec![grad_input])
    }
}

impl<T: FloatType> GELUBackward<T> {
    /// Metal GPU implementation of GELU backward
    fn backward_metal(&self, grad_output: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let grad_out_buf = grad_output.buffer().as_metal()?;
        let input_buf = self.input.buffer().as_metal()?;

        let mut device = match grad_output.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Load gradient shaders if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Create result buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(device.buffer_pool(), grad_output.numel())?;

        // Execute kernel
        let mut executor = crate::device::KernelExecutor::new(device.clone());

        let pipeline = executor.get_or_compile_pipeline("gelu_backward_f16")?;

        // Use Commands manager for command buffer (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()?;
        let encoder = command_buffer.encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&grad_out_buf.buffer), 0);
        encoder.set_buffer(1, Some(&input_buf.buffer), 0);
        encoder.set_buffer(2, Some(&result_buf.buffer), 0);

        let grid_size = metal::MTLSize::new(grad_output.numel() as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(grad_output.numel() as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Commands manager will flush and commit when needed
        // Result stays on GPU - no wait needed (batching-friendly)

        Tensor::<T>::new(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            grad_output.shape().clone(),
            grad_output.device().clone(),
        )
    }

    /// CPU fallback for GELU backward
    fn backward_cpu(&self, grad_output: &Tensor<T>) -> TensorResult<Tensor<T>> {
        let grad_output_data = grad_output.sync_and_read();
        let input_data = self.input.sync_and_read();

        let sqrt_2_over_pi_f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

        let grad_input_data: Vec<T> = grad_output_data
            .iter()
            .zip(input_data.iter())
            .map(|(&grad_out, &x)| {
                let x_f32 = x.to_f32();
                let x3 = x_f32 * x_f32 * x_f32;
                let inner = sqrt_2_over_pi_f32 * (x_f32 + 0.044715 * x3);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;

                // ∂GELU/∂x = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * derivative_of_inner
                let derivative_of_inner =
                    sqrt_2_over_pi_f32 * (1.0 + 3.0 * 0.044715 * x_f32 * x_f32);
                let gelu_derivative =
                    0.5 * (1.0 + tanh_val) + 0.5 * x_f32 * sech2 * derivative_of_inner;

                T::from_f32(grad_out.to_f32() * gelu_derivative)
            })
            .collect();

        Tensor::<T>::from_vec(grad_input_data, grad_output.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use half::f16;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_gelu_backward_basic() {
        let device = get_test_device();

        // input = [0.0, 1.0]
        let input = <Tensor<half::f16>>::from_vec_gpu(
            &device,
            vec![half::f16::from_f32(0.0), half::f16::from_f32(1.0)],
            vec![2],
        )
        .unwrap();

        // grad_output = [1.0, 1.0]
        let grad_output = <Tensor<half::f16>>::from_vec_gpu(
            &device,
            vec![half::f16::from_f32(1.0), half::f16::from_f32(1.0)],
            vec![2],
        )
        .unwrap();

        let backward = GELUBackward::new(input);
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].dims(), &[2]);

        // 勾配が計算されていることを確認（具体的な値は数値微分で検証が望ましい）
        let grad_values = grads[0].sync_and_read();
        assert!(grad_values[0].to_f32() > 0.0); // x=0での勾配は正
        assert!(grad_values[1].to_f32() > 0.0); // x=1での勾配は正
    }
}
