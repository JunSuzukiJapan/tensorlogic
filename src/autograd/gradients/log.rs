use crate::autograd::GradientFunction;
use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;

pub struct LogBackward {
    input: Tensor,
}

impl LogBackward {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        let grad_input = if grad_output.buffer().is_metal() && self.input.buffer().is_metal() {
            self.backward_metal(grad_output)?
        } else {
            self.backward_cpu(grad_output)?
        };
        Ok(vec![grad_input])
    }
}

impl LogBackward {
    fn backward_metal(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_output_buf = grad_output.buffer().as_metal()?;
        let input_buf = self.input.buffer().as_metal()?;

        let mut device = match grad_output.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        if device.library().is_none() {
            let shader_source = include_str!("../../../shaders/gradients.metal");
            device.load_library(shader_source)?;
        }

        let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), grad_output.numel())?;

        let library_ref = device.library();
        let library = library_ref
            .as_ref()
            .ok_or_else(|| TensorError::MetalError("Library not loaded".to_string()))?;
        let pipeline = library
            .get_function("log_backward_f16", None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel log_backward_f16: {:?}", e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {:?}", e)))?;

        let command_queue = device.command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(grad_output_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);

        let grid_size = metal::MTLSize::new(grad_output.numel() as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Tensor::new(
            crate::tensor::BufferHandle::Metal(result_buf),
            grad_output.shape().clone(),
            grad_output.device().clone(),
        )
    }

    fn backward_cpu(&self, grad_output: &Tensor) -> TensorResult<Tensor> {
        let grad_out = grad_output.to_vec();
        let input = self.input.to_vec();

        let grad_input: Vec<_> = grad_out
            .iter()
            .zip(input.iter())
            .map(|(g, x)| *g / *x)
            .collect();

        match grad_output.device() {
            Device::Metal(dev) => {
                Tensor::from_vec_metal(dev, grad_input, grad_output.dims().to_vec())
            }
            _ => Tensor::from_vec(grad_input, grad_output.dims().to_vec()),
        }
    }
}
