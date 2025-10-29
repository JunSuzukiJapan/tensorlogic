//! Sampling and generation operations for TensorLogic interpreter

use super::*;
use crate::tensor::TensorIO;
use rand::Rng;

impl Interpreter {
    pub(super) fn eval_sampling_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "softmax" => Some(self.eval_softmax(args)),
            "temperature_sample" => Some(self.eval_temperature_sample(args)),
            "sample" | "top_k" | "top_p" | "top_p_sample" |
            "temperature" | "argmax" | "argmin" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Sampling function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// softmax(x) or softmax(x, dim) -> tensor
    /// Softmax activation along the last dimension (or specified dimension)
    fn eval_softmax(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if args.is_empty() || args.len() > 2 {
            return Err(RuntimeError::TypeError(
                format!("softmax() expects 1 or 2 arguments (tensor, [dim]), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;

        // For now, ignore the dimension argument and apply along last dimension
        // TODO: Implement softmax with custom dimension support
        if args.len() == 2 {
            let _dim_val = self.eval_expr(&args[1])?;
            // Dimension parameter is ignored for now
        }

        Ok(match tensor_val {
            Value::TensorF16(tensor) => {
                // Apply softmax
                let result = tensor.softmax()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                result.to_value()
            }
            Value::TensorF32(tensor) => {
                // Apply softmax
                let result = tensor.softmax()
                    .map_err(|e| RuntimeError::TensorError(e))?;
                result.to_value()
            }
            _ => return Err(RuntimeError::TypeError("softmax() expects tensor (f16 or f32)".to_string()))
        })
    }

    /// temperature_sample(logits, temperature) -> int
    /// Sample a token from logits with temperature scaling (GPU-accelerated)
    /// Returns the sampled token ID as an integer
    ///
    /// For 2D logits [seq_len, vocab_size], uses only the last sequence position
    fn eval_temperature_sample(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        
        

        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("temperature_sample() expects 2 arguments (logits, temperature), got {}", args.len())
            ));
        }

        // Evaluate logits tensor
        let logits_val = self.eval_expr(&args[0])?;

        // Evaluate temperature
        let temp_val = self.eval_expr(&args[1])?;
        let temperature = match temp_val {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            _ => return Err(RuntimeError::TypeError(
                "temperature_sample() temperature must be a number".to_string()
            )),
        };

        // Process based on tensor type (f16 or f32) and device
        match logits_val {
            Value::TensorF16(ref logits) => {
                self.temperature_sample_gpu(logits, temperature)
            }
            Value::TensorF32(ref logits) => {
                self.temperature_sample_gpu(logits, temperature)
            }
            _ => Err(RuntimeError::TypeError(
                "temperature_sample() expects tensor (f16 or f32)".to_string()
            ))
        }
    }

    /// GPU-accelerated temperature sampling implementation
    fn temperature_sample_gpu<T: FloatType>(&self, logits: &crate::tensor::Tensor<T>, temperature: f32) -> RuntimeResult<Value> {
        use crate::device::Device;
        use crate::tensor::TensorAccessors;

        let dims = logits.dims();

        // For 2D tensors [seq_len, vocab_size], we'll pass the full tensor
        // and let the Metal kernel handle slicing to the last position
        let vocab_size = if dims.len() == 2 {
            dims[1]  // vocab_size is second dimension
        } else if dims.len() == 1 {
            dims[0]  // vocab_size is the only dimension
        } else {
            return Err(RuntimeError::TypeError(
                "temperature_sample() expects 1D or 2D logits tensor".to_string()
            ));
        };

        // Use CPU sampling (like llama.cpp and candle)
        // GPU sampling is too slow due to multiple kernel launches and sync overhead
        self.temperature_sample_cpu(logits, temperature, vocab_size)
    }

    /// CPU sampling implementation (following llama.cpp and candle approach)
    /// This is actually faster than GPU sampling for vocabulary-sized tensors
    fn temperature_sample_cpu<T: FloatType>(&self, logits: &crate::tensor::Tensor<T>, temperature: f32, vocab_size: usize) -> RuntimeResult<Value> {
        use crate::tensor::{TensorIO, TensorAccessors};
        use rand::Rng;

        // Get logits as f32 vector (sync all pending GPU ops before reading)
        let all_logits: Vec<f32> = logits.sync_and_read_f32();

        // For 2D tensors [seq_len, vocab_size], we only want the last row
        let logits_data: Vec<f32> = if logits.dims().len() == 2 {
            let seq_len = logits.dims()[0];
            // Take the last vocab_size elements
            all_logits[(seq_len - 1) * vocab_size .. seq_len * vocab_size].to_vec()
        } else {
            // 1D tensor - use as-is
            all_logits
        };

        // Apply temperature scaling
        let scaled: Vec<f32> = logits_data.iter().map(|&x| x / temperature).collect();

        // Compute softmax
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<f32> = Vec::with_capacity(vocab_size);

        for &logit in &scaled {
            let exp_val = (logit - max_logit).exp();
            probs.push(exp_val);
            exp_sum += exp_val;
        }

        for prob in &mut probs {
            *prob /= exp_sum;
        }

        // Sample
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();
        let mut cumulative = 0.0f32;
        let mut sampled_idx = 0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                sampled_idx = idx;
                break;
            }
        }

        if std::env::var("TL_DEBUG_SAMPLING").is_ok() {
            eprintln!("\n=== CPU Temperature Sample ===");
            eprintln!("  Sampled token: {}", sampled_idx);
        }

        Ok(Value::Integer(sampled_idx as i64))
    }

    /// Metal GPU implementation
    fn temperature_sample_metal<T: FloatType>(&self, logits: &crate::tensor::Tensor<T>, temperature: f32, vocab_size: usize) -> RuntimeResult<Value> {
        use crate::device::{Device, MetalBuffer};
        use crate::tensor::TensorAccessors;
        use rand::Rng;

        let mut device = match logits.device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(RuntimeError::TensorError(
                crate::error::TensorError::DeviceConversionError("Not on Metal device".to_string())
            )),
        };

        // Load shader library
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)
                .map_err(|e| RuntimeError::TensorError(e))?;
        }

        let input_buf = logits.buffer().as_metal()
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Calculate buffer offset for 2D tensors (to get last sequence position)
        let dims = logits.dims();
        let buffer_offset: u64 = if dims.len() == 2 {
            let seq_len = dims[0];
            let start_idx = (seq_len - 1) * vocab_size;
            (start_idx * std::mem::size_of::<T>()) as u64
        } else {
            0
        };

        // Type suffix for kernel selection
        let suffix = T::kernel_suffix();

        // Create buffers
        let probs_buf = MetalBuffer::<T>::new_uninit(device.metal_device(), vocab_size)
            .map_err(|e| RuntimeError::TensorError(e))?;
        let max_buf = MetalBuffer::<T>::new_uninit(device.metal_device(), 1)
            .map_err(|e| RuntimeError::TensorError(e))?;
        let sum_buf = MetalBuffer::<T>::new_uninit(device.metal_device(), 1)
            .map_err(|e| RuntimeError::TensorError(e))?;

        // Create parameter buffers
        let temp_buf = device.metal_device().new_buffer_with_data(
            &temperature as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let vocab_u32 = vocab_size as u32;
        let vocab_buf = device.metal_device().new_buffer_with_data(
            &vocab_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Random value for sampling
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();
        let random_buf = device.metal_device().new_buffer_with_data(
            &random_value as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Sampled token output
        let sampled_buf = device.metal_device().new_buffer_with_data(
            &0u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // Step 1: Apply temperature scaling
        let pipeline1 = executor.get_or_compile_pipeline(&format!("temperature_softmax{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline1);
        encoder.set_buffer(0, Some(&input_buf.buffer), buffer_offset);
        encoder.set_buffer(1, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(2, Some(&temp_buf), 0);
        encoder.set_buffer(3, Some(&vocab_buf), 0);

        let grid_size = metal::MTLSize::new(vocab_size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(vocab_size as u64), 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        crate::ops::async_exec::submit_async(&command_buffer); // Async!

        // Step 2: Find max value (for numerical stability)
        let pipeline2 = executor.get_or_compile_pipeline(&format!("find_max{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline2);
        encoder.set_buffer(0, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(1, Some(&max_buf.buffer), 0);
        encoder.set_buffer(2, Some(&vocab_buf), 0);
        encoder.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<f32>() as u64);

        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let grid_size = metal::MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        crate::ops::async_exec::submit_async(&command_buffer); // Async!

        // Step 3: Compute exp and sum
        let pipeline3 = executor.get_or_compile_pipeline(&format!("softmax_normalize{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline3);
        encoder.set_buffer(0, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(1, Some(&max_buf.buffer), 0);
        encoder.set_buffer(2, Some(&sum_buf.buffer), 0);
        encoder.set_buffer(3, Some(&vocab_buf), 0);
        encoder.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<f32>() as u64);

        let grid_size = metal::MTLSize::new(vocab_size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(vocab_size as u64), 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        crate::ops::async_exec::submit_async(&command_buffer); // Async!

        // Step 4: Normalize by sum
        let pipeline4 = executor.get_or_compile_pipeline(&format!("divide_by_sum{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline4);
        encoder.set_buffer(0, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(1, Some(&sum_buf.buffer), 0);
        encoder.set_buffer(2, Some(&vocab_buf), 0);

        let grid_size = metal::MTLSize::new(vocab_size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(256.min(vocab_size as u64), 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        crate::ops::async_exec::submit_async(&command_buffer); // Async!

        // Step 5: Sample from distribution
        let pipeline5 = executor.get_or_compile_pipeline(&format!("cumulative_sample{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline5);
        encoder.set_buffer(0, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(1, Some(&sampled_buf), 0);
        encoder.set_buffer(2, Some(&random_buf), 0);
        encoder.set_buffer(3, Some(&vocab_buf), 0);

        let grid_size = metal::MTLSize::new(1, 1, 1);
        let threadgroup_size = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        crate::ops::async_exec::submit_async(&command_buffer);

        // Sync all pending operations before reading result
        crate::ops::async_exec::sync_all();

        // Read result (only 4 bytes!)
        let sampled_ptr = sampled_buf.contents() as *const u32;
        let sampled_token = unsafe { *sampled_ptr };

        if std::env::var("TL_DEBUG_SAMPLING").is_ok() {
            eprintln!("\n=== GPU Temperature Sample ===");
            eprintln!("  Vocab size: {}", vocab_size);
            eprintln!("  Temperature: {}", temperature);
            eprintln!("  Random value: {}", random_value);
            eprintln!("  Sampled token: {}", sampled_token);
        }

        Ok(Value::Integer(sampled_token as i64))
    }
}
