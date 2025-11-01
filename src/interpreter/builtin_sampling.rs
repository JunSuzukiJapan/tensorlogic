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

        // OPTIMIZATION: Use argmax for temperature=0 (greedy sampling)
        // This skips expensive softmax computation entirely
        if temperature == 0.0 {
            match logits.device() {
                Device::Metal(_) => return self.argmax_sample_metal(logits, vocab_size),
                _ => {} // Fall through to CPU sampling
            }
        }

        // Use GPU sampling to avoid sync_and_read() bottleneck on every token
        // This keeps logits on GPU and performs sampling async
        match logits.device() {
            Device::Metal(_) => self.temperature_sample_metal(logits, temperature, vocab_size),
            _ => self.temperature_sample_cpu(logits, temperature, vocab_size),
        }
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

            // Show top 10 logits before softmax
            let mut indexed_logits: Vec<(usize, f32)> = logits_data.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!("  Top 10 logits (before softmax):");
            for (rank, (idx, logit)) in indexed_logits.iter().take(10).enumerate() {
                eprintln!("    {}: token {} = {:.4}", rank + 1, idx, logit);
            }

            // Show top 10 probabilities after softmax
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!("  Top 10 probs (after softmax):");
            for (rank, (idx, prob)) in indexed_probs.iter().take(10).enumerate() {
                eprintln!("    {}: token {} = {:.6}", rank + 1, idx, prob);
            }

            eprintln!("  Sampled token: {} (temperature: {})", sampled_idx, temperature);
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

        // CRITICAL: Wait for all pending command buffers to complete before creating new ones
        // Since we removed sync_and_read() from tensor creation, we must ensure
        // all GPU operations are complete before sampling
        device.wait_until_completed()
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::MetalError(format!("Failed to wait for GPU: {}", e))
            ))?;

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

        let final_command_buffer = device.command_queue().new_command_buffer();
        let encoder = final_command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline5);
        encoder.set_buffer(0, Some(&probs_buf.buffer), 0);
        encoder.set_buffer(1, Some(&sampled_buf), 0);
        encoder.set_buffer(2, Some(&random_buf), 0);
        encoder.set_buffer(3, Some(&vocab_buf), 0);

        let grid_size = metal::MTLSize::new(1, 1, 1);
        let threadgroup_size = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        final_command_buffer.commit();

        // TARGETED SYNC: Wait only for the final sampling command buffer
        // This avoids waiting for all 22 layers of transformer computation
        // Metal's dependency tracking ensures logits are ready before sampling starts
        final_command_buffer.wait_until_completed();

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

    /// GPU-accelerated argmax sampling (for temperature=0 greedy decoding)
    /// Much faster than full temperature sampling as it skips softmax computation
    fn argmax_sample_metal<T: FloatType>(&self, logits: &crate::tensor::Tensor<T>, vocab_size: usize) -> RuntimeResult<Value> {
        use crate::device::{Device, MetalBuffer};
        use crate::tensor::TensorAccessors;

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
            start_idx as u64
        } else {
            0
        };

        // Type suffix for kernel selection
        let suffix = T::kernel_suffix();

        // Create output buffer for argmax result
        let max_idx_buf = device.metal_device().new_buffer_with_data(
            &0u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create parameter buffers
        let vocab_u32 = vocab_size as u32;
        let vocab_buf = device.metal_device().new_buffer_with_data(
            &vocab_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let offset_u32 = buffer_offset as u32;
        let offset_buf = device.metal_device().new_buffer_with_data(
            &offset_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut executor = crate::device::KernelExecutor::new(device.clone());

        // CRITICAL: Wait for all pending command buffers to complete before creating new one
        // Since we removed sync_and_read() from tensor creation, we must ensure
        // all GPU operations are complete before sampling
        device.wait_until_completed()
            .map_err(|e| RuntimeError::TensorError(
                crate::error::TensorError::MetalError(format!("Failed to wait for GPU: {}", e))
            ))?;

        // Execute argmax kernel
        let pipeline = executor.get_or_compile_pipeline(&format!("argmax{}", suffix))
            .map_err(|e| RuntimeError::TensorError(e))?;

        let command_buffer = device.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&max_idx_buf), 0);
        encoder.set_buffer(2, Some(&vocab_buf), 0);
        encoder.set_buffer(3, Some(&offset_buf), 0);

        // Allocate threadgroup memory for parallel reduction
        encoder.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<T>() as u64);  // shared_max
        encoder.set_threadgroup_memory_length(1, 256 * std::mem::size_of::<u32>() as u64); // shared_idx

        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let grid_size = metal::MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();

        // TARGETED SYNC: Wait only for this argmax command buffer
        command_buffer.wait_until_completed();

        // Read result (only 4 bytes!)
        let max_idx_ptr = max_idx_buf.contents() as *const u32;
        let max_idx = unsafe { *max_idx_ptr };

        if std::env::var("TL_DEBUG_SAMPLING").is_ok() {
            eprintln!("\n=== GPU Argmax Sample (greedy) ===");
            eprintln!("  Vocab size: {}", vocab_size);
            eprintln!("  Selected token (argmax): {}", max_idx);
        }

        Ok(Value::Integer(max_idx as i64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use crate::tensor::{Tensor, TensorCreation};
    use half::f16;

    #[test]
    fn test_temperature_sample_greedy() {
        // Test that temperature=0.0 (greedy) always selects the highest logit
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Create logits with a clear maximum at index 5
        let logits_data: Vec<f16> = vec![
            f16::from_f32(1.0),  // idx 0
            f16::from_f32(2.0),  // idx 1
            f16::from_f32(0.5),  // idx 2
            f16::from_f32(3.0),  // idx 3
            f16::from_f32(1.5),  // idx 4
            f16::from_f32(10.0), // idx 5 <- maximum
            f16::from_f32(2.5),  // idx 6
            f16::from_f32(0.1),  // idx 7
        ];

        let logits = Tensor::from_vec_gpu(&device, logits_data, vec![8])
            .expect("Failed to create logits tensor");

        let interpreter = Interpreter::new();
        let result = interpreter.temperature_sample_gpu(&logits, 0.0);

        assert!(result.is_ok());
        if let Ok(Value::Integer(token_id)) = result {
            assert_eq!(token_id, 5, "Greedy sampling should select token with highest logit");
        } else {
            panic!("Expected Integer value from temperature_sample");
        }
    }

    #[test]
    fn test_temperature_sample_2d_tensor() {
        // Test that 2D logits [seq_len, vocab_size] correctly samples from last position
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Create 2D logits [2, 8] - should sample from last row
        let logits_data: Vec<f16> = vec![
            // First sequence position (should be ignored)
            f16::from_f32(10.0), f16::from_f32(1.0), f16::from_f32(1.0), f16::from_f32(1.0),
            f16::from_f32(1.0), f16::from_f32(1.0), f16::from_f32(1.0), f16::from_f32(1.0),
            // Last sequence position (should be used) - maximum at idx 3
            f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(0.5), f16::from_f32(15.0),
            f16::from_f32(1.5), f16::from_f32(2.5), f16::from_f32(0.1), f16::from_f32(1.0),
        ];

        let logits = Tensor::from_vec_gpu(&device, logits_data, vec![2, 8])
            .expect("Failed to create 2D logits tensor");

        let interpreter = Interpreter::new();
        let result = interpreter.temperature_sample_gpu(&logits, 0.0);

        assert!(result.is_ok());
        if let Ok(Value::Integer(token_id)) = result {
            assert_eq!(token_id, 3, "Should sample from last sequence position, selecting highest logit");
        } else {
            panic!("Expected Integer value from temperature_sample");
        }
    }

    #[test]
    fn test_softmax_numerical_accuracy() {
        // Test softmax computation for numerical stability
        let device = MetalDevice::new().expect("Failed to create Metal device");

        // Create logits with large values to test numerical stability
        let logits_data: Vec<f16> = vec![
            f16::from_f32(100.0),
            f16::from_f32(101.0), // slightly higher
            f16::from_f32(100.5),
        ];

        let logits = Tensor::from_vec_gpu(&device, logits_data, vec![3])
            .expect("Failed to create logits tensor");

        // With temperature=0.0, should still select argmax despite numerical challenges
        let interpreter = Interpreter::new();
        let result = interpreter.temperature_sample_gpu(&logits, 0.0);

        assert!(result.is_ok());
        if let Ok(Value::Integer(token_id)) = result {
            assert_eq!(token_id, 1, "Should handle large logits correctly");
        } else {
            panic!("Expected Integer value");
        }
    }

    #[test]
    fn test_softmax_probabilities_sum_to_one() {
        // Mathematical property: softmax probabilities must sum to 1.0
        use crate::tensor::TensorIO;

        let logits_data = vec![1.0f32, 2.0, 3.0, 0.5, 1.5];
        let temperature = 1.0;

        // Compute softmax manually
        let scaled: Vec<f32> = logits_data.iter().map(|&x| x / temperature).collect();
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut exp_sum = 0.0f32;
        let mut probs: Vec<f32> = Vec::new();

        for &logit in &scaled {
            let exp_val = (logit - max_logit).exp();
            probs.push(exp_val);
            exp_sum += exp_val;
        }

        for prob in &mut probs {
            *prob /= exp_sum;
        }

        // Verify probabilities sum to 1.0
        let prob_sum: f32 = probs.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-6,
            "Softmax probabilities must sum to 1.0, got {}",
            prob_sum
        );

        // Verify all probabilities are in [0, 1]
        for (i, &prob) in probs.iter().enumerate() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability at index {} is {}, must be in [0, 1]",
                i,
                prob
            );
        }
    }

    #[test]
    fn test_temperature_scaling_effects() {
        // Mathematical property: temperature affects distribution sharpness
        // Lower temperature -> sharper distribution (more peaked)
        // Higher temperature -> flatter distribution (more uniform)

        let logits_data = vec![1.0f32, 2.0, 3.0, 0.5];

        // Test with different temperatures
        for temperature in [0.5, 1.0, 2.0] {
            let scaled: Vec<f32> = logits_data.iter().map(|&x| x / temperature).collect();
            let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let mut exp_sum = 0.0f32;
            let mut probs: Vec<f32> = Vec::new();

            for &logit in &scaled {
                let exp_val = (logit - max_logit).exp();
                probs.push(exp_val);
                exp_sum += exp_val;
            }

            for prob in &mut probs {
                *prob /= exp_sum;
            }

            // Lower temperature should give higher probability to max logit
            let max_prob = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let expected_max_idx = logits_data.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;

            assert_eq!(
                probs.iter().position(|&p| p == max_prob).unwrap(),
                expected_max_idx,
                "Maximum probability should be at index with maximum logit"
            );
        }
    }

    #[test]
    fn test_temperature_zero_equivalence() {
        // Mathematical property: temperature=0 should be equivalent to argmax
        let device = MetalDevice::new().expect("Failed to create Metal device");

        let logits_data: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),  // maximum
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ];

        let logits = Tensor::from_vec_gpu(&device, logits_data, vec![4])
            .expect("Failed to create logits tensor");

        let interpreter = Interpreter::new();

        // With temperature=0, should always select index 1 (maximum)
        for _ in 0..10 {
            let result = interpreter.temperature_sample_gpu(&logits, 0.0);
            assert!(result.is_ok());
            if let Ok(Value::Integer(token_id)) = result {
                assert_eq!(token_id, 1, "Temperature=0 should always select argmax");
            } else {
                panic!("Expected Integer value");
            }
        }
    }

    #[test]
    fn test_exp_numerical_stability() {
        // Test that exp(x - max_x) prevents overflow/underflow
        let large_logits = vec![1000.0f32, 1001.0, 1000.5];

        // Without subtracting max, exp(1000) would overflow
        // With max subtraction: exp(0), exp(1), exp(0.5)
        let max_logit = large_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut probs: Vec<f32> = Vec::new();
        let mut exp_sum = 0.0f32;

        for &logit in &large_logits {
            let exp_val = (logit - max_logit).exp();
            assert!(
                exp_val.is_finite(),
                "exp({} - {}) should be finite, got {}",
                logit,
                max_logit,
                exp_val
            );
            probs.push(exp_val);
            exp_sum += exp_val;
        }

        // Normalize
        for prob in &mut probs {
            *prob /= exp_sum;
        }

        // Verify probabilities are valid
        let prob_sum: f32 = probs.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-5,
            "Probabilities must sum to 1.0 even with large logits, got {}",
            prob_sum
        );
    }
}
