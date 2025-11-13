//! Normalization operations (LayerNorm, RMSNorm)

use crate::device::{Device, MetalBuffer, EncoderProvider};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// RMS Normalization (Root Mean Square Normalization)
    ///
    /// ⚠️ **MATHEMATICALLY VERIFIED - DO NOT MODIFY**
    /// Verified with test input [2, 4, 6, 8]:
    /// - Expected: [0.365, 0.730, 1.096, 1.461]
    /// - Actual: [0.3652, 0.7305, 1.0957, 1.4609] ✓
    ///
    /// If you encounter incorrect output, verify OTHER operations first.
    ///
    /// Simpler than LayerNorm - used in LLaMA, TinyLlama models.
    /// Normalizes by RMS instead of mean and variance.
    ///
    /// Formula: output = (x / rms(x)) * weight
    /// where rms(x) = sqrt(mean(x^2) + eps)
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the dimensions to normalize over (from the end)
    /// * `weight` - Learnable weight (gamma) for scaling
    /// * `eps` - Small value for numerical stability (default: 1e-6 for LLaMA)
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32)).collect();
    /// let x = Tensor::from_vec(data, vec![2, 3, 4])?;
    /// let weight = Tensor::ones(vec![4])?;
    /// let normalized = x.rms_norm(vec![4], &weight, 1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn rms_norm(
        &self,
        normalized_shape: Vec<usize>,
        weight: &Tensor<T>,
        eps: f32,
    ) -> TensorResult<Self> {
        // Validate normalized_shape
        let dims = self.dims();
        let ndim = dims.len();
        let norm_ndim = normalized_shape.len();

        if norm_ndim > ndim {
            return Err(TensorError::InvalidOperation(
                format!(
                    "normalized_shape length {} exceeds tensor ndim {}",
                    norm_ndim, ndim
                ),
            ));
        }

        // Check that normalized_shape matches the last dimensions
        for i in 0..norm_ndim {
            if dims[ndim - norm_ndim + i] != normalized_shape[i] {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "normalized_shape {:?} does not match tensor shape {:?}",
                        normalized_shape, dims
                    ),
                ));
            }
        }

        // Validate weight shape
        if weight.dims() != &normalized_shape {
            return Err(TensorError::ShapeMismatch {
                expected: normalized_shape.clone(),
                actual: weight.dims().to_vec(),
            });
        }

        match self.device() {
            Device::Metal(_) if self.buffer().is_metal() => {
                self.rms_norm_metal(&normalized_shape, weight, eps)
            }
            _ => self.rms_norm_cpu(&normalized_shape, weight, eps),
        }
    }

    /// Metal GPU implementation of RMS normalization
    fn rms_norm_metal(
        &self,
        normalized_shape: &[usize],
        weight: &Tensor<T>,
        eps: f32,
    ) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if false {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let input_buf = self.buffer().as_metal()?;
        let weight_buf = weight.buffer().as_metal()?;

        let mut device = match self.device() {
            Device::Metal(dev) => dev.clone(),
            _ => {
                return Err(TensorError::DeviceConversionError(
                    "Not on Metal device".to_string(),
                ))
            }
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Calculate normalized size
        let normalized_size: usize = normalized_shape.iter().product();
        let batch_size = self.numel() / normalized_size;

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, self.numel())?;

        // Get pipeline - select kernel based on size and type
        let suffix = T::kernel_suffix();

        // Create buffers for scalar parameters using tensor type T
        let normalized_size_buf = MetalBuffer::<T>::from_slice(
            &device,
            &[T::from_f32(normalized_size as f32)],
        )?;
        let eps_buf = MetalBuffer::<T>::from_slice(&device, &[T::from_f32(eps)])?;

        let kernel_name = if normalized_size <= 256 {
            format!("rms_norm_simple{}", suffix)
        } else {
            format!("rms_norm{}", suffix)
        };

        let library_ref = device.library();
        let library = library_ref.as_ref().ok_or_else(|| {
            TensorError::MetalError("Library not loaded".to_string())
        })?;
        let pipeline = library
            .get_function(&kernel_name, None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to create pipeline: {:?}", e))
            })?;

        // Execute kernel using batched command encoder (Candle pattern)
        let (_flushed, encoder) = device.command_encoder()?;

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(weight_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(normalized_size_buf.metal_buffer()), 0);
        encoder.set_buffer(4, Some(eps_buf.metal_buffer()), 0);

        // For optimized kernel: each batch element gets one threadgroup (256 threads)
        // For simple kernel: one thread per batch element
        if normalized_size <= 256 {
            // Simple kernel: one thread per batch element
            let grid_size = metal::MTLSize::new(batch_size as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(1, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        } else {
            // Optimized kernel: each batch element gets a threadgroup
            // Use dispatch_thread_groups, not dispatch_threads
            let threadgroups = metal::MTLSize::new(batch_size as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(256, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        }
        encoder.end_encoding();

        // NOTE: No commit here - the Commands batching system handles this automatically
        // Commands will commit when batch size is reached or when data is read to CPU
        // This matches Candle's lazy batching strategy

        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    /// CPU implementation of RMS normalization
    fn rms_norm_cpu(
        &self,
        normalized_shape: &[usize],
        weight: &Tensor<T>,
        eps: f32,
    ) -> TensorResult<Self> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input_data = self.sync_and_read();
        let input_f16: Vec<f16> = unsafe { std::mem::transmute(input_data) };
        let normalized_size: usize = normalized_shape.iter().product();
        let batch_size = self.numel() / normalized_size;

        let mut output = vec![f16::ZERO; self.numel()];
        let weight_data = weight.sync_and_read();
        let weight_vec: Vec<f16> = unsafe { std::mem::transmute(weight_data) };

        for batch_idx in 0..batch_size {
            let offset = batch_idx * normalized_size;
            let slice = &input_f16[offset..offset + normalized_size];

            // Compute RMS: sqrt(mean(x^2) + eps)
            // Use f32 for accumulation to avoid precision loss
            let sq_sum: f32 = slice.iter().map(|&x| {
                let val = x.to_f32();
                val * val
            }).sum();
            let mean_sq = sq_sum / normalized_size as f32;
            let rms = (mean_sq + eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Normalize and scale by weight
            for i in 0..normalized_size {
                let normalized = slice[i].to_f32() * inv_rms;
                let scaled = normalized * weight_vec[i].to_f32();
                output[offset + i] = f16::from_f32(scaled);
            }
        }

        let output_t: Vec<T> = unsafe { std::mem::transmute(output) };
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, output_t, self.dims().to_vec()),
            _ => Tensor::from_vec(output_t, self.dims().to_vec()),
        }
    }

    /// Layer Normalization
    ///
    /// Normalizes the input over the last dimensions specified by `normalized_shape`.
    /// Applies: output = (input - mean) / sqrt(variance + eps) * weight + bias
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the dimensions to normalize over (from the end)
    /// * `weight` - Optional learnable weight (gamma) for affine transformation
    /// * `bias` - Optional learnable bias (beta) for affine transformation
    /// * `eps` - Small value for numerical stability (default: 1e-5)
    ///
    /// # Example
    /// ```
    /// use tensorlogic::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32)).collect();
    /// let x = Tensor::from_vec(data, vec![2, 3, 4])?;
    /// let normalized = x.layer_norm(vec![4], None, None, 1e-5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn layer_norm(
        &self,
        normalized_shape: Vec<usize>,
        weight: Option<&Tensor<T>>,
        bias: Option<&Tensor<T>>,
        eps: f32,
    ) -> TensorResult<Self> {
        // Validate normalized_shape
        let dims = self.dims();
        let ndim = dims.len();
        let norm_ndim = normalized_shape.len();

        if norm_ndim > ndim {
            return Err(TensorError::InvalidOperation(
                format!(
                    "normalized_shape length {} exceeds tensor ndim {}",
                    norm_ndim, ndim
                ),
            ));
        }

        // Check that normalized_shape matches the last dimensions
        for i in 0..norm_ndim {
            if dims[ndim - norm_ndim + i] != normalized_shape[i] {
                return Err(TensorError::InvalidOperation(
                    format!(
                        "normalized_shape {:?} does not match tensor shape {:?}",
                        normalized_shape, dims
                    ),
                ));
            }
        }

        // Validate weight and bias shapes
        if let Some(w) = weight {
            if w.dims() != &normalized_shape {
                return Err(TensorError::ShapeMismatch {
                    expected: normalized_shape.clone(),
                    actual: w.dims().to_vec(),
                });
            }
        }
        if let Some(b) = bias {
            if b.dims() != &normalized_shape {
                return Err(TensorError::ShapeMismatch {
                    expected: normalized_shape.clone(),
                    actual: b.dims().to_vec(),
                });
            }
        }

        match self.device() {
            Device::Metal(_) if self.buffer().is_metal() => {
                self.layer_norm_metal(&normalized_shape, weight, bias, eps)
            }
            _ => self.layer_norm_cpu(&normalized_shape, weight, bias, eps),
        }
    }

    /// Metal GPU implementation of layer normalization
    fn layer_norm_metal(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor<T>>,
        bias: Option<&Tensor<T>>,
        eps: f32,
    ) -> TensorResult<Self> {
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
                    "Not on Metal device".to_string(),
                ))
            }
        };

        // Load shader if not already loaded
        if device.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device.load_library(shader_source)?;
        }

        // Calculate normalized size (product of normalized_shape dimensions)
        let normalized_size: usize = normalized_shape.iter().product();
        let batch_size = self.numel() / normalized_size;

        // Create output buffer
        let result_buf = MetalBuffer::<T>::new_uninit_pooled(&device, self.numel())?;

        // Get weight and bias buffers (or create dummy buffers)
        let dummy_buf = MetalBuffer::<T>::from_slice(&device, &[T::zero()])?;
        let weight_buf = if let Some(w) = weight {
            w.buffer().as_metal()?
        } else {
            &dummy_buf
        };
        let bias_buf = if let Some(b) = bias {
            b.buffer().as_metal()?
        } else {
            &dummy_buf
        };

        // Choose kernel based on normalized_size and type
        let suffix = T::kernel_suffix();

        // Create buffers for scalar parameters using tensor type T
        let normalized_size_buf = MetalBuffer::<T>::from_slice(
            &device,
            &[T::from_f32(normalized_size as f32)],
        )?;
        let eps_buf = MetalBuffer::<T>::from_slice(&device, &[T::from_f32(eps)])?;
        let has_weight_buf = MetalBuffer::<T>::from_slice(
            &device,
            &[T::from_f32(if weight.is_some() { 1.0 } else { 0.0 })],
        )?;
        let has_bias_buf = MetalBuffer::<T>::from_slice(
            &device,
            &[T::from_f32(if bias.is_some() { 1.0 } else { 0.0 })],
        )?;
        let kernel_name = if normalized_size <= 256 {
            format!("layer_norm_simple{}", suffix)
        } else {
            format!("layer_norm{}", suffix)
        };

        // Get pipeline
        let library_ref = device.library();
        let library = library_ref.as_ref().ok_or_else(|| {
            TensorError::MetalError("Library not loaded".to_string())
        })?;
        let pipeline = library
            .get_function(&kernel_name, None)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
            })?;

        let pipeline_state = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&pipeline)
            .map_err(|e| {
                TensorError::MetalError(format!("Failed to create pipeline: {:?}", e))
            })?;

        // Execute kernel using batched command encoder (Candle pattern)
        let (_flushed, encoder) = device.command_encoder()?;

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(input_buf.metal_buffer()), 0);
        encoder.set_buffer(1, Some(weight_buf.metal_buffer()), 0);
        encoder.set_buffer(2, Some(bias_buf.metal_buffer()), 0);
        encoder.set_buffer(3, Some(result_buf.metal_buffer()), 0);
        encoder.set_buffer(4, Some(normalized_size_buf.metal_buffer()), 0);
        encoder.set_buffer(5, Some(eps_buf.metal_buffer()), 0);
        encoder.set_buffer(6, Some(has_weight_buf.metal_buffer()), 0);
        encoder.set_buffer(7, Some(has_bias_buf.metal_buffer()), 0);

        let grid_size = metal::MTLSize::new(batch_size as u64, 1, 1);
        let threadgroup_size = if normalized_size <= 256 {
            metal::MTLSize::new(1, 1, 1)
        } else {
            metal::MTLSize::new(256, 1, 1)
        };

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // NOTE: No commit here - the Commands batching system handles this automatically
        // Commands will commit when batch size is reached or when data is read to CPU
        // This matches Candle's lazy batching strategy

        self.new_from_pool(
            BufferHandle::Metal(unsafe { std::mem::transmute(result_buf) }),
            self.shape().clone(),
        )
    }

    /// CPU implementation of layer normalization
    fn layer_norm_cpu(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor<T>>,
        bias: Option<&Tensor<T>>,
        eps: f32,
    ) -> TensorResult<Self> {
        // Currently only f16 is supported
        if false {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input_data = self.sync_and_read();
        let input_f16: Vec<f16> = unsafe { std::mem::transmute(input_data) };
        let normalized_size: usize = normalized_shape.iter().product();
        let batch_size = self.numel() / normalized_size;

        let mut output = vec![f16::ZERO; self.numel()];

        let weight_vec = weight.map(|w| {
            let data = w.sync_and_read();
            let f16_data: Vec<f16> = unsafe { std::mem::transmute(data) };
            f16_data
        });
        let bias_vec = bias.map(|b| {
            let data = b.sync_and_read();
            let f16_data: Vec<f16> = unsafe { std::mem::transmute(data) };
            f16_data
        });

        for batch_idx in 0..batch_size {
            let offset = batch_idx * normalized_size;
            let slice = &input_f16[offset..offset + normalized_size];

            // Compute mean
            let sum: f32 = slice.iter().map(|&x| x.to_f32()).sum();
            let mean = sum / normalized_size as f32;

            // Compute variance
            let sq_sum: f32 = slice
                .iter()
                .map(|&x| {
                    let diff = x.to_f32() - mean;
                    diff * diff
                })
                .sum();
            let variance = sq_sum / normalized_size as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();

            // Normalize and apply affine transformation
            for i in 0..normalized_size {
                let mut normalized = (slice[i].to_f32() - mean) * inv_std;

                if let Some(ref w) = weight_vec {
                    normalized *= w[i].to_f32();
                }
                if let Some(ref b) = bias_vec {
                    normalized += b[i].to_f32();
                }

                output[offset + i] = f16::from_f32(normalized);
            }
        }

        let output_t: Vec<T> = unsafe { std::mem::transmute(output) };
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, output_t, self.dims().to_vec()),
            _ => Tensor::from_vec(output_t, self.dims().to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn get_test_device() -> Device {
        Device::default_metal().unwrap_or(Device::CPU)
    }

    #[test]
    fn test_layer_norm_basic() {
        let device = get_test_device();

        // Create a simple 2x3 tensor
        let x = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(
                dev,
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(2.0),
                    f16::from_f32(3.0),
                    f16::from_f32(4.0),
                    f16::from_f32(5.0),
                    f16::from_f32(6.0),
                ],
                vec![2, 3],
            )
            .unwrap(),
            _ => Tensor::from_vec(
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(2.0),
                    f16::from_f32(3.0),
                    f16::from_f32(4.0),
                    f16::from_f32(5.0),
                    f16::from_f32(6.0),
                ],
                vec![2, 3],
            )
            .unwrap(),
        };

        // Normalize over last dimension (size 3)
        let result = x.layer_norm(vec![3], None, None, 1e-5).unwrap();
        let values = result.sync_and_read();

        // Each row should have mean ≈ 0 and std ≈ 1
        for batch in 0..2 {
            let offset = batch * 3;
            let slice = &values[offset..offset + 3];

            // Check mean is close to 0
            let mean: f32 = slice.iter().map(|&x| x.to_f32()).sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-3, "Mean should be close to 0, got {}", mean);

            // Check std is close to 1
            let variance: f32 = slice
                .iter()
                .map(|&x| {
                    let diff = x.to_f32() - mean;
                    diff * diff
                })
                .sum::<f32>()
                / 3.0;
            let std = variance.sqrt();
            assert!(
                (std - 1.0).abs() < 1e-1,
                "Std should be close to 1, got {}",
                std
            );
        }
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let device = get_test_device();

        let (x, weight, bias) = match &device {
            Device::Metal(dev) => (
                Tensor::from_vec_gpu(
                    dev,
                    vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ],
                    vec![2, 2],
                )
                .unwrap(),
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(2.0), f16::from_f32(3.0)],
                    vec![2],
                )
                .unwrap(),
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.5), f16::from_f32(1.0)],
                    vec![2],
                )
                .unwrap(),
            ),
            _ => (
                Tensor::from_vec(
                    vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ],
                    vec![2, 2],
                )
                .unwrap(),
                Tensor::from_vec(
                    vec![f16::from_f32(2.0), f16::from_f32(3.0)],
                    vec![2],
                )
                .unwrap(),
                Tensor::from_vec(
                    vec![f16::from_f32(0.5), f16::from_f32(1.0)],
                    vec![2],
                )
                .unwrap(),
            ),
        };

        let result = x
            .layer_norm(vec![2], Some(&weight), Some(&bias), 1e-5)
            .unwrap();
        let values = result.sync_and_read();

        // Verify output shape
        assert_eq!(values.len(), 4);

        // Values should be transformed by weight and bias
        for &val in &values {
            let f = val.to_f32();
            assert!(f.is_finite(), "Value should be finite");
        }
    }

    #[test]
    fn test_layer_norm_3d() {
        let device = get_test_device();

        // Test with 3D tensor: [2, 3, 4]
        let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32 / 10.0)).collect();
        let x = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, data, vec![2, 3, 4]).unwrap(),
            _ => Tensor::from_vec(data, vec![2, 3, 4]).unwrap(),
        };

        // Normalize over last dimension
        let result = x.layer_norm(vec![4], None, None, 1e-5).unwrap();

        assert_eq!(result.dims(), &[2, 3, 4]);

        // Check that each normalized slice has mean ≈ 0 and std ≈ 1
        let values = result.sync_and_read();
        for batch in 0..6 {
            // 2 * 3 = 6 batches
            let offset = batch * 4;
            let slice = &values[offset..offset + 4];

            let mean: f32 = slice.iter().map(|&x| x.to_f32()).sum::<f32>() / 4.0;
            assert!(
                mean.abs() < 1e-2,
                "Mean should be close to 0, got {} for batch {}",
                mean,
                batch
            );
        }
    }

    #[test]
    fn test_layer_norm_cpu() {
        let _device = Device::CPU;

        let x = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        let result = x.layer_norm(vec![3], None, None, 1e-5).unwrap();
        let values = result.sync_and_read();

        // Verify normalization
        for batch in 0..2 {
            let offset = batch * 3;
            let slice = &values[offset..offset + 3];

            let mean: f32 = slice.iter().map(|&x| x.to_f32()).sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_mathematical_correctness() {
        // Test RMS normalization with known values
        // Formula: output = (x / rms(x)) * weight
        // where rms(x) = sqrt(mean(x^2) + eps)
        //
        // For input [2, 4, 6, 8] with weight=1.0 and eps=1e-6:
        // mean(x^2) = (4 + 16 + 36 + 64) / 4 = 30
        // rms = sqrt(30 + 1e-6) ≈ 5.477
        // output = [2/5.477, 4/5.477, 6/5.477, 8/5.477] * 1.0
        //        = [0.3652, 0.7305, 1.0957, 1.4609]

        let device = get_test_device();

        let x_data = vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(8.0),
        ];
        let weight_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];

        let x = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, x_data, vec![1, 4]).unwrap(),
            _ => Tensor::from_vec(x_data, vec![1, 4]).unwrap(),
        };

        let weight = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, weight_data, vec![4]).unwrap(),
            _ => Tensor::from_vec(weight_data, vec![4]).unwrap(),
        };

        let result = x.rms_norm(vec![4], &weight, 1e-6).unwrap();
        let values = result.sync_and_read();

        // Expected values from the formula
        let expected = vec![0.3652, 0.7305, 1.0957, 1.4609];

        for i in 0..4 {
            let actual = values[i].to_f32();
            let diff = (actual - expected[i]).abs();
            assert!(
                diff < 0.01,
                "RMS norm mismatch at index {}: expected {:.4}, got {:.4}, diff {:.4}",
                i,
                expected[i],
                actual,
                diff
            );
        }
    }

    #[test]
    fn test_rms_norm_scaling_property() {
        // RMS norm should preserve relative magnitudes
        // If input is [x, 2x, 3x], output should maintain ratios 1:2:3
        let device = get_test_device();

        let x_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ];
        let weight_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];

        let x = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, x_data, vec![1, 3]).unwrap(),
            _ => Tensor::from_vec(x_data, vec![1, 3]).unwrap(),
        };

        let weight = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, weight_data, vec![3]).unwrap(),
            _ => Tensor::from_vec(weight_data, vec![3]).unwrap(),
        };

        let result = x.rms_norm(vec![3], &weight, 1e-6).unwrap();
        let values = result.sync_and_read();

        // Check that ratios are preserved
        let v0 = values[0].to_f32();
        let v1 = values[1].to_f32();
        let v2 = values[2].to_f32();

        let ratio_1_0 = v1 / v0;
        let ratio_2_0 = v2 / v0;

        assert!(
            (ratio_1_0 - 2.0).abs() < 0.01,
            "Ratio v1/v0 should be ~2.0, got {:.4}",
            ratio_1_0
        );
        assert!(
            (ratio_2_0 - 3.0).abs() < 0.01,
            "Ratio v2/v0 should be ~3.0, got {:.4}",
            ratio_2_0
        );
    }

    #[test]
    fn test_rms_norm_weight_scaling() {
        // Weight parameter should scale the normalized output
        let device = get_test_device();

        let x_data = vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(8.0),
        ];

        // Test with different weight values
        let weight_1_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];
        let weight_2_data = vec![
            f16::from_f32(2.0),
            f16::from_f32(2.0),
            f16::from_f32(2.0),
            f16::from_f32(2.0),
        ];

        let x1 = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, x_data.clone(), vec![1, 4]).unwrap(),
            _ => Tensor::from_vec(x_data.clone(), vec![1, 4]).unwrap(),
        };
        let x2 = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, x_data, vec![1, 4]).unwrap(),
            _ => Tensor::from_vec(x_data, vec![1, 4]).unwrap(),
        };

        let weight1 = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, weight_1_data, vec![4]).unwrap(),
            _ => Tensor::from_vec(weight_1_data, vec![4]).unwrap(),
        };
        let weight2 = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, weight_2_data, vec![4]).unwrap(),
            _ => Tensor::from_vec(weight_2_data, vec![4]).unwrap(),
        };

        let result1 = x1.rms_norm(vec![4], &weight1, 1e-6).unwrap();
        let result2 = x2.rms_norm(vec![4], &weight2, 1e-6).unwrap();

        let values1 = result1.sync_and_read();
        let values2 = result2.sync_and_read();

        // result2 should be approximately 2x result1
        for i in 0..4 {
            let v1 = values1[i].to_f32();
            let v2 = values2[i].to_f32();
            let ratio = v2 / v1;

            assert!(
                (ratio - 2.0).abs() < 0.01,
                "Weight scaling: v2/v1 should be ~2.0 at index {}, got {:.4}",
                i,
                ratio
            );
        }
    }

    #[test]
    fn test_rms_norm_deterministic() {
        // Same input should always produce same output
        let device = get_test_device();

        let x_data = vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(8.0),
        ];
        let weight_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];

        let x = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, x_data, vec![1, 4]).unwrap(),
            _ => Tensor::from_vec(x_data, vec![1, 4]).unwrap(),
        };

        let weight = match &device {
            Device::Metal(dev) => Tensor::from_vec_gpu(dev, weight_data, vec![4]).unwrap(),
            _ => Tensor::from_vec(weight_data, vec![4]).unwrap(),
        };

        let result1 = x.rms_norm(vec![4], &weight, 1e-6).unwrap();
        let result2 = x.rms_norm(vec![4], &weight, 1e-6).unwrap();
        let result3 = x.rms_norm(vec![4], &weight, 1e-6).unwrap();

        let values1 = result1.sync_and_read();
        let values2 = result2.sync_and_read();
        let values3 = result3.sync_and_read();

        for i in 0..4 {
            assert_eq!(
                values1[i], values2[i],
                "RMS norm should be deterministic (result1 vs result2 at index {})",
                i
            );
            assert_eq!(
                values2[i], values3[i],
                "RMS norm should be deterministic (result2 vs result3 at index {})",
                i
            );
        }
    }
}
