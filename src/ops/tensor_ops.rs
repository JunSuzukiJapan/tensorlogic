//! Tensor manipulation operations (concat, transpose, permute, split)

use crate::device::{Device, MetalBuffer};
use crate::tensor::FloatType;
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

impl<T: FloatType> Tensor<T> {
    /// Concatenate tensors along a specified dimension
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensor references to concatenate
    /// * `dim` - Dimension along which to concatenate (0-indexed)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();
    /// let b = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();
    /// let c = Tensor::<f16>::concat(&[&a, &b], 0).unwrap(); // Shape: [4, 3]
    /// ```
    pub fn concat(tensors: &[&Tensor], dim: usize) -> TensorResult<Self> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidOperation(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        // Verify all tensors have the same number of dimensions
        let ndim = tensors[0].dims().len();
        if dim >= ndim {
            return Err(TensorError::InvalidDimension { dim });
        }

        // Verify all tensors have compatible shapes
        for tensor in tensors.iter() {
            if tensor.dims().len() != ndim {
                return Err(TensorError::InvalidOperation(
                    "All tensors must have the same number of dimensions".to_string(),
                ));
            }

            for (i, &size) in tensor.dims().iter().enumerate() {
                if i != dim && size != tensors[0].dims()[i] {
                    return Err(TensorError::ShapeMismatch {
                        expected: tensors[0].dims().to_vec(),
                        actual: tensor.dims().to_vec(),
                    });
                }
            }
        }

        // Calculate output shape
        let mut output_shape = tensors[0].dims().to_vec();
        output_shape[dim] = tensors.iter().map(|t| t.dims()[dim]).sum();

        // Use Metal if all tensors are on Metal, otherwise CPU
        let all_metal = tensors.iter().all(|t| t.buffer().is_metal());

        if all_metal {
            Self::concat_metal(tensors, dim, output_shape)
        } else {
            Self::concat_cpu(tensors, dim, output_shape)
        }
    }

    fn concat_metal(tensors: &[&Tensor], dim: usize, output_shape: Vec<usize>) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        let device = match tensors[0].device() {
            Device::Metal(dev) => dev.clone(),
            _ => return Err(TensorError::DeviceConversionError("Not on Metal device".to_string())),
        };

        // Calculate total number of elements
        let total_elements: usize = output_shape.iter().product();

        // For now, use CPU approach with GPU memory
        // This copies data to CPU, concatenates, then copies back
        // TODO: Implement proper Metal kernel for concat
        let mut result_data = Vec::with_capacity(total_elements);

        // Calculate stride for each dimension
        let mut strides = vec![1; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * output_shape[i + 1];
        }

        // Concatenate along the specified dimension
        let chunk_size: usize = output_shape[dim + 1..].iter().product();
        let num_chunks: usize = output_shape[..dim].iter().product();

        for chunk_idx in 0..num_chunks {
            for tensor in tensors {
                let data = tensor.to_vec();
                let tensor_dim_size = tensor.dims()[dim];

                for i in 0..tensor_dim_size {
                    let start = (chunk_idx * tensor_dim_size + i) * chunk_size;
                    let end = start + chunk_size;
                    result_data.extend_from_slice(&data[start..end]);
                }
            }
        }

        // Write data to Metal buffer
        let metal_buf_f16 = MetalBuffer::from_f16_slice(device.metal_device(), &result_data)?;
        let metal_buf: MetalBuffer<T> = unsafe { std::mem::transmute(metal_buf_f16) };

        Tensor::new(
            BufferHandle::Metal(metal_buf),
            output_shape.into(),
            Device::Metal(device),
        )
    }

    fn concat_cpu(tensors: &[&Tensor], dim: usize, output_shape: Vec<usize>) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let total_elements: usize = output_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_elements);

        // Calculate chunk size (elements after the concat dimension)
        let chunk_size: usize = output_shape[dim + 1..].iter().product();
        let num_chunks: usize = output_shape[..dim].iter().product();

        // Concatenate along the specified dimension
        for chunk_idx in 0..num_chunks {
            for tensor in tensors {
                let data_t = tensor.to_vec();
                let data: Vec<f16> = unsafe { std::mem::transmute(data_t) };
                let tensor_dim_size = tensor.dims()[dim];

                for i in 0..tensor_dim_size {
                    let start = (chunk_idx * tensor_dim_size + i) * chunk_size;
                    let end = start + chunk_size;
                    result_data.extend_from_slice(&data[start..end]);
                }
            }
        }

        let result_t: Vec<T> = unsafe { std::mem::transmute(result_data) };
        Tensor::from_vec(result_t, output_shape)
    }

    /// Transpose a 2D tensor (swap dimensions 0 and 1)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![2, 3]).unwrap();
    /// let b = a.transpose().unwrap(); // Shape: [3, 2]
    /// ```
    pub fn transpose(&self) -> TensorResult<Self> {
        if self.dims().len() != 2 {
            return Err(TensorError::InvalidOperation(
                "transpose() only works on 2D tensors. Use permute() for higher dimensions".to_string(),
            ));
        }

        self.permute(vec![1, 0])
    }

    /// Permute (rearrange) dimensions of a tensor
    ///
    /// # Arguments
    /// * `dims` - New order of dimensions (0-indexed)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![2, 3, 4]).unwrap();
    /// let b = a.permute(vec![2, 0, 1]).unwrap(); // Shape: [4, 2, 3]
    /// ```
    pub fn permute(&self, dims: Vec<usize>) -> TensorResult<Self> {
        // Validate dims
        if dims.len() != self.dims().len() {
            return Err(TensorError::InvalidOperation(
                format!("permute dims length {} must match tensor ndim {}", dims.len(), self.dims().len()),
            ));
        }

        let mut sorted_dims = dims.clone();
        sorted_dims.sort();
        for (i, &dim) in sorted_dims.iter().enumerate() {
            if dim != i {
                return Err(TensorError::InvalidOperation(
                    "permute dims must be a permutation of 0..ndim".to_string(),
                ));
            }
        }

        if self.buffer().is_metal() {
            self.permute_metal(&dims)
        } else {
            self.permute_cpu(&dims)
        }
    }

    fn permute_metal(&self, dims: &[usize]) -> TensorResult<Self> {
        // Currently only f16 is supported for Metal operations
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "Metal operations currently only support f16".to_string()
            ));
        }

        // For now, use CPU implementation with GPU memory
        // TODO: Implement proper Metal kernel for permute
        self.permute_cpu(dims)
    }

    fn permute_cpu(&self, dims: &[usize]) -> TensorResult<Self> {
        // Currently only f16 is supported
        if !T::is_f16() {
            return Err(TensorError::InvalidOperation(
                "CPU operations currently only support f16".to_string()
            ));
        }

        let input_data_t = self.to_vec();
        let input_data: Vec<f16> = unsafe { std::mem::transmute(input_data_t) };
        let input_shape = self.dims();

        // Calculate output shape
        let output_shape: Vec<usize> = dims.iter().map(|&i| input_shape[i]).collect();
        let total_elements: usize = output_shape.iter().product();

        let mut output_data = vec![f16::ZERO; total_elements];

        // Calculate strides for input and output
        let mut input_strides = vec![1; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        let mut output_strides = vec![1; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // Copy data with dimension permutation
        for out_idx in 0..total_elements {
            // Convert output linear index to multi-dimensional index
            let mut out_coords = vec![0; output_shape.len()];
            let mut remaining = out_idx;
            for i in 0..output_shape.len() {
                out_coords[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            // Map output coordinates to input coordinates
            let mut in_coords = vec![0; input_shape.len()];
            for i in 0..dims.len() {
                in_coords[dims[i]] = out_coords[i];
            }

            // Convert input coordinates to linear index
            let in_idx: usize = in_coords.iter()
                .zip(input_strides.iter())
                .map(|(&coord, &stride)| coord * stride)
                .sum();

            output_data[out_idx] = input_data[in_idx];
        }

        let output_t: Vec<T> = unsafe { std::mem::transmute(output_data) };
        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, output_t, output_shape),
            _ => Tensor::from_vec(output_t, output_shape),
        }
    }

    /// Add a dimension of size 1 at the specified position (unsqueeze)
    ///
    /// # Arguments
    /// * `dim` - Position where to insert the new dimension (0-indexed)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![3, 4]).unwrap();  // Shape: [3, 4]
    /// let b = a.unsqueeze(0).unwrap();  // Shape: [1, 3, 4]
    /// let c = a.unsqueeze(1).unwrap();  // Shape: [3, 1, 4]
    /// let d = a.unsqueeze(2).unwrap();  // Shape: [3, 4, 1]
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> TensorResult<Self> {
        let current_dims = self.shape().dims();
        let rank = current_dims.len();

        // dim can be 0 to rank (inclusive) to allow adding at the end
        if dim > rank {
            return Err(TensorError::InvalidDimension { dim });
        }

        // Create new shape with dimension 1 inserted at position dim
        let mut new_shape = Vec::with_capacity(rank + 1);
        for i in 0..=rank {
            if i == dim {
                new_shape.push(1);
            }
            if i < rank {
                new_shape.push(current_dims[i]);
            }
        }

        // Reshape is a view operation - no data copy
        self.reshape(new_shape)
    }

    /// Remove dimensions of size 1 (squeeze)
    ///
    /// # Arguments
    /// * `dim` - Optional specific dimension to squeeze. If None, squeeze all dimensions of size 1
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![1, 3, 1, 4]).unwrap();  // Shape: [1, 3, 1, 4]
    /// let b = a.squeeze(None).unwrap();  // Shape: [3, 4] - all 1s removed
    /// let c = a.squeeze(Some(0)).unwrap();  // Shape: [3, 1, 4] - only dim 0 removed
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> TensorResult<Self> {
        let current_dims = self.shape().dims();

        let new_shape: Vec<usize> = match dim {
            None => {
                // Remove all dimensions of size 1
                current_dims
                    .iter()
                    .filter(|&&d| d != 1)
                    .copied()
                    .collect()
            }
            Some(d) => {
                // Remove specific dimension if it has size 1
                if d >= current_dims.len() {
                    return Err(TensorError::InvalidDimension { dim: d });
                }

                if current_dims[d] != 1 {
                    return Err(TensorError::InvalidOperation(format!(
                        "Cannot squeeze dimension {} with size {}",
                        d, current_dims[d]
                    )));
                }

                current_dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != d)
                    .map(|(_, &size)| size)
                    .collect()
            }
        };

        // Handle case where all dimensions were 1 -> scalar with shape [1]
        let final_shape = if new_shape.is_empty() {
            vec![1]  // Workaround for numel() bug with empty shapes
        } else {
            new_shape
        };

        self.reshape(final_shape)
    }

    /// Split tensor into chunks along a dimension
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks to split into
    /// * `dim` - Dimension along which to split
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![6, 4]).unwrap();  // Shape: [6, 4]
    /// let parts = a.chunk(3, 0).unwrap();  // Split into 3 chunks along dim 0
    /// // parts[0]: [2, 4], parts[1]: [2, 4], parts[2]: [2, 4]
    /// ```
    pub fn chunk(&self, chunks: usize, dim: usize) -> TensorResult<Vec<Self>> {
        if chunks == 0 {
            return Err(TensorError::InvalidOperation(
                "Number of chunks must be > 0".to_string(),
            ));
        }

        let dims = self.shape().dims();
        if dim >= dims.len() {
            return Err(TensorError::InvalidDimension { dim });
        }

        let dim_size = dims[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks;  // Ceiling division

        self.split(chunk_size, dim)
    }

    /// Split tensor into parts of specified size along a dimension
    ///
    /// # Arguments
    /// * `split_size` - Size of each split (last split may be smaller)
    /// * `dim` - Dimension along which to split
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<f16>::zeros(&device, vec![7, 4]).unwrap();  // Shape: [7, 4]
    /// let parts = a.split(3, 0).unwrap();  // Split into size-3 chunks along dim 0
    /// // parts[0]: [3, 4], parts[1]: [3, 4], parts[2]: [1, 4]
    /// ```
    pub fn split(&self, split_size: usize, dim: usize) -> TensorResult<Vec<Self>> {
        if split_size == 0 {
            return Err(TensorError::InvalidOperation(
                "Split size must be > 0".to_string(),
            ));
        }

        let dims = self.shape().dims();
        if dim >= dims.len() {
            return Err(TensorError::InvalidDimension { dim });
        }

        let dim_size = dims[dim];
        let num_splits = (dim_size + split_size - 1) / split_size;

        let mut result = Vec::with_capacity(num_splits);
        let data = self.to_vec();

        // Calculate strides for indexing
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        for split_idx in 0..num_splits {
            let start = split_idx * split_size;
            let end = (start + split_size).min(dim_size);
            let current_split_size = end - start;

            // Create output shape for this split
            let mut split_dims = dims.to_vec();
            split_dims[dim] = current_split_size;

            let split_numel: usize = split_dims.iter().product();
            let mut split_data = Vec::with_capacity(split_numel);

            // Extract data for this split
            for out_idx in 0..split_numel {
                // Convert output index to coordinates in the split tensor
                let mut coords = vec![0; dims.len()];
                let mut remaining = out_idx;

                for i in 0..dims.len() {
                    let size = if i == dim {
                        current_split_size
                    } else {
                        dims[i]
                    };
                    coords[i] = remaining % size;
                    remaining /= size;
                }

                // Adjust coordinate for the split dimension
                coords[dim] += start;

                // Calculate index in original tensor
                let mut in_idx = 0;
                for (i, &coord) in coords.iter().enumerate() {
                    in_idx += coord * strides[i];
                }

                split_data.push(data[in_idx]);
            }

            // Create split tensor
            let split_tensor = match self.device() {
                Device::Metal(dev) => Tensor::from_vec_metal(dev, split_data, split_dims)?,
                _ => Tensor::from_vec(split_data, split_dims)?,
            };

            result.push(split_tensor);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_concat_dim0() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![1, 3],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![1, 3],
        )
        .unwrap();

        let c = Tensor::<f16>::concat(&[&a, &b], 0).unwrap();

        assert_eq!(c.dims(), &[2, 3]);
        let result = c.to_vec();
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(2.0));
        assert_eq!(result[2], f16::from_f32(3.0));
        assert_eq!(result[3], f16::from_f32(4.0));
        assert_eq!(result[4], f16::from_f32(5.0));
        assert_eq!(result[5], f16::from_f32(6.0));
    }

    #[test]
    fn test_concat_dim1() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            vec![2, 1],
        )
        .unwrap();

        let b = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![2, 1],
        )
        .unwrap();

        let c = Tensor::<f16>::concat(&[&a, &b], 1).unwrap();

        assert_eq!(c.dims(), &[2, 2]);
        let result = c.to_vec();
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(3.0));
        assert_eq!(result[2], f16::from_f32(2.0));
        assert_eq!(result[3], f16::from_f32(4.0));
    }

    #[test]
    fn test_concat_multiple() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(&device, vec![f16::from_f32(1.0)], vec![1, 1]).unwrap();
        let b = Tensor::from_vec_metal(&device, vec![f16::from_f32(2.0)], vec![1, 1]).unwrap();
        let c = Tensor::from_vec_metal(&device, vec![f16::from_f32(3.0)], vec![1, 1]).unwrap();

        let d = Tensor::<f16>::concat(&[&a, &b, &c], 0).unwrap();

        assert_eq!(d.dims(), &[3, 1]);
        let result = d.to_vec();
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(2.0));
        assert_eq!(result[2], f16::from_f32(3.0));
    }

    #[test]
    fn test_transpose_2d() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        let b = a.transpose().unwrap();

        assert_eq!(b.dims(), &[3, 2]);
        let result = b.to_vec();

        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(4.0));
        assert_eq!(result[2], f16::from_f32(2.0));
        assert_eq!(result[3], f16::from_f32(5.0));
        assert_eq!(result[4], f16::from_f32(3.0));
        assert_eq!(result[5], f16::from_f32(6.0));
    }

    #[test]
    fn test_permute_3d() {
        let device = get_test_device();

        // Create a 2x3x4 tensor
        let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32)).collect();
        let a = Tensor::from_vec_metal(&device, data, vec![2, 3, 4]).unwrap();

        // Permute to [4, 2, 3] (dims [2, 0, 1])
        let b = a.permute(vec![2, 0, 1]).unwrap();

        assert_eq!(b.dims(), &[4, 2, 3]);
        assert_eq!(b.numel(), 24);
    }

    #[test]
    fn test_permute_identity() {
        let device = get_test_device();

        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![2, 2],
        )
        .unwrap();

        let b = a.permute(vec![0, 1]).unwrap();

        assert_eq!(b.dims(), &[2, 2]);
        assert_eq!(a.to_vec(), b.to_vec());
    }

    #[test]
    fn test_unsqueeze() {
        let device = crate::device::MetalDevice::new().unwrap();

        // Test unsqueeze on 1D tensor [3] -> [1, 3]
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3],
        )
        .unwrap();

        let b = a.unsqueeze(0).unwrap();
        assert_eq!(b.dims(), &[1, 3]);
        assert_eq!(a.to_vec(), b.to_vec());

        let c = a.unsqueeze(1).unwrap();
        assert_eq!(c.dims(), &[3, 1]);
    }

    #[test]
    fn test_squeeze() {
        let device = crate::device::MetalDevice::new().unwrap();

        // Test squeeze on [1, 3, 1] -> [3]
        let a = Tensor::from_vec_metal(
            &device,
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![1, 3, 1],
        )
        .unwrap();

        let b = a.squeeze(None).unwrap();
        assert_eq!(b.dims(), &[3]);
        assert_eq!(a.to_vec(), b.to_vec());

        // Test squeeze specific dimension
        let c = a.squeeze(Some(0)).unwrap();
        assert_eq!(c.dims(), &[3, 1]);
    }

    #[test]
    fn test_split() {
        let device = crate::device::MetalDevice::new().unwrap();

        // Test split on [6, 4] tensor
        let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32)).collect();
        let a = Tensor::from_vec_metal(&device, data, vec![6, 4]).unwrap();

        // Split into size 2 chunks along dim 0
        let splits = a.split(2, 0).unwrap();
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].dims(), &[2, 4]);
        assert_eq!(splits[1].dims(), &[2, 4]);
        assert_eq!(splits[2].dims(), &[2, 4]);

        // Test split with uneven division
        let b = Tensor::from_vec_metal(
            &device,
            (0..28).map(|i| f16::from_f32(i as f32)).collect(),
            vec![7, 4],
        )
        .unwrap();

        let splits2 = b.split(3, 0).unwrap();
        assert_eq!(splits2.len(), 3);
        assert_eq!(splits2[0].dims(), &[3, 4]);
        assert_eq!(splits2[1].dims(), &[3, 4]);
        assert_eq!(splits2[2].dims(), &[1, 4]);  // Last split is smaller
    }

    #[test]
    fn test_chunk() {
        let device = crate::device::MetalDevice::new().unwrap();

        // Test chunk on [6, 4] tensor
        let data: Vec<f16> = (0..24).map(|i| f16::from_f32(i as f32)).collect();
        let a = Tensor::from_vec_metal(&device, data, vec![6, 4]).unwrap();

        // Split into 3 chunks along dim 0
        let chunks = a.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].dims(), &[2, 4]);
        assert_eq!(chunks[1].dims(), &[2, 4]);
        assert_eq!(chunks[2].dims(), &[2, 4]);

        // Test chunk with uneven division
        let b = Tensor::from_vec_metal(
            &device,
            (0..28).map(|i| f16::from_f32(i as f32)).collect(),
            vec![7, 4],
        )
        .unwrap();

        let chunks2 = b.chunk(3, 0).unwrap();
        assert_eq!(chunks2.len(), 3);
        assert_eq!(chunks2[0].dims(), &[3, 4]);
        assert_eq!(chunks2[1].dims(), &[3, 4]);
        assert_eq!(chunks2[2].dims(), &[1, 4]);  // Last chunk is smaller
    }
}
