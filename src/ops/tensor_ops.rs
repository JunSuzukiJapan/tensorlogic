//! Tensor manipulation operations (concat, transpose, permute, split)

use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, Tensor};
use half::f16;

impl Tensor {
    /// Concatenate tensors along a specified dimension
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensor references to concatenate
    /// * `dim` - Dimension along which to concatenate (0-indexed)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::zeros(&device, vec![2, 3]).unwrap();
    /// let b = Tensor::zeros(&device, vec![2, 3]).unwrap();
    /// let c = Tensor::concat(&[&a, &b], 0).unwrap(); // Shape: [4, 3]
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
        let metal_buf = MetalBuffer::from_f16_slice(device.metal_device(), &result_data)?;

        Tensor::new(
            BufferHandle::Metal(metal_buf),
            output_shape.into(),
            Device::Metal(device),
        )
    }

    fn concat_cpu(tensors: &[&Tensor], dim: usize, output_shape: Vec<usize>) -> TensorResult<Self> {
        let total_elements: usize = output_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_elements);

        // Calculate chunk size (elements after the concat dimension)
        let chunk_size: usize = output_shape[dim + 1..].iter().product();
        let num_chunks: usize = output_shape[..dim].iter().product();

        // Concatenate along the specified dimension
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

        Tensor::from_vec(result_data, output_shape)
    }

    /// Transpose a 2D tensor (swap dimensions 0 and 1)
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::zeros(&device, vec![2, 3]).unwrap();
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
    /// let a = Tensor::zeros(&device, vec![2, 3, 4]).unwrap();
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
        // For now, use CPU implementation with GPU memory
        // TODO: Implement proper Metal kernel for permute
        self.permute_cpu(dims)
    }

    fn permute_cpu(&self, dims: &[usize]) -> TensorResult<Self> {
        let input_data = self.to_vec();
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

        match self.device() {
            Device::Metal(dev) => Tensor::from_vec_metal(dev, output_data, output_shape),
            _ => Tensor::from_vec(output_data, output_shape),
        }
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

        let c = Tensor::concat(&[&a, &b], 0).unwrap();

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

        let c = Tensor::concat(&[&a, &b], 1).unwrap();

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

        let d = Tensor::concat(&[&a, &b, &c], 0).unwrap();

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
}
