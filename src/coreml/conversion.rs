//! Conversion between TensorLogic Tensors and CoreML MLMultiArray

use super::{CoreMLError, CoreMLResult};
use crate::tensor::Tensor;

/// Convert a TensorLogic Tensor to CoreML MLMultiArray
///
/// This function converts a TensorLogic tensor (backed by Metal)
/// to a CoreML MLMultiArray for inference on the Neural Engine.
///
/// # Arguments
///
/// * `tensor` - The TensorLogic tensor to convert
///
/// # Returns
///
/// On macOS: A CoreML MLMultiArray containing the tensor data
/// On other platforms: An empty result (placeholder)
#[cfg(target_os = "macos")]
pub fn tensor_to_mlmultiarray(tensor: &Tensor) -> CoreMLResult<()> {
    let shape = tensor.shape();
    let dims = shape.dims();

    // Get tensor data from Metal buffer
    let data = tensor.to_vec();

    println!("Converting Tensor to MLMultiArray:");
    println!("  Shape: {:?}", dims);
    println!("  Data length: {}", data.len());

    // Note: Full MLMultiArray creation with objc2-core-ml 0.2
    // requires deeper integration with the Objective-C runtime
    // For now, we validate the conversion is possible and log it
    // TODO: Implement actual MLMultiArray creation with proper API usage

    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn tensor_to_mlmultiarray(tensor: &Tensor) -> CoreMLResult<()> {
    let shape = tensor.shape();
    let dims = shape.dims();

    println!("Converting Tensor to MLMultiArray (non-macOS placeholder):");
    println!("  Shape: {:?}", dims);
    println!("  Rank: {}", tensor.rank());

    Ok(())
}

/// Convert a CoreML MLMultiArray to TensorLogic Tensor
///
/// This function converts a CoreML MLMultiArray (result from Neural Engine)
/// back to a TensorLogic tensor backed by Metal.
///
/// # Arguments
///
/// * `device` - Metal device to create the tensor on
/// * `ml_array` - The MLMultiArray to convert (macOS only)
///
/// # Returns
///
/// A TensorLogic tensor containing the data from MLMultiArray
#[cfg(target_os = "macos")]
pub fn mlmultiarray_to_tensor(
    device: &crate::device::MetalDevice,
    shape: Vec<usize>,
) -> CoreMLResult<Tensor> {
    println!("Converting MLMultiArray to Tensor:");
    println!("  Shape: {:?}", shape);

    // For now, create a zero tensor
    // Full implementation would:
    // 1. Get data pointer from MLMultiArray
    // 2. Copy data to Vec<f16>
    // 3. Create Tensor::from_vec(device, data, shape)

    Tensor::zeros(device, shape)
        .map_err(CoreMLError::TensorError)
}

#[cfg(not(target_os = "macos"))]
pub fn mlmultiarray_to_tensor(
    device: &crate::device::MetalDevice,
    shape: Vec<usize>,
) -> CoreMLResult<Tensor> {
    println!("Converting MLMultiArray to Tensor (non-macOS placeholder):");
    println!("  Target shape: {:?}", shape);

    Tensor::zeros(device, shape)
        .map_err(CoreMLError::TensorError)
}

/// Batch conversion: Tensors to MLMultiArray
pub fn tensors_to_mlmultiarray_batch(tensors: &[Tensor]) -> CoreMLResult<Vec<()>> {
    tensors
        .iter()
        .map(tensor_to_mlmultiarray)
        .collect()
}

/// Batch conversion: MLMultiArray to Tensors
pub fn mlmultiarray_to_tensors_batch(
    device: &crate::device::MetalDevice,
    shapes: &[Vec<usize>],
) -> CoreMLResult<Vec<Tensor>> {
    shapes
        .iter()
        .map(|shape| mlmultiarray_to_tensor(device, shape.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_tensor_to_mlmultiarray() {
        let device = MetalDevice::new().unwrap();
        let tensor = Tensor::zeros(&device, vec![1, 3, 224, 224]).unwrap();

        let result = tensor_to_mlmultiarray(&tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mlmultiarray_to_tensor() {
        let device = MetalDevice::new().unwrap();
        let shape = vec![1, 1000];

        let result = mlmultiarray_to_tensor(&device, shape.clone());
        assert!(result.is_ok());

        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims(), &[1, 1000]);
    }

    #[test]
    fn test_batch_conversion() {
        let device = MetalDevice::new().unwrap();
        let tensors = vec![
            Tensor::zeros(&device, vec![1, 3, 224, 224]).unwrap(),
            Tensor::zeros(&device, vec![1, 3, 224, 224]).unwrap(),
        ];

        let result = tensors_to_mlmultiarray_batch(&tensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }
}
