//! Conversion between TensorLogic Tensors and CoreML MLMultiArray

use super::{CoreMLError, CoreMLResult};
use crate::tensor::{TensorIO, TensorCreation, TensorAccessors};
use crate::tensor::Tensor;

/// Convert a TensorLogic Tensor to CoreML MLMultiArray
///
/// This function converts a TensorLogic tensor (backed by Metal)
/// to a CoreML MLMultiArray for inference on the Neural Engine.
///
/// # Implementation Notes
///
/// The objc2-core-ml 0.2 API requires:
/// 1. Create NSArray for shape using NSNumber
/// 2. Call MLMultiArray::initWithShape_dataType_error with Float16 (65552)
/// 3. Use getMutableBytesWithHandler to access and fill the buffer
/// 4. Copy f16 data from tensor.sync_and_read() into the MLMultiArray buffer
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
pub fn tensor_to_mlmultiarray(tensor: &Tensor) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLMultiArray>> {
    use objc2::rc::Retained;
    use objc2_core_ml::MLMultiArray;
    use objc2_foundation::{NSArray, NSNumber};

    let shape = tensor.shape();
    let dims = shape.dims();

    // Get tensor data from Metal buffer
    let data = tensor.sync_and_read();

    println!("Converting Tensor to MLMultiArray:");
    println!("  Shape: {:?}", dims);
    println!("  Data length: {}", data.len());

    // Validate shape and data consistency
    let expected_size: usize = dims.iter().product();
    if data.len() != expected_size {
        return Err(CoreMLError::ConversionError(
            format!("Shape {:?} doesn't match data length {}", dims, data.len())
        ));
    }

    // Create NSArray for shape
    let shape_numbers: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|&dim| NSNumber::new_usize(dim))
        .collect();
    let shape_array = NSArray::from_vec(shape_numbers);

    // Create MLMultiArray with Float16 data type
    use objc2_core_ml::MLMultiArrayDataType;
    use objc2::ClassType;

    let multi_array = unsafe {
        let allocated = MLMultiArray::alloc();
        MLMultiArray::initWithShape_dataType_error(
            allocated,
            &shape_array,
            MLMultiArrayDataType::Float16,
        )
        .map_err(|_e| {
            CoreMLError::ConversionError("Failed to create MLMultiArray".to_string())
        })?
    };

    // Copy f16 data into MLMultiArray using dataPointer
    // Note: dataPointer is deprecated but simpler than block-based handlers
    #[allow(deprecated)]
    unsafe {
        let data_ptr = multi_array.dataPointer();
        let f16_ptr = data_ptr.as_ptr() as *mut half::f16;

        // Copy data element by element
        for (i, &value) in data.iter().enumerate() {
            *f16_ptr.add(i) = value;
        }
    }

    println!("  MLMultiArray created and populated successfully");
    Ok(multi_array)
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

/// Convert MLMultiArray to MLFeatureValue
///
/// This creates an MLFeatureValue from an MLMultiArray for use with CoreML prediction.
///
/// # Arguments
///
/// * `ml_array` - The MLMultiArray to wrap
///
/// # Returns
///
/// MLFeatureValue containing the array
#[cfg(target_os = "macos")]
pub fn mlmultiarray_to_feature_value(
    ml_array: &objc2_core_ml::MLMultiArray
) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLFeatureValue>> {
    use objc2_core_ml::MLFeatureValue;

    println!("Converting MLMultiArray to MLFeatureValue");

    let feature_value = unsafe {
        MLFeatureValue::featureValueWithMultiArray(ml_array)
    };

    println!("  MLFeatureValue created successfully");
    Ok(feature_value)
}

/// Extract MLMultiArray from MLFeatureValue
///
/// # Arguments
///
/// * `feature_value` - The MLFeatureValue containing an array
///
/// # Returns
///
/// The MLMultiArray from the feature value
#[cfg(target_os = "macos")]
pub fn feature_value_to_mlmultiarray(
    feature_value: &objc2_core_ml::MLFeatureValue
) -> CoreMLResult<objc2::rc::Retained<objc2_core_ml::MLMultiArray>> {
    println!("Extracting MLMultiArray from MLFeatureValue");

    let ml_array = unsafe {
        feature_value.multiArrayValue()
            .ok_or_else(|| CoreMLError::ConversionError(
                "FeatureValue does not contain MLMultiArray".to_string()
            ))?
    };

    println!("  MLMultiArray extracted successfully");
    Ok(ml_array)
}

/// Convert a CoreML MLMultiArray to TensorLogic Tensor
///
/// This function converts a CoreML MLMultiArray (result from Neural Engine)
/// back to a TensorLogic tensor backed by Metal.
///
/// # Implementation Notes
///
/// 1. Get data pointer from ml_array.dataPointer()
/// 2. Cast to *const f16 based on dataType
/// 3. Copy data into Vec<f16>
/// 4. Create Tensor::from_vec(device, data, shape)
///
/// # Arguments
///
/// * `device` - Metal device to create the tensor on
/// * `ml_array` - The MLMultiArray to convert
/// * `shape` - Expected output shape
///
/// # Returns
///
/// A TensorLogic tensor containing the data from MLMultiArray
#[cfg(target_os = "macos")]
pub fn mlmultiarray_to_tensor(
    device: &crate::device::MetalDevice,
    ml_array: &objc2_core_ml::MLMultiArray,
    shape: Vec<usize>,
) -> CoreMLResult<Tensor> {
    println!("Converting MLMultiArray to Tensor:");
    println!("  Shape: {:?}", shape);

    // Get data pointer from MLMultiArray
    // Note: dataPointer is deprecated but simpler than block-based handlers
    #[allow(deprecated)]
    let data_ptr = unsafe { ml_array.dataPointer() };

    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Copy data from MLMultiArray to Vec<f16>
    let data: Vec<half::f16> = unsafe {
        let f16_ptr = data_ptr.as_ptr() as *const half::f16;
        std::slice::from_raw_parts(f16_ptr, total_elements).to_vec()
    };

    println!("  Copied {} f16 elements from MLMultiArray", data.len());

    // Create Tensor from data using Metal-backed constructor
    Tensor::from_vec_gpu(device, data, shape).map_err(CoreMLError::TensorError)
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
#[cfg(target_os = "macos")]
pub fn tensors_to_mlmultiarray_batch(
    tensors: &[Tensor]
) -> CoreMLResult<Vec<objc2::rc::Retained<objc2_core_ml::MLMultiArray>>> {
    tensors
        .iter()
        .map(tensor_to_mlmultiarray)
        .collect()
}

#[cfg(not(target_os = "macos"))]
pub fn tensors_to_mlmultiarray_batch(tensors: &[Tensor]) -> CoreMLResult<Vec<()>> {
    tensors
        .iter()
        .map(tensor_to_mlmultiarray)
        .collect()
}

/// Batch conversion: MLMultiArray to Tensors (non-macOS placeholder)
#[cfg(not(target_os = "macos"))]
pub fn mlmultiarray_to_tensors_batch(
    device: &crate::device::MetalDevice,
    shapes: &[Vec<usize>],
) -> CoreMLResult<Vec<Tensor>> {
    shapes
        .iter()
        .map(|shape| mlmultiarray_to_tensor(device, shape.clone()))
        .collect()
}

/// Batch conversion: MLMultiArray to Tensors (macOS version)
#[cfg(target_os = "macos")]
pub fn mlmultiarray_to_tensors_batch(
    device: &crate::device::MetalDevice,
    ml_arrays: &[&objc2_core_ml::MLMultiArray],
    shapes: &[Vec<usize>],
) -> CoreMLResult<Vec<Tensor>> {
    ml_arrays
        .iter()
        .zip(shapes.iter())
        .map(|(ml_array, shape)| mlmultiarray_to_tensor(device, ml_array, shape.clone()))
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
        #[cfg(target_os = "macos")]
        assert!(result.is_ok());
        #[cfg(not(target_os = "macos"))]
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mlmultiarray_to_feature_value() {
        let device = MetalDevice::new().unwrap();
        let tensor = Tensor::ones(&device, vec![1, 10]).unwrap();

        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        let feature_value = mlmultiarray_to_feature_value(&ml_array).unwrap();

        // Verify we can extract it back
        let _recovered = feature_value_to_mlmultiarray(&feature_value).unwrap();
        // Retained<T> is never null, just verify we got it successfully
    }

    #[test]
    #[cfg(not(target_os = "macos"))]  // Non-macOS version has different signature
    fn test_mlmultiarray_to_tensor_placeholder() {
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
