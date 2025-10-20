//! CoreML Model wrapper for Neural Engine inference

use super::{CoreMLError, CoreMLResult};
use crate::tensor::Tensor;
use std::path::Path;

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2_core_ml::MLModel;
#[cfg(target_os = "macos")]
use objc2_foundation::NSURL;

/// CoreML Model wrapper
///
/// This struct wraps a CoreML model and provides methods for loading
/// and executing inference on the Neural Engine.
pub struct CoreMLModel {
    /// Model name/identifier
    name: String,
    /// Model path
    path: String,
    /// Input shape expected by the model
    input_shape: Vec<usize>,
    /// Output shape produced by the model
    output_shape: Vec<usize>,
    /// The actual MLModel instance (macOS only)
    #[cfg(target_os = "macos")]
    ml_model: Option<Retained<MLModel>>,
}

// Manual Debug implementation since MLModel doesn't implement Debug
impl std::fmt::Debug for CoreMLModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModel")
            .field("name", &self.name)
            .field("path", &self.path)
            .field("input_shape", &self.input_shape)
            .field("output_shape", &self.output_shape)
            .finish()
    }
}

impl CoreMLModel {
    /// Load a CoreML model from a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .mlmodel or .mlmodelc file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = CoreMLModel::load("model.mlmodelc")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> CoreMLResult<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Check if file exists
        if !path.as_ref().exists() {
            return Err(CoreMLError::ModelLoadError(format!(
                "Model file not found: {}",
                path_str
            )));
        }

        let name = path.as_ref()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        #[cfg(target_os = "macos")]
        {
            // macOS: Load actual MLModel using objc2-core-ml
            use objc2_foundation::NSString;

            let path_nsstring = NSString::from_str(&path_str);
            let url = unsafe {
                NSURL::fileURLWithPath(&path_nsstring)
            };

            // Try to load MLModel
            let ml_model_result = unsafe {
                MLModel::modelWithContentsOfURL_error(&url)
            };

            match ml_model_result {
                Ok(ml_model) => {
                    // TODO: Extract input/output shapes from model description
                    // For now, use default ImageNet shapes
                    Ok(CoreMLModel {
                        name,
                        path: path_str,
                        input_shape: vec![1, 3, 224, 224],
                        output_shape: vec![1, 1000],
                        ml_model: Some(ml_model),
                    })
                }
                Err(_) => {
                    Err(CoreMLError::ModelLoadError(
                        "Failed to load MLModel".to_string()
                    ))
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: Create placeholder model
            Ok(CoreMLModel {
                name,
                path: path_str,
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 1000],
            })
        }
    }

    /// Create a CoreML model with custom input/output shapes
    pub fn with_shapes(
        name: String,
        path: String,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        CoreMLModel {
            name,
            path,
            input_shape,
            output_shape,
            #[cfg(target_os = "macos")]
            ml_model: None,
        }
    }

    /// Get the expected input shape
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Get the output shape
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the model path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Run inference on the Neural Engine
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Output tensor from the model
    ///
    /// # Example
    ///
    /// ```ignore
    /// let output = model.predict(input_tensor)?;
    /// ```
    pub fn predict(&self, input: &Tensor) -> CoreMLResult<Tensor> {
        // Validate input shape
        let input_dims = input.shape().dims();
        if input_dims != self.input_shape {
            return Err(CoreMLError::InvalidInputShape {
                expected: self.input_shape.clone(),
                actual: input_dims.to_vec(),
            });
        }

        use crate::device::MetalDevice;
        let device = MetalDevice::new().map_err(|e| CoreMLError::TensorError(e))?;

        #[cfg(target_os = "macos")]
        {
            if let Some(ref _ml_model) = self.ml_model {
                use super::conversion::tensor_to_mlmultiarray;

                println!("Running CoreML inference on Neural Engine...");
                println!("  Model: {}", self.name);
                println!("  Input shape: {:?}", input_dims);
                println!("  Output shape: {:?}", self.output_shape);

                // Convert Tensor to MLMultiArray (validation)
                let _ = tensor_to_mlmultiarray(input)?;

                // TODO: Full MLModel.prediction() integration
                // The objc2-core-ml API differs from expected
                // For now, we demonstrate the conversion layer works
                // and return a placeholder output tensor

                println!("  Note: Full MLModel.prediction() integration pending");
                println!("  Returning zero tensor as placeholder output");

                Tensor::zeros(&device, self.output_shape.clone())
                    .map_err(CoreMLError::TensorError)
            } else {
                // No MLModel loaded, return zero tensor
                println!("No MLModel loaded, returning zero tensor");
                Tensor::zeros(&device, self.output_shape.clone())
                    .map_err(CoreMLError::TensorError)
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: Return dummy output tensor
            println!("Running CoreML inference (non-macOS placeholder)...");
            println!("  Model: {}", self.name);
            println!("  Input shape: {:?}", input_dims);
            println!("  Output shape: {:?}", self.output_shape);

            Tensor::zeros(&device, self.output_shape.clone())
                .map_err(CoreMLError::TensorError)
        }
    }

    /// Run batch inference
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of input tensors
    ///
    /// # Returns
    ///
    /// Batch of output tensors
    pub fn predict_batch(&self, inputs: &[Tensor]) -> CoreMLResult<Vec<Tensor>> {
        inputs
            .iter()
            .map(|input| self.predict(input))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_coreml_model_creation() {
        let model = CoreMLModel::with_shapes(
            "test_model".to_string(),
            "test.mlmodelc".to_string(),
            vec![1, 3, 224, 224],
            vec![1, 1000],
        );

        assert_eq!(model.name(), "test_model");
        assert_eq!(model.input_shape(), &[1, 3, 224, 224]);
        assert_eq!(model.output_shape(), &[1, 1000]);
    }

    #[test]
    fn test_coreml_model_predict() {
        let device = MetalDevice::new().unwrap();
        let model = CoreMLModel::with_shapes(
            "test_model".to_string(),
            "test.mlmodelc".to_string(),
            vec![1, 3, 224, 224],
            vec![1, 1000],
        );

        let input = Tensor::zeros(&device, vec![1, 3, 224, 224]).unwrap();
        let output = model.predict(&input);

        assert!(output.is_ok());
        let output_tensor = output.unwrap();
        assert_eq!(output_tensor.shape().dims(), &[1, 1000]);
    }

    #[test]
    fn test_coreml_model_invalid_input_shape() {
        let device = MetalDevice::new().unwrap();
        let model = CoreMLModel::with_shapes(
            "test_model".to_string(),
            "test.mlmodelc".to_string(),
            vec![1, 3, 224, 224],
            vec![1, 1000],
        );

        let input = Tensor::zeros(&device, vec![1, 3, 128, 128]).unwrap();
        let output = model.predict(&input);

        assert!(output.is_err());
        if let Err(CoreMLError::InvalidInputShape { expected, actual }) = output {
            assert_eq!(expected, vec![1, 3, 224, 224]);
            assert_eq!(actual, vec![1, 3, 128, 128]);
        } else {
            panic!("Expected InvalidInputShape error");
        }
    }
}
