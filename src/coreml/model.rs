//! CoreML Model wrapper for Neural Engine inference

use super::{CoreMLError, CoreMLResult};
use crate::tensor::{Tensor, TensorAccessors, TensorCreation, TensorIO};
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
    /// Input feature name
    input_name: String,
    /// Output feature name
    output_name: String,
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
            let ml_model = unsafe {
                MLModel::modelWithContentsOfURL_error(&url)
                    .map_err(|_| CoreMLError::ModelLoadError(
                        "Failed to load MLModel".to_string()
                    ))?
            };

            // Get model description
            let description = unsafe { ml_model.modelDescription() };

            // Extract input information
            let input_dict = unsafe { description.inputDescriptionsByName() };
            let input_keys = unsafe { input_dict.allKeys() };

            if input_keys.count() == 0 {
                return Err(CoreMLError::ModelLoadError(
                    "No input descriptions found".to_string()
                ));
            }

            // Use first input
            let input_name_ns = unsafe { input_keys.objectAtIndex(0) };
            let input_name = input_name_ns.as_ref().to_string();

            // Extract output information
            let output_dict = unsafe { description.outputDescriptionsByName() };
            let output_keys = unsafe { output_dict.allKeys() };

            if output_keys.count() == 0 {
                return Err(CoreMLError::ModelLoadError(
                    "No output descriptions found".to_string()
                ));
            }

            // Use first output
            let output_name_ns = unsafe { output_keys.objectAtIndex(0) };
            let output_name = output_name_ns.as_ref().to_string();

            println!("CoreML Model loaded successfully:");
            println!("  Input: {} -> Output: {}", input_name, output_name);

            // For now, use default ImageNet shapes
            // TODO: Extract actual shapes from MLFeatureDescription
            Ok(CoreMLModel {
                name,
                path: path_str,
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 1000],
                input_name,
                output_name,
                ml_model: Some(ml_model),
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: Create placeholder model
            Ok(CoreMLModel {
                name,
                path: path_str,
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 1000],
                input_name: "input".to_string(),
                output_name: "output".to_string(),
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
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            #[cfg(target_os = "macos")]
            ml_model: None,
        }
    }

    /// Get the input feature name
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Get the output feature name
    pub fn output_name(&self) -> &str {
        &self.output_name
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
            if let Some(ref ml_model) = self.ml_model {
                use super::conversion::{
                    tensor_to_mlmultiarray,
                    mlmultiarray_to_feature_value,
                    mlmultiarray_to_tensor,
                };
                use objc2_foundation::{NSString, NSDictionary};
                use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureProvider};
                use objc2::{ClassType, runtime::ProtocolObject};

                println!("Running CoreML inference on Neural Engine...");
                println!("  Model: {}", self.name);
                println!("  Input: {} → Output: {}", self.input_name, self.output_name);

                // Step 1: Tensor → MLMultiArray
                let ml_array = tensor_to_mlmultiarray(input)?;
                println!("  ✓ MLMultiArray created");

                // Step 2: MLMultiArray → MLFeatureValue
                let feature_value = mlmultiarray_to_feature_value(&ml_array)?;
                println!("  ✓ MLFeatureValue created");

                // Step 3 & 4: Create MLDictionaryFeatureProvider directly
                // We'll use a simpler approach: create a dict with NSString keys
                let input_name_ns = NSString::from_str(&self.input_name);

                // Create a dictionary manually using objc2 msg_send
                use objc2::rc::Retained;
                use objc2::msg_send_id;

                let input_dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> = unsafe {
                    let dict_class = objc2::class!(NSMutableDictionary);
                    let dict: Retained<objc2::runtime::AnyObject> = msg_send_id![dict_class, new];

                    // Set the feature value in the dictionary
                    let _: () = objc2::msg_send![
                        &*dict,
                        setObject: &*feature_value,
                        forKey: &*input_name_ns
                    ];

                    // Convert NSMutableDictionary to NSDictionary
                    std::mem::transmute(dict)
                };
                println!("  ✓ Input dictionary created");

                // Step 4: Create MLDictionaryFeatureProvider
                let input_provider = unsafe {
                    let allocated = MLDictionaryFeatureProvider::alloc();
                    MLDictionaryFeatureProvider::initWithDictionary_error(allocated, &input_dict)
                        .map_err(|e| CoreMLError::ConversionError(
                            format!("Failed to create feature provider: {:?}", e)
                        ))?
                };
                println!("  ✓ Feature provider created");

                // Step 5: Cast to ProtocolObject<dyn MLFeatureProvider>
                let provider_protocol: &ProtocolObject<dyn MLFeatureProvider> =
                    ProtocolObject::from_ref(&*input_provider);

                // Step 6: Run prediction on Neural Engine
                println!("  → Running Neural Engine inference...");
                let output_provider = unsafe {
                    ml_model.predictionFromFeatures_error(provider_protocol)
                        .map_err(|e| CoreMLError::InferenceError(
                            format!("Prediction failed: {:?}", e)
                        ))?
                };
                println!("  ✓ Neural Engine inference completed");

                // Step 7: Extract output MLFeatureValue
                let output_name_ns = NSString::from_str(&self.output_name);
                let output_value = unsafe {
                    output_provider.featureValueForName(&output_name_ns)
                        .ok_or_else(|| CoreMLError::ConversionError(
                            format!("Output '{}' not found", self.output_name)
                        ))?
                };
                println!("  ✓ Output feature extracted: {}", self.output_name);

                // Step 8: Extract MLMultiArray from output
                let output_array = unsafe {
                    output_value.multiArrayValue()
                        .ok_or_else(|| CoreMLError::ConversionError(
                            "Output is not MLMultiArray".to_string()
                        ))?
                };
                println!("  ✓ Output MLMultiArray extracted");

                // Step 9: Convert MLMultiArray back to Tensor
                let output_tensor = mlmultiarray_to_tensor(
                    &device,
                    &output_array,
                    self.output_shape.clone(),
                )?;
                println!("  ✓ Output tensor created");

                println!("=== Neural Engine inference successful ===");
                Ok(output_tensor)
            } else {
                // No MLModel loaded: Return dummy output tensor (like non-macOS version)
                println!("Running CoreML inference (placeholder - no model loaded)...");
                println!("  Model: {}", self.name);
                println!("  Input shape: {:?}", input_dims);
                println!("  Output shape: {:?}", self.output_shape);

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
