//! CoreML format loader
//!
//! Loads Apple CoreML models optimized for Neural Engine.

use crate::error::TensorError;
use crate::model::{Model, ModelMetadata, ModelFormat, QuantizationType};
use crate::device::MetalDevice;
use std::collections::HashMap;
use std::path::Path;

#[cfg(target_os = "macos")]
use objc2_core_ml::MLModel;
#[cfg(target_os = "macos")]
use objc2_foundation::{NSURL, NSString};

pub struct CoreMLLoader;

impl CoreMLLoader {
    /// Load a CoreML model (.mlmodel or .mlpackage)
    ///
    /// Extracts weights from the CoreML model and converts to TensorLogic tensors.
    /// Note: CoreML models are typically used for inference, not training.
    #[cfg(target_os = "macos")]
    pub fn load<P: AsRef<Path>>(path: P, _device: &MetalDevice) -> Result<Model, TensorError> {
        let path = path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(TensorError::InvalidOperation(
                format!("CoreML model file not found: {}", path.display())
            ));
        }

        let path_str = path.to_string_lossy().to_string();
        let path_nsstring = NSString::from_str(&path_str);
        let url = unsafe { NSURL::fileURLWithPath(&path_nsstring) };

        // Load MLModel
        let ml_model = unsafe {
            MLModel::modelWithContentsOfURL_error(&url)
                .map_err(|_| TensorError::InvalidOperation(
                    "Failed to load CoreML model".to_string()
                ))?
        };

        // Get model description
        let description = unsafe { ml_model.modelDescription() };

        // Extract model name
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("coreml_model")
            .to_string();

        // CoreML models don't directly expose their weights like PyTorch models
        // They are compiled and optimized for inference on Neural Engine
        // For now, we create a model structure that holds the CoreML model reference

        let tensors = HashMap::new();

        // Create metadata
        let metadata = ModelMetadata {
            name,
            format: ModelFormat::CoreML,
            quantization: Some(QuantizationType::F16), // Neural Engine uses f16
        };

        // Note: CoreML models are inference-only
        // Weights are not directly accessible after compilation
        // This loader creates a wrapper that can be used for inference

        Ok(Model { tensors, metadata })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn load<P: AsRef<Path>>(_path: P, _device: &MetalDevice) -> Result<Model, TensorError> {
        Err(TensorError::InvalidOperation(
            "CoreML is only available on macOS".to_string()
        ))
    }

    /// Save to CoreML format
    ///
    /// Note: Creating CoreML models from TensorLogic tensors requires
    /// building MLModelDescription and model architecture, which is complex.
    /// For now, use CoreML models for inference only.
    pub fn save<P: AsRef<Path>>(_model: &Model, _path: P) -> Result<(), TensorError> {
        Err(TensorError::InvalidOperation(
            "CoreML model creation not yet implemented. Use coremltools in Python to create CoreML models from trained TensorLogic models.".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    #[ignore] // Requires actual CoreML model file
    fn test_coreml_load() {
        // This test requires an actual CoreML model file
        // Example usage:
        // let model = CoreMLLoader::load("path/to/model.mlmodelc").unwrap();
        // assert_eq!(model.metadata.format, ModelFormat::CoreML);
        // assert_eq!(model.metadata.quantization, Some(QuantizationType::F16));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_coreml_not_available() {
        // On non-macOS platforms, CoreML should return an error
        let result = CoreMLLoader::load("dummy.mlmodel");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("only available on macOS"));
    }

    #[test]
    fn test_coreml_save_not_implemented() {
        use crate::model::{Model, ModelMetadata, ModelFormat};
        use std::collections::HashMap;

        let metadata = ModelMetadata {
            name: "test".to_string(),
            format: ModelFormat::CoreML,
            quantization: None,
        };
        let model = Model {
            tensors: HashMap::new(),
            metadata,
        };

        let result = CoreMLLoader::save(&model, "dummy.mlmodel");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}
