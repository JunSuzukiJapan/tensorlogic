//! Model loading and saving support for multiple formats
//!
//! Supports:
//! - SafeTensors (HuggingFace standard)
//! - GGUF (LLM quantized models)
//! - CoreML (Apple Neural Engine)

pub mod metadata;
pub mod convert;
pub mod formats;

use crate::tensor::Tensor;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO};
use crate::device::{Device, MetalDevice};
use crate::error::TensorError;
use std::collections::HashMap;
use std::path::Path;

pub use metadata::{ModelMetadata, ModelFormat, QuantizationType};
pub use convert::TypeConverter;

/// Result type for model operations
pub type ModelResult<T> = Result<T, TensorError>;

/// Model structure containing multiple named tensors
///
/// Represents a complete model with all its weights and parameters.
/// Generic over tensor precision type (f16 or f32).
#[derive(Debug, Clone)]
pub struct Model<T: FloatType = half::f16> {
    /// Named tensors (e.g., "layers.0.weight", "layers.0.bias")
    pub tensors: HashMap<String, Tensor<T>>,
    /// Model metadata (format, quantization, etc.)
    pub metadata: ModelMetadata,
}

impl<T: FloatType> Model<T> {
    /// Create a new empty model
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            tensors: HashMap::new(),
            metadata,
        }
    }

    /// Create a model from a hashmap of tensors
    pub fn from_tensors(tensors: HashMap<String, Tensor<T>>, metadata: ModelMetadata) -> Self {
        Self { tensors, metadata }
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor<T>> {
        self.tensors.get(name)
    }

    /// Get a mutable reference to a tensor by name
    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut Tensor<T>> {
        self.tensors.get_mut(name)
    }

    /// Insert a tensor with a name
    pub fn insert_tensor(&mut self, name: String, tensor: Tensor<T>) {
        self.tensors.insert(name, tensor);
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Number of tensors in the model
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Load a model from a file (format auto-detected from extension)
    ///
    /// Supported extensions:
    /// - `.safetensors` → SafeTensors format
    /// - `.gguf` → GGUF format
    /// - `.mlmodel`, `.mlpackage` → CoreML format
    ///
    /// Tensors are loaded directly to Metal GPU for optimal performance.
    pub fn load<P: AsRef<Path>>(path: P, device: &MetalDevice) -> ModelResult<Model<half::f16>>
    where
        T: FloatType
    {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| TensorError::InvalidOperation(
                "File has no extension".to_string()
            ))?;

        match extension {
            "safetensors" => formats::SafeTensorsLoader::load(path, device),
            "gguf" => formats::GGUFLoader::load(path, device),
            "mlmodel" | "mlpackage" => formats::CoreMLLoader::load(path, device),
            _ => Err(TensorError::InvalidOperation(
                format!("Unsupported file extension: {}", extension)
            )),
        }
    }

}

/// Implementation specific to Model<f16> for format-specific save operations
impl Model<half::f16> {
    /// Save the model to a file
    ///
    /// Note: Save operations currently only support f16 models.
    /// For f32 models, convert to f16 first or use format-specific savers.
    pub fn save<P: AsRef<Path>>(&self, path: P, format: ModelFormat) -> ModelResult<()> {
        let path = path.as_ref();

        match format {
            ModelFormat::SafeTensors => formats::SafeTensorsLoader::save(self, path),
            ModelFormat::GGUF => formats::GGUFLoader::save(self, path),
            ModelFormat::CoreML => formats::CoreMLLoader::save(self, path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            format: ModelFormat::SafeTensors,
            quantization: None,
        };

        let model: Model<half::f16> = Model::new(metadata);
        assert_eq!(model.num_tensors(), 0);
    }

    #[test]
    fn test_model_insert_get() {
        use crate::device::MetalDevice;

        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            format: ModelFormat::SafeTensors,
            quantization: None,
        };

        let mut model: Model<half::f16> = Model::new(metadata);
        let device = MetalDevice::new().unwrap();
        let tensor = Tensor::from_vec_metal(&device, vec![half::f16::from_f32(1.0); 6], vec![2, 3]).unwrap();

        model.insert_tensor("test_tensor".to_string(), tensor);
        assert_eq!(model.num_tensors(), 1);
        assert!(model.get_tensor("test_tensor").is_some());
    }
}
