//! CoreML model management for Neural Engine inference
//!
//! Note: Full CoreML integration with model caching is deferred to future phases
//! due to Send/Sync limitations of objc2-core-ml bindings. This module provides
//! the architecture and basic model loading functionality.

use objc2::rc::Retained;
use objc2_core_ml::MLModel;
use objc2_foundation::{NSString, NSURL};
use crate::error::{TensorError, TensorResult};

/// CoreML model loader (non-caching for now due to Send/Sync limitations)
pub struct CoreMLModelManager;

impl CoreMLModelManager {
    /// Create a new CoreML model manager
    pub fn new() -> Self {
        Self
    }

    /// Load a CoreML model from file path
    ///
    /// # Arguments
    /// * `path` - Path to .mlmodel or .mlmodelc file
    ///
    /// # Returns
    /// * Compiled MLModel ready for inference
    ///
    /// # Note
    /// This implementation loads models on each call. Caching will be added
    /// in future phases when thread-safe model storage is implemented.
    pub fn load_model(&self, path: &str) -> TensorResult<Retained<MLModel>> {
        // Load model from file
        let ns_path = NSString::from_str(path);
        let url = unsafe { NSURL::fileURLWithPath(&ns_path) };

        let model = unsafe {
            MLModel::modelWithContentsOfURL_error(&url)
                .map_err(|_e| {
                    TensorError::DeviceConversionError(format!("Failed to load CoreML model from: {}", path))
                })?
        };

        Ok(model)
    }

    /// Check if a model file exists
    pub fn model_exists(&self, path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
}

impl Default for CoreMLModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let _manager = CoreMLModelManager::new();
        // Manager creation should succeed
        assert!(true);
    }

    #[test]
    fn test_model_exists_check() {
        let manager = CoreMLModelManager::new();
        // Non-existent model should return false
        assert!(!manager.model_exists("/nonexistent/model.mlmodel"));
    }

    // Note: Model loading tests require actual .mlmodel files
    // These would be integration tests in a real deployment
}
