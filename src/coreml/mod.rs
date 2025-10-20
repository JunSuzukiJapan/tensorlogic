//! CoreML Integration for Neural Engine Inference
//!
//! This module provides integration with Apple's CoreML framework,
//! enabling TensorLogic to leverage the Neural Engine for high-performance
//! inference on Apple Silicon devices.
//!
//! # Features
//!
//! - Load CoreML models from .mlmodel or .mlmodelc files
//! - Convert TensorLogic tensors to MLMultiArray and vice versa
//! - Execute inference on Neural Engine
//! - Batch processing support
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic::coreml::CoreMLModel;
//!
//! // Load a CoreML model
//! let model = CoreMLModel::load("model.mlmodelc")?;
//!
//! // Run inference
//! let output = model.predict(input_tensor)?;
//! ```

use crate::tensor::Tensor;
use crate::error::TensorError;
use std::path::Path;

pub mod model;
pub mod conversion;

pub use model::CoreMLModel;
pub use conversion::{tensor_to_mlmultiarray, mlmultiarray_to_tensor};

/// CoreML inference result
pub type CoreMLResult<T> = Result<T, CoreMLError>;

/// CoreML-specific errors
#[derive(Debug, thiserror::Error)]
pub enum CoreMLError {
    #[error("Failed to load CoreML model: {0}")]
    ModelLoadError(String),

    #[error("Failed to compile CoreML model: {0}")]
    ModelCompileError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}
