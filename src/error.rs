//! Error types for TensorLogic

use thiserror::Error;

/// Result type for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

/// Error types for tensor operations
#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid dimension: {dim}")]
    InvalidDimension { dim: usize },

    #[error("Metal error: {0}")]
    MetalError(String),

    #[error("Neural Engine error: {0}")]
    NeuralEngineError(String),

    #[error("Device conversion error: {0}")]
    DeviceConversionError(String),

    #[error("f16 precision overflow")]
    PrecisionOverflow,

    #[error("Invalid tensor operation: {0}")]
    InvalidOperation(String),

    #[error("Buffer allocation failed: {0}")]
    AllocationError(String),

    #[error("Index out of bounds: index {index}, size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Model loading error: {0}")]
    LoadError(String),

    #[error("Compilation error: {0}")]
    CompilationError(String),
}
