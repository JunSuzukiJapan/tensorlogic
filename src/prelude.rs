//! Prelude module for common imports in TensorLogic
//!
//! This module re-exports the most commonly used types and traits
//! for convenient use in examples and user code.

pub use crate::tensor::Tensor;
pub use crate::tensor::{TensorCreation, TensorAccessors, TensorTransform, TensorIO};
pub use crate::device::{Device, MetalDevice};
pub use crate::error::{TensorError, TensorResult};
pub use crate::planner::ExecutionPlanner;

// Optimizer types
pub use crate::optim::{Adam, AdamW, SGD};
pub use crate::optim::{StepLR, ExponentialLR, CosineAnnealingLR};

// Common half precision type
pub use half::f16;
