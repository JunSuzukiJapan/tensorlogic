//! TensorLogic: A unified tensor algebra and logic programming language
//!
//! This library implements the TensorLogic language with f16-only operations,
//! leveraging Metal GPU and Neural Engine for maximum performance.

pub mod autograd;
pub mod device;
pub mod error;
pub mod ops;
pub mod planner;
pub mod tensor;

// Re-export main types
pub use error::{TensorError, TensorResult};
pub use tensor::Tensor;
pub use device::{Device, MetalDevice};
pub use planner::ExecutionPlanner;
