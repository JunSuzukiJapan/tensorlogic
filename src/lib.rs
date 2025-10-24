//! TensorLogic: A unified tensor algebra and logic programming language
//!
//! This library implements the TensorLogic language with f16-only operations,
//! leveraging Metal GPU and Neural Engine for maximum performance.

pub mod ast;
pub mod autograd;
pub mod coreml;
pub mod device;
pub mod entity_registry;
pub mod relation_registry;
pub mod error;
pub mod error_reporting;
pub mod interpreter;
pub mod lexer;
pub mod logic;
pub mod model;
pub mod ops;
pub mod optim;
pub mod parser;
pub mod planner;
pub mod prelude;
pub mod tensor;
pub mod tokenizer;
pub mod typecheck;

// Python bindings (optional, enabled with "python" or "python-extension" feature)
#[cfg(any(feature = "python", feature = "python-extension"))]
pub mod python;

// Re-export main types
pub use error::{TensorError, TensorResult};
pub use tensor::Tensor;
pub use device::{Device, MetalDevice};
pub use planner::ExecutionPlanner;
