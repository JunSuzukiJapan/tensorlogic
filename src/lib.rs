//! TensorLogic: A unified tensor algebra and logic programming language
//!
//! This library implements the TensorLogic language with f16-only operations,
//! leveraging Metal GPU and Neural Engine for maximum performance.

pub mod ast;
pub mod autograd;
pub mod coreml;
pub mod device;
pub mod error;
pub mod error_reporting;
pub mod interpreter;
pub mod lexer;
pub mod logic;
pub mod ops;
pub mod optim;
pub mod parser;
pub mod planner;
pub mod prelude;
pub mod tensor;
pub mod typecheck;

// Re-export main types
pub use error::{TensorError, TensorResult};
pub use tensor::Tensor;
pub use device::{Device, MetalDevice};
pub use planner::ExecutionPlanner;
