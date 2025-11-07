//! TensorLogic: A unified tensor algebra and logic programming language
//!
//! This library implements the TensorLogic language with f16-only operations,
//! leveraging Metal GPU and Neural Engine for maximum performance.

// Allow common warnings throughout the codebase
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unreachable_patterns)]
#![allow(dead_code)]
#![allow(deprecated)]
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
// Semantic analysis is now done during parsing
// pub mod semantic;
pub mod planner;
pub mod prelude;
pub mod tensor;
pub mod tokenizer;
pub mod typecheck;

// LLVM compiler (optional, enabled with "llvm" feature)
#[cfg(feature = "llvm")]
pub mod compiler;

// Python bindings (optional, enabled with "python" or "python-extension" feature)
#[cfg(any(feature = "python", feature = "python-extension"))]
pub mod python;

// Re-export main types
pub use error::{TensorError, TensorResult};
pub use tensor::Tensor;
pub use device::{Device, MetalDevice};
pub use planner::ExecutionPlanner;
pub use model::GGUFWeightCache;
