//! Enhanced error reporting with source location information
//!
//! Provides user-friendly error messages with line/column information,
//! context snippets, and suggestions for common mistakes.

mod diagnostic;
pub mod helpers;
pub mod stack_trace;

pub use diagnostic::{Diagnostic, Severity, ErrorReporter};
pub use stack_trace::{StackTrace, StackFrame, FrameType};
