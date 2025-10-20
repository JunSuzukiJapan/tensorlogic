//! Enhanced error reporting with source location information
//!
//! Provides user-friendly error messages with line/column information,
//! context snippets, and suggestions for common mistakes.

mod diagnostic;
pub mod helpers;

pub use diagnostic::{Diagnostic, Severity, ErrorReporter};
