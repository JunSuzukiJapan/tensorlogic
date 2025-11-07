//! LLVM Compiler for TensorLogic
//!
//! This module provides LLVM-based compilation for TensorLogic programs.
//! It supports:
//! - JIT execution for improved interpreter performance
//! - LLVM IR (.ll) output
//! - Native assembly (.s) output (platform-dependent)

#[cfg(feature = "llvm")]
pub mod codegen;
#[cfg(feature = "llvm")]
pub mod jit;
#[cfg(feature = "llvm")]
pub mod output;

#[cfg(feature = "llvm")]
pub use codegen::LLVMCodeGen;
#[cfg(feature = "llvm")]
pub use jit::JITCompiler;
#[cfg(feature = "llvm")]
pub use output::{OutputFormat, OutputWriter};

/// Compilation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationMode {
    /// JIT compilation for interpreter acceleration
    JIT,
    /// Output LLVM IR to file
    LLVMAssembly,
    /// Output native assembly to file
    NativeAssembly,
}

/// Compiler options
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Compilation mode
    pub mode: CompilationMode,
    /// Output file path (for assembly outputs)
    pub output_path: Option<String>,
    /// Optimization level (0-3)
    pub opt_level: u8,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            mode: CompilationMode::JIT,
            output_path: None,
            opt_level: 2,
        }
    }
}

impl CompilerOptions {
    /// Create options for JIT compilation
    pub fn jit() -> Self {
        CompilerOptions {
            mode: CompilationMode::JIT,
            output_path: None,
            opt_level: 2,
        }
    }

    /// Create options for LLVM assembly output
    pub fn llvm_assembly(output_path: impl Into<String>) -> Self {
        CompilerOptions {
            mode: CompilationMode::LLVMAssembly,
            output_path: Some(output_path.into()),
            opt_level: 2,
        }
    }

    /// Create options for native assembly output
    pub fn native_assembly(output_path: impl Into<String>) -> Self {
        CompilerOptions {
            mode: CompilationMode::NativeAssembly,
            output_path: Some(output_path.into()),
            opt_level: 2,
        }
    }

    /// Set optimization level (0-3)
    pub fn with_opt_level(mut self, level: u8) -> Self {
        self.opt_level = level.min(3);
        self
    }
}
