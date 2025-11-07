//! Output Writer
//!
//! This module handles writing LLVM IR and native assembly to files.

use crate::ast::Program;
use crate::compiler::codegen::LLVMCodeGen;
use crate::compiler::linker::{Linker, Platform};
use crate::error::{TensorError, TensorResult};
use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::OptimizationLevel;
use std::fs;
use std::path::{Path, PathBuf};

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// LLVM IR assembly (.ll)
    LLVMAssembly,
    /// Native assembly (.s)
    NativeAssembly,
    /// Object file (.o on Unix, .obj on Windows)
    ObjectFile,
    /// Static library (.a on Unix, .lib on Windows)
    StaticLibrary,
    /// Shared library (.so on Unix, .dll on Windows, .dylib on macOS)
    SharedLibrary,
    /// Executable binary
    Executable,
}

/// Output writer
pub struct OutputWriter<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> OutputWriter<'ctx> {
    /// Create a new output writer
    pub fn new(context: &'ctx Context) -> Self {
        OutputWriter { context }
    }

    /// Compile program and write to file
    pub fn write(
        &self,
        program: &Program,
        output_path: &str,
        format: OutputFormat,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Create code generator
        let mut codegen = LLVMCodeGen::new(self.context, "tensorlogic");

        // Compile the program
        codegen.compile_program(program)?;

        // Get the module
        let module = codegen.module();

        // Verify module
        if let Err(e) = module.verify() {
            return Err(TensorError::CompilationError(format!(
                "Module verification failed: {}",
                e
            )));
        }

        match format {
            OutputFormat::LLVMAssembly => {
                self.write_llvm_assembly(module, output_path)?;
            }
            OutputFormat::NativeAssembly => {
                self.write_native_assembly(module, output_path, opt_level)?;
            }
            OutputFormat::ObjectFile => {
                self.write_object_file(module, output_path, opt_level)?;
            }
            OutputFormat::StaticLibrary => {
                self.write_static_library_from_program(program, output_path, opt_level)?;
            }
            OutputFormat::SharedLibrary => {
                self.write_shared_library_from_program(program, output_path, opt_level)?;
            }
            OutputFormat::Executable => {
                self.write_executable_from_program(program, output_path, opt_level)?;
            }
        }

        Ok(())
    }

    /// Write LLVM IR assembly to file
    fn write_llvm_assembly(
        &self,
        module: &inkwell::module::Module<'ctx>,
        output_path: &str,
    ) -> TensorResult<()> {
        let llvm_ir = module.print_to_string().to_string();

        fs::write(output_path, llvm_ir).map_err(|e| {
            TensorError::CompilationError(format!("Failed to write LLVM assembly: {}", e))
        })?;

        println!("LLVM assembly written to: {}", output_path);
        Ok(())
    }

    /// Write native assembly to file
    fn write_native_assembly(
        &self,
        module: &inkwell::module::Module<'ctx>,
        output_path: &str,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Initialize all targets
        Target::initialize_all(&InitializationConfig::default());

        // Get the host target triple
        let target_triple = TargetMachine::get_default_triple();

        // Get the target
        let target = Target::from_triple(&target_triple).map_err(|e| {
            TensorError::CompilationError(format!("Failed to get target: {}", e))
        })?;

        // Create target machine
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opt_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| {
                TensorError::CompilationError("Failed to create target machine".to_string())
            })?;

        // Check if native assembly output is supported
        let triple_str = target_triple.as_str().to_string_lossy();
        if !self.is_native_assembly_supported(&triple_str) {
            return Err(TensorError::CompilationError(format!(
                "Native assembly output is not supported on this platform: {}",
                triple_str
            )));
        }

        // Write assembly to file
        target_machine
            .write_to_file(module, FileType::Assembly, Path::new(output_path))
            .map_err(|e| {
                TensorError::CompilationError(format!("Failed to write native assembly: {}", e))
            })?;

        println!("Native assembly written to: {}", output_path);
        Ok(())
    }

    /// Write object file
    fn write_object_file(
        &self,
        module: &inkwell::module::Module<'ctx>,
        output_path: &str,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Initialize all targets
        Target::initialize_all(&InitializationConfig::default());

        // Get the host target triple
        let target_triple = TargetMachine::get_default_triple();

        // Get the target
        let target = Target::from_triple(&target_triple).map_err(|e| {
            TensorError::CompilationError(format!("Failed to get target: {}", e))
        })?;

        // Create target machine
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opt_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| {
                TensorError::CompilationError("Failed to create target machine".to_string())
            })?;

        // Write object file
        target_machine
            .write_to_file(module, FileType::Object, Path::new(output_path))
            .map_err(|e| {
                TensorError::CompilationError(format!("Failed to write object file: {}", e))
            })?;

        println!("Object file written to: {}", output_path);
        Ok(())
    }

    /// Write static library from program
    fn write_static_library_from_program(
        &self,
        program: &Program,
        output_path: &str,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Create a temporary object file
        let temp_obj = self.create_temp_object_path(output_path)?;

        // Compile to object file first
        let mut codegen = LLVMCodeGen::new(self.context, "tensorlogic");
        codegen.compile_program(program)?;
        let module = codegen.module();

        if let Err(e) = module.verify() {
            return Err(TensorError::CompilationError(format!(
                "Module verification failed: {}",
                e
            )));
        }

        self.write_object_file(module, &temp_obj, opt_level)?;

        // Use linker to create static library
        let linker = Linker::new();
        linker.create_static_library(&[&temp_obj], output_path)?;

        // Clean up temporary object file
        let _ = fs::remove_file(&temp_obj);

        Ok(())
    }

    /// Write shared library from program
    fn write_shared_library_from_program(
        &self,
        program: &Program,
        output_path: &str,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Create a temporary object file
        let temp_obj = self.create_temp_object_path(output_path)?;

        // Compile to object file first
        let mut codegen = LLVMCodeGen::new(self.context, "tensorlogic");
        codegen.compile_program(program)?;
        let module = codegen.module();

        if let Err(e) = module.verify() {
            return Err(TensorError::CompilationError(format!(
                "Module verification failed: {}",
                e
            )));
        }

        self.write_object_file(module, &temp_obj, opt_level)?;

        // Use linker to create shared library
        let linker = Linker::new();
        linker.create_shared_library(&[&temp_obj], output_path)?;

        // Clean up temporary object file
        let _ = fs::remove_file(&temp_obj);

        Ok(())
    }

    /// Write executable from program
    fn write_executable_from_program(
        &self,
        program: &Program,
        output_path: &str,
        opt_level: OptimizationLevel,
    ) -> TensorResult<()> {
        // Create a temporary object file
        let temp_obj = self.create_temp_object_path(output_path)?;

        // Compile to object file first
        let mut codegen = LLVMCodeGen::new(self.context, "tensorlogic");
        codegen.compile_program(program)?;
        let module = codegen.module();

        if let Err(e) = module.verify() {
            return Err(TensorError::CompilationError(format!(
                "Module verification failed: {}",
                e
            )));
        }

        self.write_object_file(module, &temp_obj, opt_level)?;

        // Use linker to create executable
        let linker = Linker::new();
        linker.create_executable(&[&temp_obj], output_path)?;

        // Clean up temporary object file
        let _ = fs::remove_file(&temp_obj);

        Ok(())
    }

    /// Create a temporary object file path based on output path
    fn create_temp_object_path(&self, output_path: &str) -> TensorResult<String> {
        let platform = Platform::current();
        let path = Path::new(output_path);

        let temp_name = format!(
            "{}_temp.{}",
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output"),
            platform.object_extension()
        );

        let temp_path = if let Some(parent) = path.parent() {
            parent.join(temp_name)
        } else {
            PathBuf::from(temp_name)
        };

        Ok(temp_path.to_string_lossy().to_string())
    }

    /// Check if native assembly output is supported on the current platform
    fn is_native_assembly_supported(&self, triple: &str) -> bool {
        // Native assembly output is supported on most platforms
        // but may fail on some embedded or unusual targets
        // Common supported platforms:
        // - x86_64-*-linux-*
        // - x86_64-*-darwin (macOS)
        // - x86_64-*-windows-*
        // - aarch64-*-linux-*
        // - aarch64-apple-darwin (Apple Silicon)

        triple.contains("x86_64")
            || triple.contains("aarch64")
            || triple.contains("arm64")
            || triple.contains("i686")
    }
}

// Tests will be added later when needed
