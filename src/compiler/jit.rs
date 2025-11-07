//! JIT Compiler
//!
//! This module provides JIT compilation support for TensorLogic programs.

use crate::ast::Program;
use crate::compiler::codegen::LLVMCodeGen;
use crate::error::{TensorError, TensorResult};
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;
use std::marker::PhantomData;

/// JIT Compiler
pub struct JITCompiler<'ctx> {
    context: &'ctx Context,
    execution_engine: Option<ExecutionEngine<'ctx>>,
    _phantom: PhantomData<&'ctx ()>,
}

impl<'ctx> JITCompiler<'ctx> {
    /// Create a new JIT compiler
    pub fn new(context: &'ctx Context) -> Self {
        JITCompiler {
            context,
            execution_engine: None,
            _phantom: PhantomData,
        }
    }

    /// Compile and prepare program for JIT execution
    pub fn compile(&mut self, program: &Program, opt_level: OptimizationLevel) -> TensorResult<()> {
        // Create code generator
        let mut codegen = LLVMCodeGen::new(self.context, "tensorlogic_jit");

        // Compile the program
        codegen.compile_program(program)?;

        // Get the module
        let module = codegen.module();

        // Verify module
        if let Err(e) = module.verify() {
            return Err(TensorError::CompilationError(
                format!("Module verification failed: {}", e),
            ));
        }

        // Create execution engine
        let execution_engine = module
            .create_jit_execution_engine(opt_level)
            .map_err(|e| TensorError::CompilationError(
                format!("Failed to create JIT execution engine: {}", e),
            ))?;

        self.execution_engine = Some(execution_engine);

        Ok(())
    }

    /// Get a function from the JIT-compiled module
    pub unsafe fn get_function<F>(&self, name: &str) -> TensorResult<JitFunction<F>> {
        if let Some(ref engine) = self.execution_engine {
            engine
                .get_function(name)
                .map_err(|e| TensorError::CompilationError(
                    format!("Function '{}' not found: {}", name, e),
                ))
        } else {
            Err(TensorError::CompilationError(
                "JIT execution engine not initialized".to_string(),
            ))
        }
    }

    /// Execute the main function
    pub unsafe fn execute_main(&self) -> TensorResult<i32> {
        type MainFunc = unsafe extern "C" fn() -> i32;

        let main_fn: JitFunction<MainFunc> = self.get_function("main")?;
        Ok(main_fn.call())
    }

    /// Get the execution engine (for advanced usage)
    pub fn execution_engine(&self) -> Option<&ExecutionEngine<'ctx>> {
        self.execution_engine.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_jit_simple_program() {
        let context = Context::create();
        let mut jit = JITCompiler::new(&context);

        // Create a simple program: main { let x = 42; return x; }
        let program = Program {
            declarations: vec![],
            main_block: Some(MainBlock {
                statements: vec![
                    Statement::Let {
                        target: Identifier::new("x"),
                        value: TensorExpr::int(42),
                    },
                ],
            }),
            test_blocks: vec![],
            bench_blocks: vec![],
        };

        // Compile
        let result = jit.compile(&program, OptimizationLevel::Default);
        assert!(result.is_ok());

        // Execute
        unsafe {
            let result = jit.execute_main();
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), 0); // main returns 0
        }
    }
}
