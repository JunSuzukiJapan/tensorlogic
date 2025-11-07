//! LLVM IR Code Generation
//!
//! This module translates TensorLogic AST to LLVM IR.

use crate::ast::*;
use crate::error::{TensorError, TensorResult};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;

/// LLVM Code Generator
pub struct LLVMCodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    /// Variable storage (name -> pointer to value)
    variables: HashMap<String, PointerValue<'ctx>>,
    /// Function declarations
    functions: HashMap<String, FunctionValue<'ctx>>,
    /// Current function being compiled
    current_function: Option<FunctionValue<'ctx>>,
}

impl<'ctx> LLVMCodeGen<'ctx> {
    /// Create a new code generator
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        LLVMCodeGen {
            context,
            module,
            builder,
            variables: HashMap::new(),
            functions: HashMap::new(),
            current_function: None,
        }
    }

    /// Get the generated module
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Compile a complete program
    pub fn compile_program(&mut self, program: &Program) -> TensorResult<()> {
        // First pass: declare all functions
        for decl in &program.declarations {
            if let Declaration::Function(func_decl) = decl {
                self.declare_function(func_decl)?;
            }
        }

        // Second pass: compile function bodies
        for decl in &program.declarations {
            if let Declaration::Function(func_decl) = decl {
                self.compile_function(func_decl)?;
            }
        }

        // Compile main block if present
        if let Some(main_block) = &program.main_block {
            self.compile_main_block(main_block)?;
        }

        Ok(())
    }

    /// Declare a function (without body)
    fn declare_function(&mut self, func_decl: &FunctionDecl) -> TensorResult<FunctionValue<'ctx>> {
        let param_types: Vec<BasicMetadataTypeEnum> = func_decl
            .params
            .iter()
            .map(|p| self.entity_type_to_llvm(&p.entity_type))
            .collect::<TensorResult<Vec<_>>>()?;

        let return_type = self.return_type_to_llvm(&func_decl.return_type)?;

        let fn_type = match return_type {
            Some(ret_ty) => ret_ty.fn_type(&param_types, false),
            None => self.context.void_type().fn_type(&param_types, false),
        };

        let function = self.module.add_function(&func_decl.name.0, fn_type, None);
        self.functions.insert(func_decl.name.0.clone(), function);

        Ok(function)
    }

    /// Compile a function
    fn compile_function(&mut self, func_decl: &FunctionDecl) -> TensorResult<()> {
        let function = self.functions[&func_decl.name.0];
        self.current_function = Some(function);

        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        // Clear variables for new function
        self.variables.clear();

        // Allocate space for parameters
        for (i, param) in func_decl.params.iter().enumerate() {
            let param_value = function.get_nth_param(i as u32).unwrap();
            let param_type = self.entity_type_to_llvm_basic(&param.entity_type)?;
            let alloca = self.builder.build_alloca(param_type, &param.name.0)
                .map_err(|e| TensorError::CompilationError(e.to_string()))?;
            self.builder.build_store(alloca, param_value)
                .map_err(|e| TensorError::CompilationError(e.to_string()))?;
            self.variables.insert(param.name.0.clone(), alloca);
        }

        // Compile function body
        for stmt in &func_decl.body {
            self.compile_statement(stmt)?;
        }

        // Add default return if needed
        if !self.has_terminator() {
            match &func_decl.return_type {
                ReturnType::Void => {
                    self.builder.build_return(None)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }
                ReturnType::Scalar(ScalarType::Int) => {
                    let zero = self.context.i64_type().const_int(0, false);
                    self.builder.build_return(Some(&zero))
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }
                ReturnType::Scalar(ScalarType::Float) => {
                    let zero = self.context.f64_type().const_float(0.0);
                    self.builder.build_return(Some(&zero))
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }
                _ => {
                    return Err(TensorError::CompilationError(
                        "Non-scalar return types not yet supported in LLVM compilation".to_string(),
                    ));
                }
            }
        }

        self.current_function = None;
        Ok(())
    }

    /// Compile main block
    fn compile_main_block(&mut self, main_block: &MainBlock) -> TensorResult<()> {
        // Create main function: i32 main()
        let i32_type = self.context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let function = self.module.add_function("main", fn_type, None);
        self.current_function = Some(function);

        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        self.variables.clear();

        // Compile statements
        for stmt in &main_block.statements {
            self.compile_statement(stmt)?;
        }

        // Return 0
        if !self.has_terminator() {
            let zero = i32_type.const_int(0, false);
            self.builder.build_return(Some(&zero))
                .map_err(|e| TensorError::CompilationError(e.to_string()))?;
        }

        self.current_function = None;
        Ok(())
    }

    /// Compile a statement
    fn compile_statement(&mut self, stmt: &Statement) -> TensorResult<()> {
        match stmt {
            Statement::Let { target, value } => {
                let val = self.compile_expr(value)?;
                let alloca = self.builder.build_alloca(val.get_type(), &target.0)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                self.builder.build_store(alloca, val)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                self.variables.insert(target.0.clone(), alloca);
            }
            Statement::Assignment { target, value } => {
                let val = self.compile_expr(value)?;
                if let Some(&ptr) = self.variables.get(&target.0) {
                    self.builder.build_store(ptr, val)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                } else {
                    return Err(TensorError::CompilationError(
                        format!("Variable '{}' not found", target.0),
                    ));
                }
            }
            Statement::Return { value } => {
                if let Some(expr) = value {
                    let val = self.compile_expr(expr)?;
                    self.builder.build_return(Some(&val))
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                } else {
                    self.builder.build_return(None)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }
            }
            Statement::ControlFlow(cf) => {
                self.compile_control_flow(cf)?;
            }
            Statement::FunctionCall { name, args, .. } => {
                // Compile function call for side effects (e.g., print)
                self.compile_function_call(&name.0, args)?;
            }
            _ => {
                // Other statements not yet supported in LLVM compilation
                return Err(TensorError::CompilationError(
                    format!("Statement type not yet supported: {:?}", stmt),
                ));
            }
        }
        Ok(())
    }

    /// Compile control flow
    fn compile_control_flow(&mut self, cf: &ControlFlow) -> TensorResult<()> {
        match cf {
            ControlFlow::If { condition, then_block, else_block } => {
                let cond_val = self.compile_condition(condition)?;
                let function = self.current_function.unwrap();

                let then_bb = self.context.append_basic_block(function, "then");
                let else_bb = self.context.append_basic_block(function, "else");
                let merge_bb = self.context.append_basic_block(function, "merge");

                self.builder.build_conditional_branch(cond_val, then_bb, else_bb)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                // Then block
                self.builder.position_at_end(then_bb);
                for stmt in then_block {
                    self.compile_statement(stmt)?;
                }
                if !self.has_terminator() {
                    self.builder.build_unconditional_branch(merge_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }

                // Else block
                self.builder.position_at_end(else_bb);
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.compile_statement(stmt)?;
                    }
                }
                if !self.has_terminator() {
                    self.builder.build_unconditional_branch(merge_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }

                // Merge block
                self.builder.position_at_end(merge_bb);
            }
            ControlFlow::While { condition, body } => {
                let function = self.current_function.unwrap();

                let cond_bb = self.context.append_basic_block(function, "while.cond");
                let body_bb = self.context.append_basic_block(function, "while.body");
                let end_bb = self.context.append_basic_block(function, "while.end");

                self.builder.build_unconditional_branch(cond_bb)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                // Condition block
                self.builder.position_at_end(cond_bb);
                let cond_val = self.compile_condition(condition)?;
                self.builder.build_conditional_branch(cond_val, body_bb, end_bb)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                // Body block
                self.builder.position_at_end(body_bb);
                for stmt in body {
                    self.compile_statement(stmt)?;
                }
                if !self.has_terminator() {
                    self.builder.build_unconditional_branch(cond_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }

                // End block
                self.builder.position_at_end(end_bb);
            }
            ControlFlow::For { variable, iterable, body } => {
                // For now, only support range iteration
                if let Iterable::Range(count) = iterable {
                    let function = self.current_function.unwrap();
                    let i64_type = self.context.i64_type();

                    // Allocate loop variable
                    let loop_var = self.builder.build_alloca(i64_type, &variable.0)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    let zero = i64_type.const_int(0, false);
                    self.builder.build_store(loop_var, zero)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    self.variables.insert(variable.0.clone(), loop_var);

                    let cond_bb = self.context.append_basic_block(function, "for.cond");
                    let body_bb = self.context.append_basic_block(function, "for.body");
                    let inc_bb = self.context.append_basic_block(function, "for.inc");
                    let end_bb = self.context.append_basic_block(function, "for.end");

                    self.builder.build_unconditional_branch(cond_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                    // Condition
                    self.builder.position_at_end(cond_bb);
                    let current = self.builder.build_load(i64_type, loop_var, "current")
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    let limit = i64_type.const_int(*count as u64, false);
                    let cond = self.builder.build_int_compare(
                        IntPredicate::SLT,
                        current.into_int_value(),
                        limit,
                        "cond"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    self.builder.build_conditional_branch(cond, body_bb, end_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                    // Body
                    self.builder.position_at_end(body_bb);
                    for stmt in body {
                        self.compile_statement(stmt)?;
                    }
                    if !self.has_terminator() {
                        self.builder.build_unconditional_branch(inc_bb)
                            .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    }

                    // Increment
                    self.builder.position_at_end(inc_bb);
                    let current = self.builder.build_load(i64_type, loop_var, "current")
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    let one = i64_type.const_int(1, false);
                    let next = self.builder.build_int_add(
                        current.into_int_value(),
                        one,
                        "next"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    self.builder.build_store(loop_var, next)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    self.builder.build_unconditional_branch(cond_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                    // End
                    self.builder.position_at_end(end_bb);
                } else {
                    return Err(TensorError::CompilationError(
                        "Only range iteration is supported in for loops".to_string(),
                    ));
                }
            }
            ControlFlow::Loop { body } => {
                let function = self.current_function.unwrap();

                let loop_bb = self.context.append_basic_block(function, "loop");
                let end_bb = self.context.append_basic_block(function, "loop.end");

                self.builder.build_unconditional_branch(loop_bb)
                    .map_err(|e| TensorError::CompilationError(e.to_string()))?;

                self.builder.position_at_end(loop_bb);
                for stmt in body {
                    self.compile_statement(stmt)?;
                }
                if !self.has_terminator() {
                    self.builder.build_unconditional_branch(loop_bb)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                }

                // Note: Need to handle break statement to jump to end_bb
                self.builder.position_at_end(end_bb);
            }
        }
        Ok(())
    }

    /// Compile a condition
    fn compile_condition(&mut self, condition: &Condition) -> TensorResult<inkwell::values::IntValue<'ctx>> {
        match condition {
            Condition::Tensor(expr) => {
                let val = self.compile_expr(expr)?;
                // Convert to boolean
                if val.is_int_value() {
                    let zero = val.get_type().into_int_type().const_zero();
                    self.builder.build_int_compare(
                        IntPredicate::NE,
                        val.into_int_value(),
                        zero,
                        "cond"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))
                } else if val.is_float_value() {
                    let zero = val.get_type().into_float_type().const_zero();
                    self.builder.build_float_compare(
                        FloatPredicate::ONE,
                        val.into_float_value(),
                        zero,
                        "cond"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))
                } else {
                    Err(TensorError::CompilationError(
                        "Unsupported condition type".to_string(),
                    ))
                }
            }
            Condition::Constraint(constraint) => {
                self.compile_constraint(constraint)
            }
        }
    }

    /// Compile a constraint
    fn compile_constraint(&mut self, constraint: &Constraint) -> TensorResult<inkwell::values::IntValue<'ctx>> {
        match constraint {
            Constraint::Comparison { op, left, right } => {
                let left_val = self.compile_expr(left)?;
                let right_val = self.compile_expr(right)?;

                if left_val.is_int_value() && right_val.is_int_value() {
                    let predicate = match op {
                        CompOp::Eq => IntPredicate::EQ,
                        CompOp::Ne => IntPredicate::NE,
                        CompOp::Lt => IntPredicate::SLT,
                        CompOp::Le => IntPredicate::SLE,
                        CompOp::Gt => IntPredicate::SGT,
                        CompOp::Ge => IntPredicate::SGE,
                        _ => return Err(TensorError::CompilationError("Unsupported comparison".to_string())),
                    };
                    self.builder.build_int_compare(
                        predicate,
                        left_val.into_int_value(),
                        right_val.into_int_value(),
                        "cmp"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))
                } else if left_val.is_float_value() && right_val.is_float_value() {
                    let predicate = match op {
                        CompOp::Eq => FloatPredicate::OEQ,
                        CompOp::Ne => FloatPredicate::ONE,
                        CompOp::Lt => FloatPredicate::OLT,
                        CompOp::Le => FloatPredicate::OLE,
                        CompOp::Gt => FloatPredicate::OGT,
                        CompOp::Ge => FloatPredicate::OGE,
                        _ => return Err(TensorError::CompilationError("Unsupported comparison".to_string())),
                    };
                    self.builder.build_float_compare(
                        predicate,
                        left_val.into_float_value(),
                        right_val.into_float_value(),
                        "cmp"
                    ).map_err(|e| TensorError::CompilationError(e.to_string()))
                } else {
                    Err(TensorError::CompilationError("Type mismatch in comparison".to_string()))
                }
            }
            _ => Err(TensorError::CompilationError("Constraint not yet supported".to_string())),
        }
    }

    /// Compile an expression
    fn compile_expr(&mut self, expr: &TensorExpr) -> TensorResult<BasicValueEnum<'ctx>> {
        match expr {
            TensorExpr::Variable(name) => {
                if let Some(&ptr) = self.variables.get(&name.0) {
                    let val_type = ptr.get_type().get_element_type();
                    self.builder.build_load(val_type.try_into().unwrap(), ptr, &name.0)
                        .map_err(|e| TensorError::CompilationError(e.to_string()))
                } else {
                    Err(TensorError::CompilationError(
                        format!("Variable '{}' not found", name.0),
                    ))
                }
            }
            TensorExpr::Literal(lit) => self.compile_literal(lit),
            TensorExpr::BinaryOp { op, left, right } => {
                let left_val = self.compile_expr(left)?;
                let right_val = self.compile_expr(right)?;
                self.compile_binary_op(*op, left_val, right_val)
            }
            TensorExpr::UnaryOp { op, operand } => {
                let val = self.compile_expr(operand)?;
                self.compile_unary_op(*op, val)
            }
            TensorExpr::FunctionCall { name, args, .. } => {
                self.compile_function_call(&name.0, args)
            }
            _ => Err(TensorError::CompilationError(
                format!("Expression type not yet supported: {:?}", expr),
            )),
        }
    }

    /// Compile a literal
    fn compile_literal(&self, lit: &TensorLiteral) -> TensorResult<BasicValueEnum<'ctx>> {
        match lit {
            TensorLiteral::Scalar(scalar) => match scalar {
                ScalarLiteral::Integer(i) => {
                    Ok(self.context.i64_type().const_int(*i as u64, false).into())
                }
                ScalarLiteral::Float(f) => {
                    Ok(self.context.f64_type().const_float(*f).into())
                }
                ScalarLiteral::Boolean(b) => {
                    Ok(self.context.bool_type().const_int(*b as u64, false).into())
                }
                _ => Err(TensorError::CompilationError("Literal type not supported".to_string())),
            },
            _ => Err(TensorError::CompilationError("Array literals not yet supported".to_string())),
        }
    }

    /// Compile a binary operation
    fn compile_binary_op(
        &mut self,
        op: BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
    ) -> TensorResult<BasicValueEnum<'ctx>> {
        if left.is_int_value() && right.is_int_value() {
            let left_int = left.into_int_value();
            let right_int = right.into_int_value();
            let result = match op {
                BinaryOp::Add => self.builder.build_int_add(left_int, right_int, "add"),
                BinaryOp::Sub => self.builder.build_int_sub(left_int, right_int, "sub"),
                BinaryOp::Mul => self.builder.build_int_mul(left_int, right_int, "mul"),
                BinaryOp::Div => self.builder.build_int_signed_div(left_int, right_int, "div"),
                BinaryOp::Mod => self.builder.build_int_signed_rem(left_int, right_int, "mod"),
                _ => return Err(TensorError::CompilationError(format!("Operator {:?} not supported for integers", op))),
            }.map_err(|e| TensorError::CompilationError(e.to_string()))?;
            Ok(result.into())
        } else if left.is_float_value() && right.is_float_value() {
            let left_float = left.into_float_value();
            let right_float = right.into_float_value();
            let result = match op {
                BinaryOp::Add => self.builder.build_float_add(left_float, right_float, "add"),
                BinaryOp::Sub => self.builder.build_float_sub(left_float, right_float, "sub"),
                BinaryOp::Mul => self.builder.build_float_mul(left_float, right_float, "mul"),
                BinaryOp::Div => self.builder.build_float_div(left_float, right_float, "div"),
                _ => return Err(TensorError::CompilationError(format!("Operator {:?} not supported for floats", op))),
            }.map_err(|e| TensorError::CompilationError(e.to_string()))?;
            Ok(result.into())
        } else {
            Err(TensorError::CompilationError("Type mismatch in binary operation".to_string()))
        }
    }

    /// Compile a unary operation
    fn compile_unary_op(
        &mut self,
        op: UnaryOp,
        val: BasicValueEnum<'ctx>,
    ) -> TensorResult<BasicValueEnum<'ctx>> {
        match op {
            UnaryOp::Neg => {
                if val.is_int_value() {
                    let result = self.builder.build_int_neg(val.into_int_value(), "neg")
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    Ok(result.into())
                } else if val.is_float_value() {
                    let result = self.builder.build_float_neg(val.into_float_value(), "neg")
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    Ok(result.into())
                } else {
                    Err(TensorError::CompilationError("Invalid type for negation".to_string()))
                }
            }
            UnaryOp::Not => {
                if val.is_int_value() {
                    let result = self.builder.build_not(val.into_int_value(), "not")
                        .map_err(|e| TensorError::CompilationError(e.to_string()))?;
                    Ok(result.into())
                } else {
                    Err(TensorError::CompilationError("Invalid type for logical not".to_string()))
                }
            }
            _ => Err(TensorError::CompilationError(format!("Unary operator {:?} not yet supported", op))),
        }
    }

    /// Compile a function call
    fn compile_function_call(
        &mut self,
        name: &str,
        args: &[TensorExpr],
    ) -> TensorResult<BasicValueEnum<'ctx>> {
        // Handle built-in functions
        if name == "print" {
            // For now, skip print in LLVM compilation
            return Ok(self.context.i64_type().const_int(0, false).into());
        }

        // User-defined function
        if let Some(&function) = self.functions.get(name) {
            let mut arg_values = Vec::new();
            for arg in args {
                arg_values.push(self.compile_expr(arg)?.into());
            }

            let call_site = self.builder.build_call(function, &arg_values, "call")
                .map_err(|e| TensorError::CompilationError(e.to_string()))?;

            if let Some(result) = call_site.try_as_basic_value().left() {
                Ok(result)
            } else {
                // Void function, return dummy value
                Ok(self.context.i64_type().const_int(0, false).into())
            }
        } else {
            Err(TensorError::CompilationError(
                format!("Function '{}' not found", name),
            ))
        }
    }

    /// Convert entity type to LLVM type (for metadata)
    fn entity_type_to_llvm(&self, entity_type: &EntityType) -> TensorResult<BasicMetadataTypeEnum<'ctx>> {
        match entity_type {
            EntityType::Scalar(ScalarType::Int) => {
                Ok(self.context.i64_type().into())
            }
            EntityType::Scalar(ScalarType::Float) => {
                Ok(self.context.f64_type().into())
            }
            EntityType::Scalar(ScalarType::Bool) => {
                Ok(self.context.bool_type().into())
            }
            _ => Err(TensorError::CompilationError(
                "Only scalar types are supported in function parameters".to_string(),
            )),
        }
    }

    /// Convert entity type to LLVM basic type
    fn entity_type_to_llvm_basic(&self, entity_type: &EntityType) -> TensorResult<BasicTypeEnum<'ctx>> {
        match entity_type {
            EntityType::Scalar(ScalarType::Int) => {
                Ok(self.context.i64_type().into())
            }
            EntityType::Scalar(ScalarType::Float) => {
                Ok(self.context.f64_type().into())
            }
            EntityType::Scalar(ScalarType::Bool) => {
                Ok(self.context.bool_type().into())
            }
            _ => Err(TensorError::CompilationError(
                "Only scalar types are supported".to_string(),
            )),
        }
    }

    /// Convert return type to LLVM type
    fn return_type_to_llvm(&self, return_type: &ReturnType) -> TensorResult<Option<BasicTypeEnum<'ctx>>> {
        match return_type {
            ReturnType::Void => Ok(None),
            ReturnType::Scalar(ScalarType::Int) => {
                Ok(Some(self.context.i64_type().into()))
            }
            ReturnType::Scalar(ScalarType::Float) => {
                Ok(Some(self.context.f64_type().into()))
            }
            ReturnType::Scalar(ScalarType::Bool) => {
                Ok(Some(self.context.bool_type().into()))
            }
            _ => Err(TensorError::CompilationError(
                "Only scalar return types are supported".to_string(),
            )),
        }
    }

    /// Check if current block has a terminator
    fn has_terminator(&self) -> bool {
        if let Some(block) = self.builder.get_insert_block() {
            block.get_terminator().is_some()
        } else {
            false
        }
    }
}
