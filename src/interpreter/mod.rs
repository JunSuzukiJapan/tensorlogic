//! Interpreter for TensorLogic
//!
//! This module implements the runtime execution engine for TensorLogic programs,
//! evaluating expressions and executing statements with the actual tensor library.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic::interpreter::Interpreter;
//! use tensorlogic::parser::TensorLogicParser;
//!
//! let source = r#"
//!     tensor w: float32[10] learnable
//!     main {
//!         result := w
//!     }
//! "#;
//!
//! let program = TensorLogicParser::parse_program(source)?;
//!
//! let mut interpreter = Interpreter::new();
//! interpreter.execute(&program)?;
//! ```

use std::collections::HashMap;

use crate::ast::*;
use crate::tensor::Tensor;
use crate::device::MetalDevice;
use crate::error::TensorError;
use half::f16;

/// Runtime errors
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Runtime type error: {0}")]
    TypeError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Not yet implemented: {0}")]
    NotImplemented(String),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Invalid dimensions for operation")]
    InvalidDimensions,
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Runtime value
#[derive(Debug, Clone)]
pub enum Value {
    Tensor(Tensor),
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Void,
}

impl Value {
    /// Convert to tensor if possible
    pub fn as_tensor(&self) -> RuntimeResult<&Tensor> {
        match self {
            Value::Tensor(t) => Ok(t),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected tensor, found {:?}",
                self
            ))),
        }
    }

    /// Convert to float if possible
    pub fn as_float(&self) -> RuntimeResult<f64> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Integer(i) => Ok(*i as f64),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected float, found {:?}",
                self
            ))),
        }
    }

    /// Convert to boolean if possible
    pub fn as_bool(&self) -> RuntimeResult<bool> {
        match self {
            Value::Boolean(b) => Ok(*b),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected boolean, found {:?}",
                self
            ))),
        }
    }
}

/// Runtime environment
#[derive(Debug)]
pub struct RuntimeEnvironment {
    /// Variable name → value
    variables: HashMap<String, Value>,
    /// Current Metal device for tensor operations
    metal_device: MetalDevice,
}

impl RuntimeEnvironment {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            metal_device: MetalDevice::new().unwrap(),
        }
    }

    /// Set a variable
    pub fn set_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Get a variable
    pub fn get_variable(&self, name: &str) -> RuntimeResult<&Value> {
        self.variables
            .get(name)
            .ok_or_else(|| RuntimeError::UndefinedVariable(name.to_string()))
    }

    /// Get current Metal device
    pub fn metal_device(&self) -> &MetalDevice {
        &self.metal_device
    }
}

impl Default for RuntimeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Interpreter
pub struct Interpreter {
    env: RuntimeEnvironment,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: RuntimeEnvironment::new(),
        }
    }

    /// Execute a complete program
    pub fn execute(&mut self, program: &Program) -> RuntimeResult<()> {
        // Execute all declarations
        for decl in &program.declarations {
            self.execute_declaration(decl)?;
        }

        // Execute main block if present
        if let Some(main_block) = &program.main_block {
            self.execute_main_block(main_block)?;
        }

        Ok(())
    }

    /// Execute a declaration
    fn execute_declaration(&mut self, decl: &Declaration) -> RuntimeResult<()> {
        match decl {
            Declaration::Tensor(tensor_decl) => self.execute_tensor_decl(tensor_decl),
            Declaration::Relation(_) => {
                // Relations are metadata, no runtime execution needed
                Ok(())
            }
            Declaration::Rule(_) => {
                // Rules are executed on-demand during queries
                Ok(())
            }
            Declaration::Embedding(_) => {
                // Simplified: embedding initialization deferred
                Ok(())
            }
            Declaration::Function(_) => {
                // Functions are executed when called
                Ok(())
            }
        }
    }

    /// Execute a tensor declaration
    fn execute_tensor_decl(&mut self, decl: &TensorDecl) -> RuntimeResult<()> {
        let tensor = if let Some(init_expr) = &decl.init_expr {
            // Evaluate initialization expression
            let value = self.eval_expr(init_expr)?;
            value.as_tensor()?.clone()
        } else {
            // Create zero-initialized tensor
            self.create_zero_tensor(&decl.tensor_type)?
        };

        // Set requires_grad based on learnable status
        let mut tensor = tensor;
        if decl.tensor_type.learnable == LearnableStatus::Learnable {
            tensor.set_requires_grad(true);
        }

        self.env
            .set_variable(decl.name.as_str().to_string(), Value::Tensor(tensor));

        Ok(())
    }

    /// Create a zero-initialized tensor
    fn create_zero_tensor(&self, tensor_type: &TensorType) -> RuntimeResult<Tensor> {
        // Convert dimensions to usize (resolve fixed dimensions)
        let mut shape = Vec::new();
        for dim in &tensor_type.dimensions {
            match dim {
                Dimension::Fixed(size) => shape.push(*size),
                Dimension::Variable(_) => {
                    return Err(RuntimeError::InvalidOperation(
                        "Cannot create tensor with unresolved variable dimensions".to_string(),
                    ));
                }
                Dimension::Dynamic => {
                    return Err(RuntimeError::InvalidOperation(
                        "Cannot create tensor with dynamic dimensions".to_string(),
                    ));
                }
            }
        }

        // Create tensor based on base type
        match tensor_type.base_type {
            BaseType::Float32 | BaseType::Float64 => {
                Tensor::zeros(self.env.metal_device(), shape)
                    .map_err(|e| RuntimeError::TensorError(e))
            }
            _ => Err(RuntimeError::NotImplemented(format!(
                "Base type {:?} not yet supported",
                tensor_type.base_type
            ))),
        }
    }

    /// Execute main block
    fn execute_main_block(&mut self, main_block: &MainBlock) -> RuntimeResult<()> {
        for stmt in &main_block.statements {
            self.execute_statement(stmt)?;
        }
        Ok(())
    }

    /// Execute a statement
    fn execute_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
        match stmt {
            Statement::Assignment { target, value } => {
                let evaluated_value = self.eval_expr(value)?;
                self.env
                    .set_variable(target.as_str().to_string(), evaluated_value);
                Ok(())
            }
            Statement::Equation(eq) => {
                // Execute equation (side effects only, no assignment)
                let _left = self.eval_expr(&eq.left)?;
                let _right = self.eval_expr(&eq.right)?;
                // In a full implementation, this would perform unification or constraint solving
                Ok(())
            }
            _ => Err(RuntimeError::NotImplemented(
                "Statement type not yet implemented".to_string(),
            )),
        }
    }

    /// Evaluate an expression
    fn eval_expr(&mut self, expr: &TensorExpr) -> RuntimeResult<Value> {
        match expr {
            TensorExpr::Variable(id) => {
                let value = self.env.get_variable(id.as_str())?;
                Ok(value.clone())
            }

            TensorExpr::Literal(lit) => self.eval_literal(lit),

            TensorExpr::BinaryOp { op, left, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(op, left_val, right_val)
            }

            TensorExpr::UnaryOp { op, operand } => {
                let operand_val = self.eval_expr(operand)?;
                self.eval_unary_op(op, operand_val)
            }

            TensorExpr::FunctionCall { name, args } => {
                self.eval_function_call(name, args)
            }

            TensorExpr::EinSum { .. } => {
                Err(RuntimeError::NotImplemented("einsum not yet implemented".to_string()))
            }

            TensorExpr::EmbeddingLookup { .. } => {
                Err(RuntimeError::NotImplemented(
                    "embedding lookup not yet implemented".to_string(),
                ))
            }
        }
    }

    /// Evaluate a literal
    fn eval_literal(&self, lit: &TensorLiteral) -> RuntimeResult<Value> {
        match lit {
            TensorLiteral::Scalar(scalar) => match scalar {
                ScalarLiteral::Float(f) => Ok(Value::Float(*f)),
                ScalarLiteral::Integer(i) => Ok(Value::Integer(*i)),
                ScalarLiteral::Boolean(b) => Ok(Value::Boolean(*b)),
                ScalarLiteral::Complex { .. } => {
                    Err(RuntimeError::NotImplemented("Complex numbers not yet supported".to_string()))
                }
            },
            TensorLiteral::Array(elements) => {
                // Convert array to tensor
                self.eval_array_literal(elements)
            }
        }
    }

    /// Evaluate an array literal to a tensor
    fn eval_array_literal(&self, elements: &[TensorLiteral]) -> RuntimeResult<Value> {
        if elements.is_empty() {
            return Err(RuntimeError::InvalidOperation("Empty array not allowed".to_string()));
        }

        // Recursively collect all scalar values
        let values = self.collect_scalars(elements)?;

        // Determine shape
        let shape = self.infer_shape(elements)?;

        // Convert f32 to f16
        let f16_values: Vec<f16> = values.into_iter().map(f16::from_f32).collect();

        // Create tensor from values
        let tensor = Tensor::from_vec_metal(self.env.metal_device(), f16_values, shape)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(tensor))
    }

    /// Collect all scalar values from nested arrays
    fn collect_scalars(&self, elements: &[TensorLiteral]) -> RuntimeResult<Vec<f32>> {
        let mut values = Vec::new();

        for elem in elements {
            match elem {
                TensorLiteral::Scalar(ScalarLiteral::Float(f)) => {
                    values.push(*f as f32);
                }
                TensorLiteral::Scalar(ScalarLiteral::Integer(i)) => {
                    values.push(*i as f32);
                }
                TensorLiteral::Array(nested) => {
                    values.extend(self.collect_scalars(nested)?);
                }
                _ => {
                    return Err(RuntimeError::NotImplemented(
                        "Only float/int arrays supported".to_string(),
                    ));
                }
            }
        }

        Ok(values)
    }

    /// Infer shape from nested array structure
    fn infer_shape(&self, elements: &[TensorLiteral]) -> RuntimeResult<Vec<usize>> {
        let mut shape = vec![elements.len()];

        if let TensorLiteral::Array(nested) = &elements[0] {
            let nested_shape = self.infer_shape(nested)?;
            shape.extend(nested_shape);
        }

        Ok(shape)
    }

    /// Evaluate a binary operation
    fn eval_binary_op(&self, op: &BinaryOp, left: Value, right: Value) -> RuntimeResult<Value> {
        match (left, right) {
            (Value::Tensor(l), Value::Tensor(r)) => {
                let result = match op {
                    BinaryOp::Add => l.add(&r),
                    BinaryOp::Sub => l.sub(&r),
                    BinaryOp::Mul => l.mul(&r),
                    BinaryOp::Div => l.div(&r),
                    BinaryOp::MatMul => l.matmul(&r),
                    BinaryOp::Power => {
                        return Err(RuntimeError::NotImplemented("Power not yet implemented".to_string()));
                    }
                    BinaryOp::TensorProd => {
                        return Err(RuntimeError::NotImplemented("Tensor product not yet implemented".to_string()));
                    }
                    BinaryOp::Hadamard => {
                        // Hadamard is element-wise multiplication (same as mul)
                        l.mul(&r)
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }
            (Value::Float(l), Value::Float(r)) => {
                let result = match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => {
                        if r == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        l / r
                    }
                    BinaryOp::Power => l.powf(r),
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for floats",
                            op
                        )));
                    }
                };
                Ok(Value::Float(result))
            }
            _ => Err(RuntimeError::TypeError(
                "Binary operation requires compatible types".to_string(),
            )),
        }
    }

    /// Evaluate a unary operation
    fn eval_unary_op(&self, op: &UnaryOp, operand: Value) -> RuntimeResult<Value> {
        match operand {
            Value::Tensor(t) => {
                let result = match op {
                    UnaryOp::Neg => {
                        // Negate tensor: -t
                        let zero = Tensor::zeros(self.env.metal_device(), t.shape().dims().to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        zero.sub(&t)
                    }
                    UnaryOp::Transpose => {
                        return Err(RuntimeError::NotImplemented("Transpose not yet implemented".to_string()));
                    }
                    UnaryOp::Inverse => {
                        return Err(RuntimeError::NotImplemented("Inverse not yet implemented".to_string()));
                    }
                    UnaryOp::Determinant => {
                        return Err(RuntimeError::NotImplemented("Determinant not yet implemented".to_string()));
                    }
                    UnaryOp::Not => {
                        return Err(RuntimeError::TypeError("Not operation requires boolean".to_string()));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }
            Value::Float(f) => {
                let result = match op {
                    UnaryOp::Neg => -f,
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for floats",
                            op
                        )));
                    }
                };
                Ok(Value::Float(result))
            }
            Value::Boolean(b) => {
                let result = match op {
                    UnaryOp::Not => !b,
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for booleans",
                            op
                        )));
                    }
                };
                Ok(Value::Boolean(result))
            }
            _ => Err(RuntimeError::TypeError(format!(
                "Unary operation {:?} not supported for this type",
                op
            ))),
        }
    }

    /// Evaluate a function call
    fn eval_function_call(&mut self, _name: &Identifier, _args: &[TensorExpr]) -> RuntimeResult<Value> {
        Err(RuntimeError::NotImplemented(
            "Function calls not yet implemented".to_string(),
        ))
    }

    /// Get a variable's value
    pub fn get_variable(&self, name: &str) -> RuntimeResult<&Value> {
        self.env.get_variable(name)
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
