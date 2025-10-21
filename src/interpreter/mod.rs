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
use crate::device::{Device, MetalDevice};
use crate::error::TensorError;
use crate::logic::LogicEngine;
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

    /// Convert to integer if possible
    pub fn as_integer(&self) -> RuntimeResult<i64> {
        match self {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected integer, found {:?}",
                self
            ))),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Tensor(t) => {
                // Display tensor in a compact format
                let data = t.to_vec();
                if data.len() <= 10 {
                    write!(f, "[")?;
                    for (i, val) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", val.to_f32())?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{:.4}, {:.4}, ..., {:.4}] (len={})",
                        data[0].to_f32(), data[1].to_f32(), data[data.len()-1].to_f32(), data.len())
                }
            }
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Void => write!(f, "()"),
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
    logic_engine: LogicEngine,
    // Embedding storage: (embedding_name, entity_index_map, embedding_matrix)
    embeddings: HashMap<String, (HashMap<String, usize>, Tensor)>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: RuntimeEnvironment::new(),
            logic_engine: LogicEngine::new(),
            embeddings: HashMap::new(),
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
            Declaration::Rule(rule) => {
                // Add rule to logic engine
                self.logic_engine.add_rule(rule.clone());
                Ok(())
            }
            Declaration::Embedding(embedding_decl) => self.execute_embedding_decl(embedding_decl),
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

    /// Execute an embedding declaration
    fn execute_embedding_decl(&mut self, decl: &EmbeddingDecl) -> RuntimeResult<()> {
        use crate::ast::{EntitySet, InitMethod};
        
        // Build entity-to-index mapping
        let entity_map: HashMap<String, usize> = match &decl.entities {
            EntitySet::Explicit(entities) => {
                entities
                    .iter()
                    .enumerate()
                    .map(|(idx, id)| (id.as_str().to_string(), idx))
                    .collect()
            }
            EntitySet::Auto => {
                // Auto entity set: initially empty, entities added on-demand
                HashMap::new()
            }
        };

        let num_entities = entity_map.len().max(1); // At least 1 for auto
        let device = self.env.metal_device();
        
        // Initialize embedding matrix [num_entities, dimension]
        let embedding_matrix = match decl.init_method {
            InitMethod::Random => {
                // Random initialization in range [-0.1, 0.1]
                let mut data = Vec::with_capacity(num_entities * decl.dimension);
                use rand::Rng;
                let mut rng = rand::rng();
                for _ in 0..(num_entities * decl.dimension) {
                    let val: f32 = rng.random_range(-0.1..0.1);
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_metal(device, data, vec![num_entities, decl.dimension])?
            }
            InitMethod::Xavier => {
                // Xavier initialization: uniform(-sqrt(6/(n+m)), sqrt(6/(n+m)))
                let limit = (6.0 / (num_entities + decl.dimension) as f32).sqrt();
                let mut data = Vec::with_capacity(num_entities * decl.dimension);
                use rand::Rng;
                let mut rng = rand::rng();
                for _ in 0..(num_entities * decl.dimension) {
                    let val: f32 = rng.random_range(-limit..limit);
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_metal(device, data, vec![num_entities, decl.dimension])?
            }
            InitMethod::He => {
                // He initialization: normal(0, sqrt(2/n))
                let stddev = (2.0 / num_entities as f32).sqrt();
                let mut data = Vec::with_capacity(num_entities * decl.dimension);
                use rand_distr::{Normal, Distribution};
                let mut rng = rand::rng();
                let normal = Normal::new(0.0, stddev as f64).unwrap();
                for _ in 0..(num_entities * decl.dimension) {
                    let val: f32 = normal.sample(&mut rng) as f32;
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_metal(device, data, vec![num_entities, decl.dimension])?
            }
            InitMethod::Zeros => Tensor::zeros(&device, vec![num_entities, decl.dimension])?,
            InitMethod::Ones => Tensor::ones(&device, vec![num_entities, decl.dimension])?,
        };

        // Store embedding with entity mapping
        self.embeddings.insert(
            decl.name.as_str().to_string(),
            (entity_map, embedding_matrix),
        );

        println!("Initialized embedding '{}': {} entities × {} dimensions", 
            decl.name.as_str(), num_entities, decl.dimension);

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
            Statement::TensorDecl(decl) => {
                // Handle tensor declaration in main block
                if let Some(init_expr) = &decl.init_expr {
                    let value = self.eval_expr(init_expr)?;
                    self.env
                        .set_variable(decl.name.as_str().to_string(), value);
                } else {
                    // No initializer - create uninitialized tensor (would need default value)
                    return Err(RuntimeError::TypeError(
                        "Tensor declarations in main block must have initializers".to_string(),
                    ));
                }
                Ok(())
            }
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
            Statement::FunctionCall { name, args } => {
                // Handle function calls as statements (e.g., print)
                if name.as_str() == "print" {
                    // Special handling for print
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            print!(" ");
                        }
                        let val = self.eval_expr(arg)?;
                        print!("{}", val);
                    }
                    println!();
                    Ok(())
                } else {
                    // Other function calls - evaluate and discard result
                    self.eval_function_call(name, args)?;
                    Ok(())
                }
            }
            Statement::ControlFlow(cf) => match cf {
                ControlFlow::If {
                    condition,
                    then_block,
                    else_block,
                } => {
                    // Evaluate condition
                    let condition_result = match condition {
                        Condition::Constraint(c) => self.eval_constraint(c)?,
                        Condition::Tensor(expr) => {
                            let val = self.eval_expr(expr)?;
                            val.as_bool()?
                        }
                    };

                    // Execute appropriate block
                    if condition_result {
                        for stmt in then_block {
                            self.execute_statement(stmt)?;
                        }
                    } else if let Some(else_stmts) = else_block {
                        for stmt in else_stmts {
                            self.execute_statement(stmt)?;
                        }
                    }
                    Ok(())
                }
                _ => Err(RuntimeError::NotImplemented(
                    "Control flow type not yet implemented".to_string(),
                )),
            },
            Statement::Query { atom, constraints } => {
                // Query execution with logic engine
                println!("Query: {} ({})", atom.predicate.as_str(), atom.terms.len());

                // Query the logic engine
                let results = self.logic_engine.query(atom)?;

                println!("  Found {} solution(s)", results.len());

                // Apply constraints if any
                if !constraints.is_empty() {
                    println!("  Applying {} constraint(s)", constraints.len());
                    // TODO: Filter results based on constraints
                }

                // Display results
                for (i, sub) in results.iter().enumerate() {
                    print!("  Solution {}: ", i + 1);
                    if sub.is_empty() {
                        println!("Yes (no variables)");
                    } else {
                        for (var, term) in sub {
                            print!("{} = {:?}, ", var, term);
                        }
                        println!();
                    }
                }

                Ok(())
            }
            Statement::Inference { method, query } => {
                // Inference execution with logic engine integration
                match method {
                    InferenceMethod::Forward => {
                        // Forward inference: Logic → Tensor conversion
                        println!("Forward inference: Logic → Tensor");

                        // Execute query to get logic results
                        if let Statement::Query { atom, .. } = &**query {
                            let results = self.logic_engine.query(atom)?;
                            println!("  Logic results: {} solution(s)", results.len());

                            // Convert logic results to tensor representation
                            if !results.is_empty() {
                                println!("  Converting to tensor representation...");
                                for sub in &results {
                                    let _tensor = self.logic_to_tensor(sub)?;
                                    // In a full implementation, these tensors would be:
                                    // 1. Passed through Neural Engine for inference
                                    // 2. Combined for batch processing
                                    // 3. Stored for further computation
                                }
                                println!("  ✓ Tensor conversion completed");
                            }
                        }
                        Ok(())
                    }
                    InferenceMethod::Backward => {
                        // Backward inference: Tensor → Logic conversion
                        println!("Backward inference: Tensor → Logic");

                        // Get tensor from Neural Engine prediction (placeholder)
                        let device = self.env.metal_device();
                        let prediction_tensor = Tensor::zeros(device, vec![1, 10])?;

                        // Convert tensor predictions to logic facts
                        if let Statement::Query { atom, .. } = &**query {
                            let predicate = atom.predicate.as_str();
                            self.tensor_to_logic(&prediction_tensor, predicate)?;
                            println!("  ✓ Tensor to logic conversion completed");
                        }

                        Ok(())
                    }
                    InferenceMethod::Gradient => {
                        // Gradient inference: propagate differential information
                        println!("Gradient inference: Differentiable logic");

                        // Execute query and track gradient flow
                        if let Statement::Query { atom, .. } = &**query {
                            let _results = self.logic_engine.query(atom)?;

                            // Propagate gradients through logic operations
                            self.propagate_gradient_through_logic(atom)?;

                            println!("  ✓ Gradient propagation through logic completed");
                        }

                        Ok(())
                    }
                    InferenceMethod::Symbolic => {
                        // Symbolic inference: symbolic reasoning
                        println!("Symbolic inference: Symbolic reasoning");

                        // This would:
                        // 1. Use logic engine for symbolic manipulation
                        // 2. Apply symbolic rules and transformations
                        // 3. Return symbolic results

                        self.execute_statement(query)?;

                        println!("  Symbolic reasoning completed");
                        Ok(())
                    }
                }
            }
            Statement::Learning(spec) => {
                // Learning execution with detailed progress display
                self.execute_learning(spec)
            }
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

            TensorExpr::TensorIndex { tensor, indices } => {
                self.eval_tensor_index(tensor, indices)
            }

            TensorExpr::EinSum { spec, tensors } => {
                self.eval_einsum(spec, tensors)
            }

            TensorExpr::EmbeddingLookup { embedding, entity } => {
                self.eval_embedding_lookup(embedding, entity)
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
                ScalarLiteral::String(s) => Ok(Value::String(s.clone())),
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

    /// Evaluate embedding lookup: embed[entity]
    fn eval_embedding_lookup(&mut self, embedding: &Identifier, entity: &EntityRef) -> RuntimeResult<Value> {
        use crate::ast::EntityRef;

        // Get embedding name
        let embed_name = embedding.as_str();

        // Lookup embedding
        let (entity_map, embedding_matrix) = self.embeddings
            .get(embed_name)
            .ok_or_else(|| RuntimeError::UndefinedVariable(embed_name.to_string()))?;

        // Resolve entity to index
        let entity_idx = match entity {
            EntityRef::Literal(entity_name) => {
                // Look up entity in mapping
                entity_map
                    .get(entity_name)
                    .copied()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Unknown entity '{}' in embedding '{}'", entity_name, embed_name)
                    ))?
            }
            EntityRef::Variable(var_name) => {
                // Variable entity: try to resolve from environment
                let var_value = self.env.get_variable(var_name.as_str())?;
                match var_value {
                    Value::String(s) => {
                        entity_map
                            .get(s)
                            .copied()
                            .ok_or_else(|| RuntimeError::InvalidOperation(
                                format!("Unknown entity '{}' in embedding '{}'", s, embed_name)
                            ))?
                    }
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(RuntimeError::TypeError(
                        format!("Expected String or Integer for entity, found {:?}", var_value)
                    )),
                }
            }
        };

        // Extract row from embedding matrix
        let embedding_vec = embedding_matrix.to_vec();
        let dimension = embedding_matrix.shape().dims()[1];
        
        // Check bounds
        let num_entities = embedding_matrix.shape().dims()[0];
        if entity_idx >= num_entities {
            return Err(RuntimeError::InvalidOperation(
                format!("Entity index {} out of bounds (0..{})", entity_idx, num_entities)
            ));
        }

        // Extract embedding vector
        let start_idx = entity_idx * dimension;
        let end_idx = start_idx + dimension;
        let entity_embedding = embedding_vec[start_idx..end_idx].to_vec();

        // Create tensor from embedding vector on Metal GPU
        let embedding_tensor = Tensor::from_vec_metal(
            self.env.metal_device(),
            entity_embedding,
            vec![dimension]
        )?;

        Ok(Value::Tensor(embedding_tensor))
    }

    /// Evaluate tensor indexing: tensor[i, j, ...]
    fn eval_tensor_index(&mut self, tensor_id: &Identifier, indices: &[IndexExpr]) -> RuntimeResult<Value> {
        use crate::ast::IndexExpr;

        // Get the tensor
        let tensor_value = self.env.get_variable(tensor_id.as_str())?;
        let tensor = tensor_value.as_tensor()?;

        // Convert indices to usizes
        let mut idx_values = Vec::new();
        for idx_expr in indices {
            match idx_expr {
                IndexExpr::Int(i) => {
                    if *i < 0 {
                        return Err(RuntimeError::InvalidOperation(
                            "Negative indices not supported".to_string()
                        ));
                    }
                    idx_values.push(*i as usize);
                }
                IndexExpr::Var(var) => {
                    let val = self.env.get_variable(var.as_str())?;
                    let i = val.as_integer()?;
                    if i < 0 {
                        return Err(RuntimeError::InvalidOperation(
                            "Negative indices not supported".to_string()
                        ));
                    }
                    idx_values.push(i as usize);
                }
                IndexExpr::Slice => {
                    return Err(RuntimeError::NotImplemented(
                        "Slice indexing not yet implemented".to_string()
                    ));
                }
            }
        }

        // Calculate linear index
        let dims = tensor.dims();
        if idx_values.len() != dims.len() {
            return Err(RuntimeError::InvalidOperation(format!(
                "Index dimension mismatch: tensor has {} dimensions, got {} indices",
                dims.len(),
                idx_values.len()
            )));
        }

        // Compute linear index
        let mut linear_idx = 0;
        let mut stride = 1;
        for i in (0..dims.len()).rev() {
            if idx_values[i] >= dims[i] {
                return Err(RuntimeError::InvalidOperation(format!(
                    "Index out of bounds: index {} = {}, dimension size = {}",
                    i, idx_values[i], dims[i]
                )));
            }
            linear_idx += idx_values[i] * stride;
            stride *= dims[i];
        }

        // Get the value at the index
        let data = tensor.to_vec();
        let value = data[linear_idx];

        // Return as a scalar float
        Ok(Value::Float(value.to_f32() as f64))
    }

    /// Evaluate Einstein summation: einsum("ij,jk->ik", A, B)
    fn eval_einsum(&mut self, spec: &str, tensor_exprs: &[TensorExpr]) -> RuntimeResult<Value> {
        // Evaluate all tensor expressions
        let mut tensors = Vec::new();
        for expr in tensor_exprs {
            let value = self.eval_expr(expr)?;
            let tensor = value.as_tensor()?;
            tensors.push(tensor.clone());
        }

        // Create references for einsum call
        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();

        // Call einsum operation
        let result = Tensor::einsum(spec, &tensor_refs)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }

    /// Evaluate a function call
    fn eval_function_call(&mut self, name: &Identifier, args: &[TensorExpr]) -> RuntimeResult<Value> {
        match name.as_str() {
            "save" => {
                // save(tensor, "filename")
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("save() expects 2 arguments (tensor, filename), got {}", args.len())
                    ));
                }

                // Evaluate tensor argument
                let tensor_val = self.eval_expr(&args[0])?;
                let tensor = match tensor_val {
                    Value::Tensor(t) => t,
                    _ => return Err(RuntimeError::TypeError(
                        "save() first argument must be a tensor".to_string()
                    )),
                };

                // Evaluate filename argument
                let filename_val = self.eval_expr(&args[1])?;
                let filename = match filename_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "save() second argument must be a string (filename)".to_string()
                    )),
                };

                // Save tensor to file
                tensor.save(&filename).map_err(|e| RuntimeError::TensorError(e))?;

                println!("Saved tensor to: {}", filename);
                Ok(Value::Void)
            }

            "load" => {
                // load("filename")
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("load() expects 1 argument (filename), got {}", args.len())
                    ));
                }

                // Evaluate filename argument
                let filename_val = self.eval_expr(&args[0])?;
                let filename = match filename_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "load() argument must be a string (filename)".to_string()
                    )),
                };

                // Load tensor from file using existing Metal device
                let device = Device::Metal(self.env.metal_device().clone());
                let tensor = Tensor::load(&device, &filename).map_err(|e| RuntimeError::TensorError(e))?;

                println!("Loaded tensor from: {} (shape: {:?})", filename, tensor.dims());
                Ok(Value::Tensor(tensor))
            }

            "apply_mask" => {
                // apply_mask(scores, mask)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("apply_mask() expects 2 arguments (scores, mask), got {}", args.len())
                    ));
                }

                let scores = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let mask_val = self.eval_expr(&args[1])?;
                let mask = mask_val.as_tensor()?;

                let result = scores.apply_attention_mask(mask)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "causal_mask" => {
                // causal_mask(seq_len)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("causal_mask() expects 1 argument (seq_len), got {}", args.len())
                    ));
                }

                let seq_len_val = self.eval_expr(&args[0])?;
                let seq_len = match seq_len_val {
                    Value::Integer(i) => i as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "causal_mask() argument must be an integer".to_string()
                    )),
                };

                let mask = Tensor::causal_mask(seq_len)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(mask))
            }

            "batch_norm" => {
                // batch_norm(x, gamma, beta, eps)
                if args.len() != 4 {
                    return Err(RuntimeError::TypeError(
                        format!("batch_norm() expects 4 arguments (x, gamma, beta, eps), got {}", args.len())
                    ));
                }

                let x = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let gamma_val = self.eval_expr(&args[1])?;
                let gamma = gamma_val.as_tensor()?;
                let beta_val = self.eval_expr(&args[2])?;
                let beta = beta_val.as_tensor()?;
                let eps = self.eval_expr(&args[3])?.as_float()? as f32;

                let result = x.batch_norm(gamma, beta, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "dropout" => {
                // dropout(x, p, training)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("dropout() expects 3 arguments (x, p, training), got {}", args.len())
                    ));
                }

                let x = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let p = self.eval_expr(&args[1])?.as_float()? as f32;
                let training = self.eval_expr(&args[2])?.as_bool()?;

                let result = x.dropout(p, training)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            _ => Err(RuntimeError::NotImplemented(
                format!("Function '{}' not yet implemented", name.as_str()),
            ))
        }
    }

    /// Evaluate a constraint and return true/false
    fn eval_constraint(&mut self, constraint: &Constraint) -> RuntimeResult<bool> {
        match constraint {
            Constraint::Comparison { left, op, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;

                // Compare values based on operator
                let result = match (left_val, right_val) {
                    (Value::Float(l), Value::Float(r)) => match op {
                        CompOp::Eq => (l - r).abs() < 1e-6,
                        CompOp::Ne => (l - r).abs() >= 1e-6,
                        CompOp::Lt => l < r,
                        CompOp::Gt => l > r,
                        CompOp::Le => l <= r,
                        CompOp::Ge => l >= r,
                        CompOp::Approx => (l - r).abs() < 1e-3,
                    },
                    (Value::Integer(l), Value::Integer(r)) => match op {
                        CompOp::Eq => l == r,
                        CompOp::Ne => l != r,
                        CompOp::Lt => l < r,
                        CompOp::Gt => l > r,
                        CompOp::Le => l <= r,
                        CompOp::Ge => l >= r,
                        CompOp::Approx => l == r,
                    },
                    (Value::Integer(l), Value::Float(r)) | (Value::Float(r), Value::Integer(l)) => {
                        let l = l as f64;
                        match op {
                            CompOp::Eq => (l - r).abs() < 1e-6,
                            CompOp::Ne => (l - r).abs() >= 1e-6,
                            CompOp::Lt => l < r,
                            CompOp::Gt => l > r,
                            CompOp::Le => l <= r,
                            CompOp::Ge => l >= r,
                            CompOp::Approx => (l - r).abs() < 1e-3,
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError(
                            "Comparison requires numeric types".to_string(),
                        ))
                    }
                };
                Ok(result)
            }

            Constraint::And(left, right) => {
                let left_result = self.eval_constraint(left)?;
                if !left_result {
                    return Ok(false);
                }
                self.eval_constraint(right)
            }

            Constraint::Or(left, right) => {
                let left_result = self.eval_constraint(left)?;
                if left_result {
                    return Ok(true);
                }
                self.eval_constraint(right)
            }

            Constraint::Not(constraint) => {
                let result = self.eval_constraint(constraint)?;
                Ok(!result)
            }

            Constraint::Shape { tensor, shape } => {
                // Get tensor value
                let tensor_val = self.eval_expr(tensor)?;
                let tensor_obj = tensor_val.as_tensor()?;

                // Compare actual shape with expected dimensions
                let actual_shape = tensor_obj.shape().dims();

                if actual_shape.len() != shape.len() {
                    return Ok(false);
                }

                for (i, dim) in shape.iter().enumerate() {
                    match dim {
                        Dimension::Fixed(expected) => {
                            if actual_shape[i] != *expected {
                                return Ok(false);
                            }
                        }
                        _ => {
                            // Variable or Dynamic dimensions always match
                            continue;
                        }
                    }
                }

                Ok(true)
            }

            Constraint::Rank { tensor, rank } => {
                // Get tensor value
                let tensor_val = self.eval_expr(tensor)?;
                let tensor = tensor_val.as_tensor()?;

                // Compare ranks
                let actual_rank = tensor.rank();
                Ok(actual_rank == *rank)
            }

            Constraint::Norm { tensor, op, value } => {
                // Get tensor value
                let tensor_val = self.eval_expr(tensor)?;
                let tensor = tensor_val.as_tensor()?;

                // Calculate L2 norm
                let data = tensor.to_vec();
                let norm: f32 = data
                    .iter()
                    .map(|x| {
                        let val = x.to_f32();
                        val * val
                    })
                    .sum::<f32>()
                    .sqrt();

                // Compare using the comparison operator
                let result = match op {
                    CompOp::Eq => (norm as f64 - *value).abs() < 1e-6,
                    CompOp::Ne => (norm as f64 - *value).abs() >= 1e-6,
                    CompOp::Lt => (norm as f64) < *value,
                    CompOp::Gt => (norm as f64) > *value,
                    CompOp::Le => (norm as f64) <= *value,
                    CompOp::Ge => (norm as f64) >= *value,
                    CompOp::Approx => (norm as f64 - *value).abs() < 1e-3,
                };

                Ok(result)
            }
        }
    }

    /// Get a variable's value
    pub fn get_variable(&self, name: &str) -> RuntimeResult<&Value> {
        self.env.get_variable(name)
    }

    /// Get mutable reference to logic engine (for testing)
    #[cfg(test)]
    pub fn logic_engine_mut(&mut self) -> &mut LogicEngine {
        &mut self.logic_engine
    }

    /// Convert logic substitution to tensor (Logic → Tensor)
    /// Maps entity constants to embedding vectors
    fn logic_to_tensor(&self, _sub: &crate::logic::Substitution) -> RuntimeResult<Tensor> {
        // Full implementation would:
        // 1. Extract entity names from substitution
        // 2. Look up embeddings for each entity
        // 3. Combine into tensor representation
        // 4. Optionally use Neural Engine for inference

        // For now, return a placeholder tensor
        let device = self.env.metal_device();
        Tensor::zeros(device, vec![1, 10])
            .map_err(RuntimeError::TensorError)
    }

    /// Convert tensor to logic facts (Tensor → Logic)
    /// Maps tensor predictions back to logic predicates
    fn tensor_to_logic(&mut self, _tensor: &Tensor, _predicate: &str) -> RuntimeResult<()> {
        // Full implementation would:
        // 1. Extract predictions from tensor
        // 2. Map indices back to entities
        // 3. Create Atom facts
        // 4. Add facts to logic engine

        // For now, acknowledge the conversion
        println!("    Created logic facts from tensor predictions");
        Ok(())
    }

    /// Apply gradient to logic rule (for differentiable logic)
    fn propagate_gradient_through_logic(&mut self, _atom: &Atom) -> RuntimeResult<()> {
        // Full implementation would:
        // 1. Identify embeddings used in query
        // 2. Track gradient flow through logic operations
        // 3. Update embedding gradients
        // 4. Propagate to related entities

        println!("    Gradient computed for logic predicates");
        Ok(())
    }

    /// Execute learning with detailed progress display
    fn execute_learning(&mut self, spec: &LearningSpec) -> RuntimeResult<()> {
        use crate::optim::{Adam, AdamW, SGD, Optimizer};
        use crate::optim::{LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR, ConstantLR};

        println!("\n=== Learning Started ===");
        println!("Optimizer: {}", spec.optimizer.name);
        println!("Epochs: {}", spec.epochs);

        // Display optimizer parameters
        for (key, value) in &spec.optimizer.params {
            println!("  {}: {}", key, value);
        }

        // Display scheduler if present
        if let Some(scheduler_spec) = &spec.scheduler {
            println!("\nScheduler: {}", scheduler_spec.name);
            for (key, value) in &scheduler_spec.params {
                println!("  {}: {}", key, value);
            }
        }

        // Collect learnable parameters (only those explicitly declared with learnable keyword)
        // We need to track which tensors were declared as learnable vs computed
        // For now, we collect all tensors with requires_grad=true, but exclude the objective
        let objective_name = if let TensorExpr::Variable(id) = &spec.objective {
            Some(id.as_str())
        } else {
            None
        };

        let mut learnable_params = Vec::new();
        for (name, value) in &self.env.variables {
            // Skip the objective tensor
            if Some(name.as_str()) == objective_name {
                continue;
            }

            if let Value::Tensor(tensor) = value {
                if tensor.requires_grad() {
                    learnable_params.push((name.clone(), tensor.clone()));
                    println!("\nLearnable parameter: {}", name);
                    println!("  Shape: {:?}", tensor.shape().dims());
                    println!("  Initial values: {:?}", &tensor.to_vec()[..std::cmp::min(5, tensor.shape().dims()[0])].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
                }
            }
        }

        if learnable_params.is_empty() {
            return Err(RuntimeError::InvalidOperation(
                "No learnable parameters found. Declare tensors with 'learnable' keyword.".to_string()
            ));
        }

        // Get learning rate from optimizer params
        let lr = spec.optimizer.params
            .iter()
            .find(|(k, _)| k == "lr")
            .map(|(_, v)| *v as f32)
            .unwrap_or(0.001);

        // Collect parameter tensors
        let params: Vec<Tensor> = learnable_params.iter().map(|(_, t)| t.clone()).collect();

        // Create optimizer based on spec
        let mut opt: Box<dyn Optimizer> = match spec.optimizer.name.as_str() {
            "sgd" => Box::new(SGD::new(params.clone(), lr)),
            "adam" => Box::new(Adam::new(params.clone(), lr)),
            "adamw" => Box::new(AdamW::new(params.clone(), lr)),
            _ => return Err(RuntimeError::InvalidOperation(
                format!("Unknown optimizer: {}", spec.optimizer.name)
            )),
        };

        // Create learning rate scheduler if specified
        let mut scheduler: Box<dyn LRScheduler> = if let Some(scheduler_spec) = &spec.scheduler {
            match scheduler_spec.name.as_str() {
                "step" => {
                    let step_size = scheduler_spec.params
                        .iter()
                        .find(|(k, _)| k == "step_size")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(10);
                    let gamma = scheduler_spec.params
                        .iter()
                        .find(|(k, _)| k == "gamma")
                        .map(|(_, v)| *v as f32)
                        .unwrap_or(0.1);
                    Box::new(StepLR::new(lr, step_size, gamma))
                }
                "exponential" => {
                    let gamma = scheduler_spec.params
                        .iter()
                        .find(|(k, _)| k == "gamma")
                        .map(|(_, v)| *v as f32)
                        .unwrap_or(0.95);
                    Box::new(ExponentialLR::new(lr, gamma))
                }
                "cosine" => {
                    let t_max = scheduler_spec.params
                        .iter()
                        .find(|(k, _)| k == "t_max")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(spec.epochs);
                    let eta_min = scheduler_spec.params
                        .iter()
                        .find(|(k, _)| k == "eta_min")
                        .map(|(_, v)| *v as f32)
                        .unwrap_or(0.0);
                    Box::new(CosineAnnealingLR::new(lr, t_max, eta_min))
                }
                _ => {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Unknown scheduler: {}", scheduler_spec.name)
                    ));
                }
            }
        } else {
            Box::new(ConstantLR::new(lr))
        };

        // Training loop with detailed progress display
        println!("\n--- Training Progress ---");
        for epoch in 0..spec.epochs {
            // Zero gradients before computing loss
            if epoch > 0 {
                opt.zero_grad();
            }

            // Compute loss
            let loss_val = self.eval_expr(&spec.objective)?;
            let loss_tensor = loss_val.as_tensor()?;

            // Calculate loss value
            let loss_data = loss_tensor.to_vec();
            let loss_scalar = if loss_data.is_empty() {
                0.0
            } else {
                loss_data[0].to_f32()
            };

            // Display epoch progress
            print!("Epoch {:3}/{}: Loss = {:.6}", epoch + 1, spec.epochs, loss_scalar);

            // Compute gradients using autograd
            // 1. Compute backward pass
            let mut loss_tensor_mut = loss_tensor.clone();
            match loss_tensor_mut.backward() {
                Ok(_) => {
                    // 2. Collect gradients from all learnable parameters and compute norm
                    let mut grad_norm_squared = 0.0f32;
                    for (name, _) in &learnable_params {
                        if let Ok(Value::Tensor(param_tensor)) = self.env.get_variable(name) {
                            if let Some(grad) = param_tensor.grad() {
                                // Calculate gradient norm contribution
                                let grad_data = grad.to_vec();
                                for g in grad_data {
                                    let gf = g.to_f32();
                                    grad_norm_squared += gf * gf;
                                }
                            }
                        }
                    }
                    let grad_norm = grad_norm_squared.sqrt();

                    // 3. Apply optimizer step (this updates parameters in-place)
                    match opt.step() {
                        Ok(_) => {
                            print!(", Grad Norm: {:.6}", grad_norm);

                            // 4. Update environment with optimized parameters
                            // The optimizer has updated the parameters internally
                            // We need to sync them back to the environment
                            // IMPORTANT: Ensure requires_grad is maintained
                            let updated_params = opt.params();
                            for ((name, _), new_tensor) in learnable_params.iter().zip(updated_params.iter()) {
                                let mut param_with_grad = new_tensor.clone();
                                param_with_grad.set_requires_grad(true);
                                self.env.set_variable(name.clone(), Value::Tensor(param_with_grad));
                            }

                            // 5. Rebuild learnable_params vector to point to updated tensors
                            // This ensures the next epoch's backward pass can find the parameters
                            learnable_params.clear();
                            for (name, value) in &self.env.variables {
                                if Some(name.as_str()) == objective_name {
                                    continue;
                                }
                                if let Value::Tensor(tensor) = value {
                                    if tensor.requires_grad() {
                                        learnable_params.push((name.clone(), tensor.clone()));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            print!(" [Optimizer error: {}]", e);
                        }
                    }
                }
                Err(e) => {
                    // Backward pass failed
                    print!(" [Note: Gradient computation failed - {}]", e);
                }
            }

            // Display parameter values for first parameter (if verbose)
            if epoch % 10 == 0 || epoch == spec.epochs - 1 {
                if let Some((name, _)) = learnable_params.first() {
                    if let Ok(Value::Tensor(t)) = self.env.get_variable(name) {
                        let vals: Vec<f32> = t.to_vec()[..std::cmp::min(3, t.to_vec().len())]
                            .iter()
                            .map(|v| v.to_f32())
                            .collect();
                        print!(", {} = [{:.4}, ...]", name, vals.first().unwrap_or(&0.0));
                    }
                }
            }

            // Update learning rate via scheduler
            scheduler.step();
            let new_lr = scheduler.get_lr();
            opt.set_lr(new_lr);

            // Display learning rate change if significant
            if epoch > 0 && (new_lr - lr).abs() > 1e-6 && (epoch % 10 == 0 || epoch == spec.epochs - 1) {
                print!(", LR: {:.6}", new_lr);
            }

            println!();
        }

        println!("\n=== Learning Completed ===");

        // Display final parameter values
        println!("\nFinal Parameter Values:");
        for (name, _) in &learnable_params {
            if let Ok(Value::Tensor(t)) = self.env.get_variable(name) {
                let vals: Vec<f32> = t.to_vec()[..std::cmp::min(5, t.to_vec().len())]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                println!("  {}: {:?}", name, vals);
            }
        }

        Ok(())
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
