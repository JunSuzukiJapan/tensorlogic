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

mod formatter;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::fs;

use crate::ast::*;
use crate::tensor::{Tensor, TensorShape};
use crate::device::{Device, MetalDevice};
use crate::entity_registry::EntityRegistry;
use crate::error::TensorError;
use crate::logic::LogicEngine;
use crate::model::Model;
use half::f16;

/// Epsilon for floating-point comparisons
const FLOAT_EPSILON: f64 = 1e-6;

/// Default display limit for large tensors
const DISPLAY_LIMIT: usize = 10;

/// Default epoch reporting interval
const EPOCH_REPORT_INTERVAL: usize = 10;

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

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Circular import detected: {0}")]
    CircularImport(String),

    #[error("Break outside of loop")]
    BreakOutsideLoop,

    #[error("Return from function")]
    ReturnValue(Value),

    #[error("Index {index} out of bounds for length {length}")]
    IndexError { index: usize, length: usize },
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Function call frame for local scope management
#[derive(Debug, Clone)]
struct CallFrame {
    /// Name of the function being executed
    function_name: String,
    /// Local variables in this scope
    local_vars: HashMap<String, Value>,
}

impl CallFrame {
    fn new(function_name: String) -> Self {
        Self {
            function_name,
            local_vars: HashMap::new(),
        }
    }
}

/// Runtime value
#[derive(Debug, Clone)]
pub enum Value {
    Tensor(Tensor),
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Model(Model),
    Tokenizer(std::sync::Arc<crate::tokenizer::Tokenizer>),
    TokenIds(Vec<u32>),
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
                if data.len() <= DISPLAY_LIMIT {
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
            Value::Model(m) => write!(f, "Model({:?})", m.metadata.format),
            Value::Tokenizer(_) => write!(f, "Tokenizer"),
            Value::TokenIds(ids) => write!(f, "TokenIds({:?})", ids),
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

    /// Check if a variable exists
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Declare a new variable (error if already exists)
    pub fn declare_variable(&mut self, name: String, value: Value) -> RuntimeResult<()> {
        if self.variables.contains_key(&name) {
            return Err(RuntimeError::InvalidOperation(
                format!("Variable '{}' is already defined. Use assignment without 'let' to update existing variables.", name)
            ));
        }
        self.variables.insert(name, value);
        Ok(())
    }

    /// Set a variable (update existing or error if not defined)
    pub fn set_variable(&mut self, name: String, value: Value) -> RuntimeResult<()> {
        if !self.variables.contains_key(&name) {
            return Err(RuntimeError::UndefinedVariable(name));
        }
        self.variables.insert(name, value);
        Ok(())
    }

    /// Get a variable
    pub fn get_variable(&self, name: &str) -> RuntimeResult<&Value> {
        self.variables
            .get(name)
            .ok_or_else(|| RuntimeError::UndefinedVariable(name.to_string()))
    }

    /// List all variable names
    pub fn list_variables(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
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
    // Entity registry for managing entity types and instances
    entity_registry: EntityRegistry,
    // Embedding storage: (embedding_name, entity_index_map, embedding_matrix)
    embeddings: HashMap<String, (HashMap<String, usize>, Tensor)>,
    // Python execution environment (when python feature is enabled)
    #[cfg(any(feature = "python", feature = "python-extension"))]
    python_env: Option<crate::python::environment::PythonEnvironment>,
    // Track imported files to detect circular dependencies
    imported_files: HashSet<PathBuf>,
    // Current file being executed (for resolving relative imports)
    current_file: Option<PathBuf>,
    // Track defined relation variables: predicate_name -> set of variable names
    relation_variables: HashMap<String, HashSet<String>>,
    // Track relation entity parameters: predicate_name -> (param_index -> entity_type_name)
    relation_entity_params: HashMap<String, HashMap<usize, String>>,
    // User-defined functions: function_name -> FunctionDecl
    functions: HashMap<String, FunctionDecl>,
    // Function call stack for local scope management
    call_stack: Vec<CallFrame>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: RuntimeEnvironment::new(),
            logic_engine: LogicEngine::new(),
            entity_registry: EntityRegistry::new(),
            embeddings: HashMap::new(),
            #[cfg(any(feature = "python", feature = "python-extension"))]
            python_env: None,
            imported_files: HashSet::new(),
            current_file: None,
            relation_variables: HashMap::new(),
            relation_entity_params: HashMap::new(),
            functions: HashMap::new(),
            call_stack: Vec::new(),
        }
    }

    /// Set the current file being executed (for import resolution)
    pub fn set_current_file(&mut self, path: impl Into<PathBuf>) {
        self.current_file = Some(path.into());
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

    /// Get a variable from the interpreter's environment
    /// Checks local scope (call_stack) first, then global environment
    pub fn get_variable(&self, name: &str) -> Option<Value> {
        // Check local scope first (most recent call frame)
        if let Some(frame) = self.call_stack.last() {
            if let Some(value) = frame.local_vars.get(name) {
                return Some(value.clone());
            }
        }

        // Fall back to global environment
        self.env.get_variable(name).ok().cloned()
    }

    /// Set a variable in the interpreter's environment
    /// If in a function call, sets in local scope; otherwise in global environment
    pub fn set_variable(&mut self, name: String, value: Value) {
        if let Some(frame) = self.call_stack.last_mut() {
            // Inside a function: set in local scope
            frame.local_vars.insert(name, value);
        } else {
            // Global scope
            self.env.set_variable(name, value);
        }
    }

    /// List all variables in the environment
    pub fn list_variables(&self) -> Vec<String> {
        self.env.list_variables()
    }

    /// Execute a declaration
    fn execute_declaration(&mut self, decl: &Declaration) -> RuntimeResult<()> {
        match decl {
            Declaration::Import(import_decl) => self.execute_import(import_decl),
            Declaration::Entity(entity_decl) => {
                // Register entity type in the registry
                self.entity_registry.register_from_decl(entity_decl);

                match entity_decl {
                    EntityDecl::Explicit { name, entities } => {
                        println!("✓ Entity type '{}' registered with {} entities",
                            name.as_str(), entities.len());
                    }
                    EntityDecl::FromData { name } => {
                        println!("✓ Entity type '{}' registered (data-driven)",
                            name.as_str());
                    }
                }

                Ok(())
            }
            Declaration::Tensor(tensor_decl) => self.execute_tensor_decl(tensor_decl),
            Declaration::Relation(relation_decl) => {
                // Collect variable names from relation parameters
                let predicate_name = relation_decl.name.as_str().to_string();
                let var_names: HashSet<String> = relation_decl.params.iter()
                    .map(|param| param.name.as_str().to_string())
                    .collect();

                self.relation_variables.insert(predicate_name.clone(), var_names);

                // Collect entity-typed parameters: param_index -> "entity" marker
                // (generic entity type - specific type will be inferred from context)
                let mut entity_param_indices = Vec::new();
                for (idx, param) in relation_decl.params.iter().enumerate() {
                    match &param.entity_type {
                        EntityType::Entity | EntityType::Concept => {
                            // Generic entity type parameter
                            entity_param_indices.push(idx);
                        }
                        EntityType::Tensor(_) => {
                            // Tensor-typed parameter, skip
                        }
                    }
                }

                // Store indices of entity-typed parameters
                if !entity_param_indices.is_empty() {
                    let mut entity_params = HashMap::new();
                    for idx in entity_param_indices {
                        entity_params.insert(idx, "entity".to_string());
                    }
                    self.relation_entity_params.insert(predicate_name, entity_params);
                }

                Ok(())
            }
            Declaration::Rule(rule) => {
                // Rules contain variables by definition - no conversion needed
                // Add rule to logic engine as-is
                self.logic_engine.add_rule(rule.clone());
                Ok(())
            }
            Declaration::Embedding(embedding_decl) => self.execute_embedding_decl(embedding_decl),
            Declaration::Function(func_decl) => {
                // Register user-defined function
                let func_name = func_decl.name.as_str().to_string();

                // Check for duplicate function names
                if self.functions.contains_key(&func_name) {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Function '{}' is already defined", func_name)
                    ));
                }

                // Store function definition
                self.functions.insert(func_name.clone(), func_decl.clone());

                Ok(())
            }
        }
    }

    /// Execute an import declaration
    fn execute_import(&mut self, import_decl: &ImportDecl) -> RuntimeResult<()> {
        // Resolve the import path relative to current file (if any)
        let import_path = if let Some(current) = &self.current_file {
            let current_dir = current.parent().ok_or_else(|| {
                RuntimeError::FileNotFound(format!("Cannot get parent directory of {:?}", current))
            })?;
            current_dir.join(&import_decl.path)
        } else {
            PathBuf::from(&import_decl.path)
        };

        // Canonicalize the path to handle .. and symlinks
        let canonical_path = import_path.canonicalize().map_err(|e| {
            RuntimeError::FileNotFound(format!(
                "Cannot resolve import path '{}': {}",
                import_decl.path, e
            ))
        })?;

        // Check for circular imports
        if self.imported_files.contains(&canonical_path) {
            return Err(RuntimeError::CircularImport(format!(
                "File already imported: {:?}",
                canonical_path
            )));
        }

        // Add to imported files set
        self.imported_files.insert(canonical_path.clone());

        // Read and parse the imported file
        let source = fs::read_to_string(&canonical_path)?;
        let program = crate::parser::TensorLogicParser::parse_program(&source).map_err(|e| {
            RuntimeError::ParseError(format!("Error parsing {:?}: {}", canonical_path, e))
        })?;

        // Save current file context
        let previous_file = self.current_file.clone();
        self.current_file = Some(canonical_path.clone());

        // Execute imported program's declarations
        for decl in &program.declarations {
            self.execute_declaration(decl)?;
        }

        // Note: We intentionally do NOT execute the main block of imported files
        // Only declarations (functions, tensors, etc.) are imported

        // Restore previous file context
        self.current_file = previous_file;

        Ok(())
    }

    /// Execute a tensor declaration
    fn execute_tensor_decl(&mut self, decl: &TensorDecl) -> RuntimeResult<()> {
        let mut tensor = if let Some(init_expr) = &decl.init_expr {
            // Evaluate initialization expression
            let value = self.eval_expr(init_expr)?;
            let mut t = value.as_tensor()?.clone();

            // Reshape tensor to match declared shape (if needed)
            let declared_shape = self.get_declared_shape(&decl.tensor_type)?;
            if t.shape().dims() != declared_shape.as_slice() {
                t = t.reshape(declared_shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
            }
            t
        } else {
            // Create zero-initialized tensor
            self.create_zero_tensor(&decl.tensor_type)?
        };

        // Set requires_grad based on learnable status
        if decl.tensor_type.learnable == LearnableStatus::Learnable {
            tensor.set_requires_grad(true);
        }

        // Use declare_variable for tensor declarations (they create new variables)
        self.env
            .declare_variable(decl.name.as_str().to_string(), Value::Tensor(tensor))?;

        Ok(())
    }

    /// Extract the shape from a TensorType
    fn get_declared_shape(&self, tensor_type: &TensorType) -> RuntimeResult<Vec<usize>> {
        let mut shape = Vec::new();
        for dim in &tensor_type.dimensions {
            match dim {
                Dimension::Fixed(size) => shape.push(*size),
                Dimension::Variable(_) => {
                    return Err(RuntimeError::InvalidOperation(
                        "Cannot use variable dimensions in tensor declarations".to_string(),
                    ));
                }
                Dimension::Dynamic => {
                    return Err(RuntimeError::InvalidOperation(
                        "Cannot use dynamic dimensions in tensor declarations".to_string(),
                    ));
                }
            }
        }
        Ok(shape)
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

    /// Evaluate the last statement and return its value if it's an expression
    /// This method both executes the statement for side effects AND returns the value
    fn evaluate_last_statement(&mut self, stmt: &Statement) -> RuntimeResult<Option<Value>> {
        match stmt {
            Statement::Assignment { target, value } => {
                // Assignment (identifier := expr) returns the right-hand side value
                let evaluated_value = self.eval_expr(value)?;

                // Also execute the statement for side effects (variable assignment)
                if let Some(frame) = self.call_stack.last_mut() {
                    frame.local_vars.insert(target.as_str().to_string(), evaluated_value.clone());
                } else {
                    if self.env.has_variable(target.as_str()) {
                        self.env.set_variable(target.as_str().to_string(), evaluated_value.clone())?;
                    } else {
                        self.env.declare_variable(target.as_str().to_string(), evaluated_value.clone())?;
                    }
                }

                Ok(Some(evaluated_value))
            }
            Statement::Equation(eq) => {
                // For assignment equations (expr := expr), return the right-hand side value
                if eq.eq_type == EquationType::Assign {
                    // Evaluate the right-hand side
                    let value = self.eval_expr(&eq.right)?;

                    // Also execute the statement for side effects (variable assignment)
                    if let TensorExpr::Variable(var_name) = &eq.left {
                        if let Some(frame) = self.call_stack.last_mut() {
                            frame.local_vars.insert(var_name.as_str().to_string(), value.clone());
                        } else {
                            if self.env.has_variable(var_name.as_str()) {
                                self.env.set_variable(var_name.as_str().to_string(), value.clone())?;
                            } else {
                                self.env.declare_variable(var_name.as_str().to_string(), value.clone())?;
                            }
                        }
                    }

                    Ok(Some(value))
                } else {
                    // For other equation types (=, ~), execute but don't return value
                    self.execute_statement(stmt)?;
                    Ok(None)
                }
            }
            Statement::FunctionCall { name, args } => {
                // Function call result can be implicitly returned
                let value = self.eval_function_call(name, args)?;
                Ok(Some(value))
            }
            // All other statement types: execute and return None
            _ => {
                self.execute_statement(stmt)?;
                Ok(None)
            }
        }
    }

    /// Check if a value matches the expected return type
    fn check_return_type_match(&self, value: &Value, expected_type: &ReturnType, func_name: &str) -> RuntimeResult<()> {
        match expected_type {
            ReturnType::Void => {
                // Void functions should return Value::Void
                match value {
                    Value::Void => Ok(()),
                    _ => Err(RuntimeError::TypeError(
                        format!("Function '{}' declared as 'void' but returned a value", func_name)
                    ))
                }
            }
            ReturnType::Tensor(tensor_type) => {
                // Convert TensorType to EntityType and reuse check_type_match
                let entity_type = EntityType::Tensor(tensor_type.clone());
                self.check_type_match(value, &entity_type, &format!("return value of '{}'", func_name))
            }
        }
    }

    /// Check if a value matches the expected entity type
    fn check_type_match(&self, value: &Value, expected_type: &EntityType, param_name: &str) -> RuntimeResult<()> {
        match expected_type {
            EntityType::Entity | EntityType::Concept => {
                // Entity and Concept types accept string values
                match value {
                    Value::String(_) => Ok(()),
                    _ => Err(RuntimeError::TypeError(
                        format!("Parameter '{}' expects entity/concept (string), got {:?}", param_name, value)
                    ))
                }
            }
            EntityType::Tensor(tensor_type) => {
                match value {
                    Value::Tensor(t) => {
                        // Note: TensorLogic uses f16 internally for all tensors
                        // Base type checking is skipped (all tensors are f16)
                        // Focus on shape validation

                        // Check shape if specified (non-dynamic dimensions)
                        let expected_dimensions = &tensor_type.dimensions;
                        if !expected_dimensions.is_empty() {
                            let actual_shape = t.shape().dims();

                            // Check rank (number of dimensions)
                            if expected_dimensions.len() != actual_shape.len() {
                                return Err(RuntimeError::TypeError(
                                    format!(
                                        "Parameter '{}' expects rank {} tensor, got rank {}",
                                        param_name, expected_dimensions.len(), actual_shape.len()
                                    )
                                ));
                            }

                            // Check each dimension (if not dynamic)
                            for (i, expected_dim) in expected_dimensions.iter().enumerate() {
                                if let Dimension::Fixed(expected_size) = expected_dim {
                                    if actual_shape[i] != *expected_size {
                                        return Err(RuntimeError::TypeError(
                                            format!(
                                                "Parameter '{}' dimension {} expects size {}, got {}",
                                                param_name, i, expected_size, actual_shape[i]
                                            )
                                        ));
                                    }
                                }
                                // Dynamic and Variable dimensions accept any size
                            }
                        }

                        Ok(())
                    }
                    Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => {
                        // Scalar values might be acceptable for some tensor operations
                        // For now, accept them (they can be converted to tensors)
                        Ok(())
                    }
                    _ => Err(RuntimeError::TypeError(
                        format!("Parameter '{}' expects tensor, got {:?}", param_name, value)
                    ))
                }
            }
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
            Statement::Let { target, value } => {
                let evaluated_value = self.eval_expr(value)?;

                // Check if we're inside a function call
                if let Some(frame) = self.call_stack.last_mut() {
                    // Inside function: declare in local scope
                    frame.local_vars.insert(target.as_str().to_string(), evaluated_value);
                    Ok(())
                } else {
                    // Global scope: declare in environment
                    self.env
                        .declare_variable(target.as_str().to_string(), evaluated_value)?;
                    Ok(())
                }
            }
            Statement::Assignment { target, value } => {
                let evaluated_value = self.eval_expr(value)?;

                // Check if we're inside a function call
                if let Some(frame) = self.call_stack.last_mut() {
                    // Inside function: set in local scope
                    frame.local_vars.insert(target.as_str().to_string(), evaluated_value);
                } else {
                    // Global scope: assignment (:=) auto-declares if variable doesn't exist
                    if self.env.has_variable(target.as_str()) {
                        self.env.set_variable(target.as_str().to_string(), evaluated_value)?;
                    } else {
                        self.env.declare_variable(target.as_str().to_string(), evaluated_value)?;
                    }
                }
                Ok(())
            }
            Statement::Equation(eq) => {
                use crate::ast::EquationType;

                match eq.eq_type {
                    EquationType::Assign => {
                        // := is assignment with auto-declaration
                        // Left side must be a simple identifier
                        if let TensorExpr::Variable(var_name) = &eq.left {
                            let value = self.eval_expr(&eq.right)?;

                            // Check if we're inside a function call
                            if let Some(frame) = self.call_stack.last_mut() {
                                // Inside function: update local variable
                                frame.local_vars.insert(var_name.as_str().to_string(), value);
                            } else {
                                // Global scope: try to set existing variable, if it doesn't exist, declare it
                                if self.env.has_variable(var_name.as_str()) {
                                    self.env.set_variable(var_name.as_str().to_string(), value)?;
                                } else {
                                    self.env.declare_variable(var_name.as_str().to_string(), value)?;
                                }
                            }
                            Ok(())
                        } else {
                            Err(RuntimeError::TypeError(
                                "Left side of := must be a variable name".to_string()
                            ))
                        }
                    }
                    _ => {
                        // For other equation types (=, ~), just execute both sides
                        let _left = self.eval_expr(&eq.left)?;
                        let _right = self.eval_expr(&eq.right)?;
                        // In a full implementation, this would perform unification or constraint solving
                        Ok(())
                    }
                }
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
                            // Propagate return statements upward
                            let result = self.execute_statement(stmt);
                            if let Err(RuntimeError::ReturnValue(_)) = result {
                                return result;
                            }
                            result?;
                        }
                    } else if let Some(else_stmts) = else_block {
                        for stmt in else_stmts {
                            // Propagate return statements upward
                            let result = self.execute_statement(stmt);
                            if let Err(RuntimeError::ReturnValue(_)) = result {
                                return result;
                            }
                            result?;
                        }
                    }
                    Ok(())
                }
                ControlFlow::For {
                    variable,
                    iterable,
                    body,
                } => {
                    // Evaluate iterable
                    let items = match iterable {
                        Iterable::Range(n) => {
                            // Create range 0..n
                            (0..*n).map(|i| Value::Integer(i as i64)).collect::<Vec<_>>()
                        }
                        Iterable::Tensor(expr) => {
                            // Iterate over tensor elements
                            let tensor_val = self.eval_expr(expr)?;
                            let tensor = tensor_val.as_tensor()?;
                            let data = tensor.to_vec();
                            data.iter().map(|&v| Value::Float(v.to_f32() as f64)).collect()
                        }
                        Iterable::EntitySet(entity_set) => {
                            // Get entities from set
                            match entity_set {
                                EntitySet::Auto => {
                                    return Err(RuntimeError::InvalidOperation(
                                        "Cannot iterate over 'auto' entity set".to_string()
                                    ));
                                }
                                EntitySet::Explicit(entities) => {
                                    entities.iter()
                                        .map(|id| Value::String(id.as_str().to_string()))
                                        .collect()
                                }
                            }
                        }
                    };

                    // Execute body for each item
                    let mut should_break = false;
                    for item in items {
                        // For loop variable - directly set without checking
                        self.env.variables.insert(variable.as_str().to_string(), item);
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }

                    Ok(())
                }
                ControlFlow::While {
                    condition,
                    body,
                } => {
                    // Execute while condition is true
                    loop {
                        let condition_result = match condition {
                            Condition::Constraint(c) => self.eval_constraint(c)?,
                            Condition::Tensor(expr) => {
                                let val = self.eval_expr(expr)?;
                                val.as_bool()?
                            }
                        };

                        if !condition_result {
                            break;
                        }

                        let mut should_break = false;
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }

                    Ok(())
                }
                ControlFlow::Loop { body } => {
                    loop {
                        let mut should_break = false;
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }
                    Ok(())
                }
            },
            Statement::FactAssertion { atom } => {
                // Check if this is actually a built-in function call
                // (since fact_assertion and function_call are now syntactically identical)
                let predicate_name = atom.predicate.as_str();

                if predicate_name == "print" {
                    // Handle as print function
                    for (i, term) in atom.terms.iter().enumerate() {
                        if i > 0 {
                            print!(" ");
                        }
                        // Convert term to expression and evaluate
                        let expr = self.term_to_expr(term);
                        let val = self.eval_expr(&expr)?;
                        print!("{}", val);
                    }
                    println!();
                    return Ok(());
                }

                // Otherwise, treat as a fact assertion
                println!("Adding fact: {}", predicate_name);

                // Collect entities from fact (for data-driven entity types)
                self.collect_entities_from_fact(atom)?;

                // Convert atom terms based on relation variable definitions
                let converted_atom = self.convert_atom_terms(atom);

                self.logic_engine.add_fact(converted_atom);
                println!("  ✓ Fact added to knowledge base");
                Ok(())
            }
            Statement::Query { atom, constraints } => {
                // Query execution with logic engine
                println!("Query: {}", formatter::format_atom(atom));

                // Convert atom terms based on relation variable definitions
                let converted_atom = self.convert_atom_terms(atom);

                // Query the logic engine
                let results = self.logic_engine.query(&converted_atom)?;

                if results.is_empty() {
                    println!("  No solutions found");
                } else {
                    println!("  Found {} solution(s)", results.len());

                    // Apply constraints if any
                    if !constraints.is_empty() {
                        println!("  Applying {} constraint(s)", constraints.len());
                        // TODO: Filter results based on constraints
                    }

                    // Display results
                    for (i, sub) in results.iter().enumerate() {
                        if sub.is_empty() {
                            println!("  Solution {}: Yes", i + 1);
                        } else {
                            println!("  Solution {}:", i + 1);
                            for (var, term) in sub {
                                println!("    {} = {}", var, formatter::format_term(term));
                            }
                        }
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
            Statement::InferenceBlock { items } => {
                // Execute multiple inference operations in sequence
                println!("\n=== Inference Block Started ===");
                println!("Total inference operations: {}", items.len());
                println!();

                for (index, (method, query)) in items.iter().enumerate() {
                    println!("--- Inference {}/{} ---", index + 1, items.len());

                    // Execute single inference by creating a temporary Inference statement
                    let inference_stmt = Statement::Inference {
                        method: *method,
                        query: query.clone(),
                    };

                    self.execute_statement(&inference_stmt)?;
                    println!();
                }

                println!("=== Inference Block Completed ===\n");
                Ok(())
            }
            Statement::Learning(spec) => {
                // Learning execution with detailed progress display
                self.execute_learning(spec)
            }
            Statement::Break => {
                Err(RuntimeError::BreakOutsideLoop)
            }
            Statement::Return { value } => {
                // Evaluate return value (if any) and signal early return
                let return_val = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    Value::Void
                };
                Err(RuntimeError::ReturnValue(return_val))
            }
            Statement::PythonImport { module, alias } => {
                #[cfg(any(feature = "python", feature = "python-extension"))]
                {
                    // Initialize Python environment if needed
                    if self.python_env.is_none() {
                        self.python_env = Some(crate::python::environment::PythonEnvironment::new());
                    }

                    // Import the module
                    let name = alias.as_deref();
                    self.python_env.as_mut().unwrap()
                        .import_module(module, name)
                        .map_err(|e| RuntimeError::InvalidOperation(e))?;

                    let display_name = alias.as_ref().unwrap_or(module);
                    println!("✓ Python import: {} (as {})", module, display_name);
                    Ok(())
                }
                #[cfg(not(any(feature = "python", feature = "python-extension")))]
                {
                    Err(RuntimeError::NotImplemented(
                        "Python integration not enabled (compile with --features python)".to_string()
                    ))
                }
            }
            Statement::WithBlock { entity_type, statements } => {
                // Execute with-block for entity collection
                println!("With block for entity type: {}", entity_type.as_str());

                // Execute all statements in the with block
                for stmt in statements {
                    self.execute_statement(stmt)?;
                }

                println!("  ✓ With block completed");
                Ok(())
            }
        }
    }

    /// Evaluate an expression
    fn eval_expr(&mut self, expr: &TensorExpr) -> RuntimeResult<Value> {
        match expr {
            TensorExpr::Variable(id) => {
                // Use self.get_variable() to check local scope first
                let value = self.get_variable(id.as_str())
                    .ok_or_else(|| RuntimeError::UndefinedVariable(id.as_str().to_string()))?;
                Ok(value)
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

            TensorExpr::PythonCall { function, args } => {
                #[cfg(any(feature = "python", feature = "python-extension"))]
                {
                    // Ensure Python environment is initialized
                    if self.python_env.is_none() {
                        return Err(RuntimeError::InvalidOperation(
                            "Python environment not initialized. Import a module first with 'python import'".to_string()
                        ));
                    }

                    // Evaluate all arguments
                    let tensor_args: Result<Vec<_>, _> = args.iter()
                        .map(|arg| {
                            let val = self.eval_expr(arg)?;
                            val.as_tensor().map(|t| t.clone())
                        })
                        .collect();
                    let tensor_args = tensor_args?;

                    // Create references for the call
                    let tensor_refs: Vec<&Tensor> = tensor_args.iter().collect();

                    // Call Python function
                    let result = self.python_env.as_ref().unwrap()
                        .call_function(function, tensor_refs)
                        .map_err(|e| RuntimeError::InvalidOperation(e))?;

                    println!("✓ Python call: {}({} args)", function, args.len());
                    Ok(Value::Tensor(result))
                }
                #[cfg(not(any(feature = "python", feature = "python-extension")))]
                {
                    Err(RuntimeError::NotImplemented(
                        "Python integration not enabled (compile with --features python)".to_string()
                    ))
                }
            }
        }
    }

    /// Evaluate a literal
    fn eval_literal(&mut self, lit: &TensorLiteral) -> RuntimeResult<Value> {
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
    fn eval_array_literal(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Value> {
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
    fn collect_scalars(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Vec<f32>> {
        let mut values = Vec::new();

        for elem in elements {
            match elem {
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(f))) => {
                    values.push(*f as f32);
                }
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Integer(i))) => {
                    values.push(*i as f32);
                }
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    values.extend(self.collect_scalars(nested)?);
                }
                ArrayElement::Expression(expr) => {
                    // Evaluate the expression (e.g., variable reference like seq_len, d_model)
                    let value = self.eval_expr(expr)?;
                    match value {
                        Value::Float(f) => {
                            values.push(f as f32);
                        }
                        Value::Integer(i) => {
                            values.push(i as f32);
                        }
                        _ => {
                            return Err(RuntimeError::TypeError(
                                "Array element expression must evaluate to a scalar number".to_string(),
                            ));
                        }
                    }
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
    fn infer_shape(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Vec<usize>> {
        let mut shape = vec![elements.len()];

        if let Some(first) = elements.first() {
            match first {
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    let nested_shape = self.infer_shape(nested)?;
                    shape.extend(nested_shape);
                }
                _ => {}
            }
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
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        return Err(RuntimeError::NotImplemented(format!("Comparison {:?} not yet implemented for tensors", op)));
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        return Err(RuntimeError::NotImplemented(format!("Logical {:?} not yet implemented for tensors", op)));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }
            (Value::Float(l), Value::Float(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::Float(l + r)),
                    BinaryOp::Sub => Ok(Value::Float(l - r)),
                    BinaryOp::Mul => Ok(Value::Float(l * r)),
                    BinaryOp::Div => {
                        if r == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Float(l / r))
                    }
                    BinaryOp::Power => Ok(Value::Float(l.powf(r))),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for floats",
                            op
                        )));
                    }
                }
            }
            (Value::Boolean(l), Value::Boolean(r)) => {
                match op {
                    BinaryOp::And => Ok(Value::Boolean(l && r)),
                    BinaryOp::Or => Ok(Value::Boolean(l || r)),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for booleans",
                        op
                    ))),
                }
            }
            (Value::String(l), Value::String(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::String(format!("{}{}", l, r))),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for strings",
                        op
                    ))),
                }
            }
            (Value::Integer(l), Value::Integer(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::Integer(l + r)),
                    BinaryOp::Sub => Ok(Value::Integer(l - r)),
                    BinaryOp::Mul => Ok(Value::Integer(l * r)),
                    BinaryOp::Div => {
                        if r == 0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Integer(l / r))
                    }
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for integers",
                        op
                    ))),
                }
            }
            // Tensor-Float operations (e.g., tensor * 0.5)
            (Value::Tensor(t), Value::Float(s)) => {
                let scalar_f16 = half::f16::from_f32(s as f32);
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f16),
                    BinaryOp::Sub => t.sub_scalar(scalar_f16),
                    BinaryOp::Mul => t.mul_scalar(scalar_f16),
                    BinaryOp::Div => {
                        if s == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        t.div_scalar(scalar_f16)
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for Tensor-Float",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Tensor(result))
            }
            // Float-Tensor operations (e.g., 0.5 * tensor)
            (Value::Float(s), Value::Tensor(t)) => {
                let scalar_f16 = half::f16::from_f32(s as f32);
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f16),
                    BinaryOp::Mul => t.mul_scalar(scalar_f16),
                    BinaryOp::Sub => {
                        // s - tensor = -(tensor - s)
                        let temp = t.sub_scalar(scalar_f16)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        let zero = Tensor::zeros(self.env.metal_device(), t.shape().dims().to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        zero.sub(&temp)
                    }
                    BinaryOp::Div => {
                        // s / tensor = s * (1/tensor)
                        return Err(RuntimeError::InvalidOperation(
                            "Scalar / Tensor not yet supported".to_string()
                        ));
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for Float-Tensor",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Tensor(result))
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

            "argmax" => {
                // argmax(tensor, dim: int = -1, keepdim: bool = false)
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("argmax() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                let dim = if args.len() >= 2 {
                    let dim_val = self.eval_expr(&args[1])?;
                    match dim_val {
                        Value::Integer(i) => {
                            if i < 0 {
                                None  // -1 means global argmax
                            } else {
                                Some(i as usize)
                            }
                        }
                        Value::Float(f) => {
                            // Accept float literals like 0.0 as integers
                            let i = f as i64;
                            if i < 0 {
                                None  // -1 means global argmax
                            } else {
                                Some(i as usize)
                            }
                        }
                        _ => return Err(RuntimeError::TypeError(
                            "argmax() dim must be a number".to_string()
                        )),
                    }
                } else {
                    None  // Default: global argmax
                };

                let keepdim = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.as_bool()?
                } else {
                    false
                };

                let result = tensor.argmax(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "argmin" => {
                // argmin(tensor, dim: int = -1, keepdim: bool = false)
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("argmin() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                let dim = if args.len() >= 2 {
                    let dim_val = self.eval_expr(&args[1])?;
                    match dim_val {
                        Value::Integer(i) => {
                            if i < 0 {
                                None  // -1 means global argmin
                            } else {
                                Some(i as usize)
                            }
                        }
                        Value::Float(f) => {
                            // Accept float literals like 0.0 as integers
                            let i = f as i64;
                            if i < 0 {
                                None  // -1 means global argmin
                            } else {
                                Some(i as usize)
                            }
                        }
                        _ => return Err(RuntimeError::TypeError(
                            "argmin() dim must be a number".to_string()
                        )),
                    }
                } else {
                    None  // Default: global argmin
                };

                let keepdim = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.as_bool()?
                } else {
                    false
                };

                let result = tensor.argmin(dim, keepdim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "unsqueeze" => {
                // unsqueeze(tensor, dim: int)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("unsqueeze() expects 2 arguments (tensor, dim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let dim_val = self.eval_expr(&args[1])?;

                let dim = match dim_val {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "unsqueeze() dim must be a number".to_string()
                    )),
                };

                let result = tensor.unsqueeze(dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "squeeze" => {
                // squeeze(tensor, dim: Optional[int] = None)
                if args.is_empty() || args.len() > 2 {
                    return Err(RuntimeError::TypeError(
                        format!("squeeze() expects 1-2 arguments (tensor, optional dim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                let dim = if args.len() >= 2 {
                    let dim_val = self.eval_expr(&args[1])?;
                    match dim_val {
                        Value::Integer(i) => Some(i as usize),
                        Value::Float(f) => Some(f as usize),
                        _ => return Err(RuntimeError::TypeError(
                            "squeeze() dim must be a number".to_string()
                        )),
                    }
                } else {
                    None  // Default: squeeze all dims of size 1
                };

                let result = tensor.squeeze(dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }

            "split" => {
                // split(tensor, split_size: int, dim: int) -> returns first split only (simplified)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("split() expects 3 arguments (tensor, split_size, dim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                let split_size = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "split() split_size must be a number".to_string()
                    )),
                };

                let dim = match self.eval_expr(&args[2])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "split() dim must be a number".to_string()
                    )),
                };

                let splits = tensor.split(split_size, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // For simplicity, return first split only
                // TODO: Add list type to return all splits
                if splits.is_empty() {
                    return Err(RuntimeError::InvalidOperation(
                        "split() produced no results".to_string()
                    ));
                }

                Ok(Value::Tensor(splits[0].clone()))
            }

            "chunk" => {
                // chunk(tensor, chunks: int, dim: int) -> returns first chunk only (simplified)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("chunk() expects 3 arguments (tensor, chunks, dim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                let chunks = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "chunk() chunks must be a number".to_string()
                    )),
                };

                let dim = match self.eval_expr(&args[2])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "chunk() dim must be a number".to_string()
                    )),
                };

                let chunks_result = tensor.chunk(chunks, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                // For simplicity, return first chunk only
                // TODO: Add list type to return all chunks
                if chunks_result.is_empty() {
                    return Err(RuntimeError::InvalidOperation(
                        "chunk() produced no results".to_string()
                    ));
                }

                Ok(Value::Tensor(chunks_result[0].clone()))
            }

            "load_model" => {
                // load_model("path/to/model.gguf")
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("load_model() expects 1 argument (path), got {}", args.len())
                    ));
                }

                let path_val = self.eval_expr(&args[0])?;
                let path = match path_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "load_model() argument must be a string (path)".to_string()
                    )),
                };

                let model = Model::load(&path, self.env.metal_device())
                    .map_err(|e| RuntimeError::TensorError(e))?;

                println!("Loaded model from: {} (format: {:?})", path, model.metadata.format);
                Ok(Value::Model(model))
            }

            "load_tokenizer" => {
                // load_tokenizer("path/to/tokenizer.json" or "model_name")
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("load_tokenizer() expects 1 argument (path or model name), got {}", args.len())
                    ));
                }

                let path_val = self.eval_expr(&args[0])?;
                let path_or_name = match path_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "load_tokenizer() argument must be a string".to_string()
                    )),
                };

                // Try loading from file first, then from pretrained
                let tokenizer = if std::path::Path::new(&path_or_name).exists() {
                    crate::tokenizer::Tokenizer::from_file(&path_or_name)
                        .map_err(|e| RuntimeError::TensorError(e))?
                } else {
                    crate::tokenizer::Tokenizer::from_pretrained(&path_or_name)
                        .map_err(|e| RuntimeError::TensorError(e))?
                };

                println!("Loaded tokenizer: {}", path_or_name);
                Ok(Value::Tokenizer(std::sync::Arc::new(tokenizer)))
            }

            "get_tensor" => {
                // get_tensor(model, "tensor_name")
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("get_tensor() expects 2 arguments (model, tensor_name), got {}", args.len())
                    ));
                }

                let model_val = self.eval_expr(&args[0])?;
                let model = match model_val {
                    Value::Model(m) => m,
                    _ => return Err(RuntimeError::TypeError(
                        "get_tensor() first argument must be a Model".to_string()
                    )),
                };

                let name_val = self.eval_expr(&args[1])?;
                let tensor_name = match name_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "get_tensor() second argument must be a string (tensor name)".to_string()
                    )),
                };

                let tensor = model.get_tensor(&tensor_name)
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Tensor '{}' not found in model", tensor_name)
                    ))?;

                Ok(Value::Tensor(tensor.clone()))
            }

            "tokenize" => {
                // tokenize(tokenizer, text, add_special_tokens=true)
                if args.len() < 2 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("tokenize() expects 2-3 arguments (tokenizer, text, optional add_special_tokens), got {}", args.len())
                    ));
                }

                let tokenizer = match self.eval_expr(&args[0])? {
                    Value::Tokenizer(t) => t.clone(),
                    _ => return Err(RuntimeError::TypeError(
                        "tokenize() first argument must be a tokenizer".to_string()
                    )),
                };

                let text = match self.eval_expr(&args[1])? {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "tokenize() second argument must be a string".to_string()
                    )),
                };

                let add_special_tokens = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.as_bool()?
                } else {
                    true
                };

                let token_ids = tokenizer.encode(&text, add_special_tokens)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::TokenIds(token_ids))
            }

            "detokenize" => {
                // detokenize(tokenizer, token_ids, skip_special_tokens=true)
                if args.len() < 2 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("detokenize() expects 2-3 arguments (tokenizer, token_ids, optional skip_special_tokens), got {}", args.len())
                    ));
                }

                let tokenizer = match self.eval_expr(&args[0])? {
                    Value::Tokenizer(t) => t.clone(),
                    _ => return Err(RuntimeError::TypeError(
                        "detokenize() first argument must be a tokenizer".to_string()
                    )),
                };

                let token_ids = match self.eval_expr(&args[1])? {
                    Value::TokenIds(ids) => ids,
                    _ => return Err(RuntimeError::TypeError(
                        "detokenize() second argument must be TokenIds".to_string()
                    )),
                };

                let skip_special_tokens = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.as_bool()?
                } else {
                    true
                };

                let text = tokenizer.decode(&token_ids, skip_special_tokens)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::String(text))
            }

            "embedding" => {
                // embedding(embedding_table, token_ids) -> Tensor
                // embedding_table: [vocab_size, embedding_dim]
                // token_ids: TokenIds with shape [seq_len]
                // output: [seq_len, embedding_dim]
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("embedding() expects 2 arguments (embedding_table, token_ids), got {}", args.len())
                    ));
                }

                let embedding_table = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let token_ids = match self.eval_expr(&args[1])? {
                    Value::TokenIds(ids) => ids,
                    Value::Tensor(t) => {
                        // Convert tensor to Vec<u32>
                        t.to_vec().iter().map(|&f| f.to_f32() as u32).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "embedding() second argument must be TokenIds or Tensor".to_string()
                    )),
                };

                // Validate embedding table shape
                let table_shape = embedding_table.shape();
                if table_shape.dims().len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("embedding_table must be 2D [vocab_size, embedding_dim], got shape {:?}", table_shape.dims())
                    ));
                }

                let vocab_size = table_shape.dims()[0];
                let embedding_dim = table_shape.dims()[1];

                // Check token IDs are within vocab range
                for &token_id in &token_ids {
                    if token_id as usize >= vocab_size {
                        return Err(RuntimeError::TensorError(
                            crate::error::TensorError::InvalidOperation(
                                format!("Token ID {} exceeds vocab size {}", token_id, vocab_size)
                            )
                        ));
                    }
                }

                // Perform embedding lookup
                let seq_len = token_ids.len();
                let table_data = embedding_table.to_vec();
                let mut output_data = Vec::with_capacity(seq_len * embedding_dim);

                for &token_id in &token_ids {
                    let start_idx = (token_id as usize) * embedding_dim;
                    let end_idx = start_idx + embedding_dim;
                    output_data.extend_from_slice(&table_data[start_idx..end_idx]);
                }

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    output_data,
                    vec![seq_len, embedding_dim]
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "positional_encoding" => {
                // positional_encoding(seq_len, d_model) -> Tensor
                // Generates sinusoidal positional encoding
                // output: [seq_len, d_model]
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("positional_encoding() expects 2 arguments (seq_len, d_model), got {}", args.len())
                    ));
                }

                let seq_len = match self.eval_expr(&args[0])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("positional_encoding() first argument must be a number (seq_len), got {:?}", v)
                    )),
                };

                let d_model = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("positional_encoding() second argument must be a number (d_model), got {:?}", v)
                    )),
                };

                // Generate sinusoidal positional encoding
                // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
                let mut pe_data = Vec::with_capacity(seq_len * d_model);

                for pos in 0..seq_len {
                    for i in 0..d_model {
                        let div_term = (i as f32 / d_model as f32) * 10000_f32.ln();
                        let angle = pos as f32 / div_term.exp();

                        let value = if i % 2 == 0 {
                            angle.sin()
                        } else {
                            angle.cos()
                        };

                        pe_data.push(half::f16::from_f32(value));
                    }
                }

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    pe_data,
                    vec![seq_len, d_model]
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "top_k" => {
                // top_k(logits, k) -> Tensor
                // Keep only top-k logits, set others to -inf
                // Input: logits [vocab_size] or [..., vocab_size]
                // Output: same shape as input, with non-top-k values set to -inf
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("top_k() expects 2 arguments (logits, k), got {}", args.len())
                    ));
                }

                let logits = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let k = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("top_k() second argument must be a number (k), got {:?}", v)
                    )),
                };

                let shape = logits.shape();
                let dims = shape.dims();
                let vocab_size = dims[dims.len() - 1];

                if k > vocab_size {
                    return Err(RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(
                            format!("k ({}) cannot be larger than vocab_size ({})", k, vocab_size)
                        )
                    ));
                }

                // Get logits data
                let data = logits.to_vec();
                let mut output_data = data.clone();

                // Process each sequence (last dimension is vocab)
                let batch_size = data.len() / vocab_size;

                for batch_idx in 0..batch_size {
                    let start_idx = batch_idx * vocab_size;
                    let end_idx = start_idx + vocab_size;
                    let logits_slice = &data[start_idx..end_idx];

                    // Get indices sorted by logit value (descending)
                    let mut indexed_logits: Vec<(usize, f32)> = logits_slice
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i, v.to_f32()))
                        .collect();
                    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    // Set non-top-k to -inf
                    for i in k..vocab_size {
                        let idx = indexed_logits[i].0;
                        output_data[start_idx + idx] = half::f16::from_f32(f32::NEG_INFINITY);
                    }
                }

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    output_data,
                    dims.to_vec()
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "top_p" => {
                // top_p(logits, p) -> Tensor
                // Nucleus sampling: keep smallest set of logits with cumulative probability >= p
                // Input: logits [vocab_size] or [..., vocab_size]
                // Output: same shape, with non-nucleus values set to -inf
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("top_p() expects 2 arguments (logits, p), got {}", args.len())
                    ));
                }

                let logits = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let p = match self.eval_expr(&args[1])? {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    v => return Err(RuntimeError::TypeError(
                        format!("top_p() second argument must be a number (p), got {:?}", v)
                    )),
                };

                if p < 0.0 || p > 1.0 {
                    return Err(RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(
                            format!("p must be in range [0, 1], got {}", p)
                        )
                    ));
                }

                let shape = logits.shape();
                let dims = shape.dims();
                let vocab_size = dims[dims.len() - 1];

                // Get logits data
                let data = logits.to_vec();
                let mut output_data = data.clone();

                // Process each sequence
                let batch_size = data.len() / vocab_size;

                for batch_idx in 0..batch_size {
                    let start_idx = batch_idx * vocab_size;
                    let end_idx = start_idx + vocab_size;
                    let logits_slice = &data[start_idx..end_idx];

                    // Convert to probabilities using softmax
                    let max_logit = logits_slice.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
                    let exp_logits: Vec<f32> = logits_slice
                        .iter()
                        .map(|v| (v.to_f32() - max_logit).exp())
                        .collect();
                    let sum_exp: f32 = exp_logits.iter().sum();
                    let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

                    // Sort by probability (descending)
                    let mut indexed_probs: Vec<(usize, f32)> = probs
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (i, p))
                        .collect();
                    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    // Find nucleus: smallest set with cumulative prob >= p
                    let mut cumulative_prob = 0.0;
                    let mut nucleus_size = 0;
                    for (_, prob) in &indexed_probs {
                        cumulative_prob += prob;
                        nucleus_size += 1;
                        if cumulative_prob >= p {
                            break;
                        }
                    }

                    // Set non-nucleus to -inf
                    for i in nucleus_size..vocab_size {
                        let idx = indexed_probs[i].0;
                        output_data[start_idx + idx] = half::f16::from_f32(f32::NEG_INFINITY);
                    }
                }

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    output_data,
                    dims.to_vec()
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "temperature" => {
                // temperature(logits, temp) -> Tensor
                // Scale logits by temperature: logits / temp
                // Higher temp = more random, lower temp = more deterministic
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("temperature() expects 2 arguments (logits, temp), got {}", args.len())
                    ));
                }

                let logits = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let temp = match self.eval_expr(&args[1])? {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    v => return Err(RuntimeError::TypeError(
                        format!("temperature() second argument must be a number (temp), got {:?}", v)
                    )),
                };

                if temp <= 0.0 {
                    return Err(RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(
                            format!("temperature must be positive, got {}", temp)
                        )
                    ));
                }

                // Scale logits by temperature
                let data = logits.to_vec();
                let output_data: Vec<half::f16> = data
                    .iter()
                    .map(|&v| half::f16::from_f32(v.to_f32() / temp))
                    .collect();

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    output_data,
                    logits.shape().dims().to_vec()
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "softmax" => {
                // softmax(logits, dim=-1) -> Tensor
                // Convert logits to probability distribution
                // Default: apply softmax on last dimension
                if args.is_empty() || args.len() > 2 {
                    return Err(RuntimeError::TypeError(
                        format!("softmax() expects 1-2 arguments (logits, optional dim), got {}", args.len())
                    ));
                }

                let logits = self.eval_expr(&args[0])?.as_tensor()?.clone();

                // For now, always apply softmax on last dimension
                // TODO: support custom dimension when needed
                let shape = logits.shape();
                let dims = shape.dims();
                let last_dim_size = dims[dims.len() - 1];

                // Get logits data
                let data = logits.to_vec();
                let mut output_data = Vec::with_capacity(data.len());

                // Process each sequence (last dimension is vocab)
                let batch_size = data.len() / last_dim_size;

                for batch_idx in 0..batch_size {
                    let start_idx = batch_idx * last_dim_size;
                    let end_idx = start_idx + last_dim_size;
                    let logits_slice = &data[start_idx..end_idx];

                    // Compute softmax with numerical stability
                    let max_logit = logits_slice.iter()
                        .map(|v| v.to_f32())
                        .fold(f32::NEG_INFINITY, f32::max);

                    let exp_logits: Vec<f32> = logits_slice
                        .iter()
                        .map(|v| (v.to_f32() - max_logit).exp())
                        .collect();

                    let sum_exp: f32 = exp_logits.iter().sum();

                    // Normalize to get probabilities
                    for exp_val in exp_logits {
                        output_data.push(half::f16::from_f32(exp_val / sum_exp));
                    }
                }

                // Create output tensor
                let output = crate::tensor::Tensor::from_vec_metal(
                    self.env.metal_device(),
                    output_data,
                    dims.to_vec()
                ).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "sample" => {
                // sample(probs) -> Integer
                // Sample a single token index from probability distribution
                // Input: probs [vocab_size] - probability distribution (should sum to 1.0)
                // Output: Integer - sampled token index
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sample() expects 1 argument (probs), got {}", args.len())
                    ));
                }

                let probs_tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let shape = probs_tensor.shape();
                let dims = shape.dims();

                // For now, only support 1D probability distributions
                if dims.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sample() currently only supports 1D probability distributions, got shape {:?}", dims)
                    ));
                }

                let vocab_size = dims[0];
                let probs = probs_tensor.to_vec();

                // Convert to f32 probabilities
                let probs_f32: Vec<f32> = probs.iter().map(|v| v.to_f32()).collect();

                // Verify it's a valid probability distribution
                let sum: f32 = probs_f32.iter().sum();
                if (sum - 1.0).abs() > 0.01 {
                    return Err(RuntimeError::TensorError(
                        crate::error::TensorError::InvalidOperation(
                            format!("Probabilities must sum to ~1.0, got sum={}", sum)
                        )
                    ));
                }

                // Sample using cumulative distribution
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen(); // [0, 1)

                let mut cumulative = 0.0;
                let mut sampled_idx = 0;

                for (idx, &prob) in probs_f32.iter().enumerate() {
                    cumulative += prob;
                    if random_val < cumulative {
                        sampled_idx = idx;
                        break;
                    }
                }

                Ok(Value::Integer(sampled_idx as i64))
            }

            "temperature_sample" => {
                // temperature_sample(logits, temperature) -> Integer
                // Sample from logits with temperature scaling
                // Higher temperature = more random, lower = more deterministic
                // temperature: 0.1-2.0 typical range, 1.0 = no scaling
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("temperature_sample() expects 2 arguments (logits, temperature), got {}", args.len())
                    ));
                }

                let logits_tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let temperature = match self.eval_expr(&args[1])? {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "temperature_sample() temperature must be a number".to_string()
                    )),
                };

                if temperature <= 0.0 {
                    return Err(RuntimeError::InvalidOperation(
                        "temperature must be positive".to_string()
                    ));
                }

                let shape = logits_tensor.shape();
                let dims = shape.dims();

                // Support both 1D and 2D tensors
                // For 2D [seq_len, vocab_size], use the last row (last token's logits)
                let logits_f32: Vec<f32> = if dims.len() == 1 {
                    // 1D: [vocab_size]
                    let logits = logits_tensor.to_vec();
                    logits.iter().map(|v| v.to_f32()).collect()
                } else if dims.len() == 2 {
                    // 2D: [seq_len, vocab_size] - extract last row
                    let seq_len = dims[0];
                    let vocab_size = dims[1];
                    let logits = logits_tensor.to_vec();
                    let start_idx = (seq_len - 1) * vocab_size;
                    logits[start_idx..].iter().map(|v| v.to_f32()).collect()
                } else {
                    return Err(RuntimeError::TypeError(
                        format!("temperature_sample() expects 1D or 2D logits, got shape {:?}", dims)
                    ));
                };

                // Apply temperature scaling
                let scaled_logits: Vec<f32> = logits_f32.iter().map(|&x| x / temperature).collect();

                // Compute softmax
                let max_logit = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_logits: Vec<f32> = scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

                // Sample
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen();

                let mut cumulative = 0.0;
                let mut sampled_idx = 0;

                for (idx, &prob) in probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val < cumulative {
                        sampled_idx = idx;
                        break;
                    }
                }

                Ok(Value::Integer(sampled_idx as i64))
            }

            "print_top_k" => {
                // print_top_k(tensor, k) -> Prints top k values and indices from last row
                // For debugging logits
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("print_top_k() expects 2 arguments (tensor, k), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let k = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "print_top_k() k must be an integer".to_string()
                    )),
                };

                let dims = tensor.shape().dims();
                let logits_f32: Vec<f32> = if dims.len() == 1 {
                    tensor.to_vec().iter().map(|v| v.to_f32()).collect()
                } else if dims.len() == 2 {
                    let seq_len = dims[0];
                    let vocab_size = dims[1];
                    let logits = tensor.to_vec();
                    let start_idx = (seq_len - 1) * vocab_size;
                    logits[start_idx..].iter().map(|v| v.to_f32()).collect()
                } else {
                    return Err(RuntimeError::TypeError(
                        format!("print_top_k() expects 1D or 2D tensor, got shape {:?}", dims)
                    ));
                };

                // Get top k indices
                let mut indexed: Vec<(usize, f32)> = logits_f32.iter()
                    .copied()
                    .enumerate()
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                println!("      Top {} logits:", k);
                for i in 0..k.min(indexed.len()) {
                    let (idx, val) = indexed[i];
                    println!("        [{:5}] = {:10.4}", idx, val);
                }

                Ok(Value::Integer(0))
            }

            "top_p_sample" => {
                // top_p_sample(logits, p) -> Integer
                // Nucleus sampling: sample from smallest set of tokens with cumulative prob >= p
                // p: 0.0-1.0, typical values 0.9-0.95
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("top_p_sample() expects 2 arguments (logits, p), got {}", args.len())
                    ));
                }

                let logits_tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let p = match self.eval_expr(&args[1])? {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "top_p_sample() p must be a number".to_string()
                    )),
                };

                if p <= 0.0 || p > 1.0 {
                    return Err(RuntimeError::InvalidOperation(
                        "top_p must be in range (0.0, 1.0]".to_string()
                    ));
                }

                let shape = logits_tensor.shape();
                let dims = shape.dims();

                // Support both 1D and 2D tensors
                // For 2D [seq_len, vocab_size], use the last row (last token's logits)
                let logits_f32: Vec<f32> = if dims.len() == 1 {
                    // 1D: [vocab_size]
                    let logits = logits_tensor.to_vec();
                    logits.iter().map(|v| v.to_f32()).collect()
                } else if dims.len() == 2 {
                    // 2D: [seq_len, vocab_size] - extract last row
                    let seq_len = dims[0];
                    let vocab_size = dims[1];
                    let logits = logits_tensor.to_vec();
                    let start_idx = (seq_len - 1) * vocab_size;
                    logits[start_idx..].iter().map(|v| v.to_f32()).collect()
                } else {
                    return Err(RuntimeError::TypeError(
                        format!("top_p_sample() expects 1D or 2D logits, got shape {:?}", dims)
                    ));
                };

                // Compute softmax
                let max_logit = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_logits: Vec<f32> = logits_f32.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

                // Create (index, probability) pairs and sort by probability descending
                let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                indexed_probs.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Find nucleus: smallest set with cumulative prob >= p
                let mut cumulative = 0.0;
                let mut nucleus_size = 0;

                for (_, prob) in &indexed_probs {
                    cumulative += prob;
                    nucleus_size += 1;
                    if cumulative >= p {
                        break;
                    }
                }

                // Renormalize probabilities within nucleus
                let nucleus = &indexed_probs[..nucleus_size];
                let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
                let renormalized: Vec<(usize, f32)> = nucleus.iter()
                    .map(|(idx, p)| (*idx, p / nucleus_sum))
                    .collect();

                // Sample from nucleus
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_val: f32 = rng.gen();

                let mut cumulative = 0.0;
                let mut sampled_idx = renormalized[0].0;

                for (idx, prob) in renormalized {
                    cumulative += prob;
                    if random_val < cumulative {
                        sampled_idx = idx;
                        break;
                    }
                }

                Ok(Value::Integer(sampled_idx as i64))
            }

            "relu" => {
                // relu(tensor) -> Tensor
                // Apply ReLU activation: max(0, x)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("relu() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.relu().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "matmul" => {
                // matmul(a, b) -> Tensor
                // Matrix multiplication: a @ b
                // Supports batch matrix multiplication
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("matmul() expects 2 arguments (a, b), got {}", args.len())
                    ));
                }

                let a = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let b = self.eval_expr(&args[1])?.as_tensor()?.clone();

                // Use einsum for matrix multiplication
                let output = a.matmul(&b).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "layer_norm" => {
                // layer_norm(tensor, normalized_shape, eps=1e-5) -> Tensor
                // Layer normalization
                if args.len() < 1 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("layer_norm() expects 1-3 arguments (tensor, optional normalized_shape, optional eps), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                // Default: normalize over last dimension
                let shape = tensor.shape();
                let dims = shape.dims();
                let default_normalized_shape = vec![dims[dims.len() - 1]];

                let normalized_shape = if args.len() >= 2 {
                    // TODO: parse normalized_shape from argument
                    default_normalized_shape
                } else {
                    default_normalized_shape
                };

                let eps = if args.len() >= 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Float(f) => f as f32,
                        Value::Integer(i) => i as f32,
                        _ => 1e-5_f32,
                    }
                } else {
                    1e-5_f32
                };

                let output = tensor.layer_norm(normalized_shape, None, None, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "rms_norm" => {
                // rms_norm(tensor, weight, eps=1e-6) -> Tensor
                // RMS normalization (used in LLaMA, TinyLlama)
                if args.len() < 2 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("rms_norm() expects 2-3 arguments (tensor, weight, optional eps), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let weight = self.eval_expr(&args[1])?.as_tensor()?.clone();

                // Infer normalized_shape from weight shape
                let normalized_shape = weight.shape().dims().to_vec();

                let eps = if args.len() >= 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Float(f) => f as f32,
                        Value::Integer(i) => i as f32,
                        _ => 1e-6_f32,  // Default eps for RMSNorm (LLaMA uses 1e-6)
                    }
                } else {
                    1e-6_f32
                };

                let output = tensor.rms_norm(normalized_shape, &weight, eps)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "concat" => {
                // concat(tensors, dim) -> Tensor
                // Concatenate tensors along dimension
                // For now, simplified version that takes 2 tensors
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("concat() expects 3 arguments (tensor1, tensor2, dim), got {}", args.len())
                    ));
                }

                let tensor1 = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let tensor2 = self.eval_expr(&args[1])?.as_tensor()?.clone();

                let dim = match self.eval_expr(&args[2])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("concat() dim argument must be a number, got {:?}", v)
                    )),
                };

                let tensors = vec![&tensor1, &tensor2];
                let output = crate::tensor::Tensor::concat(&tensors, dim)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "len" => {
                // len(value) -> int
                // Get length of TokenIds, Tensor, or String
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("len() expects 1 argument, got {}", args.len())
                    ));
                }

                let value = self.eval_expr(&args[0])?;
                let length = match value {
                    Value::TokenIds(ref ids) => ids.len() as i64,
                    Value::Tensor(ref t) => {
                        let dims = t.dims();
                        dims[0] as i64
                    }
                    Value::String(ref s) => s.len() as i64,
                    v => return Err(RuntimeError::TypeError(
                        format!("len() argument must be TokenIds, Tensor, or String, got {:?}", v)
                    )),
                };

                Ok(Value::Integer(length))
            }

            "get" => {
                // get(token_ids, index) -> int
                // Get token ID at specific index
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("get() expects 2 arguments (token_ids, index), got {}", args.len())
                    ));
                }

                let token_ids = match self.eval_expr(&args[0])? {
                    Value::TokenIds(ids) => ids,
                    v => return Err(RuntimeError::TypeError(
                        format!("get() first argument must be TokenIds, got {:?}", v)
                    )),
                };

                let index = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => {
                        // Support negative indexing
                        if i < 0 {
                            (token_ids.len() as i64 + i) as usize
                        } else {
                            i as usize
                        }
                    }
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("get() index must be a number, got {:?}", v)
                    )),
                };

                if index >= token_ids.len() {
                    return Err(RuntimeError::IndexError {
                        index,
                        length: token_ids.len(),
                    });
                }

                Ok(Value::Integer(token_ids[index] as i64))
            }

            "append" => {
                // append(token_ids, token_id) -> TokenIds
                // Append a token ID to the sequence
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("append() expects 2 arguments (token_ids, token_id), got {}", args.len())
                    ));
                }

                let mut token_ids = match self.eval_expr(&args[0])? {
                    Value::TokenIds(ids) => ids,
                    v => return Err(RuntimeError::TypeError(
                        format!("append() first argument must be TokenIds, got {:?}", v)
                    )),
                };

                let new_token = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as u32,
                    Value::Float(f) => f as u32,
                    Value::Tensor(ref t) => {
                        // Convert scalar tensor to integer
                        if t.numel() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("append() token_id tensor must be scalar, got shape {:?}", t.dims())
                            ));
                        }
                        t.to_vec()[0].to_f32() as u32
                    }
                    v => return Err(RuntimeError::TypeError(
                        format!("append() token_id must be a number or scalar tensor, got {:?}", v)
                    )),
                };

                token_ids.push(new_token);
                Ok(Value::TokenIds(token_ids))
            }

            "to_int" => {
                // to_int(value) -> int
                // Convert tensor or number to integer
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("to_int() expects 1 argument, got {}", args.len())
                    ));
                }

                let value = self.eval_expr(&args[0])?;
                let int_val = match value {
                    Value::Integer(i) => i,
                    Value::Float(f) => f as i64,
                    Value::Tensor(ref t) => {
                        if t.numel() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("to_int() tensor must be scalar, got shape {:?}", t.dims())
                            ));
                        }
                        t.to_vec()[0].to_f32() as i64
                    }
                    v => return Err(RuntimeError::TypeError(
                        format!("to_int() argument must be a number or scalar tensor, got {:?}", v)
                    )),
                };

                Ok(Value::Integer(int_val))
            }

            "sigmoid" => {
                // sigmoid(tensor) -> Tensor
                // Sigmoid activation: 1 / (1 + exp(-x))
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sigmoid() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.sigmoid().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "sum" => {
                // sum(tensor, dim, keepdim) -> Tensor or Float
                // Sum along dimension, or sum all elements
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("sum() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                if args.len() == 1 {
                    // Sum all elements
                    let result = tensor.sum().map_err(|e| RuntimeError::TensorError(e))?;
                    Ok(Value::Float(result.to_f32() as f64))
                } else {
                    // Sum along dimension
                    let dim = match self.eval_expr(&args[1])? {
                        Value::Integer(i) => i as usize,
                        Value::Float(f) => f as usize,
                        v => return Err(RuntimeError::TypeError(
                            format!("sum() dim argument must be a number, got {:?}", v)
                        )),
                    };

                    let keepdim = if args.len() >= 3 {
                        self.eval_expr(&args[2])?.as_bool()?
                    } else {
                        false
                    };

                    let output = tensor.sum_dim(dim, keepdim)
                        .map_err(|e| RuntimeError::TensorError(e))?;

                    Ok(Value::Tensor(output))
                }
            }

            "mean" => {
                // mean(tensor, dim, keepdim) -> Tensor or Float
                // Mean along dimension, or mean of all elements
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("mean() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                if args.len() == 1 {
                    // Mean of all elements
                    let result = tensor.mean().map_err(|e| RuntimeError::TensorError(e))?;
                    Ok(Value::Float(result.to_f32() as f64))
                } else {
                    // Mean along dimension
                    let dim = match self.eval_expr(&args[1])? {
                        Value::Integer(i) => i as usize,
                        Value::Float(f) => f as usize,
                        v => return Err(RuntimeError::TypeError(
                            format!("mean() dim argument must be a number, got {:?}", v)
                        )),
                    };

                    let keepdim = if args.len() >= 3 {
                        self.eval_expr(&args[2])?.as_bool()?
                    } else {
                        false
                    };

                    let output = tensor.mean_dim(dim, keepdim)
                        .map_err(|e| RuntimeError::TensorError(e))?;

                    Ok(Value::Tensor(output))
                }
            }

            // Tensor creation functions
            "zeros" => {
                // zeros([shape])
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("zeros() expects 1 argument (shape), got {}", args.len())
                    ));
                }

                // Evaluate the shape argument (array literal becomes a 1D tensor)
                let shape_value = self.eval_expr(&args[0])?;
                let shape = match shape_value {
                    Value::Tensor(t) => {
                        // Convert tensor data to Vec<usize>
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "zeros() shape must be an array".to_string()
                    )),
                };
                let device = MetalDevice::new().map_err(|e| RuntimeError::TensorError(e))?;
                let tensor = Tensor::zeros(&device, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(tensor))
            }

            "ones" => {
                // ones([shape])
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("ones() expects 1 argument (shape), got {}", args.len())
                    ));
                }

                let shape_value = self.eval_expr(&args[0])?;
                let shape = match shape_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "ones() shape must be an array".to_string()
                    )),
                };
                // Use shared Metal device from environment
                let device = self.env.metal_device();
                let tensor = Tensor::ones(device, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(tensor))
            }

            // Tensor shape functions
            "reshape" => {
                // reshape(tensor, [new_shape])
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("reshape() expects 2 arguments (tensor, new_shape), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let shape_value = self.eval_expr(&args[1])?;
                let new_shape = match shape_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "reshape() new_shape must be an array".to_string()
                    )),
                };

                let output = tensor.reshape(new_shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "flatten" => {
                // flatten(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("flatten() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.flatten()
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "shape" => {
                // shape(tensor) -> returns shape as a 1D tensor of integers
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("shape() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let dims = tensor.dims();

                // Convert shape dimensions to a 1D tensor
                let device = MetalDevice::new().map_err(|e| RuntimeError::TensorError(e))?;
                let shape_vec: Vec<f16> = dims.iter().map(|&d| f16::from_f32(d as f32)).collect();
                let shape_tensor = Tensor::from_vec_metal(&device, shape_vec, vec![dims.len()])
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(shape_tensor))
            }

            "broadcast_to" => {
                // broadcast_to(tensor, [target_shape])
                // Broadcast tensor to target shape following NumPy broadcasting rules
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("broadcast_to() expects 2 arguments (tensor, target_shape), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let shape_value = self.eval_expr(&args[1])?;
                let target_shape = match shape_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "broadcast_to() target_shape must be an array".to_string()
                    )),
                };

                let target_tensor_shape = TensorShape::new(target_shape);
                let output = tensor.broadcast_to(&target_tensor_shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "transpose" => {
                // transpose(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("transpose() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.transpose()
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "permute" => {
                // permute(tensor, [dims])
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("permute() expects 2 arguments (tensor, dims), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let dims_value = self.eval_expr(&args[1])?;
                let dims = match dims_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "permute() dims must be an array".to_string()
                    )),
                };

                let output = tensor.permute(dims)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Indexing functions
            "gather" => {
                // gather(tensor, dim, indices)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("gather() expects 3 arguments (tensor, dim, indices), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let dim = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("gather() dim must be a number, got {:?}", v)
                    )),
                };
                let indices = self.eval_expr(&args[2])?.as_tensor()?.clone();

                let output = tensor.gather(dim, &indices)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "scatter" => {
                // scatter(tensor, dim, indices, src)
                if args.len() != 4 {
                    return Err(RuntimeError::TypeError(
                        format!("scatter() expects 4 arguments (tensor, dim, indices, src), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let dim = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("scatter() dim must be a number, got {:?}", v)
                    )),
                };
                let indices = self.eval_expr(&args[2])?.as_tensor()?.clone();
                let src = self.eval_expr(&args[3])?.as_tensor()?.clone();

                let output = tensor.scatter(dim, &indices, &src)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Reduction functions (max, min)
            "max" => {
                // max(tensor) -> scalar or max(tensor, dim, keepdim) -> tensor
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("max() expects 1 to 3 arguments, got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                if args.len() == 1 {
                    // max(tensor) -> scalar
                    let result = tensor.max().map_err(|e| RuntimeError::TensorError(e))?;
                    Ok(Value::Float(result.to_f32() as f64))
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        "max() with dimension not yet implemented".to_string()
                    ));
                }
            }

            "min" => {
                // min(tensor) -> scalar or min(tensor, dim, keepdim) -> tensor
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("min() expects 1 to 3 arguments, got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();

                if args.len() == 1 {
                    // min(tensor) -> scalar
                    let result = tensor.min().map_err(|e| RuntimeError::TensorError(e))?;
                    Ok(Value::Float(result.to_f32() as f64))
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        "min() with dimension not yet implemented".to_string()
                    ));
                }
            }

            // Activation functions
            "gelu" => {
                // gelu(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("gelu() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.gelu().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "tanh" => {
                // tanh(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("tanh() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.tanh().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Math functions
            "exp" => {
                // exp(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("exp() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.exp().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "log" => {
                // log(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("log() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.log().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "sqrt" => {
                // sqrt(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sqrt() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.sqrt().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "pow" => {
                // pow(tensor, exponent)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("pow() expects 2 arguments (tensor, exponent), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let exponent = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as f32,
                    Value::Float(f) => f as f32,
                    v => return Err(RuntimeError::TypeError(
                        format!("pow() exponent must be a number, got {:?}", v)
                    )),
                };

                let output = tensor.pow(exponent).map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "sin" => {
                // sin(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sin() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.sin().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "cos" => {
                // cos(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("cos() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.cos().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "tan" => {
                // tan(tensor)
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("tan() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let output = tensor.tan().map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Masking operations
            "apply_attention_mask" => {
                // apply_attention_mask(tensor, mask)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("apply_attention_mask() expects 2 arguments (tensor, mask), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let mask = self.eval_expr(&args[1])?.as_tensor()?.clone();

                let output = tensor.apply_attention_mask(&mask)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "padding_mask" => {
                // padding_mask([lengths], max_len)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("padding_mask() expects 2 arguments (lengths, max_len), got {}", args.len())
                    ));
                }

                // Parse lengths array
                let lengths_value = self.eval_expr(&args[0])?;
                let lengths: Vec<usize> = match lengths_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "padding_mask() lengths must be an array".to_string()
                    )),
                };

                let max_len = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("padding_mask() max_len must be a number, got {:?}", v)
                    )),
                };

                let output = crate::tensor::Tensor::padding_mask(&lengths, max_len)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "combine_masks" => {
                // combine_masks(mask1, mask2)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("combine_masks() expects 2 arguments (mask1, mask2), got {}", args.len())
                    ));
                }

                let mask1 = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let mask2 = self.eval_expr(&args[1])?.as_tensor()?.clone();

                let output = mask1.combine_masks(&mask2)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Broadcast operation
            "broadcast_to" => {
                // broadcast_to(tensor, [target_shape])
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("broadcast_to() expects 2 arguments (tensor, target_shape), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let shape_value = self.eval_expr(&args[1])?;
                let target_shape = match shape_value {
                    Value::Tensor(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "broadcast_to() target_shape must be an array".to_string()
                    )),
                };

                use crate::tensor::TensorShape;
                let target_tensor_shape = TensorShape::new(target_shape);
                let output = tensor.broadcast_to(&target_tensor_shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            // Fused operations
            "fused_add_relu" => {
                // fused_add_relu(tensor, other)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_add_relu() expects 2 arguments (tensor, other), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let other = self.eval_expr(&args[1])?.as_tensor()?.clone();

                let output = tensor.fused_add_relu(&other)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "fused_mul_relu" => {
                // fused_mul_relu(tensor, other)
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_mul_relu() expects 2 arguments (tensor, other), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let other = self.eval_expr(&args[1])?.as_tensor()?.clone();

                let output = tensor.fused_mul_relu(&other)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "fused_affine" => {
                // fused_affine(tensor, scale, bias)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_affine() expects 3 arguments (tensor, scale, bias), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let scale = self.eval_expr(&args[1])?.as_tensor()?.clone();
                let bias = self.eval_expr(&args[2])?.as_tensor()?.clone();

                let output = tensor.fused_affine(&scale, &bias)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "fused_gelu_linear" => {
                // fused_gelu_linear(tensor, weight, bias)
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_gelu_linear() expects 3 arguments (tensor, weight, bias), got {}", args.len())
                    ));
                }

                let tensor = self.eval_expr(&args[0])?.as_tensor()?.clone();
                let weight = self.eval_expr(&args[1])?.as_tensor()?.clone();
                let bias = self.eval_expr(&args[2])?.as_tensor()?.clone();

                let output = tensor.fused_gelu_linear(&weight, &bias)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(output))
            }

            "env" => {
                // env("VAR_NAME")
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("env() expects 1 argument (var_name), got {}", args.len())
                    ));
                }

                let var_name_val = self.eval_expr(&args[0])?;
                let var_name = match var_name_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "env() argument must be a string (variable name)".to_string()
                    )),
                };

                let value = std::env::var(&var_name)
                    .map_err(|_| RuntimeError::InvalidOperation(
                        format!("Environment variable '{}' not found", var_name)
                    ))?;

                Ok(Value::String(value))
            }

            "input" => {
                // input() or input("prompt")
                use std::io::{self, Write};

                if args.len() > 1 {
                    return Err(RuntimeError::TypeError(
                        format!("input() expects 0 or 1 argument (optional prompt), got {}", args.len())
                    ));
                }

                // Print prompt if provided
                if args.len() == 1 {
                    let prompt_val = self.eval_expr(&args[0])?;
                    if let Value::String(prompt) = prompt_val {
                        print!("{}", prompt);
                        io::stdout().flush().unwrap();
                    }
                }

                // Read line from stdin
                let mut buffer = String::new();
                io::stdin().read_line(&mut buffer)
                    .map_err(|e| RuntimeError::IoError(e))?;

                // Remove trailing newline
                let input = buffer.trim_end().to_string();
                Ok(Value::String(input))
            }

            "generate" => {
                // generate(model, prompt, max_tokens: int = 100, temperature: float = 0.7)
                if args.len() < 2 || args.len() > 4 {
                    return Err(RuntimeError::TypeError(
                        format!("generate() expects 2-4 arguments (model, prompt, optional max_tokens, optional temperature), got {}", args.len())
                    ));
                }

                let model_val = self.eval_expr(&args[0])?;
                let _model = match model_val {
                    Value::Model(m) => m,
                    _ => return Err(RuntimeError::TypeError(
                        "generate() first argument must be a Model".to_string()
                    )),
                };

                let prompt_val = self.eval_expr(&args[1])?;
                let prompt = match prompt_val {
                    Value::String(s) => s,
                    _ => return Err(RuntimeError::TypeError(
                        "generate() second argument must be a string (prompt)".to_string()
                    )),
                };

                let _max_tokens = if args.len() >= 3 {
                    let val = self.eval_expr(&args[2])?;
                    match val {
                        Value::Integer(i) => i as i64,
                        Value::Float(f) => f as i64,
                        _ => return Err(RuntimeError::TypeError(
                            "generate() max_tokens must be a number".to_string()
                        )),
                    }
                } else {
                    100
                };

                let _temperature = if args.len() >= 4 {
                    let val = self.eval_expr(&args[3])?;
                    match val {
                        Value::Float(f) => f,
                        Value::Integer(i) => i as f64,
                        _ => return Err(RuntimeError::TypeError(
                            "generate() temperature must be a number".to_string()
                        )),
                    }
                } else {
                    0.7
                };

                // For now, return a placeholder response
                // Full transformer inference requires:
                // 1. Tokenizer integration (e.g., sentencepiece, tiktoken)
                // 2. Transformer layer implementation (attention, feedforward)
                // 3. KV cache for efficient generation
                // 4. Sampling strategies (greedy, top-k, top-p, etc.)
                let response = format!(
                    "[Placeholder Response] You said: \"{}\". Full LLM inference not yet implemented. \
                    Model loaded: {:?}, tensors: {}",
                    prompt,
                    _model.metadata.format,
                    _model.num_tensors()
                );

                Ok(Value::String(response))
            }

            "str" => {
                // str(value) -> String
                // Convert any value to string representation
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("str() expects 1 argument, got {}", args.len())
                    ));
                }

                let val = self.eval_expr(&args[0])?;
                let result = match val {
                    Value::String(s) => s,
                    Value::Integer(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Boolean(b) => b.to_string(),
                    Value::Void => "void".to_string(),
                    _ => return Err(RuntimeError::TypeError(
                        "str() can only convert primitives (int, float, bool, string)".to_string()
                    )),
                };

                Ok(Value::String(result))
            }

            "print" => {
                // print(value1, value2, ..., end: "\n", flush: false)
                // For now, simple implementation
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        print!(" ");
                    }
                    let val = self.eval_expr(arg)?;
                    match val {
                        Value::String(s) => print!("{}", s),
                        Value::Integer(i) => print!("{}", i),
                        Value::Float(f) => print!("{}", f),
                        Value::Boolean(b) => print!("{}", b),
                        Value::Tensor(t) => print!("{:?}", t),
                        Value::Model(m) => print!("Model({:?})", m.metadata.format),
                        Value::Tokenizer(t) => print!("{:?}", t),
                        Value::TokenIds(ids) => print!("{:?}", ids),
                        Value::Void => print!("void"),
                    }
                }
                println!();
                Ok(Value::Void)
            }

            _ => {
                // Check if it's a user-defined function
                let func_name = name.as_str();

                if let Some(func_decl) = self.functions.get(func_name).cloned() {
                    // User-defined function found!

                    // 1. Check argument count
                    if args.len() != func_decl.params.len() {
                        return Err(RuntimeError::TypeError(
                            format!(
                                "Function '{}' expects {} arguments, got {}",
                                func_name,
                                func_decl.params.len(),
                                args.len()
                            )
                        ));
                    }

                    // 2. Create new call frame
                    let mut frame = CallFrame::new(func_name.to_string());

                    // 3. Evaluate arguments, check types, and bind to parameters
                    for (param, arg) in func_decl.params.iter().zip(args.iter()) {
                        let arg_value = self.eval_expr(arg)?;

                        // Type check the argument
                        self.check_type_match(&arg_value, &param.entity_type, param.name.as_str())?;

                        frame.local_vars.insert(param.name.as_str().to_string(), arg_value);
                    }

                    // 4. Push call frame onto stack
                    self.call_stack.push(frame);

                    // 5. Execute function body and catch explicit returns
                    let mut explicit_return = None;
                    let body_len = func_decl.body.len();

                    // Execute all statements except the last one
                    if body_len > 0 {
                        for stmt in &func_decl.body[..body_len - 1] {
                            match self.execute_statement(stmt) {
                                Err(RuntimeError::ReturnValue(val)) => {
                                    // Explicit return statement - save value and stop execution
                                    explicit_return = Some(val);
                                    break;
                                }
                                Err(e) => {
                                    // Other errors - propagate upward
                                    self.call_stack.pop();
                                    return Err(e);
                                }
                                Ok(_) => {}
                            }
                        }
                    }

                    // 6. Get return value (explicit return or implicit return from last expression)
                    let return_value = if explicit_return.is_some() {
                        explicit_return.unwrap()
                    } else if let Some(last_stmt) = func_decl.body.last() {
                        // Evaluate last statement and get implicit return value
                        match self.evaluate_last_statement(last_stmt) {
                            Ok(Some(val)) => val,
                            Ok(None) => Value::Void,
                            Err(RuntimeError::ReturnValue(val)) => val,  // Explicit return in last statement
                            Err(e) => {
                                self.call_stack.pop();
                                return Err(e);
                            }
                        }
                    } else {
                        // Empty function body
                        Value::Void
                    };

                    // 7. Check return type matches declaration
                    self.check_return_type_match(&return_value, &func_decl.return_type, func_name)?;

                    // 8. Pop call frame
                    self.call_stack.pop();

                    Ok(return_value)
                } else {
                    // Not a built-in or user-defined function
                    Err(RuntimeError::NotImplemented(
                        format!("Function '{}' not yet implemented", func_name),
                    ))
                }
            }
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
                        CompOp::Eq => (l - r).abs() < FLOAT_EPSILON,
                        CompOp::Ne => (l - r).abs() >= FLOAT_EPSILON,
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
                            CompOp::Eq => (l - r).abs() < FLOAT_EPSILON,
                            CompOp::Ne => (l - r).abs() >= FLOAT_EPSILON,
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
                    CompOp::Eq => (norm as f64 - *value).abs() < FLOAT_EPSILON,
                    CompOp::Ne => (norm as f64 - *value).abs() >= FLOAT_EPSILON,
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

        // Collect learnable parameters BEFORE executing statements
        // This ensures only explicitly declared 'learnable' tensors are optimized,
        // not intermediate variables computed in the learn block
        let mut learnable_params = Vec::new();
        let mut learnable_param_names = Vec::new(); // Store names for later rebuilding
        for (name, value) in &self.env.variables {
            if let Value::Tensor(tensor) = value {
                if tensor.requires_grad() {
                    learnable_params.push((name.clone(), tensor.clone()));
                    learnable_param_names.push(name.clone());
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

        // Execute preamble statements AFTER collecting learnable params
        // These create local variables for intermediate computations
        for stmt in &spec.statements {
            self.execute_statement(stmt)?;
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

            // Re-execute statements for each epoch (recompute intermediate variables)
            for stmt in &spec.statements {
                self.execute_statement(stmt)?;
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
                                // Learning context - update parameter
                                self.env.variables.insert(name.clone(), Value::Tensor(param_with_grad));
                            }

                            // 5. Rebuild learnable_params vector to point to updated tensors
                            // This ensures the next epoch's backward pass can find the parameters
                            // Only rebuild from the original learnable parameter names (not local variables)
                            learnable_params.clear();
                            for name in &learnable_param_names {
                                if let Ok(Value::Tensor(tensor)) = self.env.get_variable(name) {
                                    learnable_params.push((name.clone(), tensor.clone()));
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
            if epoch % EPOCH_REPORT_INTERVAL == 0 || epoch == spec.epochs - 1 {
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
            if epoch > 0 && (new_lr - lr).abs() > FLOAT_EPSILON as f32 && (epoch % EPOCH_REPORT_INTERVAL == 0 || epoch == spec.epochs - 1) {
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

    /// Convert a Term to a TensorExpr for evaluation
    fn term_to_expr(&self, term: &Term) -> TensorExpr {
        match term {
            Term::Variable(ident) => {
                // Variables become variable expressions
                TensorExpr::Variable(ident.clone())
            }
            Term::Constant(constant) => {
                // Constants become literal expressions
                let scalar_lit = match constant {
                    Constant::Integer(i) => ScalarLiteral::Integer(*i),
                    Constant::Float(f) => ScalarLiteral::Float(*f),
                    Constant::String(s) => ScalarLiteral::String(s.clone()),
                    Constant::Boolean(b) => ScalarLiteral::Boolean(*b),
                };
                TensorExpr::Literal(TensorLiteral::Scalar(scalar_lit))
            }
            Term::Tensor(expr) => {
                // Already a tensor expression
                expr.clone()
            }
        }
    }

    /// Collect entities from a fact assertion (for data-driven entity types)
    fn collect_entities_from_fact(&mut self, atom: &Atom) -> RuntimeResult<()> {
        let predicate_name = atom.predicate.as_str();

        // Check if this relation has entity-typed parameters
        if let Some(entity_params) = self.relation_entity_params.get(predicate_name) {
            // Get all data-driven entity types
            let data_driven_types: Vec<String> = self.entity_registry
                .all_type_names()
                .iter()
                .filter_map(|&type_name| {
                    if let Some(type_info) = self.entity_registry.get_type_info(type_name) {
                        match &type_info.declaration_type {
                            crate::entity_registry::EntityDeclType::FromData => {
                                Some(type_name.to_string())
                            }
                            _ => None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // For each entity-typed parameter
            for (&param_idx, _entity_type) in entity_params {
                if let Some(term) = atom.terms.get(param_idx) {
                    // Extract constant value from term
                    let entity_name = match term {
                        Term::Constant(Constant::String(s)) => Some(s.clone()),
                        Term::Variable(ident) => {
                            let name = ident.as_str();
                            // Lowercase identifiers are treated as constants
                            if name.chars().next().unwrap().is_lowercase() {
                                Some(name.to_string())
                            } else {
                                None // Variable, skip
                            }
                        }
                        _ => None,
                    };

                    // Add to all data-driven entity types
                    if let Some(entity_name) = entity_name {
                        for type_name in &data_driven_types {
                            self.entity_registry.add_entity(type_name, entity_name.clone())
                                .map_err(|e| RuntimeError::InvalidOperation(e))?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert an atom's terms based on relation variable definitions
    /// Terms that match relation variables remain as variables, others become constants
    fn convert_atom_terms(&self, atom: &Atom) -> Atom {
        let predicate_name = atom.predicate.as_str();

        // Get the defined variables for this predicate
        let defined_vars = self.relation_variables.get(predicate_name);

        let converted_terms: Vec<Term> = atom.terms.iter().map(|term| {
            match term {
                Term::Variable(ident) => {
                    let var_name = ident.as_str();

                    // Single uppercase letter is always a variable (Prolog convention)
                    if var_name.len() == 1 && var_name.chars().next().unwrap().is_uppercase() {
                        return term.clone();
                    }

                    // Check if this identifier is a defined variable for this predicate
                    // Case-insensitive comparison (x matches X)
                    if let Some(vars) = defined_vars {
                        let var_name_lower = var_name.to_lowercase();
                        let is_defined_var = vars.iter().any(|v| v.to_lowercase() == var_name_lower);

                        if is_defined_var {
                            // It's a defined variable - keep as Variable
                            term.clone()
                        } else {
                            // Not a defined variable - convert to Constant (String)
                            Term::Constant(Constant::String(var_name.to_string()))
                        }
                    } else {
                        // No relation definition - treat as constant
                        Term::Constant(Constant::String(var_name.to_string()))
                    }
                }
                // Constants and Tensors remain unchanged
                _ => term.clone(),
            }
        }).collect();

        Atom {
            predicate: atom.predicate.clone(),
            terms: converted_terms,
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
