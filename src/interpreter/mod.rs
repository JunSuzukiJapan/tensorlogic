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

// Sub-modules
mod formatter;
mod value;
mod environment;
mod eval;        // Expression and statement evaluation (work in progress)

// Builtin function modules (organized by category)
mod builtin_tensor;    // Basic tensor operations
mod builtin_math;      // Math operations
mod builtin_nn;        // Neural network operations
mod builtin_kg;        // Knowledge graph embeddings
mod builtin_gnn;       // Graph neural networks
mod builtin_model;     // Model and I/O operations
mod builtin_sampling;  // Sampling and generation
mod builtin_util;      // Utility functions

// Re-export public types
pub use value::{Value, ModelLayerCollection, ModelLayer, ModelFeature};
pub use environment::{RuntimeEnvironment, CallFrame, ScopeType};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::fs;

use crate::ast::*;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO, TensorTransform, TensorAutograd};
use crate::tensor::{Tensor, TensorShape};
use crate::device::MetalDevice;
use crate::entity_registry::EntityRegistry;
use crate::relation_registry::RelationRegistry;
use crate::error::TensorError;
use crate::logic::LogicEngine;
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

/// Interpreter
pub struct Interpreter {
    env: RuntimeEnvironment,
    logic_engine: LogicEngine,
    // Entity registry for managing entity types and instances
    entity_registry: EntityRegistry,
    // Relation registry for managing relation types
    relation_registry: RelationRegistry,
    // Embedding storage: (embedding_name, entity_index_map, embedding_matrix)
    embeddings: HashMap<String, (HashMap<String, usize>, Tensor)>,
    // Relation embedding storage: (embedding_name, relation_index_map, embedding_matrix)
    relation_embeddings: HashMap<String, (HashMap<String, usize>, Tensor)>,
    // Python execution environment (when python feature is enabled)
    #[cfg(any(feature = "python", feature = "python-extension"))]
    python_env: Option<crate::python::environment::PythonEnvironment>,
    // Track imported files to detect circular dependencies
    imported_files: HashSet<PathBuf>,
    // Current file being executed (for resolving relative imports)
    current_file: Option<PathBuf>,
    // Current source location (for error reporting)
    current_span: Option<Span>,
    // Track defined relation variables: predicate_name -> set of variable names
    relation_variables: HashMap<String, HashSet<String>>,
    // Track relation entity parameters: predicate_name -> (param_index -> entity_type_name)
    relation_entity_params: HashMap<String, HashMap<usize, String>>,
    // User-defined functions: function_name -> FunctionDecl
    functions: HashMap<String, FunctionDecl>,
    // Function call stack for local scope management
    call_stack: Vec<CallFrame>,
    // Track learnable tensors declared with 'learnable' keyword
    // These are the only tensors that should be optimized in learn blocks
    learnable_params: HashSet<String>,
    // Struct definitions: struct_name -> StructDecl
    structs: HashMap<String, StructDecl>,
    // Drop trait implementations: struct_name -> drop method
    drop_impls: HashMap<String, MethodDecl>,
    // Shared Metal device for GPU operations (singleton per interpreter)
    device: crate::device::MetalDevice,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: RuntimeEnvironment::new(),
            logic_engine: LogicEngine::new(),
            entity_registry: EntityRegistry::new(),
            relation_registry: RelationRegistry::new(),
            embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            #[cfg(any(feature = "python", feature = "python-extension"))]
            python_env: None,
            imported_files: HashSet::new(),
            current_file: None,
            current_span: None,
            relation_variables: HashMap::new(),
            relation_entity_params: HashMap::new(),
            functions: HashMap::new(),
            call_stack: Vec::new(),
            learnable_params: HashSet::new(),
            structs: HashMap::new(),
            drop_impls: HashMap::new(),
            device: crate::device::MetalDevice::new()
                .expect("Failed to create Metal device - Metal may not be available on this system"),
        }
    }

    /// Set the current file being executed (for import resolution)
    pub fn set_current_file(&mut self, path: impl Into<PathBuf>) {
        self.current_file = Some(path.into());
    }

    /// Execute a complete program
    pub fn execute(&mut self, program: &Program) -> RuntimeResult<()> {
        // Function references are now resolved during parsing
        // No need for separate semantic analysis pass

        // Execute all declarations
        for decl in &program.declarations {
            self.execute_declaration(decl)?;
        }

        // Execute main block if present
        if let Some(main_block) = &program.main_block {
            self.execute_main_block(main_block)?;
        }

        // Print buffer pool statistics if enabled via environment variable
        if std::env::var("TL_BUFFER_STATS").is_ok() {
            self.env.metal_device().print_buffer_pool_stats("Program Execution");
        }

        Ok(())
    }

    /// Execute test blocks
    pub fn execute_tests(&mut self, program: &Program) -> RuntimeResult<()> {
        // Execute all declarations first
        for decl in &program.declarations {
            self.execute_declaration(decl)?;
        }

        if program.test_blocks.is_empty() {
            println!("No test blocks found");
            return Ok(());
        }

        let mut passed = 0;
        let mut failed = 0;

        for test_block in &program.test_blocks {
            print!("test {} ... ", test_block.name.as_str());
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            match self.execute_test_block(test_block) {
                Ok(_) => {
                    println!("✓ ok");
                    passed += 1;
                }
                Err(e) => {
                    println!("✗ FAILED");
                    eprintln!("  Error: {}", e);
                    failed += 1;
                }
            }
        }

        println!("\ntest result: {}. {} passed; {} failed",
            if failed == 0 { "ok" } else { "FAILED" },
            passed,
            failed
        );

        if failed > 0 {
            std::process::exit(1);
        }

        Ok(())
    }

    /// Execute benchmark blocks with timing
    pub fn execute_benchmarks(&mut self, program: &Program) -> RuntimeResult<()> {
        // Execute all declarations first
        for decl in &program.declarations {
            self.execute_declaration(decl)?;
        }

        if program.bench_blocks.is_empty() {
            println!("No benchmark blocks found");
            return Ok(());
        }

        // Enable profiling for benchmarks
        std::env::set_var("TL_PROFILE", "1");

        for bench_block in &program.bench_blocks {
            println!("bench {} ...", bench_block.name.as_str());

            let start = std::time::Instant::now();
            self.execute_bench_block(bench_block)?;
            let duration = start.elapsed();

            println!("  Total time: {:.3}ms\n", duration.as_secs_f64() * 1000.0);
        }

        Ok(())
    }

    fn execute_test_block(&mut self, test_block: &TestBlock) -> RuntimeResult<()> {
        for stmt in &test_block.statements {
            self.execute_statement(stmt)?;
        }
        Ok(())
    }

    fn execute_bench_block(&mut self, bench_block: &BenchBlock) -> RuntimeResult<()> {
        for stmt in &bench_block.statements {
            self.execute_statement(stmt)?;
        }
        Ok(())
    }

    /// Get a variable from the interpreter's environment
    /// Variables are managed by the scope stack
    pub fn get_variable(&self, name: &str) -> RuntimeResult<Value> {
        self.env.get_variable(name)
    }

    pub fn get_all_variables(&self) -> Option<&HashMap<String, Value>> {
        self.env.get_all_variables()
    }

    /// Set a variable in the interpreter's environment
    /// Variables are managed by the scope stack
    pub fn set_variable(&mut self, name: String, value: Value) {
        let _ = self.env.set_variable(&name, value);
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
                // Register relation in the registry
                self.relation_registry.register_from_decl(relation_decl);

                // Collect variable names from relation parameters
                let predicate_name = relation_decl.name.as_str().to_string();
                let var_names: HashSet<String> = relation_decl.params.iter()
                    .map(|param| param.name.as_str().to_string())
                    .collect();

                self.relation_variables.insert(predicate_name.clone(), var_names);

                // Collect entity-typed parameters: param_index -> entity_type_name
                let mut entity_params = HashMap::new();
                for (idx, param) in relation_decl.params.iter().enumerate() {
                    match &param.entity_type {
                        EntityType::Entity | EntityType::Concept => {
                            // Generic entity type - will collect to all data-driven types
                            entity_params.insert(idx, "entity".to_string());
                        }
                        EntityType::NamedEntity(type_name) => {
                            // Specific entity type - will collect only to this type
                            entity_params.insert(idx, type_name.as_str().to_string());
                        }
                        EntityType::Tensor(_) => {
                            // Tensor-typed parameter, skip
                        }
                        EntityType::Scalar(_) => {
                            // Scalar-typed parameter, skip
                        }
                        EntityType::Struct(_) => {
                            // Struct-typed parameter, skip
                        }
                    }
                }

                // Store entity-typed parameters
                if !entity_params.is_empty() {
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
            Declaration::RelationEmbedding(rel_embedding_decl) => self.execute_relation_embedding_decl(rel_embedding_decl),
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
            Declaration::Struct(struct_decl) => {
                // Store struct definition
                let struct_name = struct_decl.name.as_str().to_string();

                // Check for duplicate struct names
                if self.structs.contains_key(&struct_name) {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Struct '{}' is already defined", struct_name)
                    ));
                }

                self.structs.insert(struct_name, struct_decl.clone());
                Ok(())
            }
            Declaration::Impl(impl_block) => {
                // Register methods as functions with qualified names
                let struct_name = impl_block.struct_type.name.as_str().to_string();

                // Check if this is a Drop trait implementation
                let is_drop_impl = impl_block.trait_name.as_ref()
                    .map(|t| t.as_str() == "Drop")
                    .unwrap_or(false);

                for method in &impl_block.methods {
                    if is_drop_impl {
                        // Store Drop implementation separately
                        if method.name.as_str() == "drop" {
                            self.drop_impls.insert(struct_name.clone(), method.clone());
                        }
                    } else {
                        // Regular method: store as function with qualified name
                        let qualified_name = format!("{}::{}", struct_name, method.name.as_str());

                        // Convert method to function declaration
                        let func_decl = self.method_to_function(method, &struct_name)?;
                        self.functions.insert(qualified_name, func_decl);
                    }
                }

                Ok(())
            }
        }
    }

    /// Convert a method declaration to a function declaration
    fn method_to_function(&self, method: &MethodDecl, _struct_name: &str) -> RuntimeResult<FunctionDecl> {
        // Convert method parameters to function parameters
        let mut params = Vec::new();
        for param in &method.params {
            match param {
                MethodParam::SelfParam => {
                    // Skip self parameter for now - we'll handle it in method calls
                }
                MethodParam::Regular(func_param) => {
                    params.push(func_param.clone());
                }
            }
        }

        Ok(FunctionDecl {
            name: method.name.clone(),
            params,
            return_type: method.return_type.clone(),
            body: method.body.clone(),
        })
    }

    /// Call destructors for all struct values in the current scope
    fn call_scope_destructors(&mut self) -> RuntimeResult<()> {
        // Get variables from current scope in environment
        // Variables are stored in env, not in call frames
        let variables: Vec<(String, Value)> = self.env.list_variables().iter()
            .filter_map(|name| {
                self.env.get_variable(name)
                    .ok()
                    .map(|v| (name.clone(), v.clone()))
            })
            .collect();

        // Call destructors for struct values in reverse order (LIFO)
        for (_name, value) in variables.iter().rev() {
            if let Value::Struct { struct_type, .. } = value {
                self.call_drop_method(struct_type, value)?;
            }
        }

        Ok(())
    }

    /// Call the drop method for a struct value if it has a Drop implementation
    fn call_drop_method(&mut self, struct_type: &StructType, value: &Value) -> RuntimeResult<()> {
        let struct_name = struct_type.name.as_str();

        // Check if there's a Drop implementation for this struct
        if let Some(drop_method) = self.drop_impls.get(struct_name).cloned() {
            // Create a new call frame for the drop method
            let drop_fn_name = format!("{}::drop", struct_name);
            let frame = CallFrame::new(drop_fn_name.clone());

            // Push call frame and new scope
            self.call_stack.push(frame);
            self.env.push_scope(ScopeType::Function(drop_fn_name));

            // Bind self parameter to the struct value in environment
            self.env.set_variable("self", value.clone())?;

            // Execute drop method body
            for stmt in &drop_method.body {
                match self.execute_statement(stmt) {
                    Err(RuntimeError::ReturnValue(_)) => {
                        // Early return from drop method - just stop execution
                        break;
                    }
                    Err(e) => {
                        // Error in drop method - pop frame/scope and propagate
                        self.env.pop_scope();
                        self.call_stack.pop();
                        return Err(e);
                    }
                    Ok(_) => {}
                }
            }

            // Pop scope and call frame
            self.env.pop_scope();
            self.call_stack.pop();
        }

        Ok(())
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
        use crate::ast::BaseType;

        let value = if let Some(init_expr) = &decl.init_expr {
            // Evaluate initialization expression
            let init_value = self.eval_expr(init_expr)?;

            // Reshape tensor to match declared shape (if needed)
            let declared_shape = self.get_declared_shape(&decl.tensor_type)?;

            match (&init_value, &decl.tensor_type.base_type) {
                (Value::TensorF16(t), BaseType::Float32) => {
                    // f16 tensor
                    let mut tensor = t.clone();
                    if tensor.shape().dims() != declared_shape.as_slice() {
                        tensor = tensor.reshape(declared_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                    }
                    if decl.tensor_type.learnable == LearnableStatus::Learnable {
                        tensor.set_requires_grad(true);
                    }
                    Value::TensorF16(tensor)
                }
                (Value::TensorF32(t), BaseType::Float64) => {
                    // f32 tensor
                    let mut tensor = t.clone();
                    if tensor.shape().dims() != declared_shape.as_slice() {
                        tensor = tensor.reshape(declared_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                    }
                    if decl.tensor_type.learnable == LearnableStatus::Learnable {
                        tensor.set_requires_grad(true);
                    }
                    Value::TensorF32(tensor)
                }
                _ => {
                    return Err(RuntimeError::TypeError(
                        format!("Type mismatch in tensor declaration: expected {:?}, got {:?}",
                            decl.tensor_type.base_type, init_value)
                    ));
                }
            }
        } else {
            // Create zero-initialized tensor based on base type
            let mut value = self.create_zero_tensor(&decl.tensor_type)?;

            // Set requires_grad if learnable
            if decl.tensor_type.learnable == LearnableStatus::Learnable {
                match &mut value {
                    Value::TensorF16(t) => t.set_requires_grad(true),
                    Value::TensorF32(t) => t.set_requires_grad(true),
                    _ => {}
                }
            }
            value
        };

        // Track learnable parameters (only if explicitly declared with 'learnable' keyword)
        if decl.tensor_type.learnable == LearnableStatus::Learnable {
            self.learnable_params.insert(decl.name.as_str().to_string());
        }

        // Use declare_variable for tensor declarations (they create new variables)
        self.env.declare_variable(decl.name.as_str().to_string(), value)?;

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
            EntitySet::Type(type_name) => {
                // Get entities from entity registry
                let type_name_str = type_name.as_str();
                if let Some(type_info) = self.entity_registry.get_type_info(type_name_str) {
                    let entities = type_info.all_entities();
                    if entities.is_empty() {
                        println!("⚠️  Warning: Entity type '{}' has no entities yet. Embedding will be initialized with placeholder.", type_name_str);
                        println!("   Note: Entities are collected during fact evaluation in main block.");
                    }
                    entities
                        .iter()
                        .enumerate()
                        .map(|(idx, name)| (name.clone(), idx))
                        .collect()
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Entity type '{}' not found for embedding", type_name_str)
                    ));
                }
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
                Tensor::from_vec_gpu(device, data, vec![num_entities, decl.dimension])?
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
                Tensor::from_vec_gpu(device, data, vec![num_entities, decl.dimension])?
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
                Tensor::from_vec_gpu(device, data, vec![num_entities, decl.dimension])?
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

    /// Execute a relation embedding declaration
    fn execute_relation_embedding_decl(&mut self, decl: &RelationEmbeddingDecl) -> RuntimeResult<()> {
        use crate::ast::{RelationSet, InitMethod};

        // Build relation-to-index mapping
        let relation_map: HashMap<String, usize> = match &decl.relations {
            RelationSet::Explicit(relations) => {
                relations
                    .iter()
                    .enumerate()
                    .map(|(idx, id)| (id.as_str().to_string(), idx))
                    .collect()
            }
            RelationSet::All => {
                // All relations: get from registry
                let all_relations = self.relation_registry.all_relation_names();
                all_relations
                    .iter()
                    .enumerate()
                    .map(|(idx, name)| (name.clone(), idx))
                    .collect()
            }
        };

        let num_relations = relation_map.len().max(1); // At least 1
        let device = self.env.metal_device();

        // Initialize relation embedding matrix [num_relations, dimension]
        let embedding_matrix = match decl.init_method {
            InitMethod::Random => {
                // Random initialization in range [-0.1, 0.1]
                let mut data = Vec::with_capacity(num_relations * decl.dimension);
                use rand::Rng;
                let mut rng = rand::rng();
                for _ in 0..(num_relations * decl.dimension) {
                    let val: f32 = rng.random_range(-0.1..0.1);
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_gpu(device, data, vec![num_relations, decl.dimension])?
            }
            InitMethod::Xavier => {
                // Xavier initialization: uniform(-sqrt(6/(n+m)), sqrt(6/(n+m)))
                let limit = (6.0 / (num_relations + decl.dimension) as f32).sqrt();
                let mut data = Vec::with_capacity(num_relations * decl.dimension);
                use rand::Rng;
                let mut rng = rand::rng();
                for _ in 0..(num_relations * decl.dimension) {
                    let val: f32 = rng.random_range(-limit..limit);
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_gpu(device, data, vec![num_relations, decl.dimension])?
            }
            InitMethod::He => {
                // He initialization: normal(0, sqrt(2/n))
                let stddev = (2.0 / num_relations as f32).sqrt();
                let mut data = Vec::with_capacity(num_relations * decl.dimension);
                use rand_distr::{Normal, Distribution};
                let mut rng = rand::rng();
                let normal = Normal::new(0.0, stddev as f64).unwrap();
                for _ in 0..(num_relations * decl.dimension) {
                    let val: f32 = normal.sample(&mut rng) as f32;
                    data.push(half::f16::from_f32(val));
                }
                Tensor::from_vec_gpu(device, data, vec![num_relations, decl.dimension])?
            }
            InitMethod::Zeros => Tensor::zeros(&device, vec![num_relations, decl.dimension])?,
            InitMethod::Ones => Tensor::ones(&device, vec![num_relations, decl.dimension])?,
        };

        // Store relation embedding with relation mapping
        self.relation_embeddings.insert(
            decl.name.as_str().to_string(),
            (relation_map, embedding_matrix),
        );

        println!("Initialized relation embedding '{}': {} relations × {} dimensions",
            decl.name.as_str(), num_relations, decl.dimension);

        Ok(())
    }

    /// Create a zero-initialized tensor
    fn create_zero_tensor(&self, tensor_type: &TensorType) -> RuntimeResult<Value> {
        use crate::ast::BaseType;

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
            BaseType::Float32 => {
                // float16 -> f16 tensor
                let tensor = Tensor::<half::f16>::zeros(self.env.metal_device(), shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF16(tensor))
            }
            BaseType::Float64 => {
                // float32 -> f32 tensor
                let tensor = Tensor::<f32>::zeros(self.env.metal_device(), shape)
                    .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(tensor))
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

                // Execute the statement for side effects (variable assignment)
                // Scope stack handles variable assignment automatically
                if self.env.has_variable(target.as_str()) {
                    self.env.set_variable(target.as_str(), evaluated_value.clone())?;
                } else {
                    self.env.declare_variable(target.as_str().to_string(), evaluated_value.clone())?;
                }

                Ok(Some(evaluated_value))
            }
            Statement::Equation(_eq) => {
                // For equation types (~), execute but don't return value
                self.execute_statement(stmt)?;
                Ok(None)
            }
            Statement::FunctionCall { name, args, resolved, span } => {
                // Save span for error reporting
                self.current_span = Some(span.clone());
                // Function call result can be implicitly returned
                let value = self.eval_function_call(None, name, args, resolved.as_ref())?;
                Ok(Some(value))
            }
            Statement::Expr { expr } => {
                // Expression statement as last statement: return its value (implicit return)
                let value = self.eval_expr(expr)?;
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
            ReturnType::Scalar(scalar_type) => {
                // Convert ScalarType to EntityType and reuse check_type_match
                let entity_type = EntityType::Scalar(scalar_type.clone());
                self.check_type_match(value, &entity_type, &format!("return value of '{}'", func_name))
            }
            ReturnType::Tensor(tensor_type) => {
                // Convert TensorType to EntityType and reuse check_type_match
                let entity_type = EntityType::Tensor(tensor_type.clone());
                self.check_type_match(value, &entity_type, &format!("return value of '{}'", func_name))
            }
            ReturnType::Struct(struct_type) => {
                // Convert StructType to EntityType and reuse check_type_match
                let entity_type = EntityType::Struct(struct_type.clone());
                self.check_type_match(value, &entity_type, &format!("return value of '{}'", func_name))
            }
        }
    }

    /// Check if a value matches the expected entity type
    fn check_type_match(&self, value: &Value, expected_type: &EntityType, param_name: &str) -> RuntimeResult<()> {
        match expected_type {
            EntityType::Entity | EntityType::Concept | EntityType::NamedEntity(_) => {
                // Entity, Concept, and Named Entity types accept string values
                match value {
                    Value::String(_) => Ok(()),
                    _ => Err(RuntimeError::TypeError(
                        format!("Parameter '{}' expects entity/concept (string), got {:?}", param_name, value)
                    ))
                }
            }
            EntityType::Tensor(tensor_type) => {
                match value {
                    Value::TensorF16(t) => {
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
                    Value::TensorF32(t) => {
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
            EntityType::Scalar(scalar_type) => {
                use crate::ast::ScalarType;
                match (scalar_type, value) {
                    (ScalarType::Int, Value::Integer(_)) => Ok(()),
                    (ScalarType::Float, Value::Float(_)) => Ok(()),
                    (ScalarType::Bool, Value::Boolean(_)) => Ok(()),
                    (ScalarType::String, Value::String(_)) => Ok(()),
                    (ScalarType::Int, Value::Float(f)) => {
                        // Accept float if it's a whole number
                        if f.fract() == 0.0 {
                            Ok(())
                        } else {
                            Err(RuntimeError::TypeError(
                                format!("Parameter '{}' expects int, got non-integer float {}", param_name, f)
                            ))
                        }
                    }
                    (ScalarType::Float, Value::Integer(_)) => {
                        // Accept int for float (will be promoted)
                        Ok(())
                    }
                    _ => Err(RuntimeError::TypeError(
                        format!("Parameter '{}' expects {:?}, got {:?}", param_name, scalar_type, value)
                    ))
                }
            }
            EntityType::Struct(struct_type) => {
                // Check if value is a struct of the correct type
                match value {
                    Value::Struct { struct_type: value_type, .. } => {
                        if value_type.name == struct_type.name {
                            Ok(())
                        } else {
                            Err(RuntimeError::TypeError(
                                format!("Parameter '{}' expects struct {}, got struct {}",
                                    param_name, struct_type.name.as_str(), value_type.name.as_str())
                            ))
                        }
                    }
                    _ => Err(RuntimeError::TypeError(
                        format!("Parameter '{}' expects struct {}, got {:?}",
                            param_name, struct_type.name.as_str(), value)
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

        // Call destructors for all struct values in main scope before exiting
        self.call_scope_destructors()?;

        Ok(())
    }

    /// Evaluate typed function call (e.g., f32::zeros, f16::ones)
    fn eval_typed_function_call(&mut self, type_namespace: &str, name: &str, args: &[TensorExpr]) -> RuntimeResult<Value> {
        
        

        match type_namespace {
            "f32" => {
                // f32-specific function calls
                match name {
                    "zeros" => self.eval_zeros_f32(args),
                    "ones" => self.eval_ones_f32(args),
                    "arange" => self.eval_range_f32(args),
                    // Most tensor operations already support both f16/f32, so we can call them directly
                    _ => {
                        // Try calling the function normally - it should handle f32 tensors
                        if let Some(result) = self.eval_tensor_function(name, args) {
                            result
                        } else if let Some(result) = self.eval_math_function(name, args) {
                            result
                        } else if let Some(result) = self.eval_nn_function(name, args) {
                            result
                        } else {
                            Err(RuntimeError::TypeError(
                                format!("f32::{} is not implemented", name)
                            ))
                        }
                    }
                }
            }
            "f16" => {
                // f16-specific function calls
                match name {
                    "zeros" => self.eval_zeros_f16(args),
                    "ones" => self.eval_ones_f16(args),
                    "arange" => self.eval_range_f16(args),
                    // Most tensor operations already support both f16/f32, so we can call them directly
                    _ => {
                        // Try calling the function normally - it should handle f16 tensors
                        if let Some(result) = self.eval_tensor_function(name, args) {
                            result
                        } else if let Some(result) = self.eval_math_function(name, args) {
                            result
                        } else if let Some(result) = self.eval_nn_function(name, args) {
                            result
                        } else {
                            Err(RuntimeError::TypeError(
                                format!("f16::{} is not implemented", name)
                            ))
                        }
                    }
                }
            }
            "Tensor" => {
                // Tensor static methods
                match name {
                    "create_cache" => self.eval_create_cache_tensor(args),
                    _ => Err(RuntimeError::TypeError(
                        format!("Tensor::{} is not implemented", name)
                    ))
                }
            }
            "KVCache" => {
                // KVCache static methods
                match name {
                    "new" | "new_f16" => {
                        // KVCache::new(num_layers: int) -> KVCacheF16
                        // KVCache::new_f16(num_layers: int) -> KVCacheF16
                        if args.len() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("KVCache::{}() expects 1 argument, got {}", name, args.len())
                            ));
                        }

                        let num_layers = match self.eval_expr(&args[0])? {
                            Value::Integer(n) => n as usize,
                            v => return Err(RuntimeError::TypeError(
                                format!("KVCache::{}() expects Integer, got {}", name, v.type_name())
                            )),
                        };

                        let cache = crate::model::llama::Cache::<half::f16>::new(num_layers);
                        Ok(Value::KVCacheF16(std::sync::Arc::new(std::sync::Mutex::new(cache))))
                    }
                    "new_f32" => {
                        // KVCache::new_f32(num_layers: int) -> KVCacheF32
                        if args.len() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("KVCache::new_f32() expects 1 argument, got {}", args.len())
                            ));
                        }

                        let num_layers = match self.eval_expr(&args[0])? {
                            Value::Integer(n) => n as usize,
                            v => return Err(RuntimeError::TypeError(
                                format!("KVCache::new_f32() expects Integer, got {}", v.type_name())
                            )),
                        };

                        let cache = crate::model::llama::Cache::<f32>::new(num_layers);
                        Ok(Value::KVCacheF32(std::sync::Arc::new(std::sync::Mutex::new(cache))))
                    }
                    _ => Err(RuntimeError::TypeError(
                        format!("KVCache::{} is not implemented", name)
                    ))
                }
            }
            _ => Err(RuntimeError::TypeError(
                format!("Unknown type namespace: {}", type_namespace)
            ))
        }
    }

    /// Evaluate a resolved function (optimized dispatch without HashMap lookup)
    /// This is called when semantic analysis has already resolved the function reference
    fn eval_resolved_function(&mut self, resolved: &crate::ast::ResolvedFunction, args: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::ast::{BuiltinFunctionId, ResolvedFunction};

        // eprintln!("[DEBUG] eval_resolved_function: Entry");

        match resolved {
            ResolvedFunction::Builtin(id) => {
                // eprintln!("[DEBUG] eval_resolved_function: Builtin function id={:?}", id);

                // IMPORTANT: Directly dispatch to builtin functions to avoid infinite recursion
                // The fallback mechanism was causing hangs due to re-evaluation loops
                use crate::ast::BuiltinFunctionId;

                let result = match id {
                    // CRITICAL FIX: Direct dispatch for reshape to avoid infinite recursion
                    BuiltinFunctionId::TensorReshape => {
                        if let Some(result) = self.eval_tensor_function("reshape", args) {
                            result
                        } else {
                            return Err(RuntimeError::NotImplemented("reshape not found".to_string()));
                        }
                    }

                    // For any other builtin, fall back to string-based dispatch
                    _ => {
                        // eprintln!("[DEBUG] eval_resolved_function: Builtin id={:?} not implemented in fast dispatch, using fallback", id);
                        return Err(RuntimeError::NotImplemented(
                            format!("Fast builtin dispatch not yet implemented for {:?}", id)
                        ));
                    }
                };

                // eprintln!("[DEBUG] eval_resolved_function: Builtin dispatch completed");
                result
            }
            ResolvedFunction::UserDefined(func_decl) => {
                // eprintln!("[DEBUG] eval_resolved_function: UserDefined function name={}", func_decl.name.as_str());
                // Direct call to user-defined function (no HashMap lookup)
                self.call_user_defined_function(func_decl, args)
            }
        }
    }

    /// Call a user-defined function directly (optimized, no HashMap lookup)
    fn call_user_defined_function(&mut self, func_decl: &crate::ast::FunctionDecl, args: &[TensorExpr]) -> RuntimeResult<Value> {
        let func_name = func_decl.name.as_str();

        // eprintln!("[DEBUG] call_user_defined_function: name={}, args.len={}", func_name, args.len());

        // Check argument count
        if args.len() != func_decl.params.len() {
            return Err(RuntimeError::TypeError(format!(
                "Function '{}' expects {} arguments, got {}",
                func_name,
                func_decl.params.len(),
                args.len()
            )));
        }

        // eprintln!("[DEBUG] call_user_defined_function: Argument count OK");

        // Create call frame for stack trace
        let frame = crate::interpreter::environment::CallFrame::new(func_name.to_string());

        // eprintln!("[DEBUG] call_user_defined_function: Evaluating {} arguments...", args.len());

        // Push call frame for stack trace
        self.call_stack.push(frame);

        // Push function scope
        self.env.push_scope(ScopeType::Function(func_name.to_string()));

        // Evaluate arguments and bind to parameters in function scope
        for (i, (param, arg)) in func_decl.params.iter().zip(args.iter()).enumerate() {
            // eprintln!("[DEBUG] call_user_defined_function: Evaluating arg[{}] for param '{}'", i, param.name.as_str());
            let arg_value = self.eval_expr(arg)?;
            // eprintln!("[DEBUG] call_user_defined_function: Arg[{}] evaluated successfully", i);
            // Note: Type checking could be optimized here in Phase 5
            self.check_type_match(&arg_value, &param.entity_type, param.name.as_str())?;
            self.env.declare_variable(param.name.as_str().to_string(), arg_value)?;
        }

        // eprintln!("[DEBUG] call_user_defined_function: All arguments evaluated");

        // Execute function body
        let body_len = func_decl.body.len();
        if body_len == 0 {
            self.env.pop_scope();
            self.call_stack.pop();
            return Err(RuntimeError::TypeError(format!(
                "Function '{}' has empty body",
                func_name
            )));
        }

        // Execute all statements except the last
        for stmt in &func_decl.body[..body_len - 1] {
            match self.execute_statement(stmt) {
                Ok(_) => {}
                Err(RuntimeError::ReturnValue(val)) => {
                    self.env.pop_scope();
                    self.call_stack.pop();
                    return Ok(val);
                }
                Err(e) => {
                    self.env.pop_scope();
                    self.call_stack.pop();
                    return Err(e);
                }
            }
        }

        // Handle last statement (implicit return)
        let return_value = match self.evaluate_last_statement(&func_decl.body[body_len - 1]) {
            Ok(Some(val)) => val,
            Ok(None) => Value::Void,
            Err(RuntimeError::ReturnValue(val)) => val,
            Err(e) => {
                self.env.pop_scope();
                self.call_stack.pop();
                return Err(e);
            }
        };

        // Type check return value
        self.check_return_type_match(&return_value, &func_decl.return_type, func_name)?;

        // Pop function scope and call frame, then return
        self.env.pop_scope();
        self.call_stack.pop();
        Ok(return_value)
    }

    /// Execute a statement
    fn eval_function_call(&mut self, type_namespace: Option<&str>, name: &Identifier, args: &[TensorExpr], resolved: Option<&crate::ast::ResolvedFunction>) -> RuntimeResult<Value> {
        let name_str = name.as_str();

        // Profiling: check if TL_PROFILE is set OR always profile key functions
        let key_functions = ["transformer_layer", "linear", "rope", "embedding", "attention_with_cache",
                             "softmax", "matmul", "einsum", "concat", "reshape"];
        let should_profile = std::env::var("TL_PROFILE").is_ok() || key_functions.contains(&name_str);
        let profile_enabled = std::env::var("TL_PROFILE").is_ok();
        let start_time = if should_profile {
            let full_name = if let Some(ns) = type_namespace {
                format!("{}::{}", ns, name_str)
            } else {
                name_str.to_string()
            };
            // eprintln!("[PROFILE] → {}", full_name);
            Some((full_name, std::time::Instant::now()))
        } else {
            None
        };

        // ========================================================================
        // OPTIMIZATION: Use resolved function reference if available
        // This eliminates HashMap lookup and string comparison overhead (~5μs)
        // ========================================================================
        if let Some(resolved_func) = resolved {
            match self.eval_resolved_function(resolved_func, args) {
                Ok(value) => {
                    if let Some((name, start)) = start_time {
                        let elapsed = start.elapsed();
                        // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
                    }
                    return Ok(value);
                }
                Err(RuntimeError::NotImplemented(_)) => {
                    // Fall through to string-based dispatch
                }
                Err(e) => {
                    if let Some((name, start)) = start_time {
                        let elapsed = start.elapsed();
                        // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
                    }
                    return Err(e);
                }
            }
        }

        // If type namespace is specified (e.g., f32::zeros), handle typed function call
        if let Some(type_ns) = type_namespace {
            let result = self.eval_typed_function_call(type_ns, name_str, args);
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }

        // Try dispatching to category-specific builtin modules
        // Each returns Option<RuntimeResult<Value>>: Some if handled, None if not in that category

        if let Some(result) = self.eval_tensor_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_math_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_nn_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_kg_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_gnn_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_model_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_sampling_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }
        if let Some(result) = self.eval_util_function(name_str, args) {
            if let Some((name, start)) = start_time {
                let elapsed = start.elapsed();
                // eprintln!("[PROFILE] ← {} ({:.3}ms)", name, elapsed.as_secs_f64() * 1000.0);
            }
            return result;
        }

        // Legacy builtin functions (to be migrated)
        match name_str {
            "apply_mask" => {
                // apply_mask(scores, mask)
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("apply_mask() expects 2 arguments (scores, mask), got {}", args.len())
                    ));
                }

                let scores_val = self.eval_expr(&args[0])?;
                let mask_val = self.eval_expr(&args[1])?;

                match (scores_val, mask_val) {
                    (Value::TensorF16(scores), Value::TensorF16(mask)) => {
                        let result = scores.apply_attention_mask(&mask)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    (Value::TensorF32(scores), Value::TensorF32(mask)) => {
                        let result = scores.apply_attention_mask(&mask)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "apply_mask() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
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

                let mask = Tensor::<half::f16>::causal_mask(seq_len)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::TensorF16(mask))
            }

            "batch_norm" => {
                // batch_norm(x, gamma, beta, eps) - f16 only
                use crate::interpreter::value::ToValue;
                if args.len() != 4 {
                    return Err(RuntimeError::TypeError(
                        format!("batch_norm() expects 4 arguments (x, gamma, beta, eps), got {}", args.len())
                    ));
                }

                let x_val = self.eval_expr(&args[0])?;
                let gamma_val = self.eval_expr(&args[1])?;
                let beta_val = self.eval_expr(&args[2])?;
                let eps = self.eval_expr(&args[3])?.as_float()? as f32;

                match (x_val, gamma_val, beta_val) {
                    (Value::TensorF16(x), Value::TensorF16(gamma), Value::TensorF16(beta)) => {
                        let result = x.batch_norm(&gamma, &beta, eps)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "batch_norm() requires all tensors to be f16 (f32 not yet supported)".to_string()
                    ))
                }
            }

            "dropout" => {
                // dropout(x, p, training)
                use crate::interpreter::value::ToValue;
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("dropout() expects 3 arguments (x, p, training), got {}", args.len())
                    ));
                }

                let x_val = self.eval_expr(&args[0])?;
                let p = self.eval_expr(&args[1])?.as_float()? as f32;
                let training = self.eval_expr(&args[2])?.as_bool()?;

                match x_val {
                    Value::TensorF16(x) => {
                        let result = x.dropout(p, training)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    Value::TensorF32(x) => {
                        let result = x.dropout(p, training)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("dropout() expects tensor (f16 or f32)".to_string()))
                }
            }

            "argmax" => {
                // argmax(tensor, dim: int = -1, keepdim: bool = false)
                use crate::interpreter::value::ToValue;
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("argmax() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

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

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let result = tensor.argmax(dim, keepdim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let result = tensor.argmax(dim, keepdim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("argmax() expects tensor (f16 or f32)".to_string()))
                }
            }

            "argmin" => {
                // argmin(tensor, dim: int = -1, keepdim: bool = false)
                use crate::interpreter::value::ToValue;
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("argmin() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

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

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let result = tensor.argmin(dim, keepdim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let result = tensor.argmin(dim, keepdim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(result.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("argmin() expects tensor (f16 or f32)".to_string()))
                }
            }

            "unsqueeze" => {
                // unsqueeze(tensor, dim)
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("unsqueeze() expects 2 arguments (tensor, dim), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                let dim = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError("unsqueeze() dim must be integer".to_string())),
                };

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.unsqueeze(dim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.unsqueeze(dim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("unsqueeze() expects tensor".to_string()))
                }
            }

            "top_k" => {
                // top_k(logits, k) -> Tensor
                // Keep only top-k logits, set others to -inf
                // Input: logits [vocab_size] or [..., vocab_size]
                // Output: same shape as input, with non-top-k values set to -inf
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("top_k() expects 2 arguments (logits, k), got {}", args.len())
                    ));
                }

                let logits_val = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("top_k() second argument must be a number (k), got {:?}", v)
                    )),
                };

                match logits_val {
                    Value::TensorF16(logits) => {
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
                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            dims.to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    Value::TensorF32(logits) => {
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
                                .map(|(i, &v)| (i, v))
                                .collect();
                            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                            // Set non-top-k to -inf
                            for i in k..vocab_size {
                                let idx = indexed_logits[i].0;
                                output_data[start_idx + idx] = f32::NEG_INFINITY;
                            }
                        }

                        // Create output tensor
                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            dims.to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("top_k() expects tensor (f16 or f32)".to_string()))
                }
            }

            "top_p" => {
                // top_p(logits, p) -> Tensor
                // Nucleus sampling: keep smallest set of logits with cumulative probability >= p
                // Input: logits [vocab_size] or [..., vocab_size]
                // Output: same shape, with non-nucleus values set to -inf
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("top_p() expects 2 arguments (logits, p), got {}", args.len())
                    ));
                }

                let logits_val = self.eval_expr(&args[0])?;
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

                match logits_val {
                    Value::TensorF16(logits) => {
                        let shape = logits.shape();
                        let dims = shape.dims();
                        let vocab_size = dims[dims.len() - 1];
                        let data = logits.to_vec();
                        let mut output_data = data.clone();
                        let batch_size = data.len() / vocab_size;

                        for batch_idx in 0..batch_size {
                            let start_idx = batch_idx * vocab_size;
                            let end_idx = start_idx + vocab_size;
                            let logits_slice = &data[start_idx..end_idx];

                            let max_logit = logits_slice.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
                            let exp_logits: Vec<f32> = logits_slice.iter().map(|v| (v.to_f32() - max_logit).exp()).collect();
                            let sum_exp: f32 = exp_logits.iter().sum();
                            let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

                            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                            let mut cumulative_prob = 0.0;
                            let mut nucleus_size = 0;
                            for (_, prob) in &indexed_probs {
                                cumulative_prob += prob;
                                nucleus_size += 1;
                                if cumulative_prob >= p {
                                    break;
                                }
                            }

                            for i in nucleus_size..vocab_size {
                                let idx = indexed_probs[i].0;
                                output_data[start_idx + idx] = half::f16::from_f32(f32::NEG_INFINITY);
                            }
                        }

                        let output = crate::tensor::Tensor::from_vec_gpu(self.env.metal_device(), output_data, dims.to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(logits) => {
                        let shape = logits.shape();
                        let dims = shape.dims();
                        let vocab_size = dims[dims.len() - 1];
                        let data = logits.to_vec();
                        let mut output_data = data.clone();
                        let batch_size = data.len() / vocab_size;

                        for batch_idx in 0..batch_size {
                            let start_idx = batch_idx * vocab_size;
                            let end_idx = start_idx + vocab_size;
                            let logits_slice = &data[start_idx..end_idx];

                            let max_logit = logits_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let exp_logits: Vec<f32> = logits_slice.iter().map(|&v| (v - max_logit).exp()).collect();
                            let sum_exp: f32 = exp_logits.iter().sum();
                            let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

                            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                            let mut cumulative_prob = 0.0;
                            let mut nucleus_size = 0;
                            for (_, prob) in &indexed_probs {
                                cumulative_prob += prob;
                                nucleus_size += 1;
                                if cumulative_prob >= p {
                                    break;
                                }
                            }

                            for i in nucleus_size..vocab_size {
                                let idx = indexed_probs[i].0;
                                output_data[start_idx + idx] = f32::NEG_INFINITY;
                            }
                        }

                        let output = crate::tensor::Tensor::from_vec_gpu(self.env.metal_device(), output_data, dims.to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("top_p() expects tensor (f16 or f32)".to_string()))
                }
            }

            "temperature" => {
                // temperature(logits, temp) -> Tensor
                // Scale logits by temperature: logits / temp
                // Higher temp = more random, lower temp = more deterministic
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("temperature() expects 2 arguments (logits, temp), got {}", args.len())
                    ));
                }

                let logits_val = self.eval_expr(&args[0])?;
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

                match logits_val {
                    Value::TensorF16(logits) => {
                        let data = logits.to_vec();
                        let output_data: Vec<half::f16> = data
                            .iter()
                            .map(|&v| half::f16::from_f32(v.to_f32() / temp))
                            .collect();

                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            logits.shape().dims().to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    Value::TensorF32(logits) => {
                        let data = logits.to_vec();
                        let output_data: Vec<f32> = data
                            .iter()
                            .map(|&v| v / temp)
                            .collect();

                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            logits.shape().dims().to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("temperature() expects tensor (f16 or f32)".to_string()))
                }
            }

            "softmax" => {
                // softmax(logits, dim=-1) -> Tensor
                // Convert logits to probability distribution
                // Default: apply softmax on last dimension
                use crate::interpreter::value::ToValue;

                if args.is_empty() || args.len() > 2 {
                    return Err(RuntimeError::TypeError(
                        format!("softmax() expects 1-2 arguments (logits, optional dim), got {}", args.len())
                    ));
                }

                let logits_val = self.eval_expr(&args[0])?;

                match logits_val {
                    Value::TensorF16(logits) => {
                        // For now, always apply softmax on last dimension
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
                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            dims.to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    Value::TensorF32(logits) => {
                        // For now, always apply softmax on last dimension
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
                                .copied()
                                .fold(f32::NEG_INFINITY, f32::max);

                            let exp_logits: Vec<f32> = logits_slice
                                .iter()
                                .map(|v| (v - max_logit).exp())
                                .collect();

                            let sum_exp: f32 = exp_logits.iter().sum();

                            // Normalize to get probabilities
                            for exp_val in exp_logits {
                                output_data.push(exp_val / sum_exp);
                            }
                        }

                        // Create output tensor
                        let output = crate::tensor::Tensor::from_vec_gpu(
                            self.env.metal_device(),
                            output_data,
                            dims.to_vec()
                        ).map_err(|e| RuntimeError::TensorError(e))?;

                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("softmax() expects tensor (f16 or f32)".to_string()))
                }
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

                let probs_val = self.eval_expr(&args[0])?;

                let probs_f32: Vec<f32> = match probs_val {
                    Value::TensorF16(probs_tensor) => {
                        let shape = probs_tensor.shape();
                        let dims = shape.dims();

                        // For now, only support 1D probability distributions
                        if dims.len() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("sample() currently only supports 1D probability distributions, got shape {:?}", dims)
                            ));
                        }

                        let probs = probs_tensor.to_vec();
                        probs.iter().map(|v| v.to_f32()).collect()
                    }
                    Value::TensorF32(probs_tensor) => {
                        let shape = probs_tensor.shape();
                        let dims = shape.dims();

                        // For now, only support 1D probability distributions
                        if dims.len() != 1 {
                            return Err(RuntimeError::TypeError(
                                format!("sample() currently only supports 1D probability distributions, got shape {:?}", dims)
                            ));
                        }

                        probs_tensor.to_vec()
                    }
                    _ => return Err(RuntimeError::TypeError("sample() expects tensor (f16 or f32)".to_string()))
                };

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

                let logits_val = self.eval_expr(&args[0])?;
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

                let logits_f32: Vec<f32> = match logits_val {
                    Value::TensorF16(logits_tensor) => {
                        let shape = logits_tensor.shape();
                        let dims = shape.dims();

                        if dims.len() == 1 {
                            let logits = logits_tensor.to_vec();
                            logits.iter().map(|v| v.to_f32()).collect()
                        } else if dims.len() == 2 {
                            let seq_len = dims[0];
                            let vocab_size = dims[1];
                            let logits = logits_tensor.to_vec();
                            let start_idx = (seq_len - 1) * vocab_size;
                            logits[start_idx..].iter().map(|v| v.to_f32()).collect()
                        } else {
                            return Err(RuntimeError::TypeError(
                                format!("temperature_sample() expects 1D or 2D logits, got shape {:?}", dims)
                            ));
                        }
                    }
                    Value::TensorF32(logits_tensor) => {
                        let shape = logits_tensor.shape();
                        let dims = shape.dims();

                        if dims.len() == 1 {
                            logits_tensor.to_vec()
                        } else if dims.len() == 2 {
                            let seq_len = dims[0];
                            let vocab_size = dims[1];
                            let logits = logits_tensor.to_vec();
                            let start_idx = (seq_len - 1) * vocab_size;
                            logits[start_idx..].to_vec()
                        } else {
                            return Err(RuntimeError::TypeError(
                                format!("temperature_sample() expects 1D or 2D logits, got shape {:?}", dims)
                            ));
                        }
                    }
                    _ => return Err(RuntimeError::TypeError("temperature_sample() expects tensor (f16 or f32)".to_string()))
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

                let tensor_val = self.eval_expr(&args[0])?;
                let k = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "print_top_k() k must be an integer".to_string()
                    )),
                };

                let logits_f32: Vec<f32> = match tensor_val {
                    Value::TensorF16(tensor) => {
                        let dims = tensor.shape().dims();
                        if dims.len() == 1 {
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
                        }
                    }
                    Value::TensorF32(tensor) => {
                        let dims = tensor.shape().dims();
                        if dims.len() == 1 {
                            tensor.to_vec()
                        } else if dims.len() == 2 {
                            let seq_len = dims[0];
                            let vocab_size = dims[1];
                            let logits = tensor.to_vec();
                            let start_idx = (seq_len - 1) * vocab_size;
                            logits[start_idx..].to_vec()
                        } else {
                            return Err(RuntimeError::TypeError(
                                format!("print_top_k() expects 1D or 2D tensor, got shape {:?}", dims)
                            ));
                        }
                    }
                    _ => return Err(RuntimeError::TypeError("print_top_k() expects tensor (f16 or f32)".to_string()))
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

                let logits_val = self.eval_expr(&args[0])?;
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

                // Support both 1D and 2D tensors
                // For 2D [seq_len, vocab_size], use the last row (last token's logits)
                let logits_f32: Vec<f32> = match logits_val {
                    Value::TensorF16(logits_tensor) => {
                        let shape = logits_tensor.shape();
                        let dims = shape.dims();

                        if dims.len() == 1 {
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
                        }
                    }
                    Value::TensorF32(logits_tensor) => {
                        let shape = logits_tensor.shape();
                        let dims = shape.dims();

                        if dims.len() == 1 {
                            logits_tensor.to_vec()
                        } else if dims.len() == 2 {
                            let seq_len = dims[0];
                            let vocab_size = dims[1];
                            let logits = logits_tensor.to_vec();
                            let start_idx = (seq_len - 1) * vocab_size;
                            logits[start_idx..].to_vec()
                        } else {
                            return Err(RuntimeError::TypeError(
                                format!("top_p_sample() expects 1D or 2D logits, got shape {:?}", dims)
                            ));
                        }
                    }
                    _ => return Err(RuntimeError::TypeError("top_p_sample() expects tensor (f16 or f32)".to_string()))
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
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("relu() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.relu().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.relu().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("relu() expects tensor (f16 or f32)".to_string()))
                }
            }

            "matmul" => {
                // matmul(a, b) -> Tensor
                // Matrix multiplication: a @ b
                // Supports batch matrix multiplication
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("matmul() expects 2 arguments (a, b), got {}", args.len())
                    ));
                }

                let a_val = self.eval_expr(&args[0])?;
                let b_val = self.eval_expr(&args[1])?;

                match (a_val, b_val) {
                    (Value::TensorF16(a), Value::TensorF16(b)) => {
                        let output = a.matmul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(a), Value::TensorF32(b)) => {
                        let output = a.matmul(&b).map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "matmul() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "layer_norm" => {
                // layer_norm(tensor, normalized_shape, eps=1e-5) -> Tensor
                // Layer normalization
                use crate::interpreter::value::ToValue;
                if args.len() < 1 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("layer_norm() expects 1-3 arguments (tensor, optional normalized_shape, optional eps), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                let eps = if args.len() >= 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Float(f) => f as f32,
                        Value::Integer(i) => i as f32,
                        _ => 1e-5_f32,
                    }
                } else {
                    1e-5_f32
                };

                match tensor_val {
                    Value::TensorF16(tensor) => {
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

                        let output = tensor.layer_norm(normalized_shape, None, None, eps)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
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

                        let output = tensor.layer_norm(normalized_shape, None, None, eps)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("layer_norm() expects tensor (f16 or f32)".to_string()))
                }
            }

            "rms_norm" => {
                // rms_norm(tensor, weight, eps=1e-6) -> Tensor
                // RMS normalization (used in LLaMA, TinyLlama)
                use crate::interpreter::value::ToValue;

                if args.len() < 2 || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("rms_norm() expects 2-3 arguments (tensor, weight, optional eps), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let weight_val = self.eval_expr(&args[1])?;

                let eps = if args.len() >= 3 {
                    match self.eval_expr(&args[2])? {
                        Value::Float(f) => f as f32,
                        Value::Integer(i) => i as f32,
                        _ => 1e-6_f32,  // Default eps for RMSNorm (LLaMA uses 1e-6)
                    }
                } else {
                    1e-6_f32
                };

                match (tensor_val, weight_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(weight)) => {
                        let normalized_shape = weight.shape().dims().to_vec();
                        let output = tensor.rms_norm(normalized_shape, &weight, eps)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor), Value::TensorF32(weight)) => {
                        let normalized_shape = weight.shape().dims().to_vec();
                        let output = tensor.rms_norm(normalized_shape, &weight, eps)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "rms_norm() requires tensor and weight to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "concat" => {
                // concat(tensors, dim) -> Tensor
                // Concatenate tensors along dimension
                // For now, simplified version that takes 2 tensors
                use crate::interpreter::value::ToValue;

                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("concat() expects 3 arguments (tensor1, tensor2, dim), got {}", args.len())
                    ));
                }

                let tensor1_val = self.eval_expr(&args[0])?;
                let tensor2_val = self.eval_expr(&args[1])?;

                let dim = match self.eval_expr(&args[2])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("concat() dim argument must be a number, got {:?}", v)
                    )),
                };

                match (tensor1_val, tensor2_val) {
                    (Value::TensorF16(tensor1), Value::TensorF16(tensor2)) => {
                        let tensors = vec![&tensor1, &tensor2];
                        let output = crate::tensor::Tensor::concat(&tensors[..], dim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor1), Value::TensorF32(tensor2)) => {
                        let tensors = vec![&tensor1, &tensor2];
                        let output = crate::tensor::Tensor::concat(&tensors[..], dim)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "concat() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "sigmoid" => {
                // sigmoid(tensor) -> Tensor
                // Sigmoid activation: 1 / (1 + exp(-x))
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sigmoid() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                Ok(match tensor_val {
                    Value::TensorF16(tensor) => {
                        tensor.sigmoid().map_err(|e| RuntimeError::TensorError(e))?.to_value()
                    }
                    Value::TensorF32(tensor) => {
                        tensor.sigmoid().map_err(|e| RuntimeError::TensorError(e))?.to_value()
                    }
                    _ => return Err(RuntimeError::TypeError("sigmoid() expects tensor (f16 or f32)".to_string()))
                })
            }

            "sum" => {
                // sum(tensor, dim, keepdim) -> Tensor or Float
                // Sum along dimension, or sum all elements
                use crate::interpreter::value::ToValue;

                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("sum() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                if args.len() == 1 {
                    // Sum all elements
                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let result = tensor.sum().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        Value::TensorF32(tensor) => {
                            let result = tensor.sum().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        _ => Err(RuntimeError::TypeError("sum() expects tensor (f16 or f32)".to_string()))
                    }
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

                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let output = tensor.sum_dim(dim, keepdim)
                                .map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(output.to_value())
                        }
                        Value::TensorF32(tensor) => {
                            let output = tensor.sum_dim(dim, keepdim)
                                .map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(output.to_value())
                        }
                        _ => Err(RuntimeError::TypeError("sum() expects tensor (f16 or f32)".to_string()))
                    }
                }
            }

            "mean" => {
                // mean(tensor, dim, keepdim) -> Tensor or Float
                // Mean along dimension, or mean of all elements
                use crate::interpreter::value::ToValue;

                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("mean() expects 1-3 arguments (tensor, optional dim, optional keepdim), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                if args.len() == 1 {
                    // Mean of all elements
                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let result = tensor.mean().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        Value::TensorF32(tensor) => {
                            let result = tensor.mean().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        _ => Err(RuntimeError::TypeError("mean() expects tensor (f16 or f32)".to_string()))
                    }
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

                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let output = tensor.mean_dim(dim, keepdim)
                                .map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(output.to_value())
                        }
                        Value::TensorF32(tensor) => {
                            let output = tensor.mean_dim(dim, keepdim)
                                .map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(output.to_value())
                        }
                        _ => Err(RuntimeError::TypeError("mean() expects tensor (f16 or f32)".to_string()))
                    }
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
                    Value::TensorF16(t) => {
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

                Ok(Value::TensorF16(tensor))
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
                    Value::TensorF16(t) => {
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

                Ok(Value::TensorF16(tensor))
            }

            // Tensor shape functions
            "reshape" => {
                // reshape(tensor, [new_shape])
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("reshape() expects 2 arguments (tensor, new_shape), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let shape_value = self.eval_expr(&args[1])?;

                // Extract new_shape from TensorF16 or TensorF32
                let new_shape = match shape_value {
                    Value::TensorF16(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    Value::TensorF32(t) => {
                        t.to_vec().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "reshape() new_shape must be an array".to_string()
                    )),
                };

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.reshape(new_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.reshape(new_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("reshape() expects tensor".to_string()))
                }
            }

            "flatten" => {
                // flatten(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("flatten() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.flatten()
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.flatten()
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("flatten() expects tensor".to_string()))
                }
            }

            "shape" => {
                // shape(tensor) -> returns shape as a 1D tensor of integers
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("shape() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                use crate::interpreter::value::ToValue;
                let tensor_val = self.eval_expr(&args[0])?;

                let device = MetalDevice::new().map_err(|e| RuntimeError::TensorError(e))?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let dims = tensor.dims();
                        let shape_vec: Vec<f16> = dims.iter().map(|&d| f16::from_f32(d as f32)).collect();
                        let shape_tensor = Tensor::from_vec_gpu(&device, shape_vec, vec![dims.len()])
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(shape_tensor.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let dims = tensor.dims();
                        let shape_vec: Vec<f32> = dims.iter().map(|&d| d as f32).collect();
                        let shape_tensor = Tensor::from_vec_gpu(&device, shape_vec, vec![dims.len()])
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(shape_tensor.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("shape() expects tensor (f16 or f32)".to_string()))
                }
            }

            "broadcast_to" => {
                // broadcast_to(tensor, [target_shape])
                // Broadcast tensor to target shape following NumPy broadcasting rules
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("broadcast_to() expects 2 arguments (tensor, target_shape), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let shape_value = self.eval_expr(&args[1])?;

                let target_shape = match shape_value {
                    Value::TensorF16(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    Value::TensorF32(t) => {
                        t.to_vec().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "broadcast_to() target_shape must be an array".to_string()
                    )),
                };

                let target_tensor_shape = TensorShape::new(target_shape);

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.broadcast_to(&target_tensor_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.broadcast_to(&target_tensor_shape)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("broadcast_to() expects tensor (f16 or f32)".to_string()))
                }
            }

            "transpose" => {
                // transpose(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("transpose() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.transpose()
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.transpose()
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("transpose() expects tensor".to_string()))
                }
            }

            "permute" => {
                // permute(tensor, [dims])
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("permute() expects 2 arguments (tensor, dims), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let dims_value = self.eval_expr(&args[1])?;

                // Extract dims from TensorF16 or TensorF32
                let dims = match dims_value {
                    Value::TensorF16(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    Value::TensorF32(t) => {
                        t.to_vec().iter().map(|&v| v as usize).collect()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "permute() dims must be an array".to_string()
                    )),
                };

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.permute(dims)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.permute(dims)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("permute() expects tensor (f16 or f32)".to_string()))
                }
            }

            // Indexing functions
            "gather" => {
                // gather(tensor, dim, indices)
                use crate::interpreter::value::ToValue;

                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("gather() expects 3 arguments (tensor, dim, indices), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let dim = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("gather() dim must be a number, got {:?}", v)
                    )),
                };
                let indices_val = self.eval_expr(&args[2])?;

                match (tensor_val, indices_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(indices)) => {
                        let output = tensor.gather(dim, &indices)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor), Value::TensorF32(indices)) => {
                        let output = tensor.gather(dim, &indices)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "gather() requires tensor and indices to be same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "scatter" => {
                // scatter(tensor, dim, indices, src)
                use crate::interpreter::value::ToValue;

                if args.len() != 4 {
                    return Err(RuntimeError::TypeError(
                        format!("scatter() expects 4 arguments (tensor, dim, indices, src), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let dim = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    v => return Err(RuntimeError::TypeError(
                        format!("scatter() dim must be a number, got {:?}", v)
                    )),
                };
                let indices_val = self.eval_expr(&args[2])?;
                let src_val = self.eval_expr(&args[3])?;

                match (tensor_val, indices_val, src_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(indices), Value::TensorF16(src)) => {
                        let output = tensor.scatter(dim, &indices, &src)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(_), Value::TensorF32(_), Value::TensorF32(_)) => {
                        // scatter() for f32 tensors is not yet implemented due to API constraints
                        Err(RuntimeError::TypeError(
                            "scatter() for f32 tensors is not yet implemented. Please use f16 tensors for scatter operations.".to_string()
                        ))
                    }
                    _ => Err(RuntimeError::TypeError(
                        "scatter() requires all tensors (tensor, indices, src) to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            // Reduction functions (max, min)
            "max" => {
                // max(tensor) -> scalar or max(tensor, dim, keepdim) -> tensor
                if args.is_empty() || args.len() > 3 {
                    return Err(RuntimeError::TypeError(
                        format!("max() expects 1 to 3 arguments, got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                if args.len() == 1 {
                    // max(tensor) -> scalar
                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let result = tensor.max().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        Value::TensorF32(tensor) => {
                            let result = tensor.max().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        _ => Err(RuntimeError::TypeError("max() expects tensor (f16 or f32)".to_string()))
                    }
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

                let tensor_val = self.eval_expr(&args[0])?;

                if args.len() == 1 {
                    // min(tensor) -> scalar
                    match tensor_val {
                        Value::TensorF16(tensor) => {
                            let result = tensor.min().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        Value::TensorF32(tensor) => {
                            let result = tensor.min().map_err(|e| RuntimeError::TensorError(e))?;
                            Ok(Value::Float(result.to_f32() as f64))
                        }
                        _ => Err(RuntimeError::TypeError("min() expects tensor (f16 or f32)".to_string()))
                    }
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        "min() with dimension not yet implemented".to_string()
                    ));
                }
            }

            // Activation functions
            "gelu" => {
                // gelu(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("gelu() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.gelu().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("gelu() expects tensor (f16 or f32)".to_string()))
                }
            }

            "tanh" => {
                // tanh(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("tanh() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.tanh().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.tanh().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("tanh() expects tensor (f16 or f32)".to_string()))
                }
            }

            // Math functions
            "exp" => {
                // exp(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("exp() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.exp().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.exp().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("exp() expects tensor (f16 or f32)".to_string()))
                }
            }

            "log" => {
                // log(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("log() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.log().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.log().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("log() expects tensor (f16 or f32)".to_string()))
                }
            }

            "sqrt" => {
                // sqrt(tensor)
                use crate::interpreter::value::ToValue;

                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sqrt() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.sqrt().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.sqrt().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("sqrt() expects tensor (f16 or f32)".to_string()))
                }
            }

            "pow" => {
                // pow(tensor, exponent)
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("pow() expects 2 arguments (tensor, exponent), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let exponent = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as f32,
                    Value::Float(f) => f as f32,
                    v => return Err(RuntimeError::TypeError(
                        format!("pow() exponent must be a number, got {:?}", v)
                    )),
                };

                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.pow(exponent).map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.pow(exponent).map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("pow() expects tensor (f16 or f32)".to_string()))
                }
            }

            "sin" => {
                // sin(tensor)
                use crate::interpreter::value::ToValue;
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("sin() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.sin().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.sin().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("sin() expects tensor (f16 or f32)".to_string()))
                }
            }

            "cos" => {
                // cos(tensor)
                use crate::interpreter::value::ToValue;
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("cos() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.cos().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.cos().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("cos() expects tensor (f16 or f32)".to_string()))
                }
            }

            "tan" => {
                // tan(tensor)
                use crate::interpreter::value::ToValue;
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("tan() expects 1 argument (tensor), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                match tensor_val {
                    Value::TensorF16(tensor) => {
                        let output = tensor.tan().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    Value::TensorF32(tensor) => {
                        let output = tensor.tan().map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError("tan() expects tensor (f16 or f32)".to_string()))
                }
            }

            // Masking operations
            "apply_attention_mask" => {
                // apply_attention_mask(tensor, mask)
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("apply_attention_mask() expects 2 arguments (tensor, mask), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let mask_val = self.eval_expr(&args[1])?;

                match (tensor_val, mask_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(mask)) => {
                        let output = tensor.apply_attention_mask(&mask)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor), Value::TensorF32(mask)) => {
                        let output = tensor.apply_attention_mask(&mask)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "apply_attention_mask() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "padding_mask" => {
                // padding_mask([lengths], max_len)
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("padding_mask() expects 2 arguments (lengths, max_len), got {}", args.len())
                    ));
                }

                // Parse lengths array
                let lengths_value = self.eval_expr(&args[0])?;
                let lengths: Vec<usize> = match lengths_value {
                    Value::TensorF16(t) => {
                        t.to_vec_f32().iter().map(|&v| v as usize).collect()
                    }
                    Value::TensorF32(t) => {
                        t.to_vec().iter().map(|&v| v as usize).collect()
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

                // Return f16 version by default for masks
                let output = crate::tensor::Tensor::<half::f16>::padding_mask(&lengths, max_len)
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(output.to_value())
            }

            "combine_masks" => {
                // combine_masks(mask1, mask2)
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("combine_masks() expects 2 arguments (mask1, mask2), got {}", args.len())
                    ));
                }

                let mask1_val = self.eval_expr(&args[0])?;
                let mask2_val = self.eval_expr(&args[1])?;

                match (mask1_val, mask2_val) {
                    (Value::TensorF16(mask1), Value::TensorF16(mask2)) => {
                        let output = mask1.combine_masks(&mask2)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(mask1), Value::TensorF32(mask2)) => {
                        let output = mask1.combine_masks(&mask2)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "combine_masks() requires both masks to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            // Fused operations
            "fused_add_relu" => {
                // fused_add_relu(tensor, other)
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_add_relu() expects 2 arguments (tensor, other), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let other_val = self.eval_expr(&args[1])?;

                match (tensor_val, other_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(other)) => {
                        let output = tensor.fused_add_relu(&other)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor), Value::TensorF32(other)) => {
                        let output = tensor.fused_add_relu(&other)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "fused_add_relu() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "fused_mul_relu" => {
                // fused_mul_relu(tensor, other)
                use crate::interpreter::value::ToValue;
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_mul_relu() expects 2 arguments (tensor, other), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let other_val = self.eval_expr(&args[1])?;

                match (tensor_val, other_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(other)) => {
                        let output = tensor.fused_mul_relu(&other)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    (Value::TensorF32(tensor), Value::TensorF32(other)) => {
                        let output = tensor.fused_mul_relu(&other)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "fused_mul_relu() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "fused_affine" => {
                // fused_affine(tensor, scale, bias) - f16 only
                use crate::interpreter::value::ToValue;
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_affine() expects 3 arguments (tensor, scale, bias), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let scale_val = self.eval_expr(&args[1])?;
                let bias_val = self.eval_expr(&args[2])?;

                match (tensor_val, scale_val, bias_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(scale), Value::TensorF16(bias)) => {
                        let output = tensor.fused_affine(&scale, &bias)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "fused_affine() requires all tensors to be f16 (f32 not yet supported for fused operations)".to_string()
                    ))
                }
            }

            "fused_gelu_linear" => {
                // fused_gelu_linear(tensor, weight, bias) - f16 only
                use crate::interpreter::value::ToValue;
                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("fused_gelu_linear() expects 3 arguments (tensor, weight, bias), got {}", args.len())
                    ));
                }

                let tensor_val = self.eval_expr(&args[0])?;
                let weight_val = self.eval_expr(&args[1])?;
                let bias_val = self.eval_expr(&args[2])?;

                match (tensor_val, weight_val, bias_val) {
                    (Value::TensorF16(tensor), Value::TensorF16(weight), Value::TensorF16(bias)) => {
                        let output = tensor.fused_gelu_linear(&weight, &bias)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        Ok(output.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "fused_gelu_linear() requires all tensors to be f16 (f32 not yet supported for fused operations)".to_string()
                    ))
                }
            }

            "generate" => {
                // generate(model, prompt, max_tokens: int = 100, temperature: float = 0.7)
                if args.len() < 2 || args.len() > 4 {
                    return Err(RuntimeError::TypeError(
                        format!("generate() expects 2-4 arguments (model, prompt, optional max_tokens, optional temperature), got {}", args.len())
                    ));
                }

                let model_val = self.eval_expr(&args[0])?;
                match model_val {
                    Value::ModelF16(_) | Value::ModelF32(_) => {},
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
                    "[Placeholder Response] You said: \"{}\". Full LLM inference not yet implemented.",
                    prompt
                );

                Ok(Value::String(response))
            }

            "entity_onehot" => {
                // entity_onehot(type_name, entity_name) -> Tensor
                // Convert an entity to a one-hot tensor
                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("entity_onehot() expects 2 arguments (type_name, entity_name), got {}", args.len())
                    ));
                }

                let type_name_val = self.eval_expr(&args[0])?;
                let type_name = match type_name_val {
                    Value::String(s) => s,
                    Value::Type(t) => t,
                    _ => return Err(RuntimeError::TypeError(
                        "entity_onehot() first argument must be a Type or string (type_name)".to_string()
                    )),
                };

                // Handle entity name: can be a variable (string value), symbol (identifier), or string literal
                let entity_name = match &args[1] {
                    TensorExpr::Variable(id) => {
                        // Try to evaluate as variable first
                        match self.eval_expr(&args[1]) {
                            Ok(Value::String(s)) => s,
                            Err(RuntimeError::UndefinedVariable(_)) => {
                                // Undefined variable - treat identifier as entity name (symbol)
                                id.as_str().to_string()
                            }
                            Err(e) => return Err(e),
                            Ok(_) => return Err(RuntimeError::TypeError(
                                "entity_onehot() second argument must be a string or entity symbol".to_string()
                            )),
                        }
                    }
                    _ => {
                        // Evaluate normally (e.g., string literal)
                        let val = self.eval_expr(&args[1])?;
                        match val {
                            Value::String(s) => s,
                            _ => return Err(RuntimeError::TypeError(
                                "entity_onehot() second argument must be a string or entity symbol".to_string()
                            )),
                        }
                    }
                };

                // Get one-hot vector from entity registry
                let onehot_vec = self.entity_registry.entity_to_onehot(&type_name, &entity_name)
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Entity '{}' not found in type '{}'", entity_name, type_name)
                    ))?;

                // Convert f32 to f16 for Metal
                let dim = onehot_vec.len();
                let f16_vec: Vec<half::f16> = onehot_vec.iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();

                // Convert to tensor
                let tensor = Tensor::from_vec_gpu(self.env.metal_device(), f16_vec, vec![dim])
                    .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::TensorF16(tensor))
            }

            "entity_dim" => {
                // entity_dim(type_name) -> Integer
                // Get the dimension (number of entities) of an entity type
                if args.len() != 1 {
                    return Err(RuntimeError::TypeError(
                        format!("entity_dim() expects 1 argument (type_name), got {}", args.len())
                    ));
                }

                let type_name_val = self.eval_expr(&args[0])?;
                let type_name = match type_name_val {
                    Value::String(s) => s,
                    Value::Type(t) => t,
                    _ => return Err(RuntimeError::TypeError(
                        "entity_dim() argument must be a Type or string (type_name)".to_string()
                    )),
                };

                // Get dimension from entity registry
                let dimension = self.entity_registry.get_entity_dimension(&type_name)
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Entity type '{}' not found", type_name)
                    ))?;

                Ok(Value::Integer(dimension as i64))
            }

            "transe_score" => {
                // transe_score(head, relation, tail, norm: "L1" or "L2") -> Tensor
                // TransE scoring function: score = -||h + r - t||
                use crate::interpreter::value::ToValue;

                if args.len() < 3 || args.len() > 4 {
                    return Err(RuntimeError::TypeError(
                        format!("transe_score() expects 3-4 arguments (head, relation, tail, norm?), got {}", args.len())
                    ));
                }

                // Evaluate arguments
                let head_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_val = self.eval_expr(&args[2])?;

                // Parse norm type (default: L2)
                let norm_type = if args.len() == 4 {
                    match self.eval_expr(&args[3])? {
                        Value::String(s) => s,
                        _ => return Err(RuntimeError::TypeError(
                            "transe_score() norm argument must be a string (\"L1\" or \"L2\")".to_string()
                        )),
                    }
                } else {
                    "L2".to_string()
                };

                match (head_val, relation_val, tail_val) {
                    (Value::TensorF16(head), Value::TensorF16(relation), Value::TensorF16(tail)) => {
                        // Compute h + r - t
                        let h_plus_r = head.add(&relation)?;
                        let diff = h_plus_r.sub(&tail)?;

                        // Compute norm
                        let score = match norm_type.as_str() {
                            "L1" => {
                                // L1 norm: sum(|x|)
                                // TODO: Implement abs() method for Tensor
                                return Err(RuntimeError::NotImplemented(
                                    "L1 norm not yet implemented (requires Tensor.abs())".to_string()
                                ));
                            }
                            "L2" => {
                                // L2 norm: sqrt(sum(x^2))
                                let squared = diff.mul(&diff)?;
                                let sum_squared_f16 = squared.sum()?;
                                let sum_squared_f32 = sum_squared_f16.to_f32();
                                let l2_norm_f32 = sum_squared_f32.sqrt();
                                let l2_norm_f16 = half::f16::from_f32(-l2_norm_f32);

                                // Create scalar tensor
                                let device = self.env.metal_device();
                                Tensor::from_vec_gpu(device, vec![l2_norm_f16], vec![1])?
                            }
                            _ => return Err(RuntimeError::InvalidOperation(
                                format!("transe_score() norm must be \"L1\" or \"L2\", got \"{}\"", norm_type)
                            )),
                        };

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(head), Value::TensorF32(relation), Value::TensorF32(tail)) => {
                        // Compute h + r - t
                        let h_plus_r = head.add(&relation)?;
                        let diff = h_plus_r.sub(&tail)?;

                        // Compute norm
                        let score = match norm_type.as_str() {
                            "L1" => {
                                // L1 norm: sum(|x|)
                                // TODO: Implement abs() method for Tensor
                                return Err(RuntimeError::NotImplemented(
                                    "L1 norm not yet implemented (requires Tensor.abs())".to_string()
                                ));
                            }
                            "L2" => {
                                // L2 norm: sqrt(sum(x^2))
                                let squared = diff.mul(&diff)?;
                                let sum_squared = squared.sum()?;
                                let l2_norm = sum_squared.sqrt();
                                let score_value = -l2_norm;

                                // Create scalar tensor
                                let device = self.env.metal_device();
                                Tensor::from_vec_gpu(device, vec![score_value], vec![1])?
                            }
                            _ => return Err(RuntimeError::InvalidOperation(
                                format!("transe_score() norm must be \"L1\" or \"L2\", got \"{}\"", norm_type)
                            )),
                        };

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "transe_score() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "distmult_score" => {
                // distmult_score(head, relation, tail) -> Tensor
                // DistMult scoring function: score = sum(h * r * t)
                use crate::interpreter::value::ToValue;

                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("distmult_score() expects 3 arguments (head, relation, tail), got {}", args.len())
                    ));
                }

                // Evaluate arguments
                let head_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_val = self.eval_expr(&args[2])?;

                match (head_val, relation_val, tail_val) {
                    (Value::TensorF16(head), Value::TensorF16(relation), Value::TensorF16(tail)) => {
                        // Compute element-wise product: h * r * t
                        let h_mul_r = head.mul(&relation)?;
                        let product = h_mul_r.mul(&tail)?;

                        // Sum all elements
                        let score_f16 = product.sum()?;

                        // Create scalar tensor
                        let device = self.env.metal_device();
                        let score_tensor = Tensor::from_vec_gpu(device, vec![score_f16], vec![1])?;

                        Ok(score_tensor.to_value())
                    }
                    (Value::TensorF32(head), Value::TensorF32(relation), Value::TensorF32(tail)) => {
                        // Compute element-wise product: h * r * t
                        let h_mul_r = head.mul(&relation)?;
                        let product = h_mul_r.mul(&tail)?;

                        // Sum all elements
                        let score = product.sum()?;

                        // Create scalar tensor
                        let device = self.env.metal_device();
                        let score_tensor = Tensor::from_vec_gpu(device, vec![score], vec![1])?;

                        Ok(score_tensor.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "distmult_score() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "complex_score" => {
                // complex_score(h_re, h_im, r_re, r_im, t_re, t_im) -> Tensor
                // ComplEx scoring function using complex embeddings
                // Formula: Re(<h, r, conj(t)>) = sum over i of:
                //   h_re[i] * r_re[i] * t_re[i]
                // + h_re[i] * r_im[i] * t_im[i]
                // + h_im[i] * r_re[i] * t_im[i]
                // - h_im[i] * r_im[i] * t_re[i]
                use crate::interpreter::value::ToValue;

                if args.len() != 6 {
                    return Err(RuntimeError::TypeError(
                        format!("complex_score() expects 6 arguments (h_re, h_im, r_re, r_im, t_re, t_im), got {}", args.len())
                    ));
                }

                // Evaluate arguments
                let h_re_val = self.eval_expr(&args[0])?;
                let h_im_val = self.eval_expr(&args[1])?;
                let r_re_val = self.eval_expr(&args[2])?;
                let r_im_val = self.eval_expr(&args[3])?;
                let t_re_val = self.eval_expr(&args[4])?;
                let t_im_val = self.eval_expr(&args[5])?;

                match (h_re_val, h_im_val, r_re_val, r_im_val, t_re_val, t_im_val) {
                    (Value::TensorF16(h_re), Value::TensorF16(h_im), Value::TensorF16(r_re),
                     Value::TensorF16(r_im), Value::TensorF16(t_re), Value::TensorF16(t_im)) => {
                        // Compute the four trilinear products

                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();

                        // Create scalar tensors for each term
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        // Add first three terms
                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;

                        // Subtract fourth term
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(h_re), Value::TensorF32(h_im), Value::TensorF32(r_re),
                     Value::TensorF32(r_im), Value::TensorF32(t_re), Value::TensorF32(t_im)) => {
                        // Compute the four trilinear products

                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();

                        // Create scalar tensors for each term
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        // Add first three terms
                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;

                        // Subtract fourth term
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "complex_score() requires all 6 tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "margin_ranking_loss" => {
                // margin_ranking_loss(pos_score, neg_score, margin) -> Tensor
                // Margin ranking loss: loss = max(0, margin + neg_score - pos_score)
                // Used in TransE training
                use crate::interpreter::value::ToValue;

                if args.len() != 3 {
                    return Err(RuntimeError::TypeError(
                        format!("margin_ranking_loss() expects 3 arguments (pos_score, neg_score, margin), got {}", args.len())
                    ));
                }

                // Evaluate arguments
                let pos_score_val = self.eval_expr(&args[0])?;
                let neg_score_val = self.eval_expr(&args[1])?;
                let margin_val = self.eval_expr(&args[2])?;

                // Parse margin
                let margin = match margin_val {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "margin_ranking_loss() margin must be a number".to_string()
                    )),
                };

                match (pos_score_val, neg_score_val) {
                    (Value::TensorF16(pos_score), Value::TensorF16(neg_score)) => {
                        // Compute: margin + neg_score - pos_score
                        let neg_minus_pos = neg_score.sub(&pos_score)?;

                        // Add margin
                        let margin_f16 = half::f16::from_f32(margin);
                        let device = self.env.metal_device();
                        let margin_tensor = Tensor::from_vec_gpu(device, vec![margin_f16], vec![1])?;
                        let diff_plus_margin = neg_minus_pos.add(&margin_tensor)?;

                        // Apply max(0, x) = ReLU
                        let loss = diff_plus_margin.relu()?;

                        Ok(loss.to_value())
                    }
                    (Value::TensorF32(pos_score), Value::TensorF32(neg_score)) => {
                        // Compute: margin + neg_score - pos_score
                        let neg_minus_pos = neg_score.sub(&pos_score)?;

                        // Add margin
                        let device = self.env.metal_device();
                        let margin_tensor = Tensor::from_vec_gpu(device, vec![margin], vec![1])?;
                        let diff_plus_margin = neg_minus_pos.add(&margin_tensor)?;

                        // Apply max(0, x) = ReLU
                        let loss = diff_plus_margin.relu()?;

                        Ok(loss.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "margin_ranking_loss() requires both tensors to be the same type (both f16 or both f32)".to_string()
                    ))
                }
            }

            "binary_cross_entropy" => {
                // binary_cross_entropy(score, target) -> Tensor
                // BCE loss: -target * log(sigmoid(score)) - (1-target) * log(1-sigmoid(score))
                // Used for binary classification of triples
                use crate::interpreter::value::ToValue;

                if args.len() != 2 {
                    return Err(RuntimeError::TypeError(
                        format!("binary_cross_entropy() expects 2 arguments (score, target), got {}", args.len())
                    ));
                }

                // Evaluate arguments
                let score_val = self.eval_expr(&args[0])?;
                let target_val = self.eval_expr(&args[1])?;

                // Parse target (0 or 1)
                let target_f32 = match target_val {
                    Value::Float(f) => f as f32,
                    Value::Integer(i) => i as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "binary_cross_entropy() target must be a number (0 or 1)".to_string()
                    )),
                };

                match score_val {
                    Value::TensorF16(score) => {
                        // Apply sigmoid to score
                        let prob = score.sigmoid()?;

                        // Compute BCE: -target * log(prob) - (1-target) * log(1-prob)
                        // For numerical stability, use: target * log_sigmoid(score) + (1-target) * log_sigmoid(-score)
                        let device = self.env.metal_device();

                        // Get prob value as f32
                        let prob_data = prob.to_vec();
                        let prob_f32 = prob_data[0].to_f32();

                        // Compute log(prob) and log(1-prob) with numerical stability
                        let log_prob = if prob_f32 > 0.0 { prob_f32.ln() } else { -100.0 }; // Clamp to avoid -inf
                        let log_one_minus_prob = if prob_f32 < 1.0 { (1.0 - prob_f32).ln() } else { -100.0 };

                        // BCE = -target * log(prob) - (1-target) * log(1-prob)
                        let bce_f32 = -target_f32 * log_prob - (1.0 - target_f32) * log_one_minus_prob;
                        let bce_f16 = half::f16::from_f32(bce_f32);

                        // Create scalar tensor
                        let loss_tensor = Tensor::from_vec_gpu(device, vec![bce_f16], vec![1])?;

                        Ok(loss_tensor.to_value())
                    }
                    Value::TensorF32(score) => {
                        // Apply sigmoid to score
                        let prob = score.sigmoid()?;

                        // Compute BCE: -target * log(prob) - (1-target) * log(1-prob)
                        // For numerical stability, use: target * log_sigmoid(score) + (1-target) * log_sigmoid(-score)
                        let device = self.env.metal_device();

                        // Get prob value as f32
                        let prob_data = prob.to_vec();
                        let prob_f32 = prob_data[0];

                        // Compute log(prob) and log(1-prob) with numerical stability
                        let log_prob = if prob_f32 > 0.0 { prob_f32.ln() } else { -100.0 }; // Clamp to avoid -inf
                        let log_one_minus_prob = if prob_f32 < 1.0 { (1.0 - prob_f32).ln() } else { -100.0 };

                        // BCE = -target * log(prob) - (1-target) * log(1-prob)
                        let bce = -target_f32 * log_prob - (1.0 - target_f32) * log_one_minus_prob;

                        // Create scalar tensor
                        let loss_tensor = Tensor::from_vec_gpu(device, vec![bce], vec![1])?;

                        Ok(loss_tensor.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "binary_cross_entropy() score must be a tensor (f16 or f32)".to_string()
                    ))
                }
            }

            "predict_tail_transe" => {
                // predict_tail_transe(head, relation, tail_candidates, model: "L2")
                // Computes TransE scores for multiple tail candidates
                // Returns list of scores (for now, just computes one at a time)
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_tail_transe() expects at least 3 arguments (head, relation, tail_candidate), got {}", args.len())
                    ));
                }

                let head_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_candidate_val = self.eval_expr(&args[2])?;

                let model = if args.len() > 3 {
                    match self.eval_expr(&args[3])? {
                        Value::String(s) => s,
                        _ => "L2".to_string(),
                    }
                } else {
                    "L2".to_string()
                };

                match (head_val, relation_val, tail_candidate_val) {
                    (Value::TensorF16(head), Value::TensorF16(relation), Value::TensorF16(tail_candidate)) => {
                        // Compute TransE score for this candidate
                        let device = self.env.metal_device();

                        // h + r
                        let h_plus_r = head.add(&relation)?;

                        // h + r - t
                        let diff = h_plus_r.sub(&tail_candidate)?;

                        // Compute norm based on model type
                        let score = if model == "L2" {
                            // L2 norm: -sqrt(sum(x^2))
                            let squared = diff.mul(&diff)?;
                            let sum_squared_f16 = squared.sum()?;
                            let sum_squared_f32 = sum_squared_f16.to_f32();
                            let l2_norm_f32 = sum_squared_f32.sqrt();
                            let score_f16 = half::f16::from_f32(-l2_norm_f32);
                            Tensor::from_vec_gpu(device, vec![score_f16], vec![1])?
                        } else {
                            return Err(RuntimeError::NotImplemented(
                                format!("predict_tail_transe: model '{}' not yet implemented (only L2 supported)", model)
                            ));
                        };

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(head), Value::TensorF32(relation), Value::TensorF32(tail_candidate)) => {
                        // Compute TransE score for this candidate
                        let device = self.env.metal_device();

                        // h + r
                        let h_plus_r = head.add(&relation)?;

                        // h + r - t
                        let diff = h_plus_r.sub(&tail_candidate)?;

                        // Compute norm based on model type
                        let score = if model == "L2" {
                            // L2 norm: -sqrt(sum(x^2))
                            let squared = diff.mul(&diff)?;
                            let sum_squared = squared.sum()?;
                            let l2_norm = sum_squared.sqrt();
                            let score_value = -l2_norm;
                            Tensor::from_vec_gpu(device, vec![score_value], vec![1])?
                        } else {
                            return Err(RuntimeError::NotImplemented(
                                format!("predict_tail_transe: model '{}' not yet implemented (only L2 supported)", model)
                            ));
                        };

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_tail_transe() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "predict_head_transe" => {
                // predict_head_transe(head_candidate, relation, tail, model: "L2")
                // Computes TransE scores for head candidates
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_head_transe() expects at least 3 arguments (head_candidate, relation, tail), got {}", args.len())
                    ));
                }

                let head_candidate_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_val = self.eval_expr(&args[2])?;

                let model = if args.len() > 3 {
                    match self.eval_expr(&args[3])? {
                        Value::String(s) => s,
                        _ => "L2".to_string(),
                    }
                } else {
                    "L2".to_string()
                };

                match (head_candidate_val, relation_val, tail_val) {
                    (Value::TensorF16(head_candidate), Value::TensorF16(relation), Value::TensorF16(tail)) => {
                        // Compute TransE score: -(||h_candidate + r - t||)
                        let device = self.env.metal_device();

                        let h_plus_r = head_candidate.add(&relation)?;
                        let diff = h_plus_r.sub(&tail)?;

                        let score = if model == "L2" {
                            let squared = diff.mul(&diff)?;
                            let sum_squared_f16 = squared.sum()?;
                            let sum_squared_f32 = sum_squared_f16.to_f32();
                            let l2_norm_f32 = sum_squared_f32.sqrt();
                            let score_f16 = half::f16::from_f32(-l2_norm_f32);
                            Tensor::from_vec_gpu(device, vec![score_f16], vec![1])?
                        } else {
                            return Err(RuntimeError::NotImplemented(
                                format!("predict_head_transe: model '{}' not yet implemented", model)
                            ));
                        };

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(head_candidate), Value::TensorF32(relation), Value::TensorF32(tail)) => {
                        // Compute TransE score: -(||h_candidate + r - t||)
                        let device = self.env.metal_device();

                        let h_plus_r = head_candidate.add(&relation)?;
                        let diff = h_plus_r.sub(&tail)?;

                        let score = if model == "L2" {
                            let squared = diff.mul(&diff)?;
                            let sum_squared = squared.sum()?;
                            let l2_norm = sum_squared.sqrt();
                            let score_value = -l2_norm;
                            Tensor::from_vec_gpu(device, vec![score_value], vec![1])?
                        } else {
                            return Err(RuntimeError::NotImplemented(
                                format!("predict_head_transe: model '{}' not yet implemented", model)
                            ));
                        };

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_head_transe() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "predict_tail_distmult" => {
                // predict_tail_distmult(head, relation, tail_candidate)
                // Computes DistMult scores for tail candidates
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_tail_distmult() expects 3 arguments (head, relation, tail_candidate), got {}", args.len())
                    ));
                }

                let head_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_candidate_val = self.eval_expr(&args[2])?;

                match (head_val, relation_val, tail_candidate_val) {
                    (Value::TensorF16(head), Value::TensorF16(relation), Value::TensorF16(tail_candidate)) => {
                        // DistMult: score = sum(h * r * t)
                        let device = self.env.metal_device();

                        let h_mul_r = head.mul(&relation)?;
                        let product = h_mul_r.mul(&tail_candidate)?;
                        let score_f16 = product.sum()?;

                        let score_tensor = Tensor::from_vec_gpu(device, vec![score_f16], vec![1])?;
                        Ok(score_tensor.to_value())
                    }
                    (Value::TensorF32(head), Value::TensorF32(relation), Value::TensorF32(tail_candidate)) => {
                        // DistMult: score = sum(h * r * t)
                        let device = self.env.metal_device();

                        let h_mul_r = head.mul(&relation)?;
                        let product = h_mul_r.mul(&tail_candidate)?;
                        let score = product.sum()?;

                        let score_tensor = Tensor::from_vec_gpu(device, vec![score], vec![1])?;
                        Ok(score_tensor.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_tail_distmult() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "predict_head_distmult" => {
                // predict_head_distmult(head_candidate, relation, tail)
                // Computes DistMult scores for head candidates
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_head_distmult() expects 3 arguments (head_candidate, relation, tail), got {}", args.len())
                    ));
                }

                let head_candidate_val = self.eval_expr(&args[0])?;
                let relation_val = self.eval_expr(&args[1])?;
                let tail_val = self.eval_expr(&args[2])?;

                match (head_candidate_val, relation_val, tail_val) {
                    (Value::TensorF16(head_candidate), Value::TensorF16(relation), Value::TensorF16(tail)) => {
                        // DistMult: score = sum(h * r * t)
                        let device = self.env.metal_device();

                        let h_mul_r = head_candidate.mul(&relation)?;
                        let product = h_mul_r.mul(&tail)?;
                        let score_f16 = product.sum()?;

                        let score_tensor = Tensor::from_vec_gpu(device, vec![score_f16], vec![1])?;
                        Ok(score_tensor.to_value())
                    }
                    (Value::TensorF32(head_candidate), Value::TensorF32(relation), Value::TensorF32(tail)) => {
                        // DistMult: score = sum(h * r * t)
                        let device = self.env.metal_device();

                        let h_mul_r = head_candidate.mul(&relation)?;
                        let product = h_mul_r.mul(&tail)?;
                        let score = product.sum()?;

                        let score_tensor = Tensor::from_vec_gpu(device, vec![score], vec![1])?;
                        Ok(score_tensor.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_head_distmult() requires all tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "predict_tail_complex" => {
                // predict_tail_complex(h_re, h_im, r_re, r_im, t_candidate_re, t_candidate_im)
                // Computes ComplEx scores for tail candidates
                // Uses ComplEx formula: Re(<h, r, conj(t)>)
                use crate::interpreter::value::ToValue;

                if args.len() < 6 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_tail_complex() expects 6 arguments (h_re, h_im, r_re, r_im, t_candidate_re, t_candidate_im), got {}", args.len())
                    ));
                }

                let h_re_val = self.eval_expr(&args[0])?;
                let h_im_val = self.eval_expr(&args[1])?;
                let r_re_val = self.eval_expr(&args[2])?;
                let r_im_val = self.eval_expr(&args[3])?;
                let t_candidate_re_val = self.eval_expr(&args[4])?;
                let t_candidate_im_val = self.eval_expr(&args[5])?;

                match (h_re_val, h_im_val, r_re_val, r_im_val, t_candidate_re_val, t_candidate_im_val) {
                    (Value::TensorF16(h_re), Value::TensorF16(h_im), Value::TensorF16(r_re),
                     Value::TensorF16(r_im), Value::TensorF16(t_candidate_re), Value::TensorF16(t_candidate_im)) => {
                        // Compute ComplEx score using the formula
                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_candidate_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_candidate_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_candidate_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_candidate_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(h_re), Value::TensorF32(h_im), Value::TensorF32(r_re),
                     Value::TensorF32(r_im), Value::TensorF32(t_candidate_re), Value::TensorF32(t_candidate_im)) => {
                        // Compute ComplEx score using the formula
                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_candidate_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_candidate_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_candidate_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_candidate_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_tail_complex() requires all 6 tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "predict_head_complex" => {
                // predict_head_complex(h_candidate_re, h_candidate_im, r_re, r_im, t_re, t_im)
                // Computes ComplEx scores for head candidates
                // Uses ComplEx formula: Re(<h, r, conj(t)>)
                use crate::interpreter::value::ToValue;

                if args.len() < 6 {
                    return Err(RuntimeError::TypeError(
                        format!("predict_head_complex() expects 6 arguments (h_candidate_re, h_candidate_im, r_re, r_im, t_re, t_im), got {}", args.len())
                    ));
                }

                let h_candidate_re_val = self.eval_expr(&args[0])?;
                let h_candidate_im_val = self.eval_expr(&args[1])?;
                let r_re_val = self.eval_expr(&args[2])?;
                let r_im_val = self.eval_expr(&args[3])?;
                let t_re_val = self.eval_expr(&args[4])?;
                let t_im_val = self.eval_expr(&args[5])?;

                match (h_candidate_re_val, h_candidate_im_val, r_re_val, r_im_val, t_re_val, t_im_val) {
                    (Value::TensorF16(h_candidate_re), Value::TensorF16(h_candidate_im), Value::TensorF16(r_re),
                     Value::TensorF16(r_im), Value::TensorF16(t_re), Value::TensorF16(t_im)) => {
                        // Compute ComplEx score using the formula
                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_candidate_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_candidate_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_candidate_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_candidate_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    (Value::TensorF32(h_candidate_re), Value::TensorF32(h_candidate_im), Value::TensorF32(r_re),
                     Value::TensorF32(r_im), Value::TensorF32(t_re), Value::TensorF32(t_im)) => {
                        // Compute ComplEx score using the formula
                        // Term 1: h_re * r_re * t_re
                        let h_re_r_re = h_candidate_re.mul(&r_re)?;
                        let term1_product = h_re_r_re.mul(&t_re)?;
                        let term1 = term1_product.sum()?;

                        // Term 2: h_re * r_im * t_im
                        let h_re_r_im = h_candidate_re.mul(&r_im)?;
                        let term2_product = h_re_r_im.mul(&t_im)?;
                        let term2 = term2_product.sum()?;

                        // Term 3: h_im * r_re * t_im
                        let h_im_r_re = h_candidate_im.mul(&r_re)?;
                        let term3_product = h_im_r_re.mul(&t_im)?;
                        let term3 = term3_product.sum()?;

                        // Term 4: h_im * r_im * t_re
                        let h_im_r_im = h_candidate_im.mul(&r_im)?;
                        let term4_product = h_im_r_im.mul(&t_re)?;
                        let term4 = term4_product.sum()?;

                        // Combine: term1 + term2 + term3 - term4
                        let device = self.env.metal_device();
                        let term1_tensor = Tensor::from_vec_gpu(device, vec![term1], vec![1])?;
                        let term2_tensor = Tensor::from_vec_gpu(device, vec![term2], vec![1])?;
                        let term3_tensor = Tensor::from_vec_gpu(device, vec![term3], vec![1])?;
                        let term4_tensor = Tensor::from_vec_gpu(device, vec![term4], vec![1])?;

                        let sum12 = term1_tensor.add(&term2_tensor)?;
                        let sum123 = sum12.add(&term3_tensor)?;
                        let score = sum123.sub(&term4_tensor)?;

                        Ok(score.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "predict_head_complex() requires all 6 tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "compute_rank" => {
                // compute_rank(target_score, candidate_scores_list)
                // Returns the rank of target_score among all candidates
                // Lower rank = better (rank 1 = highest score)
                if args.len() < 2 {
                    return Err(RuntimeError::TypeError(
                        format!("compute_rank() expects 2 arguments (target_score, num_higher_scores), got {}", args.len())
                    ));
                }

                let target_score_val = self.eval_expr(&args[0])?;
                let target_score_f32 = match target_score_val {
                    Value::TensorF16(ref t) => t.to_vec()[0].to_f32(),
                    Value::TensorF32(ref t) => t.to_vec()[0],
                    _ => return Err(RuntimeError::TypeError(
                        "compute_rank() target_score must be a tensor (f16 or f32)".to_string()
                    ))
                };

                // For simplicity, second arg is the count of candidates with higher scores
                let num_higher = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_rank: num_higher_scores must be a number".to_string()
                    )),
                };

                // Rank = number of candidates with higher score + 1
                let rank = num_higher + 1;

                Ok(Value::Integer(rank as i64))
            }

            "compute_mrr" => {
                // compute_mrr(rank) -> mean reciprocal rank
                // MRR = 1 / rank
                if args.is_empty() {
                    return Err(RuntimeError::TypeError(
                        "compute_mrr() expects 1 argument (rank)".to_string()
                    ));
                }

                let rank = match self.eval_expr(&args[0])? {
                    Value::Integer(i) => i as f32,
                    Value::Float(f) => f as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_mrr: rank must be a number".to_string()
                    )),
                };

                if rank <= 0.0 {
                    return Err(RuntimeError::InvalidOperation(
                        "compute_mrr: rank must be positive".to_string()
                    ));
                }

                let mrr = 1.0 / rank;
                Ok(Value::Float(mrr as f64))
            }

            "compute_hits_at_k" => {
                // compute_hits_at_k(rank, k) -> 1 if rank <= k, else 0
                // Hits@k metric: whether correct answer is in top-k
                if args.len() < 2 {
                    return Err(RuntimeError::TypeError(
                        format!("compute_hits_at_k() expects 2 arguments (rank, k), got {}", args.len())
                    ));
                }

                let rank = match self.eval_expr(&args[0])? {
                    Value::Integer(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_hits_at_k: rank must be a number".to_string()
                    )),
                };

                let k = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i,
                    Value::Float(f) => f as i64,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_hits_at_k: k must be a number".to_string()
                    )),
                };

                let hits = if rank <= k && rank > 0 { 1 } else { 0 };
                Ok(Value::Integer(hits))
            }

            "compute_mean_rank" => {
                // compute_mean_rank(sum_of_ranks, num_queries)
                // Mean Rank = sum of ranks / number of queries
                if args.len() < 2 {
                    return Err(RuntimeError::TypeError(
                        format!("compute_mean_rank() expects 2 arguments (sum_of_ranks, num_queries), got {}", args.len())
                    ));
                }

                let sum_ranks = match self.eval_expr(&args[0])? {
                    Value::Integer(i) => i as f32,
                    Value::Float(f) => f as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_mean_rank: sum_of_ranks must be a number".to_string()
                    )),
                };

                let num_queries = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as f32,
                    Value::Float(f) => f as f32,
                    _ => return Err(RuntimeError::TypeError(
                        "compute_mean_rank: num_queries must be a number".to_string()
                    )),
                };

                if num_queries <= 0.0 {
                    return Err(RuntimeError::InvalidOperation(
                        "compute_mean_rank: num_queries must be positive".to_string()
                    ));
                }

                let mean_rank = sum_ranks / num_queries;
                Ok(Value::Float(mean_rank as f64))
            }

            "aggregate_neighbors" => {
                // aggregate_neighbors(node_features, neighbor_indices, aggregation: "mean"|"sum")
                // Aggregates features from neighboring nodes
                // This is a simplified version for demonstration
                use crate::interpreter::value::ToValue;

                if args.len() < 2 {
                    return Err(RuntimeError::TypeError(
                        format!("aggregate_neighbors() expects at least 2 arguments (node_features, num_neighbors), got {}", args.len())
                    ));
                }

                let node_features_val = self.eval_expr(&args[0])?;

                let num_neighbors = match self.eval_expr(&args[1])? {
                    Value::Integer(i) => i as usize,
                    _ => return Err(RuntimeError::TypeError(
                        "aggregate_neighbors: num_neighbors must be an integer".to_string()
                    )),
                };

                let aggregation = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::String(s) => s,
                        _ => "mean".to_string(),
                    }
                } else {
                    "mean".to_string()
                };

                // For demonstration: return the node features
                // In a full implementation, this would aggregate from actual neighbor tensors
                let device = self.env.metal_device();

                match node_features_val {
                    Value::TensorF16(node_features) => {
                        if aggregation == "mean" && num_neighbors > 0 {
                            // Divide by number of neighbors for mean aggregation
                            let scale = 1.0 / (num_neighbors as f32);
                            let scale_f16 = half::f16::from_f32(scale);
                            let scale_tensor = Tensor::from_vec_gpu(device, vec![scale_f16], vec![1])?;
                            let aggregated = node_features.mul(&scale_tensor)?;
                            Ok(aggregated.to_value())
                        } else {
                            // Sum aggregation (or no neighbors)
                            Ok(node_features.to_value())
                        }
                    }
                    Value::TensorF32(node_features) => {
                        if aggregation == "mean" && num_neighbors > 0 {
                            // Divide by number of neighbors for mean aggregation
                            let scale = 1.0 / (num_neighbors as f32);
                            let scale_tensor = Tensor::from_vec_gpu(device, vec![scale], vec![1])?;
                            let aggregated = node_features.mul(&scale_tensor)?;
                            Ok(aggregated.to_value())
                        } else {
                            // Sum aggregation (or no neighbors)
                            Ok(node_features.to_value())
                        }
                    }
                    _ => Err(RuntimeError::TypeError(
                        "aggregate_neighbors() expects node_features to be a tensor (f16 or f32)".to_string()
                    ))
                }
            }

            "relational_aggregate" => {
                // relational_aggregate(node_emb, relation_emb, neighbor_emb, relation_weight)
                // R-GCN style aggregation: considers relation types
                // Formula: h_i^(l+1) = σ(Σ_r Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j^(l))
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("relational_aggregate() expects at least 3 arguments (node_emb, relation_emb, neighbor_emb), got {}", args.len())
                    ));
                }

                let node_emb_val = self.eval_expr(&args[0])?;
                let relation_emb_val = self.eval_expr(&args[1])?;
                let neighbor_emb_val = self.eval_expr(&args[2])?;

                match (node_emb_val, relation_emb_val, neighbor_emb_val) {
                    (Value::TensorF16(node_emb), Value::TensorF16(relation_emb), Value::TensorF16(neighbor_emb)) => {
                        // Simplified R-GCN: relation-specific transformation
                        // message = relation_emb * neighbor_emb (element-wise)
                        let message = relation_emb.mul(&neighbor_emb)?;

                        // Combine with node's own embedding
                        let combined = node_emb.add(&message)?;

                        Ok(combined.to_value())
                    }
                    (Value::TensorF32(node_emb), Value::TensorF32(relation_emb), Value::TensorF32(neighbor_emb)) => {
                        // Simplified R-GCN: relation-specific transformation
                        // message = relation_emb * neighbor_emb (element-wise)
                        let message = relation_emb.mul(&neighbor_emb)?;

                        // Combine with node's own embedding
                        let combined = node_emb.add(&message)?;

                        Ok(combined.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "relational_aggregate() requires all 3 tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "graph_attention" => {
                // graph_attention(query, key, value, num_neighbors)
                // GAT-style attention mechanism for graph
                // Simplified version: computes attention-weighted aggregation
                use crate::interpreter::value::ToValue;

                if args.len() < 3 {
                    return Err(RuntimeError::TypeError(
                        format!("graph_attention() expects at least 3 arguments (query, key, value), got {}", args.len())
                    ));
                }

                let query_val = self.eval_expr(&args[0])?;
                let key_val = self.eval_expr(&args[1])?;
                let value_val = self.eval_expr(&args[2])?;

                let device = self.env.metal_device();

                match (query_val, key_val, value_val) {
                    (Value::TensorF16(query), Value::TensorF16(key), Value::TensorF16(value)) => {
                        // Compute attention score: dot product of query and key
                        let qk = query.mul(&key)?;
                        let attention_score = qk.sum()?;

                        // Apply softmax (simplified: just use sigmoid for single neighbor)
                        let score_f32 = attention_score.to_f32();
                        let sigmoid_val = 1.0 / (1.0 + (-score_f32).exp());

                        // For simplicity, just scale the value by the attention weight
                        // In a full implementation, this would use proper broadcasting
                        let weight_f16 = half::f16::from_f32(sigmoid_val);
                        let weight_tensor = Tensor::from_vec_gpu(device, vec![weight_f16], vec![1])?;

                        // Simplified: return weighted value (scalar multiplication)
                        let attended_value = value.mul(&weight_tensor)?;

                        Ok(attended_value.to_value())
                    }
                    (Value::TensorF32(query), Value::TensorF32(key), Value::TensorF32(value)) => {
                        // Compute attention score: dot product of query and key
                        let qk = query.mul(&key)?;
                        let score_f32: f32 = qk.sum()?.into();  // Convert f16 to f32

                        // Apply softmax (simplified: just use sigmoid for single neighbor)
                        let sigmoid_val = 1.0 / (1.0 + (-score_f32).exp());

                        // For simplicity, just scale the value by the attention weight
                        // In a full implementation, this would use proper broadcasting
                        let weight_tensor = Tensor::from_vec_gpu(device, vec![sigmoid_val], vec![1])?;

                        // Simplified: return weighted value (scalar multiplication)
                        let attended_value = value.mul(&weight_tensor)?;

                        Ok(attended_value.to_value())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "graph_attention() requires all 3 tensors to be the same type (all f16 or all f32)".to_string()
                    ))
                }
            }

            "normalize_features" => {
                // normalize_features(features, norm_type: "l2"|"layer")
                // Normalizes node features
                use crate::interpreter::value::ToValue;

                if args.is_empty() {
                    return Err(RuntimeError::TypeError(
                        "normalize_features() expects at least 1 argument (features)".to_string()
                    ));
                }

                let features_val = self.eval_expr(&args[0])?;

                let norm_type = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::String(s) => s,
                        _ => "l2".to_string(),
                    }
                } else {
                    "l2".to_string()
                };

                let device = self.env.metal_device();

                match features_val {
                    Value::TensorF16(features) => {
                        if norm_type == "l2" {
                            // L2 normalization: x / ||x||_2
                            let squared = features.mul(&features)?;
                            let sum_squared_f16 = squared.sum()?;
                            let norm_f32 = sum_squared_f16.to_f32().sqrt();

                            if norm_f32 > 1e-8 {
                                let inv_norm_f16 = half::f16::from_f32(1.0 / norm_f32);
                                let inv_norm_tensor = Tensor::from_vec_gpu(device, vec![inv_norm_f16], vec![1])?;
                                let normalized = features.mul(&inv_norm_tensor)?;
                                Ok(normalized.to_value())
                            } else {
                                // Avoid division by zero
                                Ok(features.to_value())
                            }
                        } else {
                            // For other normalization types, just return features for now
                            Ok(features.to_value())
                        }
                    }
                    Value::TensorF32(features) => {
                        if norm_type == "l2" {
                            // L2 normalization: x / ||x||_2
                            let squared = features.mul(&features)?;
                            let sum_squared_f32: f32 = squared.sum()?.into();  // Convert f16 to f32
                            let norm_f32 = sum_squared_f32.sqrt();

                            if norm_f32 > 1e-8 {
                                let inv_norm = 1.0 / norm_f32;
                                let inv_norm_tensor = Tensor::from_vec_gpu(device, vec![inv_norm], vec![1])?;
                                let normalized = features.mul(&inv_norm_tensor)?;
                                Ok(normalized.to_value())
                            } else {
                                // Avoid division by zero
                                Ok(features.to_value())
                            }
                        } else {
                            // For other normalization types, just return features for now
                            Ok(features.to_value())
                        }
                    }
                    _ => Err(RuntimeError::TypeError(
                        "normalize_features() expects features to be a tensor (f16 or f32)".to_string()
                    ))
                }
            }

            "print" => {
                // print(value1, value2, ...) - simple mode
                // print("format {}", arg1, arg2, ...) - format string mode

                if args.is_empty() {
                    println!();
                    return Ok(Value::Void);
                }

                // Check if first argument is a string literal (format string mode)
                let first_val = self.eval_expr(&args[0])?;

                if let Value::String(format_str) = first_val {
                    // Format string mode: print("Hello {}", name)
                    if args.len() > 1 {
                        // Evaluate remaining arguments
                        let mut format_args = Vec::new();
                        for arg in &args[1..] {
                            format_args.push(self.eval_expr(arg)?);
                        }

                        // Use format_string helper from eval.rs
                        let formatted = self.format_string(&format_str, &format_args)?;
                        println!("{}", formatted);
                    } else {
                        // Just a string, print it
                        println!("{}", format_str);
                    }
                } else {
                    // Simple mode: print(value1, value2, ...)
                    print!("{}", self.value_to_display(&first_val));
                    for arg in &args[1..] {
                        print!(" ");
                        let val = self.eval_expr(arg)?;
                        print!("{}", self.value_to_display(&val));
                    }
                    println!();
                }

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

                    // 2. Create new call frame for stack trace
                    let frame = CallFrame::new(func_name.to_string());

                    // 3. Push call frame and function scope
                    self.call_stack.push(frame);
                    self.env.push_scope(ScopeType::Function(func_name.to_string()));

                    // 4. Evaluate arguments, check types, and bind to parameters
                    for (param, arg) in func_decl.params.iter().zip(args.iter()) {
                        let arg_value = self.eval_expr(arg)?;

                        // Type check the argument
                        self.check_type_match(&arg_value, &param.entity_type, param.name.as_str())?;

                        self.env.declare_variable(param.name.as_str().to_string(), arg_value)?;
                    }

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
                                    // Other errors - call destructors, pop scope and propagate upward
                                    let _ = self.call_scope_destructors();
                                    self.env.pop_scope();
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
                                let _ = self.call_scope_destructors();
                                self.env.pop_scope();
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

                    // 8. Call destructors for local variables before popping scope/frame
                    self.call_scope_destructors()?;

                    // 9. Pop function scope and call frame
                    self.env.pop_scope();
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

                let actual_shape = match tensor_val {
                    Value::TensorF16(ref t) => t.shape().dims(),
                    Value::TensorF32(ref t) => t.shape().dims(),
                    _ => return Err(RuntimeError::TypeError(
                        "Shape constraint requires a tensor (f16 or f32)".to_string()
                    ))
                };

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

                let actual_rank = match tensor_val {
                    Value::TensorF16(ref t) => t.rank(),
                    Value::TensorF32(ref t) => t.rank(),
                    _ => return Err(RuntimeError::TypeError(
                        "Rank constraint requires a tensor (f16 or f32)".to_string()
                    ))
                };

                Ok(actual_rank == *rank)
            }

            Constraint::Norm { tensor, op, value } => {
                // Get tensor value
                let tensor_val = self.eval_expr(tensor)?;

                // Calculate L2 norm based on tensor type
                let norm: f32 = match tensor_val {
                    Value::TensorF16(ref t) => {
                        let data = t.to_vec();
                        data.iter()
                            .map(|x| {
                                let val = x.to_f32();
                                val * val
                            })
                            .sum::<f32>()
                            .sqrt()
                    }
                    Value::TensorF32(ref t) => {
                        let data = t.to_vec();
                        data.iter()
                            .map(|&x| x * x)
                            .sum::<f32>()
                            .sqrt()
                    }
                    _ => return Err(RuntimeError::TypeError(
                        "Norm constraint requires a tensor (f16 or f32)".to_string()
                    ))
                };

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
        // Only collect tensors that were explicitly declared with 'learnable' keyword
        // This prevents intermediate variables (like loss) from being treated as parameters
        let mut learnable_params = Vec::new();
        let mut learnable_param_names = Vec::new(); // Store names for later rebuilding
        let mut learnable_param_node_ids = Vec::new(); // Store node IDs for gradient retrieval
        for name in &self.learnable_params {
            if let Ok(value) = self.env.get_variable(name) {
                match value {
                    Value::TensorF16(tensor) => {
                        let node_id = tensor.grad_node().ok_or_else(|| {
                            RuntimeError::InvalidOperation(format!(
                                "Learnable parameter '{}' does not have a computation graph node", name
                            ))
                        })?;
                        learnable_params.push((name.clone(), tensor.clone()));
                        learnable_param_names.push(name.clone());
                        learnable_param_node_ids.push(node_id);
                        println!("\nLearnable parameter: {}", name);
                        println!("  Shape: {:?}", tensor.shape().dims());
                        println!("  Initial values: {:?}", &tensor.to_vec()[..std::cmp::min(5, tensor.shape().dims()[0])].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
                    }
                    Value::TensorF32(_) => {
                        return Err(RuntimeError::InvalidOperation(
                            format!("Learnable parameter '{}' is float32, but autograd currently only supports float16. Please use 'float16' type for learnable tensors.", name)
                        ));
                    }
                    _ => {}
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

            // Update node_ids from optimizer parameters (they may change after step())
            // Also update environment with latest parameter values
            learnable_param_node_ids.clear();
            let current_params = opt.params();
            for (idx, param) in current_params.iter().enumerate() {
                if let Some(node_id) = param.grad_node() {
                    learnable_param_node_ids.push(node_id);
                }
                // Update environment with current parameter
                if idx < learnable_param_names.len() {
                    let name = &learnable_param_names[idx];
                    let mut param_clone = param.clone();
                    param_clone.requires_grad = true;
                    if let Some(node_id) = param.grad_node() {
                        use crate::tensor::TensorAutograd;
                        param_clone.set_grad_node(node_id);
                    }
                    let _ = self.env.set_variable(name, Value::TensorF16(param_clone));
                }
            }

            // Re-execute statements for each epoch (recompute intermediate variables)
            for stmt in &spec.statements {
                self.execute_statement(stmt)?;
            }

            // Compute loss
            let loss_val = self.eval_expr(&spec.objective)?;

            // Calculate loss value and get loss tensor (support both f16 and f32)
            let (loss_scalar, loss_tensor) = match loss_val {
                Value::TensorF16(ref t) => {
                    let loss_data = t.to_vec();
                    let scalar = if loss_data.is_empty() {
                        0.0
                    } else {
                        loss_data[0].to_f32()
                    };
                    (scalar, t.clone())
                }
                Value::TensorF32(ref t) => {
                    let loss_data = t.to_vec();
                    let scalar = if loss_data.is_empty() {
                        0.0
                    } else {
                        loss_data[0]
                    };
                    // For now, training only supports f16 tensors for backward pass
                    // Convert f32 loss to f16 for gradient computation
                    return Err(RuntimeError::TypeError(
                        "Training currently only supports f16 tensors. Please use f16 for trainable parameters and loss.".to_string()
                    ));
                }
                _ => return Err(RuntimeError::TypeError(
                    "Loss must be a tensor (f16 or f32)".to_string()
                ))
            };

            // Display epoch progress
            print!("Epoch {:3}/{}: Loss = {:.6}", epoch + 1, spec.epochs, loss_scalar);

            // Compute gradients using autograd
            // 1. Compute backward pass
            let mut loss_tensor_mut = loss_tensor.clone();
            match loss_tensor_mut.backward() {
                Ok(_) => {
                    // 2. Collect gradients from all learnable parameters and compute norm
                    // Get gradients from TENSOR_REGISTRY using node IDs
                    use crate::autograd::AutogradContext;
                    let mut grad_norm_squared = 0.0f32;
                    for node_id in learnable_param_node_ids.iter() {
                        if let Some(param_tensor) = AutogradContext::get_tensor_generic::<half::f16>(*node_id) {
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
                            // IMPORTANT: Preserve the original node_id to maintain gradient information
                            let updated_params = opt.params();
                            for (((name, _), new_tensor), node_id) in learnable_params.iter()
                                .zip(updated_params.iter())
                                .zip(learnable_param_node_ids.iter()) {
                                let mut param_with_grad = new_tensor.clone();
                                // Manually set requires_grad without allocating a new node
                                use crate::tensor::TensorAutograd;
                                param_with_grad.requires_grad = true;
                                // Restore the original node_id to maintain gradient connection
                                param_with_grad.set_grad_node(*node_id);
                                // Learning context - update parameter
                                let _ = self.env.set_variable(name, Value::TensorF16(param_with_grad));
                            }

                            // Note: We don't rebuild learnable_params from the environment
                            // The optimizer maintains the authoritative parameter state
                            // and we sync to the environment above
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
                    if let Ok(Value::TensorF16(t)) = self.env.get_variable(name) {
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
            if let Ok(Value::TensorF16(t)) = self.env.get_variable(name) {
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
            // For each entity-typed parameter
            for (&param_idx, entity_type_name) in entity_params {
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

                    if let Some(entity_name) = entity_name {
                        if entity_type_name == "entity" {
                            // Generic entity type - add to all data-driven types
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

                            for type_name in &data_driven_types {
                                self.entity_registry.add_entity(type_name, entity_name.clone())
                                    .map_err(|e| RuntimeError::InvalidOperation(e))?;
                            }
                        } else {
                            // Specific entity type - add only to this type if it exists and is data-driven
                            if let Some(type_info) = self.entity_registry.get_type_info(entity_type_name) {
                                match &type_info.declaration_type {
                                    crate::entity_registry::EntityDeclType::FromData => {
                                        self.entity_registry.add_entity(entity_type_name, entity_name)
                                            .map_err(|e| RuntimeError::InvalidOperation(e))?;
                                    }
                                    _ => {
                                        // Explicit entity type - no need to collect
                                    }
                                }
                            }
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
