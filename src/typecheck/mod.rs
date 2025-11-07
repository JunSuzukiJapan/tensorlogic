//! Type Checker for TensorLogic
//!
//! This module implements static type checking for TensorLogic programs using
//! the visitor pattern to traverse the AST and infer/validate types.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic::typecheck::TypeChecker;
//! use tensorlogic::parser::TensorLogicParser;
//!
//! let source = "tensor w: float32[10, 20] learnable";
//! let program = TensorLogicParser::parse_program(source)?;
//!
//! let mut checker = TypeChecker::new();
//! checker.check_program(&program)?;
//! ```

use std::collections::HashMap;

use crate::ast::*;

/// Type checking errors
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Incompatible dimensions: {left:?} vs {right:?}")]
    DimensionMismatch { left: Vec<Dimension>, right: Vec<Dimension> },

    #[error("Incompatible base types: {left:?} vs {right:?}")]
    BaseTypeMismatch { left: BaseType, right: BaseType },

    #[error("Invalid operation {op} for types {left} and {right}")]
    InvalidOperation { op: String, left: String, right: String },

    #[error("Duplicate declaration: {0}")]
    DuplicateDeclaration(String),

    #[error("Relation {0} not found")]
    UndefinedRelation(String),

    #[error("Function {0} not found")]
    UndefinedFunction(String),

    #[error("Struct {0} not found")]
    UndefinedStruct(String),

    #[error("Method {0}::{1} not found")]
    UndefinedMethod(String, String),

    #[error("Field {0} not found in struct {1}")]
    UndefinedField(String, String),

    #[error("Wrong number of arguments: expected {expected}, found {found}")]
    ArgumentCountMismatch { expected: usize, found: usize },

    #[error("Wrong number of type arguments: expected {expected}, found {found}")]
    TypeArgumentCountMismatch { expected: usize, found: usize },

    #[error("Cannot infer type for expression")]
    CannotInferType,

    #[error("Dimension variable {0} not in scope")]
    UndefinedDimensionVariable(String),

    #[error("Type parameter {0} not in scope")]
    UndefinedTypeParameter(String),
}

pub type TypeResult<T> = Result<T, TypeError>;

/// Type information for a tensor
#[derive(Debug, Clone, PartialEq)]
pub struct TensorTypeInfo {
    pub base_type: BaseType,
    pub dimensions: Vec<Dimension>,
    pub learnable: LearnableStatus,
}

impl TensorTypeInfo {
    pub fn new(base_type: BaseType, dimensions: Vec<Dimension>) -> Self {
        Self {
            base_type,
            dimensions,
            learnable: LearnableStatus::Default,
        }
    }

    pub fn from_tensor_type(tensor_type: &TensorType) -> Self {
        Self {
            base_type: tensor_type.base_type.clone(),
            dimensions: tensor_type.dimensions.clone(),
            learnable: tensor_type.learnable.clone(),
        }
    }

    /// Check if this type is compatible with another for assignment
    pub fn is_compatible_with(&self, other: &TensorTypeInfo) -> bool {
        self.base_type == other.base_type && self.dimensions_match(&other.dimensions)
    }

    /// Check if dimensions match (accounting for dynamic dimensions)
    pub fn dimensions_match(&self, other_dims: &[Dimension]) -> bool {
        if self.dimensions.len() != other_dims.len() {
            return false;
        }

        self.dimensions.iter().zip(other_dims.iter()).all(|(d1, d2)| {
            match (d1, d2) {
                (Dimension::Dynamic, _) | (_, Dimension::Dynamic) => true,
                (Dimension::Fixed(n1), Dimension::Fixed(n2)) => n1 == n2,
                (Dimension::Variable(v1), Dimension::Variable(v2)) => v1.as_str() == v2.as_str(),
                _ => false,
            }
        })
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

/// Type environment for tracking variable types
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    /// Variable name → type info
    variables: HashMap<String, TensorTypeInfo>,
    /// Relation name → parameter types
    relations: HashMap<String, Vec<EntityType>>,
    /// Function name → (parameter types, return type)
    functions: HashMap<String, (Vec<TensorTypeInfo>, Option<TensorTypeInfo>)>,
    /// Dimension variables in scope
    dimension_vars: HashMap<String, ()>,
    /// Struct name → (type params, fields)
    structs: HashMap<String, StructDecl>,
    /// (Struct name, Method name) → method declaration
    methods: HashMap<(String, String), (Vec<TypeParam>, MethodDecl)>,
    /// Type parameter context (scoped)
    type_params: HashMap<String, ()>,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            relations: HashMap::new(),
            functions: HashMap::new(),
            dimension_vars: HashMap::new(),
            structs: HashMap::new(),
            methods: HashMap::new(),
            type_params: HashMap::new(),
        }
    }

    /// Add a variable to the environment
    pub fn add_variable(&mut self, name: String, type_info: TensorTypeInfo) -> TypeResult<()> {
        if self.variables.contains_key(&name) {
            return Err(TypeError::DuplicateDeclaration(name));
        }
        self.variables.insert(name, type_info);
        Ok(())
    }

    /// Get a variable's type
    pub fn get_variable(&self, name: &str) -> TypeResult<&TensorTypeInfo> {
        self.variables
            .get(name)
            .ok_or_else(|| TypeError::UndefinedVariable(name.to_string()))
    }

    /// Add a relation to the environment
    pub fn add_relation(&mut self, name: String, params: Vec<EntityType>) -> TypeResult<()> {
        if self.relations.contains_key(&name) {
            return Err(TypeError::DuplicateDeclaration(name));
        }
        self.relations.insert(name, params);
        Ok(())
    }

    /// Get a relation's parameter types
    pub fn get_relation(&self, name: &str) -> TypeResult<&Vec<EntityType>> {
        self.relations
            .get(name)
            .ok_or_else(|| TypeError::UndefinedRelation(name.to_string()))
    }

    /// Add a function to the environment
    pub fn add_function(
        &mut self,
        name: String,
        params: Vec<TensorTypeInfo>,
        return_type: Option<TensorTypeInfo>,
    ) -> TypeResult<()> {
        if self.functions.contains_key(&name) {
            return Err(TypeError::DuplicateDeclaration(name));
        }
        self.functions.insert(name, (params, return_type));
        Ok(())
    }

    /// Get a function's signature
    pub fn get_function(
        &self,
        name: &str,
    ) -> TypeResult<&(Vec<TensorTypeInfo>, Option<TensorTypeInfo>)> {
        self.functions
            .get(name)
            .ok_or_else(|| TypeError::UndefinedFunction(name.to_string()))
    }

    /// Add a dimension variable
    pub fn add_dimension_var(&mut self, name: String) {
        self.dimension_vars.insert(name, ());
    }

    /// Check if a dimension variable is in scope
    pub fn has_dimension_var(&self, name: &str) -> bool {
        self.dimension_vars.contains_key(name)
    }

    /// Add a struct to the environment
    pub fn add_struct(&mut self, struct_decl: StructDecl) -> TypeResult<()> {
        let name = struct_decl.name.as_str().to_string();
        if self.structs.contains_key(&name) {
            return Err(TypeError::DuplicateDeclaration(name));
        }
        self.structs.insert(name, struct_decl);
        Ok(())
    }

    /// Get a struct declaration
    pub fn get_struct(&self, name: &str) -> TypeResult<&StructDecl> {
        self.structs
            .get(name)
            .ok_or_else(|| TypeError::UndefinedStruct(name.to_string()))
    }

    /// Add a method to the environment
    pub fn add_method(
        &mut self,
        struct_name: String,
        type_params: Vec<TypeParam>,
        method: MethodDecl,
    ) -> TypeResult<()> {
        let method_name = method.name.as_str().to_string();
        let key = (struct_name.clone(), method_name.clone());
        if self.methods.contains_key(&key) {
            return Err(TypeError::DuplicateDeclaration(format!(
                "{}::{}",
                struct_name, method_name
            )));
        }
        self.methods.insert(key, (type_params, method));
        Ok(())
    }

    /// Get a method declaration
    pub fn get_method(
        &self,
        struct_name: &str,
        method_name: &str,
    ) -> TypeResult<&(Vec<TypeParam>, MethodDecl)> {
        let key = (struct_name.to_string(), method_name.to_string());
        self.methods
            .get(&key)
            .ok_or_else(|| TypeError::UndefinedMethod(struct_name.to_string(), method_name.to_string()))
    }

    /// Add a type parameter
    pub fn add_type_param(&mut self, name: String) {
        self.type_params.insert(name, ());
    }

    /// Check if a type parameter is in scope
    pub fn has_type_param(&self, name: &str) -> bool {
        self.type_params.contains_key(name)
    }

    /// Clear type parameters (when exiting a generic scope)
    pub fn clear_type_params(&mut self) {
        self.type_params.clear();
    }
}

/// Type checker implementation
pub struct TypeChecker {
    env: TypeEnvironment,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnvironment::new(),
        }
    }

    /// Type check a complete program
    pub fn check_program(&mut self, program: &Program) -> TypeResult<()> {
        // First pass: collect all declarations
        for decl in &program.declarations {
            self.check_declaration(decl)?;
        }

        // Second pass: check main block if present
        if let Some(main_block) = &program.main_block {
            self.check_main_block(main_block)?;
        }

        Ok(())
    }

    /// Type check a declaration
    fn check_declaration(&mut self, decl: &Declaration) -> TypeResult<()> {
        match decl {
            Declaration::Import(_) => {
                // Import declarations don't need type checking
                // Type checking happens when the imported file is parsed
                Ok(())
            }
            Declaration::Entity(_) => {
                // Entity declarations don't need type checking for now
                // They will be validated at runtime
                Ok(())
            }
            Declaration::Tensor(tensor_decl) => self.check_tensor_decl(tensor_decl),
            Declaration::Relation(relation_decl) => self.check_relation_decl(relation_decl),
            Declaration::Rule(rule_decl) => self.check_rule_decl(rule_decl),
            Declaration::Embedding(embedding_decl) => self.check_embedding_decl(embedding_decl),
            Declaration::RelationEmbedding(_) => {
                // Relation embedding declarations don't need type checking for now
                // They will be validated at runtime
                Ok(())
            }
            Declaration::Function(function_decl) => self.check_function_decl(function_decl),
            Declaration::Struct(struct_decl) => self.check_struct_decl(struct_decl),
            Declaration::Impl(impl_block) => self.check_impl_block(impl_block),
        }
    }

    /// Type check a tensor declaration
    fn check_tensor_decl(&mut self, decl: &TensorDecl) -> TypeResult<()> {
        let type_info = TensorTypeInfo::from_tensor_type(&decl.tensor_type);

        // Validate dimension variables
        for dim in &type_info.dimensions {
            if let Dimension::Variable(var) = dim {
                if !self.env.has_dimension_var(var.as_str()) {
                    self.env.add_dimension_var(var.as_str().to_string());
                }
            }
        }

        // If there's an initialization expression, check it
        if let Some(init_expr) = &decl.init_expr {
            let expr_type = self.infer_expr_type(init_expr)?;
            if !type_info.is_compatible_with(&expr_type) {
                return Err(TypeError::TypeMismatch {
                    expected: format!("{:?}", type_info),
                    found: format!("{:?}", expr_type),
                });
            }
        }

        self.env
            .add_variable(decl.name.as_str().to_string(), type_info)?;

        Ok(())
    }

    /// Type check a relation declaration
    fn check_relation_decl(&mut self, decl: &RelationDecl) -> TypeResult<()> {
        let param_types = decl.params.iter().map(|p| p.entity_type.clone()).collect();

        self.env
            .add_relation(decl.name.as_str().to_string(), param_types)?;

        Ok(())
    }

    /// Type check a rule declaration
    fn check_rule_decl(&mut self, _decl: &RuleDecl) -> TypeResult<()> {
        // Simplified: rule type checking deferred
        Ok(())
    }

    /// Type check an embedding declaration
    fn check_embedding_decl(&mut self, _decl: &EmbeddingDecl) -> TypeResult<()> {
        // Simplified: embedding type checking deferred
        Ok(())
    }

    /// Type check a function declaration
    fn check_function_decl(&mut self, decl: &FunctionDecl) -> TypeResult<()> {
        // Extract parameter types
        let param_types: Vec<TensorTypeInfo> = decl
            .params
            .iter()
            .filter_map(|p| {
                if let EntityType::Tensor(tensor_type) = &p.entity_type {
                    Some(TensorTypeInfo::from_tensor_type(tensor_type))
                } else {
                    None
                }
            })
            .collect();

        // Extract return type
        let return_type = match &decl.return_type {
            ReturnType::Scalar(scalar_type) => {
                // Treat scalar as 0-dimensional tensor
                let base_type = match scalar_type {
                    ScalarType::Int | ScalarType::Bool => BaseType::Int32,
                    ScalarType::Float => BaseType::Float32,
                    ScalarType::String => BaseType::Float32, // String treated as generic
                };
                Some(TensorTypeInfo::new(base_type, vec![]))
            }
            ReturnType::Tensor(tensor_type) => {
                Some(TensorTypeInfo::from_tensor_type(tensor_type))
            }
            ReturnType::Void => None,
        };

        // Add function to environment
        self.env.add_function(
            decl.name.as_str().to_string(),
            param_types,
            return_type,
        )?;

        // TODO: Type check function body statements

        Ok(())
    }

    /// Type check a struct declaration
    fn check_struct_decl(&mut self, decl: &StructDecl) -> TypeResult<()> {
        // Add type parameters to scope
        for type_param in &decl.type_params {
            self.env.add_type_param(type_param.name.as_str().to_string());
        }

        // Check that all field types are valid
        for field in &decl.fields {
            self.check_field_type(&field.field_type)?;
        }

        // Add struct to environment
        self.env.add_struct(decl.clone())?;

        // Clear type parameters after struct definition
        self.env.clear_type_params();

        Ok(())
    }

    /// Type check a field type
    fn check_field_type(&self, field_type: &FieldType) -> TypeResult<()> {
        match field_type {
            FieldType::Scalar(_) => Ok(()),
            FieldType::Tensor(tensor_type) => {
                // Validate tensor dimensions
                for dim in &tensor_type.dimensions {
                    if let Dimension::Variable(var) = dim {
                        if !self.env.has_dimension_var(var.as_str()) {
                            // Dimension variables in fields are OK (they'll be checked later)
                        }
                    }
                }
                Ok(())
            }
            FieldType::Struct(struct_type) => {
                // Check that the struct exists
                self.env.get_struct(struct_type.name.as_str())?;
                Ok(())
            }
            FieldType::TypeParam(type_param) => {
                // Check that the type parameter is in scope
                if !self.env.has_type_param(type_param.as_str()) {
                    return Err(TypeError::UndefinedTypeParameter(
                        type_param.as_str().to_string(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Type check an impl block
    fn check_impl_block(&mut self, impl_block: &ImplBlock) -> TypeResult<()> {
        // Get the struct declaration
        let struct_name = impl_block.struct_type.name.as_str();
        let _struct_decl = self.env.get_struct(struct_name)?;

        // Add type parameters from impl block to scope
        for type_param in &impl_block.type_params {
            self.env.add_type_param(type_param.name.as_str().to_string());
        }

        // If this is a Drop trait implementation, perform special checks
        if let Some(ref trait_name) = impl_block.trait_name {
            if trait_name.as_str() == "Drop" {
                self.check_drop_impl(impl_block)?;
            }
        }

        // Check each method
        for method in &impl_block.methods {
            // Add method to environment
            self.env.add_method(
                struct_name.to_string(),
                impl_block.type_params.clone(),
                method.clone(),
            )?;

            // TODO: Type check method body
        }

        // Clear type parameters after impl block
        self.env.clear_type_params();

        Ok(())
    }

    /// Type check a Drop trait implementation
    fn check_drop_impl(&self, impl_block: &ImplBlock) -> TypeResult<()> {
        // Drop trait must have exactly one method named "drop"
        if impl_block.methods.len() != 1 {
            return Err(TypeError::TypeMismatch {
                expected: "exactly one method named 'drop'".to_string(),
                found: format!("{} methods", impl_block.methods.len()),
            });
        }

        let method = &impl_block.methods[0];

        // Method must be named "drop"
        if method.name.as_str() != "drop" {
            return Err(TypeError::TypeMismatch {
                expected: "method named 'drop'".to_string(),
                found: format!("method named '{}'", method.name.as_str()),
            });
        }

        // drop method must have exactly one parameter: self
        if method.params.len() != 1 {
            return Err(TypeError::ArgumentCountMismatch {
                expected: 1,
                found: method.params.len(),
            });
        }

        // The parameter must be self
        if !matches!(method.params[0], MethodParam::SelfParam) {
            return Err(TypeError::TypeMismatch {
                expected: "self parameter".to_string(),
                found: "regular parameter".to_string(),
            });
        }

        // Return type must be void
        if !matches!(method.return_type, ReturnType::Void) {
            return Err(TypeError::TypeMismatch {
                expected: "void return type".to_string(),
                found: format!("{:?}", method.return_type),
            });
        }

        Ok(())
    }

    /// Type check main block
    fn check_main_block(&mut self, main_block: &MainBlock) -> TypeResult<()> {
        for stmt in &main_block.statements {
            self.check_statement(stmt)?;
        }
        Ok(())
    }

    /// Type check a statement
    fn check_statement(&mut self, stmt: &Statement) -> TypeResult<()> {
        match stmt {
            Statement::Assignment { target, value } => {
                let value_type = self.infer_expr_type(value)?;
                self.env
                    .add_variable(target.as_str().to_string(), value_type)?;
                Ok(())
            }
            Statement::Equation(eq) => {
                let left_type = self.infer_expr_type(&eq.left)?;
                let right_type = self.infer_expr_type(&eq.right)?;

                if !left_type.is_compatible_with(&right_type) {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{:?}", left_type),
                        found: format!("{:?}", right_type),
                    });
                }

                Ok(())
            }
            _ => Ok(()), // Other statements deferred
        }
    }

    /// Infer the type of an expression
    fn infer_expr_type(&self, expr: &TensorExpr) -> TypeResult<TensorTypeInfo> {
        match expr {
            TensorExpr::Variable(id) => {
                let type_info = self.env.get_variable(id.as_str())?;
                Ok(type_info.clone())
            }

            TensorExpr::Literal(lit) => self.infer_literal_type(lit),

            TensorExpr::BinaryOp { op, left, right } => {
                let left_type = self.infer_expr_type(left)?;
                let right_type = self.infer_expr_type(right)?;
                self.infer_binary_op_type(op, &left_type, &right_type)
            }

            TensorExpr::UnaryOp { op, operand } => {
                let operand_type = self.infer_expr_type(operand)?;
                self.infer_unary_op_type(op, &operand_type)
            }

            TensorExpr::FunctionCall { type_namespace, name, args, .. } => {
                let (param_types, return_type) = self.env.get_function(name.as_str())?;

                // Check argument count
                if args.len() != param_types.len() {
                    return Err(TypeError::ArgumentCountMismatch {
                        expected: param_types.len(),
                        found: args.len(),
                    });
                }

                // Check argument types
                for (arg, param_type) in args.iter().zip(param_types.iter()) {
                    let arg_type = self.infer_expr_type(arg)?;
                    if !arg_type.is_compatible_with(param_type) {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{:?}", param_type),
                            found: format!("{:?}", arg_type),
                        });
                    }
                }

                return_type
                    .clone()
                    .ok_or(TypeError::CannotInferType)
            }

            TensorExpr::EinSum { .. } => {
                // Simplified: einsum type inference deferred
                Ok(TensorTypeInfo::new(
                    BaseType::Float32,
                    vec![Dimension::Dynamic],
                ))
            }

            TensorExpr::TensorIndex { tensor, .. } => {
                // Tensor indexing returns a scalar value
                // Recursively infer the type of the tensor expression
                let tensor_type = self.infer_expr_type(tensor)?;
                Ok(TensorTypeInfo::new(
                    tensor_type.base_type,
                    vec![], // Scalar result
                ))
            }

            TensorExpr::EmbeddingLookup { .. } => {
                // Simplified: embedding lookup type inference deferred
                Ok(TensorTypeInfo::new(
                    BaseType::Float32,
                    vec![Dimension::Dynamic],
                ))
            }

            TensorExpr::PythonCall { .. } => {
                // TODO: Python function type inference
                // Phase 2: Infer return type from Python function signature
                Ok(TensorTypeInfo::new(
                    BaseType::Float32,
                    vec![Dimension::Dynamic],
                ))
            }

            TensorExpr::PropertyAccess { object, property: _ } => {
                // For now, infer property access returns a tensor
                // TODO: More precise type inference based on property name
                let _obj_type = self.infer_expr_type(object)?;
                Ok(TensorTypeInfo::new(
                    BaseType::Float32,
                    vec![Dimension::Dynamic],
                ))
            }

            TensorExpr::MethodCall { object, method, args: _ } => {
                // For now, infer method call return type based on method name
                let _obj_type = self.infer_expr_type(object)?;
                match method.as_str() {
                    "shape" => {
                        // shape() returns an integer array
                        Ok(TensorTypeInfo::new(
                            BaseType::Int32,
                            vec![Dimension::Dynamic],
                        ))
                    }
                    _ => {
                        // Default: assume returns a tensor
                        Ok(TensorTypeInfo::new(
                            BaseType::Float32,
                            vec![Dimension::Dynamic],
                        ))
                    }
                }
            }

            TensorExpr::StructLiteral { struct_type, fields } => {
                // Check that struct exists
                let struct_decl = self.env.get_struct(struct_type.name.as_str())?;

                // Check that all required fields are provided
                for struct_field in &struct_decl.fields {
                    if !fields.iter().any(|f| f.name.as_str() == struct_field.name.as_str()) {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("field {}", struct_field.name.as_str()),
                            found: "missing field".to_string(),
                        });
                    }
                }

                // For now, return a placeholder type (struct types aren't tensors)
                // In a full implementation, we'd need a different type system
                Ok(TensorTypeInfo::new(
                    BaseType::Int32,
                    vec![], // Struct as scalar placeholder
                ))
            }

            TensorExpr::AssociatedCall { struct_type, function, args: _ } => {
                // Check that struct exists
                let _struct_decl = self.env.get_struct(struct_type.name.as_str())?;

                // Check that method exists
                let (_type_params, method_decl) = self.env.get_method(
                    struct_type.name.as_str(),
                    function.as_str(),
                )?;

                // Return the method's return type
                match &method_decl.return_type {
                    ReturnType::Tensor(tensor_type) => {
                        Ok(TensorTypeInfo::from_tensor_type(tensor_type))
                    }
                    ReturnType::Scalar(scalar_type) => {
                        let base_type = match scalar_type {
                            ScalarType::Int | ScalarType::Bool => BaseType::Int32,
                            ScalarType::Float => BaseType::Float32,
                            ScalarType::String => BaseType::Int32,
                        };
                        Ok(TensorTypeInfo::new(base_type, vec![]))
                    }
                    ReturnType::Struct(_) => {
                        // Struct type as placeholder
                        Ok(TensorTypeInfo::new(BaseType::Int32, vec![]))
                    }
                    ReturnType::Void => Err(TypeError::CannotInferType),
                }
            }
        }
    }

    /// Infer type of a literal
    fn infer_literal_type(&self, lit: &TensorLiteral) -> TypeResult<TensorTypeInfo> {
        match lit {
            TensorLiteral::Scalar(scalar) => {
                let base_type = match scalar {
                    ScalarLiteral::Float(_) => BaseType::Float32,
                    ScalarLiteral::Integer(_) => BaseType::Int32,
                    ScalarLiteral::Boolean(_) => BaseType::Bool,
                    ScalarLiteral::Complex { .. } => BaseType::Complex64,
                    ScalarLiteral::String(_) => BaseType::Int32, // Placeholder for string type
                };
                Ok(TensorTypeInfo::new(base_type, vec![]))
            }
            TensorLiteral::Array(elements) => {
                if elements.is_empty() {
                    return Err(TypeError::CannotInferType);
                }

                // Infer element type from first element
                let elem_type = match &elements[0] {
                    ArrayElement::Literal(lit) => self.infer_literal_type(lit)?,
                    ArrayElement::Expression(expr) => self.infer_expr_type(expr)?,
                };

                // Construct array type with one more dimension
                let mut dimensions = vec![Dimension::Fixed(elements.len())];
                dimensions.extend(elem_type.dimensions);

                Ok(TensorTypeInfo::new(elem_type.base_type, dimensions))
            }
        }
    }

    /// Infer type of binary operation
    fn infer_binary_op_type(
        &self,
        op: &BinaryOp,
        left: &TensorTypeInfo,
        right: &TensorTypeInfo,
    ) -> TypeResult<TensorTypeInfo> {
        // Check base type compatibility
        if left.base_type != right.base_type {
            return Err(TypeError::BaseTypeMismatch {
                left: left.base_type.clone(),
                right: right.base_type.clone(),
            });
        }

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                // Element-wise operations: same shape required
                if !left.dimensions_match(&right.dimensions) {
                    return Err(TypeError::DimensionMismatch {
                        left: left.dimensions.clone(),
                        right: right.dimensions.clone(),
                    });
                }
                Ok(left.clone())
            }

            BinaryOp::MatMul => {
                // Matrix multiplication: [M, K] @ [K, N] -> [M, N]
                if left.rank() < 2 || right.rank() < 2 {
                    return Err(TypeError::InvalidOperation {
                        op: "@".to_string(),
                        left: format!("{:?}", left),
                        right: format!("{:?}", right),
                    });
                }

                // Simplified dimension checking
                let mut result_dims = left.dimensions[..left.dimensions.len() - 1].to_vec();
                result_dims.push(right.dimensions[right.dimensions.len() - 1].clone());

                Ok(TensorTypeInfo::new(left.base_type.clone(), result_dims))
            }

            BinaryOp::Power => {
                // Power operation: same shape
                if !left.dimensions_match(&right.dimensions) {
                    return Err(TypeError::DimensionMismatch {
                        left: left.dimensions.clone(),
                        right: right.dimensions.clone(),
                    });
                }
                Ok(left.clone())
            }

            BinaryOp::TensorProd | BinaryOp::Hadamard => {
                // Simplified: return left type
                Ok(left.clone())
            }

            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                // Comparison operators return boolean scalar
                Ok(TensorTypeInfo::new(
                    BaseType::Bool,
                    vec![]
                ))
            }

            BinaryOp::And | BinaryOp::Or => {
                // Logical operators work on booleans
                if left.base_type != BaseType::Bool || right.base_type != BaseType::Bool {
                    return Err(TypeError::InvalidOperation {
                        op: format!("{:?}", op),
                        left: format!("{:?}", left),
                        right: format!("{:?}", right),
                    });
                }
                Ok(left.clone())
            }
        }
    }

    /// Infer type of unary operation
    fn infer_unary_op_type(
        &self,
        op: &UnaryOp,
        operand: &TensorTypeInfo,
    ) -> TypeResult<TensorTypeInfo> {
        match op {
            UnaryOp::Neg | UnaryOp::Not => Ok(operand.clone()),

            UnaryOp::Transpose => {
                // Transpose: reverse dimensions
                if operand.rank() < 2 {
                    return Err(TypeError::InvalidOperation {
                        op: "transpose".to_string(),
                        left: format!("{:?}", operand),
                        right: "none".to_string(),
                    });
                }

                let mut dims = operand.dimensions.clone();
                dims.reverse();
                Ok(TensorTypeInfo::new(operand.base_type.clone(), dims))
            }

            UnaryOp::Inverse | UnaryOp::Determinant => {
                // Inverse/determinant: square matrix required
                if operand.rank() != 2 {
                    return Err(TypeError::InvalidOperation {
                        op: format!("{:?}", op),
                        left: format!("{:?}", operand),
                        right: "none".to_string(),
                    });
                }
                Ok(operand.clone())
            }
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
