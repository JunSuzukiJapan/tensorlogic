//! Abstract Syntax Tree (AST) for TensorLogic Language
//!
//! This module defines the AST nodes for the TensorLogic programming language,
//! which unifies tensor algebra with logic programming.
//!
//! # Module Structure
//!
//! - `mod.rs` - Core AST node definitions
//! - `span.rs` - Source location tracking
//! - `visitor.rs` - Visitor pattern for AST traversal

pub mod span;
pub mod visitor;

pub use span::{Position, Span, Spanned};
pub use visitor::{Visitor, VisitorMut};

use std::fmt;

/// A complete TensorLogic program
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Top-level declarations
    pub declarations: Vec<Declaration>,
    /// Optional main block
    pub main_block: Option<MainBlock>,
    /// Test blocks
    pub test_blocks: Vec<TestBlock>,
    /// Benchmark blocks
    pub bench_blocks: Vec<BenchBlock>,
}

/// Main execution block
#[derive(Debug, Clone, PartialEq)]
pub struct MainBlock {
    pub statements: Vec<Statement>,
}

/// Test block
#[derive(Debug, Clone, PartialEq)]
pub struct TestBlock {
    pub name: Identifier,
    pub statements: Vec<Statement>,
}

/// Benchmark block
#[derive(Debug, Clone, PartialEq)]
pub struct BenchBlock {
    pub name: Identifier,
    pub statements: Vec<Statement>,
}

/// Top-level declarations
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Import(ImportDecl),
    Entity(EntityDecl),
    Tensor(TensorDecl),
    Relation(RelationDecl),
    Rule(RuleDecl),
    Embedding(EmbeddingDecl),
    RelationEmbedding(RelationEmbeddingDecl),
    Function(FunctionDecl),
    Struct(StructDecl),
    Impl(ImplBlock),
}

// ============================================================================
// Import Declarations
// ============================================================================

/// Import declaration: import "path/to/module.tl"
#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub path: String,
}

// ============================================================================
// Entity Declarations
// ============================================================================

/// Entity declaration: entity T or entity T = {e1, e2, ...}
#[derive(Debug, Clone, PartialEq)]
pub enum EntityDecl {
    /// Explicit enumeration: entity Person = {alice, bob, charlie}
    Explicit {
        name: Identifier,
        entities: Vec<Identifier>,
    },
    /// From data: entity Person
    FromData {
        name: Identifier,
    },
}

// ============================================================================
// Tensor Declarations
// ============================================================================

/// Tensor declaration: tensor name: type = expr?
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDecl {
    pub name: Identifier,
    pub tensor_type: TensorType,
    pub init_expr: Option<TensorExpr>,
}

/// Tensor type specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub base_type: BaseType,
    pub dimensions: Vec<Dimension>,
    pub learnable: LearnableStatus,
}

/// Base scalar types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseType {
    Float32,  // Used for float16 (16-bit float)
    Float64,
    Int16,    // 16-bit integer
    Int32,
    Int64,
    Bool,
    Complex64,  // Used for complex16 (2x 16-bit floats)
}

/// Dimension specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Fixed(usize),
    Variable(Identifier),
    Dynamic, // ?
}

/// Learnable parameter status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LearnableStatus {
    Learnable,
    Frozen,
    Default, // Not specified
}

// ============================================================================
// Relation Declarations
// ============================================================================

/// Relation declaration
#[derive(Debug, Clone, PartialEq)]
pub struct RelationDecl {
    pub name: Identifier,
    pub params: Vec<Param>,
    pub embedding_spec: Option<TensorType>,
}

/// Parameter in relation or function
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Identifier,
    pub entity_type: EntityType,
}

/// Scalar types for function parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Int,
    Float,
    Bool,
    String,
}

/// Entity type for parameters
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    /// Generic entity type (backward compatibility)
    Entity,
    /// Named entity type (e.g., Person, City)
    NamedEntity(Identifier),
    /// Generic concept type
    Concept,
    /// Scalar type (int, float, bool, string)
    Scalar(ScalarType),
    /// Tensor type with shape and base type
    Tensor(TensorType),
    /// Struct type
    Struct(StructType),
}

// ============================================================================
// Rule Declarations
// ============================================================================

/// Rule declaration: head <- body
#[derive(Debug, Clone, PartialEq)]
pub struct RuleDecl {
    pub head: RuleHead,
    pub body: Vec<BodyTerm>,
}

/// Rule head (consequent)
#[derive(Debug, Clone, PartialEq)]
pub enum RuleHead {
    Atom(Atom),
    Equation(TensorEquation),
}

/// Body term (antecedent)
#[derive(Debug, Clone, PartialEq)]
pub enum BodyTerm {
    Atom(Atom),
    Equation(TensorEquation),
    Constraint(Constraint),
}

/// Logical atom: pred(term1, term2, ...)
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub predicate: Identifier,
    pub terms: Vec<Term>,
}

/// Term in atom
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Variable(Identifier),
    Constant(Constant),
    Tensor(TensorExpr),
}

// ============================================================================
// Embedding Declarations
// ============================================================================

/// Entity embedding declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDecl {
    pub name: Identifier,
    pub entities: EntitySet,
    pub dimension: usize,
    pub init_method: InitMethod,
}

/// Relation embedding declaration
#[derive(Debug, Clone, PartialEq)]
pub struct RelationEmbeddingDecl {
    pub name: Identifier,
    pub relations: RelationSet,
    pub dimension: usize,
    pub init_method: InitMethod,
}

/// Entity set specification
#[derive(Debug, Clone, PartialEq)]
pub enum EntitySet {
    Explicit(Vec<Identifier>),
    Auto,
    /// From a specific entity type
    Type(Identifier),
}

/// Relation set specification
#[derive(Debug, Clone, PartialEq)]
pub enum RelationSet {
    /// Explicit relation list: {lives_in, owns, friend_of}
    Explicit(Vec<Identifier>),
    /// All registered relations
    All,
}

/// Initialization method for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMethod {
    Random,
    Xavier,
    He,
    Zeros,
    Ones,
}

// ============================================================================
// Function Declarations
// ============================================================================

/// Function declaration
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: Identifier,
    pub params: Vec<Param>,
    pub return_type: ReturnType,
    pub body: Vec<Statement>,
}

/// Return type
#[derive(Debug, Clone, PartialEq)]
pub enum ReturnType {
    Scalar(ScalarType),
    Tensor(TensorType),
    Struct(StructType),
    Void,
}

// ============================================================================
// Struct Declarations
// ============================================================================

/// Type parameter (for generics)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub name: Identifier,
}

/// Struct declaration
#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: Identifier,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<StructField>,
}

/// Struct field
#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: Identifier,
    pub field_type: FieldType,
}

/// Field type (can be scalar, tensor, or struct)
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Scalar(ScalarType),
    Tensor(TensorType),
    Struct(StructType),
    TypeParam(Identifier),
}

/// Struct type with optional type arguments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    pub name: Identifier,
    pub type_args: Vec<TypeArg>,
}

/// Type argument for generic structs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeArg {
    Tensor(TensorType),
    Scalar(ScalarType),
    Struct(Box<StructType>),
}

// ============================================================================
// Impl Blocks
// ============================================================================

/// Implementation block
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub type_params: Vec<TypeParam>,
    /// Trait name for trait implementations (e.g., "Drop")
    /// None for regular impl blocks, Some(name) for "impl Trait for Struct"
    pub trait_name: Option<Identifier>,
    pub struct_type: StructType,
    pub methods: Vec<MethodDecl>,
}

/// Method declaration
#[derive(Debug, Clone, PartialEq)]
pub struct MethodDecl {
    pub name: Identifier,
    pub params: Vec<MethodParam>,
    pub return_type: ReturnType,
    pub body: Vec<Statement>,
}

/// Method parameter (includes self)
#[derive(Debug, Clone, PartialEq)]
pub enum MethodParam {
    SelfParam,
    Regular(Param),
}

// ============================================================================
// Tensor Expressions
// ============================================================================

/// Tensor expression
#[derive(Debug, Clone, PartialEq)]
pub enum TensorExpr {
    /// Variable reference
    Variable(Identifier),
    /// Tensor literal
    Literal(TensorLiteral),
    /// Binary operation
    BinaryOp {
        op: BinaryOp,
        left: Box<TensorExpr>,
        right: Box<TensorExpr>,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        operand: Box<TensorExpr>,
    },
    /// Einstein summation
    EinSum {
        spec: String,
        tensors: Vec<TensorExpr>,
    },
    /// Function call
    FunctionCall {
        type_namespace: Option<String>,
        name: Identifier,
        args: Vec<TensorExpr>,
        /// Resolved function reference (populated during semantic analysis)
        /// None means not yet resolved (fallback to runtime lookup)
        resolved: Option<ResolvedFunction>,
    },
    /// Type-qualified function call: Type::method(args)
    /// Examples: KVCache::new_f32(22), f32::zeros([3, 4])
    TypeFunctionCall {
        type_namespace: String,
        name: Identifier,
        args: Vec<TensorExpr>,
    },
    /// Tensor indexing: tensor[i, j, ...] or expression[i]
    TensorIndex {
        tensor: Box<TensorExpr>,
        indices: Vec<IndexExpr>,
    },
    /// Embedding lookup: embed[entity]
    EmbeddingLookup {
        embedding: Identifier,
        entity: EntityRef,
    },
    /// Python function call: python.call("function", args)
    PythonCall {
        function: String,
        args: Vec<TensorExpr>,
    },
    /// Property access: object.property
    PropertyAccess {
        object: Box<TensorExpr>,
        property: Identifier,
    },
    /// Method call: object.method(args)
    MethodCall {
        object: Box<TensorExpr>,
        method: Identifier,
        args: Vec<TensorExpr>,
    },
    /// Struct literal: StructName { field1: expr1, field2: expr2, ... }
    StructLiteral {
        struct_type: StructType,
        fields: Vec<FieldInit>,
    },
    /// Associated function call: Type::function(args)
    AssociatedCall {
        struct_type: StructType,
        function: Identifier,
        args: Vec<TensorExpr>,
    },
}

/// Index expression for tensor indexing
#[derive(Debug, Clone, PartialEq)]
pub enum IndexExpr {
    /// Integer index
    Int(i64),
    /// Variable index
    Var(Identifier),
    /// Slice (colon)
    Slice,
}

/// Field initialization in struct literal
#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    pub name: Identifier,
    pub value: TensorExpr,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,          // +
    Sub,          // -
    Mul,          // *
    Div,          // /
    Mod,          // %
    MatMul,       // @
    Power,        // **
    TensorProd,   // ⊗
    Hadamard,     // ⊙
    // Comparison operators
    Eq,           // ==
    Ne,           // !=
    Lt,           // <
    Le,           // <=
    Gt,           // >
    Ge,           // >=
    // Logical operators
    And,          // &&
    Or,           // ||
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,        // -
    Not,        // !
    Transpose,  // transpose
    Inverse,    // inv
    Determinant, // det
}

/// Tensor literal
#[derive(Debug, Clone, PartialEq)]
pub enum TensorLiteral {
    Scalar(ScalarLiteral),
    Array(Vec<ArrayElement>),
}

/// Array element - can be a literal or expression (supports variables in arrays)
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayElement {
    Literal(TensorLiteral),
    Expression(TensorExpr),
}

/// Scalar literal
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarLiteral {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Complex { real: f64, imag: f64 },
    String(String),
}

/// Entity reference in embedding lookup
#[derive(Debug, Clone, PartialEq)]
pub enum EntityRef {
    Variable(Identifier),
    Literal(String),
}

// ============================================================================
// Constraints
// ============================================================================

/// Constraints in logic programming
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Comparison constraint
    Comparison {
        op: CompOp,
        left: TensorExpr,
        right: TensorExpr,
    },
    /// Shape constraint
    Shape {
        tensor: TensorExpr,
        shape: Vec<Dimension>,
    },
    /// Rank constraint
    Rank {
        tensor: TensorExpr,
        rank: usize,
    },
    /// Norm constraint
    Norm {
        tensor: TensorExpr,
        op: CompOp,
        value: f64,
    },
    /// Logical negation
    Not(Box<Constraint>),
    /// Logical AND
    And(Box<Constraint>, Box<Constraint>),
    /// Logical OR
    Or(Box<Constraint>, Box<Constraint>),
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Eq,      // ==
    Ne,      // !=
    Lt,      // <
    Gt,      // >
    Le,      // <=
    Ge,      // >=
    Approx,  // ≈
}

// ============================================================================
// Tensor Equations
// ============================================================================

/// Tensor equation
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEquation {
    pub left: TensorExpr,
    pub right: TensorExpr,
    pub eq_type: EquationType,
}

/// Equation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquationType {
    Approx,     // ~
}

// ============================================================================
// Statements
// ============================================================================

/// Statements in imperative code
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Tensor declaration: tensor name: type = expr?
    TensorDecl(TensorDecl),
    /// Let statement: let x = expr (declare new variable)
    Let {
        target: Identifier,
        value: TensorExpr,
    },
    /// Assignment: x := expr (update existing variable)
    Assignment {
        target: Identifier,
        value: TensorExpr,
    },
    /// Tensor equation
    Equation(TensorEquation),
    /// Function call (e.g., print)
    FunctionCall {
        name: Identifier,
        args: Vec<TensorExpr>,
        /// Resolved function reference (populated during semantic analysis)
        resolved: Option<ResolvedFunction>,
        /// Source location of the function call
        span: Span,
    },
    /// Fact assertion: <- pred(a, b)
    FactAssertion {
        atom: Atom,
    },
    /// Query: ?- pred(x, y) where constraints
    Query {
        atom: Atom,
        constraints: Vec<Constraint>,
    },
    /// Inference call: infer method query
    Inference {
        method: InferenceMethod,
        query: Box<Statement>, // Must be Query
    },
    /// Inference block: infer { method query* }
    InferenceBlock {
        items: Vec<(InferenceMethod, Box<Statement>)>,
    },
    /// With block: with EntityType { statements }
    WithBlock {
        entity_type: Identifier,
        statements: Vec<Statement>,
    },
    /// Learning call
    Learning(LearningSpec),
    /// Control flow
    ControlFlow(ControlFlow),
    /// Block statement: { statements }
    Block {
        statements: Vec<Statement>,
    },
    /// Break statement
    Break,
    /// Return statement: return [expr]
    Return {
        value: Option<TensorExpr>,
    },
    /// Panic statement: panic("format", args...)
    Panic {
        format: String,
        args: Vec<TensorExpr>,
    },
    /// Python import: python import module [as alias]
    PythonImport {
        module: String,
        alias: Option<String>,
    },
    /// Expression statement (e.g., method calls like cache.set(...))
    Expr {
        expr: TensorExpr,
    },
}

/// Inference methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMethod {
    Forward,
    Backward,
    Gradient,
    Symbolic,
}

/// Learning specification
#[derive(Debug, Clone, PartialEq)]
pub struct LearningSpec {
    pub statements: Vec<Statement>,
    pub objective: TensorExpr,
    pub optimizer: OptimizerSpec,
    pub epochs: usize,
    pub scheduler: Option<SchedulerSpec>,
}

/// Optimizer specification
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerSpec {
    pub name: String,
    pub params: Vec<(String, f64)>, // e.g., [("lr", 0.001)]
}

/// Learning rate scheduler specification
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerSpec {
    pub name: String, // "step", "exponential", "cosine"
    pub params: Vec<(String, f64)>, // e.g., [("step_size", 10), ("gamma", 0.1)]
}

// ============================================================================
// Control Flow
// ============================================================================

/// Control flow statements
#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlow {
    If {
        condition: Condition,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
    },
    For {
        variable: Identifier,
        iterable: Iterable,
        body: Vec<Statement>,
    },
    While {
        condition: Condition,
        body: Vec<Statement>,
    },
    Loop {
        body: Vec<Statement>,
    },
}

/// Condition for control flow
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    Constraint(Constraint),
    Tensor(TensorExpr),
}

/// Iterable for for loops
#[derive(Debug, Clone, PartialEq)]
pub enum Iterable {
    Tensor(TensorExpr),
    EntitySet(EntitySet),
    Range(usize),
}

// ============================================================================
// Constants and Identifiers
// ============================================================================

/// Constant value
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

/// Identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier(pub String);

impl Identifier {
    pub fn new(s: impl Into<String>) -> Self {
        Identifier(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// Semantic Analysis Types
// ============================================================================

/// Resolved function reference after semantic analysis
/// This is populated during the semantic analysis pass to avoid runtime HashMap lookups
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedFunction {
    /// Builtin function with direct ID-based dispatch
    Builtin(BuiltinFunctionId),
    /// User-defined function with shared reference
    UserDefined(std::rc::Rc<FunctionDecl>),
}

/// Builtin function identifiers for direct dispatch without string comparison
/// Organized by category for maintainability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFunctionId {
    // Tensor creation functions
    TensorZeros,
    TensorOnes,
    TensorRand,
    TensorRandn,
    TensorEye,
    TensorArange,
    TensorLinspace,

    // Tensor shape operations
    TensorReshape,
    TensorFlatten,
    TensorTranspose,
    TensorPermute,
    TensorSqueeze,
    TensorUnsqueeze,
    TensorConcat,
    TensorStack,
    TensorSplit,
    TensorChunk,

    // Mathematical functions
    MathSin,
    MathCos,
    MathTan,
    MathAsin,
    MathAcos,
    MathAtan,
    MathSinh,
    MathCosh,
    MathTanh,
    MathExp,
    MathLog,
    MathLog10,
    MathSqrt,
    MathAbs,
    MathPow,
    MathFloor,
    MathCeil,
    MathRound,

    // Neural network operations
    NNLinear,
    NNRmsNorm,
    NNLayerNorm,
    NNBatchNorm,
    NNSoftmax,
    NNLogSoftmax,
    NNReLU,
    NNGeLU,
    NNSiLU,
    NNTanh,
    NNSigmoid,
    NNDropout,
    NNAttention,
    NNAttentionWithCache,
    NNRoPE,

    // Embedding operations
    NNEmbedding,
    NNEmbeddingLookup,

    // Activation functions
    ActReLU,
    ActLeakyReLU,
    ActELU,
    ActSELU,
    ActGeLU,
    ActSiLU,
    ActMish,
    ActSwish,

    // Loss functions
    LossMSE,
    LossCrossEntropy,
    LossBCE,
    LossKLDiv,

    // Sampling functions
    SampleTemperature,
    SampleTopK,
    SampleTopP,
    SampleGreedy,
    SampleArgmax,

    // Model I/O functions
    ModelLoad,
    ModelLoadF16,
    ModelLoadF32,
    ModelSave,
    ModelGetTensor,

    // Tokenizer functions
    TokenizerLoad,
    TokenizerTokenize,
    TokenizerDetokenize,
    TokenizerDetokenizeSingle,

    // Utility functions
    UtilShape,
    UtilRank,
    UtilSize,
    UtilPrint,
    UtilEnv,
    UtilIntToTokenIds,

    // Knowledge graph functions (for future)
    KGTransE,
    KGDistMult,
    KGComplEx,

    // Graph neural network functions (for future)
    GNNGCNLayer,
    GNNGATLayer,
}

// ============================================================================
// Helper implementations
// ============================================================================

impl TensorType {
    /// Create a simple float32 tensor type (internally float16)
    pub fn float32(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Float32,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }

    /// Create a simple float16 tensor type (alias for float32)
    pub fn float16(dims: Vec<usize>) -> Self {
        Self::float32(dims)
    }

    /// Create a learnable float32 tensor type (internally float16)
    pub fn learnable_float32(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Float32,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Learnable,
        }
    }

    /// Create a learnable float16 tensor type (alias for learnable_float32)
    pub fn learnable_float16(dims: Vec<usize>) -> Self {
        Self::learnable_float32(dims)
    }

    /// Create a simple int16 tensor type
    pub fn int16(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Int16,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }

    /// Create a simple int32 tensor type
    pub fn int32(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Int32,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }

    /// Create a simple int64 tensor type
    pub fn int64(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Int64,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }

    /// Create a simple bool tensor type
    pub fn bool(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Bool,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }

    /// Create a simple complex16 tensor type (uses Complex64 internally)
    pub fn complex16(dims: Vec<usize>) -> Self {
        TensorType {
            base_type: BaseType::Complex64,
            dimensions: dims.into_iter().map(Dimension::Fixed).collect(),
            learnable: LearnableStatus::Default,
        }
    }
}

impl TensorExpr {
    /// Create a variable reference
    pub fn var(name: impl Into<String>) -> Self {
        TensorExpr::Variable(Identifier::new(name))
    }

    /// Create a scalar literal
    pub fn scalar(value: f64) -> Self {
        TensorExpr::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(value)))
    }

    /// Create an integer literal
    pub fn int(value: i64) -> Self {
        TensorExpr::Literal(TensorLiteral::Scalar(ScalarLiteral::Integer(value)))
    }

    /// Create a binary operation
    pub fn binary(op: BinaryOp, left: TensorExpr, right: TensorExpr) -> Self {
        TensorExpr::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation
    pub fn unary(op: UnaryOp, operand: TensorExpr) -> Self {
        TensorExpr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }
}

impl Atom {
    /// Create a new atom
    pub fn new(predicate: impl Into<String>, terms: Vec<Term>) -> Self {
        Atom {
            predicate: Identifier::new(predicate),
            terms,
        }
    }
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaseType::Float32 => write!(f, "float16"),  // Display as float16
            BaseType::Float64 => write!(f, "float64"),
            BaseType::Int16 => write!(f, "int16"),
            BaseType::Int32 => write!(f, "int32"),
            BaseType::Int64 => write!(f, "int64"),
            BaseType::Bool => write!(f, "bool"),
            BaseType::Complex64 => write!(f, "complex16"),  // Display as complex16
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::MatMul => write!(f, "@"),
            BinaryOp::Power => write!(f, "**"),
            BinaryOp::TensorProd => write!(f, "⊗"),
            BinaryOp::Hadamard => write!(f, "⊙"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::Transpose => write!(f, "transpose"),
            UnaryOp::Inverse => write!(f, "inv"),
            UnaryOp::Determinant => write!(f, "det"),
        }
    }
}
#[cfg(test)]
mod tests;
