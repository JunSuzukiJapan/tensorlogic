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
}

/// Main execution block
#[derive(Debug, Clone, PartialEq)]
pub struct MainBlock {
    pub statements: Vec<Statement>,
}

/// Top-level declarations
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Tensor(TensorDecl),
    Relation(RelationDecl),
    Rule(RuleDecl),
    Embedding(EmbeddingDecl),
    Function(FunctionDecl),
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
#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    pub base_type: BaseType,
    pub dimensions: Vec<Dimension>,
    pub learnable: LearnableStatus,
}

/// Base scalar types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dimension {
    Fixed(usize),
    Variable(Identifier),
    Dynamic, // ?
}

/// Learnable parameter status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Entity type for parameters
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    Entity,
    Concept,
    Tensor(TensorType),
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

/// Embedding declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDecl {
    pub name: Identifier,
    pub entities: EntitySet,
    pub dimension: usize,
    pub init_method: InitMethod,
}

/// Entity set specification
#[derive(Debug, Clone, PartialEq)]
pub enum EntitySet {
    Explicit(Vec<Identifier>),
    Auto,
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
    Tensor(TensorType),
    Void,
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
        name: Identifier,
        args: Vec<TensorExpr>,
    },
    /// Tensor indexing: tensor[i, j, ...]
    TensorIndex {
        tensor: Identifier,
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

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,          // +
    Sub,          // -
    Mul,          // *
    Div,          // /
    MatMul,       // @
    Power,        // **
    TensorProd,   // ⊗
    Hadamard,     // ⊙
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
    Array(Vec<TensorLiteral>),
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
    Exact,      // =
    Approx,     // ~
    Assign,     // :=
}

// ============================================================================
// Statements
// ============================================================================

/// Statements in imperative code
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Tensor declaration: tensor name: type = expr?
    TensorDecl(TensorDecl),
    /// Assignment: x := expr
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
    },
    /// Query: query pred(x, y) where constraints
    Query {
        atom: Atom,
        constraints: Vec<Constraint>,
    },
    /// Inference call: infer method query
    Inference {
        method: InferenceMethod,
        query: Box<Statement>, // Must be Query
    },
    /// Learning call
    Learning(LearningSpec),
    /// Control flow
    ControlFlow(ControlFlow),
    /// Python import: python import module [as alias]
    PythonImport {
        module: String,
        alias: Option<String>,
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
            BinaryOp::MatMul => write!(f, "@"),
            BinaryOp::Power => write!(f, "**"),
            BinaryOp::TensorProd => write!(f, "⊗"),
            BinaryOp::Hadamard => write!(f, "⊙"),
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
