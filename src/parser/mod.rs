//! Parser for TensorLogic Language
//!
//! This module provides a Pest-based parser that converts TensorLogic source code
//! into the Abstract Syntax Tree (AST) defined in the `ast` module.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic::parser::TensorLogicParser;
//!
//! let source = "tensor w: float32[10, 20] learnable";
//! let program = TensorLogicParser::parse_program(source)?;
//! ```

use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "parser/grammar.pest"]
pub struct TensorLogicParser;

/// Parse errors
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Parse error: {0}")]
    PestError(String),

    #[error("Unexpected rule: expected {expected}, found {found}")]
    UnexpectedRule { expected: String, found: String },

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(err: pest::error::Error<Rule>) -> Self {
        ParseError::PestError(err.to_string())
    }
}

impl TensorLogicParser {
    /// Parse a complete TensorLogic program from source code
    pub fn parse_program(source: &str) -> Result<Program, ParseError> {
        let pairs = Self::parse(Rule::program, source)?;

        let mut declarations = Vec::new();
        let mut main_block = None;

        for pair in pairs {
            match pair.as_rule() {
                Rule::program => {
                    for inner in pair.into_inner() {
                        match inner.as_rule() {
                            Rule::declaration => {
                                declarations.push(Self::parse_declaration(inner)?);
                            }
                            Rule::main_block => {
                                main_block = Some(Self::parse_main_block(inner)?);
                            }
                            Rule::EOI => {}
                            _ => {
                                return Err(ParseError::UnexpectedRule {
                                    expected: "declaration or main_block".to_string(),
                                    found: format!("{:?}", inner.as_rule()),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Program {
            declarations,
            main_block,
        })
    }

    fn parse_main_block(pair: pest::iterators::Pair<Rule>) -> Result<MainBlock, ParseError> {
        let mut statements = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::statement {
                statements.push(Self::parse_statement(inner)?);
            }
        }

        Ok(MainBlock { statements })
    }

    fn parse_declaration(pair: pest::iterators::Pair<Rule>) -> Result<Declaration, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("declaration type".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_decl => Ok(Declaration::Tensor(Self::parse_tensor_decl(inner)?)),
            Rule::relation_decl => Ok(Declaration::Relation(Self::parse_relation_decl(inner)?)),
            Rule::rule_decl => Ok(Declaration::Rule(Self::parse_rule_decl(inner)?)),
            Rule::embedding_decl => Ok(Declaration::Embedding(Self::parse_embedding_decl(inner)?)),
            Rule::function_decl => Ok(Declaration::Function(Self::parse_function_decl(inner)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "declaration type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_decl(pair: pest::iterators::Pair<Rule>) -> Result<TensorDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("tensor name".to_string())
        })?)?;

        let tensor_type = Self::parse_tensor_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("tensor type".to_string())
        })?)?;

        let init_expr = if let Some(expr_pair) = inner.next() {
            Some(Self::parse_tensor_expr(expr_pair)?)
        } else {
            None
        };

        Ok(TensorDecl {
            name,
            tensor_type,
            init_expr,
        })
    }

    fn parse_tensor_type(pair: pest::iterators::Pair<Rule>) -> Result<TensorType, ParseError> {
        let mut inner = pair.into_inner();

        let base_type = Self::parse_base_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("base type".to_string())
        })?)?;

        let dimension_list = inner.next().ok_or_else(|| {
            ParseError::MissingField("dimension list".to_string())
        })?;
        let dimensions = Self::parse_dimension_list(dimension_list)?;

        let learnable = if let Some(learnable_pair) = inner.next() {
            Self::parse_learnable(learnable_pair)?
        } else {
            LearnableStatus::Default
        };

        Ok(TensorType {
            base_type,
            dimensions,
            learnable,
        })
    }

    fn parse_base_type(pair: pest::iterators::Pair<Rule>) -> Result<BaseType, ParseError> {
        match pair.as_str() {
            "float32" => Ok(BaseType::Float32),
            "float64" => Ok(BaseType::Float64),
            "int32" => Ok(BaseType::Int32),
            "int64" => Ok(BaseType::Int64),
            "bool" => Ok(BaseType::Bool),
            "complex64" => Ok(BaseType::Complex64),
            s => Err(ParseError::InvalidValue(format!("Unknown base type: {}", s))),
        }
    }

    fn parse_dimension_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Dimension>, ParseError> {
        pair.into_inner()
            .map(|dim_pair| Self::parse_dimension(dim_pair))
            .collect()
    }

    fn parse_dimension(pair: pest::iterators::Pair<Rule>) -> Result<Dimension, ParseError> {
        // Check if this is a "?" directly
        if pair.as_str() == "?" {
            return Ok(Dimension::Dynamic);
        }

        let inner_opt = pair.into_inner().next();

        if let Some(inner) = inner_opt {
            match inner.as_rule() {
                Rule::integer => {
                    let size = inner.as_str().parse::<usize>().map_err(|e| {
                        ParseError::InvalidValue(format!("Invalid dimension size: {}", e))
                    })?;
                    Ok(Dimension::Fixed(size))
                }
                Rule::identifier => Ok(Dimension::Variable(Self::parse_identifier(inner)?)),
                _ => Err(ParseError::InvalidValue(format!("Invalid dimension: {}", inner.as_str()))),
            }
        } else {
            // No inner value, treat as identifier or try to parse the whole string
            Err(ParseError::MissingField("dimension value".to_string()))
        }
    }

    fn parse_learnable(pair: pest::iterators::Pair<Rule>) -> Result<LearnableStatus, ParseError> {
        match pair.as_str() {
            "learnable" => Ok(LearnableStatus::Learnable),
            "frozen" => Ok(LearnableStatus::Frozen),
            s => Err(ParseError::InvalidValue(format!("Unknown learnable status: {}", s))),
        }
    }

    fn parse_relation_decl(pair: pest::iterators::Pair<Rule>) -> Result<RelationDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("relation name".to_string())
        })?)?;

        let mut params = Vec::new();
        let mut embedding_spec = None;

        for param_or_embed in inner {
            match param_or_embed.as_rule() {
                Rule::param_list => {
                    params = Self::parse_param_list(param_or_embed)?;
                }
                Rule::embedding_spec => {
                    embedding_spec = Some(Self::parse_embedding_spec(param_or_embed)?);
                }
                _ => {}
            }
        }

        Ok(RelationDecl {
            name,
            params,
            embedding_spec,
        })
    }

    fn parse_param_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Param>, ParseError> {
        pair.into_inner()
            .map(|param_pair| Self::parse_param(param_pair))
            .collect()
    }

    fn parse_param(pair: pest::iterators::Pair<Rule>) -> Result<Param, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("parameter name".to_string())
        })?)?;

        let entity_type = Self::parse_entity_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("entity type".to_string())
        })?)?;

        Ok(Param { name, entity_type })
    }

    fn parse_entity_type(pair: pest::iterators::Pair<Rule>) -> Result<EntityType, ParseError> {
        // Check the string directly first
        match pair.as_str() {
            "entity" => return Ok(EntityType::Entity),
            "concept" => return Ok(EntityType::Concept),
            _ => {}
        }

        // Try to get inner rule
        if let Some(inner) = pair.into_inner().next() {
            match inner.as_rule() {
                Rule::tensor_type => Ok(EntityType::Tensor(Self::parse_tensor_type(inner)?)),
                _ => Err(ParseError::InvalidValue(format!("Unknown entity type: {}", inner.as_str()))),
            }
        } else {
            Err(ParseError::MissingField("entity type value".to_string()))
        }
    }

    fn parse_embedding_spec(pair: pest::iterators::Pair<Rule>) -> Result<TensorType, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("embedding tensor type".to_string())
        })?;
        Self::parse_tensor_type(inner)
    }

    fn parse_rule_decl(pair: pest::iterators::Pair<Rule>) -> Result<RuleDecl, ParseError> {
        let mut inner = pair.into_inner();

        let head = Self::parse_rule_head(inner.next().ok_or_else(|| {
            ParseError::MissingField("rule head".to_string())
        })?)?;

        let body = Self::parse_rule_body(inner.next().ok_or_else(|| {
            ParseError::MissingField("rule body".to_string())
        })?)?;

        Ok(RuleDecl { head, body })
    }

    fn parse_rule_head(pair: pest::iterators::Pair<Rule>) -> Result<RuleHead, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("rule head content".to_string())
        })?;

        match inner.as_rule() {
            Rule::atom => Ok(RuleHead::Atom(Self::parse_atom(inner)?)),
            Rule::tensor_equation => Ok(RuleHead::Equation(Self::parse_tensor_equation(inner)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "atom or equation".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_rule_body(pair: pest::iterators::Pair<Rule>) -> Result<Vec<BodyTerm>, ParseError> {
        pair.into_inner()
            .map(|term_pair| Self::parse_body_term(term_pair))
            .collect()
    }

    fn parse_body_term(pair: pest::iterators::Pair<Rule>) -> Result<BodyTerm, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("body term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::atom => Ok(BodyTerm::Atom(Self::parse_atom(inner)?)),
            Rule::tensor_equation => Ok(BodyTerm::Equation(Self::parse_tensor_equation(inner)?)),
            Rule::constraint => Ok(BodyTerm::Constraint(Self::parse_constraint(inner)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "atom, equation, or constraint".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_atom(pair: pest::iterators::Pair<Rule>) -> Result<Atom, ParseError> {
        let mut inner = pair.into_inner();

        let predicate = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("atom predicate".to_string())
        })?)?;

        let terms = if let Some(term_list) = inner.next() {
            Self::parse_term_list(term_list)?
        } else {
            Vec::new()
        };

        Ok(Atom { predicate, terms })
    }

    fn parse_term_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Term>, ParseError> {
        pair.into_inner()
            .map(|term_pair| Self::parse_term(term_pair))
            .collect()
    }

    fn parse_term(pair: pest::iterators::Pair<Rule>) -> Result<Term, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::identifier => Ok(Term::Variable(Self::parse_identifier(inner)?)),
            Rule::constant => Ok(Term::Constant(Self::parse_constant(inner)?)),
            Rule::tensor_expr => Ok(Term::Tensor(Self::parse_tensor_expr(inner)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "identifier, constant, or tensor_expr".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_embedding_decl(pair: pest::iterators::Pair<Rule>) -> Result<EmbeddingDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("embedding name".to_string())
        })?)?;

        let mut entities = EntitySet::Auto;
        let mut dimension = 0;
        let mut init_method = InitMethod::Random;

        for field in inner {
            match field.as_rule() {
                Rule::entity_set => {
                    entities = Self::parse_entity_set(field)?;
                }
                Rule::integer => {
                    dimension = field.as_str().parse::<usize>().map_err(|e| {
                        ParseError::InvalidValue(format!("Invalid dimension: {}", e))
                    })?;
                }
                Rule::init_method => {
                    init_method = Self::parse_init_method(field)?;
                }
                _ => {}
            }
        }

        Ok(EmbeddingDecl {
            name,
            entities,
            dimension,
            init_method,
        })
    }

    fn parse_entity_set(pair: pest::iterators::Pair<Rule>) -> Result<EntitySet, ParseError> {
        // Check if it's "auto" directly
        if pair.as_str() == "auto" {
            return Ok(EntitySet::Auto);
        }

        // Try to find identifier_list
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier_list => {
                    let identifiers = inner
                        .into_inner()
                        .map(|id_pair| Self::parse_identifier(id_pair))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(EntitySet::Explicit(identifiers));
                }
                _ if inner.as_str() == "auto" => {
                    return Ok(EntitySet::Auto);
                }
                _ => {}
            }
        }

        Err(ParseError::MissingField("entity set content".to_string()))
    }

    fn parse_init_method(pair: pest::iterators::Pair<Rule>) -> Result<InitMethod, ParseError> {
        match pair.as_str() {
            "random" => Ok(InitMethod::Random),
            "xavier" => Ok(InitMethod::Xavier),
            "he" => Ok(InitMethod::He),
            "zeros" => Ok(InitMethod::Zeros),
            "ones" => Ok(InitMethod::Ones),
            s => Err(ParseError::InvalidValue(format!("Unknown init method: {}", s))),
        }
    }

    fn parse_function_decl(pair: pest::iterators::Pair<Rule>) -> Result<FunctionDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("function name".to_string())
        })?)?;

        let mut params = Vec::new();
        let mut return_type = ReturnType::Void;
        let mut body = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::param_list => {
                    params = Self::parse_param_list(item)?;
                }
                Rule::return_type => {
                    return_type = Self::parse_return_type(item)?;
                }
                Rule::statement => {
                    body.push(Self::parse_statement(item)?);
                }
                _ => {}
            }
        }

        Ok(FunctionDecl {
            name,
            params,
            return_type,
            body,
        })
    }

    fn parse_return_type(pair: pest::iterators::Pair<Rule>) -> Result<ReturnType, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("return type value".to_string())
        })?;

        match inner.as_rule() {
            _ if inner.as_str() == "void" => Ok(ReturnType::Void),
            Rule::tensor_type => Ok(ReturnType::Tensor(Self::parse_tensor_type(inner)?)),
            _ => Err(ParseError::InvalidValue(format!("Invalid return type: {}", inner.as_str()))),
        }
    }

    fn parse_tensor_expr(pair: pest::iterators::Pair<Rule>) -> Result<TensorExpr, ParseError> {
        // Expression parser with operator precedence handling
        let mut pairs = pair.into_inner();

        // Parse the first term
        let first_term = pairs.next().ok_or_else(|| {
            ParseError::MissingField("tensor expression content".to_string())
        })?;
        let mut expr = Self::parse_tensor_term(first_term)?;

        // Parse remaining (binary_op, tensor_term) pairs
        while let Some(op_pair) = pairs.next() {
            let op = Self::parse_binary_op(op_pair)?;
            let right_term = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right = Self::parse_tensor_term(right_term)?;

            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_binary_op(pair: pest::iterators::Pair<Rule>) -> Result<BinaryOp, ParseError> {
        match pair.as_str() {
            "+" => Ok(BinaryOp::Add),
            "-" => Ok(BinaryOp::Sub),
            "*" => Ok(BinaryOp::Mul),
            "/" => Ok(BinaryOp::Div),
            "@" => Ok(BinaryOp::MatMul),
            "**" => Ok(BinaryOp::Power),
            "⊗" => Ok(BinaryOp::TensorProd),
            "⊙" => Ok(BinaryOp::Hadamard),
            _ => Err(ParseError::InvalidValue(format!("unknown binary operator: {}", pair.as_str()))),
        }
    }

    fn parse_tensor_term(pair: pest::iterators::Pair<Rule>) -> Result<TensorExpr, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::identifier => Ok(TensorExpr::Variable(Self::parse_identifier(inner)?)),
            Rule::tensor_literal => {
                let lit = Self::parse_tensor_literal(inner)?;
                Ok(TensorExpr::Literal(lit))
            }
            Rule::einstein_sum => {
                let mut inner_pairs = inner.into_inner();
                let spec = Self::parse_string_literal(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("einsum spec".to_string())
                })?)?;

                let tensor_list = inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("einsum tensor list".to_string())
                })?;
                let tensors = Self::parse_tensor_list(tensor_list)?;

                Ok(TensorExpr::EinSum { spec, tensors })
            }
            Rule::function_call => {
                let mut inner_pairs = inner.into_inner();
                let name = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("function name".to_string())
                })?)?;

                let args = if let Some(tensor_list) = inner_pairs.next() {
                    Self::parse_tensor_list(tensor_list)?
                } else {
                    Vec::new()
                };

                Ok(TensorExpr::FunctionCall { name, args })
            }
            Rule::embedding_lookup => {
                let mut inner_pairs = inner.into_inner();
                let embedding = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("embedding name".to_string())
                })?)?;

                let entity = Self::parse_entity_ref(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("entity reference".to_string())
                })?)?;

                Ok(TensorExpr::EmbeddingLookup { embedding, entity })
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "tensor term".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<TensorExpr>, ParseError> {
        pair.into_inner()
            .map(|expr_pair| Self::parse_tensor_expr(expr_pair))
            .collect()
    }

    fn parse_tensor_literal(pair: pest::iterators::Pair<Rule>) -> Result<TensorLiteral, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor literal content".to_string())
        })?;

        match inner.as_rule() {
            Rule::scalar_literal => Ok(TensorLiteral::Scalar(Self::parse_scalar_literal(inner)?)),
            Rule::tensor_elements => {
                let elements = inner
                    .into_inner()
                    .map(|elem| Self::parse_tensor_element(elem))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TensorLiteral::Array(elements))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "scalar or array".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_element(pair: pest::iterators::Pair<Rule>) -> Result<TensorLiteral, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor element content".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_literal => Self::parse_tensor_literal(inner),
            Rule::number => {
                let value = Self::parse_number(inner)?;
                Ok(TensorLiteral::Scalar(ScalarLiteral::Float(value)))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "tensor literal or number".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_scalar_literal(pair: pest::iterators::Pair<Rule>) -> Result<ScalarLiteral, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("scalar value".to_string())
        })?;

        match inner.as_rule() {
            Rule::number => Ok(ScalarLiteral::Float(Self::parse_number(inner)?)),
            Rule::boolean => Ok(ScalarLiteral::Boolean(Self::parse_boolean(inner)?)),
            Rule::complex_number => {
                // Simplified complex parsing
                Ok(ScalarLiteral::Complex { real: 0.0, imag: 0.0 })
            }
            _ => Err(ParseError::InvalidValue(format!("Invalid scalar: {}", inner.as_str()))),
        }
    }

    fn parse_entity_ref(pair: pest::iterators::Pair<Rule>) -> Result<EntityRef, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("entity reference value".to_string())
        })?;

        match inner.as_rule() {
            Rule::identifier => Ok(EntityRef::Variable(Self::parse_identifier(inner)?)),
            Rule::string_literal => Ok(EntityRef::Literal(Self::parse_string_literal(inner)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "identifier or string".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_constraint(_pair: pest::iterators::Pair<Rule>) -> Result<Constraint, ParseError> {
        // Simplified constraint parsing
        Ok(Constraint::Comparison {
            op: CompOp::Eq,
            left: TensorExpr::scalar(0.0),
            right: TensorExpr::scalar(0.0),
        })
    }

    fn parse_tensor_equation(pair: pest::iterators::Pair<Rule>) -> Result<TensorEquation, ParseError> {
        let mut inner = pair.into_inner();

        let left = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation left side".to_string())
        })?)?;

        let eq_type = Self::parse_eq_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation type".to_string())
        })?)?;

        let right = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation right side".to_string())
        })?)?;

        Ok(TensorEquation { left, right, eq_type })
    }

    fn parse_eq_type(pair: pest::iterators::Pair<Rule>) -> Result<EquationType, ParseError> {
        match pair.as_str() {
            "=" => Ok(EquationType::Exact),
            "~" => Ok(EquationType::Approx),
            ":=" => Ok(EquationType::Assign),
            s => Err(ParseError::InvalidValue(format!("Unknown equation type: {}", s))),
        }
    }

    fn parse_statement(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("statement content".to_string())
        })?;

        match inner.as_rule() {
            Rule::assignment => {
                let mut inner_pairs = inner.into_inner();
                let target = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("assignment target".to_string())
                })?)?;
                let value = Self::parse_tensor_expr(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("assignment value".to_string())
                })?)?;
                Ok(Statement::Assignment { target, value })
            }
            Rule::tensor_equation => {
                Ok(Statement::Equation(Self::parse_tensor_equation(inner)?))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "statement type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    // Helper parsers
    fn parse_identifier(pair: pest::iterators::Pair<Rule>) -> Result<Identifier, ParseError> {
        Ok(Identifier::new(pair.as_str()))
    }

    fn parse_constant(pair: pest::iterators::Pair<Rule>) -> Result<Constant, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("constant value".to_string())
        })?;

        match inner.as_rule() {
            Rule::number => Ok(Constant::Float(Self::parse_number(inner)?)),
            Rule::string_literal => Ok(Constant::String(Self::parse_string_literal(inner)?)),
            Rule::boolean => Ok(Constant::Boolean(Self::parse_boolean(inner)?)),
            _ => Err(ParseError::InvalidValue(format!("Invalid constant: {}", inner.as_str()))),
        }
    }

    fn parse_number(pair: pest::iterators::Pair<Rule>) -> Result<f64, ParseError> {
        pair.as_str().parse::<f64>().map_err(|e| {
            ParseError::InvalidValue(format!("Invalid number: {}", e))
        })
    }

    fn parse_boolean(pair: pest::iterators::Pair<Rule>) -> Result<bool, ParseError> {
        match pair.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            s => Err(ParseError::InvalidValue(format!("Invalid boolean: {}", s))),
        }
    }

    fn parse_string_literal(pair: pest::iterators::Pair<Rule>) -> Result<String, ParseError> {
        let s = pair.as_str();
        // Remove surrounding quotes
        if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
            Ok(s[1..s.len() - 1].to_string())
        } else {
            Err(ParseError::InvalidValue(format!("Invalid string literal: {}", s)))
        }
    }
}

#[cfg(test)]
mod tests;
