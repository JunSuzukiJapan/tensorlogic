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

    fn parse_constraint(pair: pest::iterators::Pair<Rule>) -> Result<Constraint, ParseError> {
        // constraint = { constraint_term ~ (logical_op ~ constraint_term)* }
        let mut pairs = pair.into_inner();

        let first_term = pairs.next().ok_or_else(|| {
            ParseError::MissingField("constraint term".to_string())
        })?;
        let mut constraint = Self::parse_constraint_term(first_term)?;

        // Parse logical operators (and, or)
        while let Some(op_pair) = pairs.next() {
            let op = op_pair.as_str();
            let right_term = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right constraint term".to_string())
            })?;
            let right = Self::parse_constraint_term(right_term)?;

            constraint = match op {
                "and" => Constraint::And(Box::new(constraint), Box::new(right)),
                "or" => Constraint::Or(Box::new(constraint), Box::new(right)),
                _ => return Err(ParseError::InvalidValue(format!("unknown logical operator: {}", op))),
            };
        }

        Ok(constraint)
    }

    fn parse_constraint_term(pair: pest::iterators::Pair<Rule>) -> Result<Constraint, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("constraint term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::constraint_term => {
                // Negation: "not" ~ constraint_term
                let mut inner_pairs = inner.into_inner();
                let first = inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("negation content".to_string())
                })?;

                if first.as_str() == "not" {
                    let negated = inner_pairs.next().ok_or_else(|| {
                        ParseError::MissingField("negated constraint".to_string())
                    })?;
                    Ok(Constraint::Not(Box::new(Self::parse_constraint_term(negated)?)))
                } else {
                    Self::parse_constraint_term(first)
                }
            }
            Rule::constraint => {
                // Parenthesized constraint
                Self::parse_constraint(inner)
            }
            Rule::tensor_constraint => {
                Self::parse_tensor_constraint(inner)
            }
            Rule::comparison => {
                Self::parse_comparison(inner)
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "constraint term".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_constraint(pair: pest::iterators::Pair<Rule>) -> Result<Constraint, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor constraint content".to_string())
        })?;

        let constraint_str = inner.as_str();

        if constraint_str.starts_with("shape") {
            // shape(tensor) == shape_spec
            let mut parts = inner.into_inner();
            let tensor_expr = Self::parse_tensor_expr(parts.next().ok_or_else(|| {
                ParseError::MissingField("tensor in shape constraint".to_string())
            })?)?;

            let shape_spec = parts.next().ok_or_else(|| {
                ParseError::MissingField("shape spec".to_string())
            })?;
            let shape = Self::parse_dimension_list(shape_spec)?;

            Ok(Constraint::Shape { tensor: tensor_expr, shape })
        } else if constraint_str.starts_with("rank") {
            // rank(tensor) comp_op integer
            let mut parts = inner.into_inner();
            let tensor_expr = Self::parse_tensor_expr(parts.next().ok_or_else(|| {
                ParseError::MissingField("tensor in rank constraint".to_string())
            })?)?;

            let comp_op_pair = parts.next().ok_or_else(|| {
                ParseError::MissingField("comparison operator".to_string())
            })?;
            let op = Self::parse_comp_op(comp_op_pair)?;

            let rank_val = parts.next().ok_or_else(|| {
                ParseError::MissingField("rank value".to_string())
            })?;
            let rank = Self::parse_number(rank_val)? as usize;

            // For now, only support == for rank
            // Store the comparison in the AST for future extension
            if op != CompOp::Eq {
                return Err(ParseError::InvalidValue(
                    "rank constraint only supports == operator".to_string()
                ));
            }

            Ok(Constraint::Rank { tensor: tensor_expr, rank })
        } else if constraint_str.starts_with("norm") {
            // norm(tensor) comp_op number
            let mut parts = inner.into_inner();
            let tensor_expr = Self::parse_tensor_expr(parts.next().ok_or_else(|| {
                ParseError::MissingField("tensor in norm constraint".to_string())
            })?)?;

            let comp_op_pair = parts.next().ok_or_else(|| {
                ParseError::MissingField("comparison operator".to_string())
            })?;
            let op = Self::parse_comp_op(comp_op_pair)?;

            let value_pair = parts.next().ok_or_else(|| {
                ParseError::MissingField("norm value".to_string())
            })?;
            let value = Self::parse_number(value_pair)?;

            Ok(Constraint::Norm { tensor: tensor_expr, op, value })
        } else {
            Err(ParseError::InvalidValue(format!("unknown tensor constraint: {}", constraint_str)))
        }
    }

    fn parse_comparison(pair: pest::iterators::Pair<Rule>) -> Result<Constraint, ParseError> {
        let mut inner = pair.into_inner();

        let left = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("left side of comparison".to_string())
        })?)?;

        let op_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("comparison operator".to_string())
        })?;
        let op = Self::parse_comp_op(op_pair)?;

        let right = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("right side of comparison".to_string())
        })?)?;

        Ok(Constraint::Comparison { op, left, right })
    }

    fn parse_comp_op(pair: pest::iterators::Pair<Rule>) -> Result<CompOp, ParseError> {
        match pair.as_str() {
            "==" => Ok(CompOp::Eq),
            "!=" => Ok(CompOp::Ne),
            "<" => Ok(CompOp::Lt),
            ">" => Ok(CompOp::Gt),
            "<=" => Ok(CompOp::Le),
            ">=" => Ok(CompOp::Ge),
            "≈" => Ok(CompOp::Approx),
            _ => Err(ParseError::InvalidValue(format!("unknown comparison operator: {}", pair.as_str()))),
        }
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
            Rule::query => {
                Self::parse_query(inner)
            }
            Rule::inference_call => {
                Self::parse_inference_call(inner)
            }
            Rule::learning_call => {
                Self::parse_learning_call(inner)
            }
            Rule::control_flow => {
                Self::parse_control_flow(inner).map(Statement::ControlFlow)
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "statement type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_query(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let atom = Self::parse_atom(inner.next().ok_or_else(|| {
            ParseError::MissingField("query atom".to_string())
        })?)?;

        let constraints = if let Some(constraint_list) = inner.next() {
            Self::parse_constraint_list(constraint_list)?
        } else {
            Vec::new()
        };

        Ok(Statement::Query { atom, constraints })
    }

    fn parse_constraint_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Constraint>, ParseError> {
        pair.into_inner()
            .map(|constraint_pair| Self::parse_constraint(constraint_pair))
            .collect()
    }

    fn parse_inference_call(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let method = Self::parse_inference_method(inner.next().ok_or_else(|| {
            ParseError::MissingField("inference method".to_string())
        })?)?;

        let query = Self::parse_query(inner.next().ok_or_else(|| {
            ParseError::MissingField("query in inference call".to_string())
        })?)?;

        Ok(Statement::Inference {
            method,
            query: Box::new(query),
        })
    }

    fn parse_inference_method(pair: pest::iterators::Pair<Rule>) -> Result<InferenceMethod, ParseError> {
        match pair.as_str() {
            "forward" => Ok(InferenceMethod::Forward),
            "backward" => Ok(InferenceMethod::Backward),
            "gradient" => Ok(InferenceMethod::Gradient),
            "symbolic" => Ok(InferenceMethod::Symbolic),
            s => Err(ParseError::InvalidValue(format!("Unknown inference method: {}", s))),
        }
    }

    fn parse_learning_call(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("learning spec".to_string())
        })?;

        let learning_spec = Self::parse_learning_spec(inner)?;
        Ok(Statement::Learning(learning_spec))
    }

    fn parse_learning_spec(pair: pest::iterators::Pair<Rule>) -> Result<LearningSpec, ParseError> {
        let mut inner = pair.into_inner();

        // Parse objective: tensor_expr
        let objective_expr = inner.next().ok_or_else(|| {
            ParseError::MissingField("objective expression".to_string())
        })?;
        let objective = Self::parse_tensor_expr(objective_expr)?;

        // Parse optimizer: optimizer_spec
        let optimizer_spec = inner.next().ok_or_else(|| {
            ParseError::MissingField("optimizer spec".to_string())
        })?;
        let optimizer = Self::parse_optimizer_spec(optimizer_spec)?;

        // Parse epochs: integer
        let epochs_val = inner.next().ok_or_else(|| {
            ParseError::MissingField("epochs value".to_string())
        })?;
        let epochs = Self::parse_number(epochs_val)? as usize;

        Ok(LearningSpec {
            objective,
            optimizer,
            epochs,
        })
    }

    fn parse_optimizer_spec(pair: pest::iterators::Pair<Rule>) -> Result<OptimizerSpec, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("optimizer name".to_string())
        })?)?.as_str().to_string();

        let params = if let Some(params_pair) = inner.next() {
            Self::parse_optimizer_params(params_pair)?
        } else {
            Vec::new()
        };

        Ok(OptimizerSpec { name, params })
    }

    fn parse_optimizer_params(pair: pest::iterators::Pair<Rule>) -> Result<Vec<(String, f64)>, ParseError> {
        pair.into_inner()
            .map(|param_pair| {
                let mut inner = param_pair.into_inner();
                let name = Self::parse_identifier(inner.next().ok_or_else(|| {
                    ParseError::MissingField("parameter name".to_string())
                })?)?.as_str().to_string();
                let value = Self::parse_number(inner.next().ok_or_else(|| {
                    ParseError::MissingField("parameter value".to_string())
                })?)?;
                Ok((name, value))
            })
            .collect()
    }

    fn parse_control_flow(pair: pest::iterators::Pair<Rule>) -> Result<ControlFlow, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("control flow statement".to_string())
        })?;

        match inner.as_rule() {
            Rule::if_statement => Self::parse_if_statement(inner),
            Rule::for_statement => Self::parse_for_statement(inner),
            Rule::while_statement => Self::parse_while_statement(inner),
            _ => Err(ParseError::InvalidValue(format!("Invalid control flow: {}", inner.as_str()))),
        }
    }

    fn parse_if_statement(pair: pest::iterators::Pair<Rule>) -> Result<ControlFlow, ParseError> {
        // Get the input string before consuming pair
        let input_str = pair.as_str();
        let has_else = input_str.contains("} else {");
        
        let mut inner = pair.into_inner();
        
        // Parse condition (always first)
        let condition_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("if condition".to_string())
        })?;
        let condition = Self::parse_condition(condition_pair)?;
        
        // Collect all statement pairs
        let remaining: Vec<_> = inner
            .filter(|p| p.as_rule() == Rule::statement)
            .collect();
        
        let mut then_block = Vec::new();
        let mut else_block = None;
        
        if has_else {
            // Count statements in then block by counting := before "} else {"
            let before_else = input_str.split("} else {").next().unwrap();
            let then_stmt_count = before_else.matches(":=").count();
            
            // Parse then block
            for i in 0..then_stmt_count.min(remaining.len()) {
                then_block.push(Self::parse_statement(remaining[i].clone())?);
            }
            
            // Parse else block
            let mut else_stmts = Vec::new();
            for i in then_stmt_count..remaining.len() {
                else_stmts.push(Self::parse_statement(remaining[i].clone())?);
            }
            
            if !else_stmts.is_empty() {
                else_block = Some(else_stmts);
            }
        } else {
            // No else block - all statements go to then block
            for stmt_pair in remaining {
                then_block.push(Self::parse_statement(stmt_pair)?);
            }
        }
        
        Ok(ControlFlow::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_for_statement(pair: pest::iterators::Pair<Rule>) -> Result<ControlFlow, ParseError> {
        let mut inner = pair.into_inner();
        
        // Parse loop variable
        let var_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("for loop variable".to_string())
        })?;
        let variable = Self::parse_identifier(var_pair)?;
        
        // Parse iterable
        let iterable_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("for loop iterable".to_string())
        })?;
        let iterable = Self::parse_iterable(iterable_pair)?;
        
        // Parse body
        let mut body = Vec::new();
        for stmt_pair in inner {
            if stmt_pair.as_rule() == Rule::statement {
                body.push(Self::parse_statement(stmt_pair)?);
            }
        }
        
        Ok(ControlFlow::For {
            variable,
            iterable,
            body,
        })
    }

    fn parse_while_statement(pair: pest::iterators::Pair<Rule>) -> Result<ControlFlow, ParseError> {
        let mut inner = pair.into_inner();
        
        // Parse condition
        let condition_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("while condition".to_string())
        })?;
        let condition = Self::parse_condition(condition_pair)?;
        
        // Parse body
        let mut body = Vec::new();
        for stmt_pair in inner {
            if stmt_pair.as_rule() == Rule::statement {
                body.push(Self::parse_statement(stmt_pair)?);
            }
        }
        
        Ok(ControlFlow::While {
            condition,
            body,
        })
    }

    fn parse_condition(pair: pest::iterators::Pair<Rule>) -> Result<Condition, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("condition value".to_string())
        })?;

        match inner.as_rule() {
            Rule::constraint => Ok(Condition::Constraint(Self::parse_constraint(inner)?)),
            Rule::tensor_expr => Ok(Condition::Tensor(Self::parse_tensor_expr(inner)?)),
            _ => Err(ParseError::InvalidValue(format!("Invalid condition: {}", inner.as_str()))),
        }
    }

    fn parse_iterable(pair: pest::iterators::Pair<Rule>) -> Result<Iterable, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("iterable value".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_expr => Ok(Iterable::Tensor(Self::parse_tensor_expr(inner)?)),
            Rule::entity_set => Ok(Iterable::EntitySet(Self::parse_entity_set(inner)?)),
            Rule::range_expr => {
                // range(n) -> extract n
                let mut range_inner = inner.into_inner();
                let n_pair = range_inner.next().ok_or_else(|| {
                    ParseError::MissingField("range size".to_string())
                })?;
                let n = Self::parse_number(n_pair)? as usize;
                Ok(Iterable::Range(n))
            },
            _ => Err(ParseError::InvalidValue(format!("Invalid iterable: {}", inner.as_str()))),
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
