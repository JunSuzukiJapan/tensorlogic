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
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "parser/grammar.pest"]
pub struct TensorLogicParser;

/// Function registry for resolving function calls during parsing
struct FunctionRegistry {
    /// Builtin functions
    builtins: HashMap<String, BuiltinFunctionId>,
    /// User-defined functions (accumulated during parsing)
    user_functions: HashMap<String, Rc<FunctionDecl>>,
}

impl FunctionRegistry {
    fn new() -> Self {
        let mut builtins = HashMap::new();
        Self::register_builtins(&mut builtins);
        FunctionRegistry {
            builtins,
            user_functions: HashMap::new(),
        }
    }

    fn register_builtins(builtins: &mut HashMap<String, BuiltinFunctionId>) {
        use BuiltinFunctionId::*;

        // Tensor operations
        builtins.insert("zeros".to_string(), TensorZeros);
        builtins.insert("ones".to_string(), TensorOnes);
        builtins.insert("reshape".to_string(), TensorReshape);
        builtins.insert("concat".to_string(), TensorConcat);
        builtins.insert("split".to_string(), TensorSplit);
        builtins.insert("transpose".to_string(), TensorTranspose);
        builtins.insert("permute".to_string(), TensorPermute);

        // Neural network operations
        builtins.insert("linear".to_string(), NNLinear);
        builtins.insert("rms_norm".to_string(), NNRmsNorm);
        builtins.insert("softmax".to_string(), NNSoftmax);
        builtins.insert("rope".to_string(), NNRoPE);
        builtins.insert("attention_with_cache".to_string(), NNAttentionWithCache);
        builtins.insert("silu".to_string(), NNSiLU);
        builtins.insert("gelu".to_string(), NNGeLU);
        builtins.insert("sigmoid".to_string(), NNSigmoid);

        // Math functions
        builtins.insert("sin".to_string(), MathSin);
        builtins.insert("cos".to_string(), MathCos);
        builtins.insert("exp".to_string(), MathExp);
        builtins.insert("log".to_string(), MathLog);
        builtins.insert("sqrt".to_string(), MathSqrt);
        builtins.insert("abs".to_string(), MathAbs);
        builtins.insert("pow".to_string(), MathPow);
        builtins.insert("tanh".to_string(), MathTanh);

        // Sampling functions
        builtins.insert("sample_temperature".to_string(), SampleTemperature);
        builtins.insert("sample_top_k".to_string(), SampleTopK);
        builtins.insert("sample_greedy".to_string(), SampleGreedy);

        // Model I/O
        builtins.insert("load_f16".to_string(), ModelLoadF16);
        builtins.insert("load_f32".to_string(), ModelLoadF32);
        builtins.insert("load_tokenizer".to_string(), TokenizerLoad);

        // Utilities
        builtins.insert("shape".to_string(), UtilShape);
        builtins.insert("print".to_string(), UtilPrint);
        builtins.insert("env".to_string(), UtilEnv);
    }

    fn register_user_function(&mut self, func_decl: FunctionDecl) {
        let name = func_decl.name.as_str().to_string();
        self.user_functions.insert(name, Rc::new(func_decl));
    }

    fn resolve(&self, name: &str) -> Option<ResolvedFunction> {
        if let Some(builtin_id) = self.builtins.get(name) {
            return Some(ResolvedFunction::Builtin(*builtin_id));
        }
        if let Some(func_decl) = self.user_functions.get(name) {
            return Some(ResolvedFunction::UserDefined(Rc::clone(func_decl)));
        }
        None
    }
}

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
        // Validate with lexer to ensure no keywords are used as identifiers
        use crate::lexer::Lexer;
        Lexer::validate_identifiers(source)
            .map_err(|e| ParseError::PestError(format!("Lexer validation error: {}", e)))?;

        // Create function registry with builtin functions
        let mut registry = FunctionRegistry::new();

        // Parse with Pest
        let pairs = Self::parse(Rule::program, source)?;

        let mut declarations = Vec::new();
        let mut main_block = None;
        let mut test_blocks = Vec::new();
        let mut bench_blocks = Vec::new();

        for pair in pairs {
            match pair.as_rule() {
                Rule::program => {
                    for inner in pair.into_inner() {
                        match inner.as_rule() {
                            Rule::declaration => {
                                declarations.push(Self::parse_declaration(inner, &mut registry)?);
                            }
                            Rule::main_block => {
                                main_block = Some(Self::parse_main_block(inner, &registry)?);
                            }
                            Rule::test_block => {
                                test_blocks.push(Self::parse_test_block(inner, &registry)?);
                            }
                            Rule::bench_block => {
                                bench_blocks.push(Self::parse_bench_block(inner, &registry)?);
                            }
                            Rule::EOI => {}
                            _ => {
                                return Err(ParseError::UnexpectedRule {
                                    expected: "declaration, main_block, test_block, or bench_block".to_string(),
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
            test_blocks,
            bench_blocks,
        })
    }

    fn parse_main_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<MainBlock, ParseError> {
        let mut statements = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::statement {
                statements.push(Self::parse_statement(inner, registry)?);
            }
        }

        Ok(MainBlock { statements })
    }

    fn parse_test_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TestBlock, ParseError> {
        let mut name = None;
        let mut statements = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    name = Some(Identifier(inner.as_str().to_string()));
                }
                Rule::statement => {
                    statements.push(Self::parse_statement(inner, registry)?);
                }
                _ => {}
            }
        }

        Ok(TestBlock {
            name: name.ok_or_else(|| ParseError::UnexpectedRule {
                expected: "test name".to_string(),
                found: "none".to_string(),
            })?,
            statements,
        })
    }

    fn parse_bench_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<BenchBlock, ParseError> {
        let mut name = None;
        let mut statements = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    name = Some(Identifier(inner.as_str().to_string()));
                }
                Rule::statement => {
                    statements.push(Self::parse_statement(inner, registry)?);
                }
                _ => {}
            }
        }

        Ok(BenchBlock {
            name: name.ok_or_else(|| ParseError::UnexpectedRule {
                expected: "bench name".to_string(),
                found: "none".to_string(),
            })?,
            statements,
        })
    }

    fn parse_declaration(pair: pest::iterators::Pair<Rule>, registry: &mut FunctionRegistry) -> Result<Declaration, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("declaration type".to_string())
        })?;

        match inner.as_rule() {
            Rule::import_decl => Ok(Declaration::Import(Self::parse_import_decl(inner)?)),
            Rule::entity_decl => Ok(Declaration::Entity(Self::parse_entity_decl(inner)?)),
            Rule::tensor_decl => Ok(Declaration::Tensor(Self::parse_tensor_decl(inner, registry)?)),
            Rule::relation_decl => Ok(Declaration::Relation(Self::parse_relation_decl(inner)?)),
            Rule::rule_decl => Ok(Declaration::Rule(Self::parse_rule_decl(inner, registry)?)),
            Rule::embedding_decl => Ok(Declaration::Embedding(Self::parse_embedding_decl(inner)?)),
            Rule::relation_embedding_decl => Ok(Declaration::RelationEmbedding(Self::parse_relation_embedding_decl(inner)?)),
            Rule::function_decl => {
                let func_decl = Self::parse_function_decl(inner, registry)?;
                // Register the function immediately after parsing
                registry.register_user_function(func_decl.clone());
                Ok(Declaration::Function(func_decl))
            },
            Rule::struct_decl => Ok(Declaration::Struct(Self::parse_struct_decl(inner)?)),
            Rule::impl_block => Ok(Declaration::Impl(Self::parse_impl_block(inner, registry)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "declaration type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_import_decl(pair: pest::iterators::Pair<Rule>) -> Result<ImportDecl, ParseError> {
        let mut inner = pair.into_inner();

        let path_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("import path".to_string())
        })?;

        let path = Self::parse_string_literal(path_pair)?;

        Ok(ImportDecl { path })
    }

    fn parse_entity_decl(pair: pest::iterators::Pair<Rule>) -> Result<EntityDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("entity name".to_string())
        })?)?;

        // Check if there's an entity_list (explicit enumeration)
        if let Some(entity_list_pair) = inner.next() {
            // Explicit: entity Person = {alice, bob, charlie}
            let entities = Self::parse_entity_list(entity_list_pair)?;
            Ok(EntityDecl::Explicit { name, entities })
        } else {
            // From data: entity Person
            Ok(EntityDecl::FromData { name })
        }
    }

    fn parse_entity_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Identifier>, ParseError> {
        let entities = pair
            .into_inner()
            .map(|p| Self::parse_identifier(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(entities)
    }

    fn parse_tensor_decl(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("tensor name".to_string())
        })?)?;

        let tensor_type = Self::parse_tensor_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("tensor type".to_string())
        })?)?;

        let init_expr = if let Some(expr_pair) = inner.next() {
            Some(Self::parse_tensor_expr(expr_pair, registry)?)
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
            "float32" => Ok(BaseType::Float64), // Float64 enum represents float32 (32-bit float)
            "float16" => Ok(BaseType::Float32), // Float32 enum represents float16 (16-bit float)
            "int16" => Ok(BaseType::Int16),
            "int32" => Ok(BaseType::Int32),
            "int64" => Ok(BaseType::Int64),
            "bool" => Ok(BaseType::Bool),
            "complex16" => Ok(BaseType::Complex64), // Complex64 enum represents complex16 (2x 16-bit floats)
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
        // Check the string directly first for keywords
        match pair.as_str() {
            "entity" => return Ok(EntityType::Entity),
            "concept" => return Ok(EntityType::Concept),
            _ => {}
        }

        // Try to get inner rule
        if let Some(inner) = pair.into_inner().next() {
            match inner.as_rule() {
                Rule::scalar_type => Ok(EntityType::Scalar(Self::parse_scalar_type(inner)?)),
                Rule::tensor_type => Ok(EntityType::Tensor(Self::parse_tensor_type(inner)?)),
                Rule::struct_type => Ok(EntityType::Struct(Self::parse_struct_type(inner)?)),
                Rule::identifier => {
                    // Named entity type (e.g., Person, City)
                    Ok(EntityType::NamedEntity(Self::parse_identifier(inner)?))
                }
                _ => Err(ParseError::InvalidValue(format!("Unknown entity type: {}", inner.as_str()))),
            }
        } else {
            Err(ParseError::MissingField("entity type value".to_string()))
        }
    }

    fn parse_scalar_type(pair: pest::iterators::Pair<Rule>) -> Result<ScalarType, ParseError> {
        match pair.as_str() {
            "int" => Ok(ScalarType::Int),
            "float" => Ok(ScalarType::Float),
            "bool" => Ok(ScalarType::Bool),
            "string" => Ok(ScalarType::String),
            _ => Err(ParseError::InvalidValue(format!("Unknown scalar type: {}", pair.as_str()))),
        }
    }

    fn parse_embedding_spec(pair: pest::iterators::Pair<Rule>) -> Result<TensorType, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("embedding tensor type".to_string())
        })?;
        Self::parse_tensor_type(inner)
    }

    fn parse_rule_decl(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<RuleDecl, ParseError> {
        let mut inner = pair.into_inner();

        let head = Self::parse_rule_head(inner.next().ok_or_else(|| {
            ParseError::MissingField("rule head".to_string())
        })?, registry)?;

        let body = Self::parse_rule_body(inner.next().ok_or_else(|| {
            ParseError::MissingField("rule body".to_string())
        })?, registry)?;

        Ok(RuleDecl { head, body })
    }

    fn parse_rule_head(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<RuleHead, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("rule head content".to_string())
        })?;

        match inner.as_rule() {
            Rule::atom => Ok(RuleHead::Atom(Self::parse_atom(inner, registry)?)),
            Rule::tensor_equation => Ok(RuleHead::Equation(Self::parse_tensor_equation(inner, registry)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "atom or equation".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_rule_body(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Vec<BodyTerm>, ParseError> {
        pair.into_inner()
            .map(|term_pair| Self::parse_body_term(term_pair, registry))
            .collect()
    }

    fn parse_body_term(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<BodyTerm, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("body term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::atom => Ok(BodyTerm::Atom(Self::parse_atom(inner, registry)?)),
            Rule::tensor_equation => Ok(BodyTerm::Equation(Self::parse_tensor_equation(inner, registry)?)),
            Rule::constraint => Ok(BodyTerm::Constraint(Self::parse_constraint(inner, registry)?)),
            _ => Err(ParseError::UnexpectedRule {
                expected: "atom, equation, or constraint".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_atom(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Atom, ParseError> {
        let mut inner = pair.into_inner();

        let predicate = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("atom predicate".to_string())
        })?)?;

        let terms = if let Some(term_list) = inner.next() {
            Self::parse_term_list(term_list, registry)?
        } else {
            Vec::new()
        };

        Ok(Atom { predicate, terms })
    }

    fn parse_term_list(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Vec<Term>, ParseError> {
        pair.into_inner()
            .map(|term_pair| Self::parse_term(term_pair, registry))
            .collect()
    }

    fn parse_term(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Term, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::identifier => Ok(Term::Variable(Self::parse_identifier(inner)?)),
            Rule::constant => Ok(Term::Constant(Self::parse_constant(inner)?)),
            Rule::tensor_expr => Ok(Term::Tensor(Self::parse_tensor_expr(inner, registry)?)),
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

        // Try to find identifier_list or identifier
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier_list => {
                    let identifiers = inner
                        .into_inner()
                        .map(|id_pair| Self::parse_identifier(id_pair))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(EntitySet::Explicit(identifiers));
                }
                Rule::identifier => {
                    // Entity type name
                    return Ok(EntitySet::Type(Self::parse_identifier(inner)?));
                }
                _ if inner.as_str() == "auto" => {
                    return Ok(EntitySet::Auto);
                }
                _ => {}
            }
        }

        Err(ParseError::MissingField("entity set content".to_string()))
    }

    fn parse_relation_embedding_decl(pair: pest::iterators::Pair<Rule>) -> Result<RelationEmbeddingDecl, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("relation embedding name".to_string())
        })?)?;

        let mut relations = RelationSet::All;
        let mut dimension = 0;
        let mut init_method = InitMethod::Random;

        for field in inner {
            match field.as_rule() {
                Rule::relation_set => {
                    relations = Self::parse_relation_set(field)?;
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

        Ok(RelationEmbeddingDecl {
            name,
            relations,
            dimension,
            init_method,
        })
    }

    fn parse_relation_set(pair: pest::iterators::Pair<Rule>) -> Result<RelationSet, ParseError> {
        // Check if it's "all" directly
        if pair.as_str() == "all" {
            return Ok(RelationSet::All);
        }

        // Try to find identifier_list
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier_list => {
                    let identifiers = inner
                        .into_inner()
                        .map(|id_pair| Self::parse_identifier(id_pair))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(RelationSet::Explicit(identifiers));
                }
                _ if inner.as_str() == "all" => {
                    return Ok(RelationSet::All);
                }
                _ => {}
            }
        }

        Err(ParseError::MissingField("relation set content".to_string()))
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

    fn parse_function_decl(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<FunctionDecl, ParseError> {
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
                    body.push(Self::parse_statement(item, registry)?);
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
        // Check if return type is directly "void" (no inner nodes)
        if pair.as_str() == "void" {
            return Ok(ReturnType::Void);
        }

        // Otherwise, it should be a scalar_type or tensor_type
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("return type value".to_string())
        })?;

        match inner.as_rule() {
            Rule::scalar_type => Ok(ReturnType::Scalar(Self::parse_scalar_type(inner)?)),
            Rule::tensor_type => Ok(ReturnType::Tensor(Self::parse_tensor_type(inner)?)),
            Rule::struct_type => Ok(ReturnType::Struct(Self::parse_struct_type(inner)?)),
            _ => Err(ParseError::InvalidValue(format!("Invalid return type: {}", inner.as_str()))),
        }
    }

    fn parse_tensor_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        // Expression parser with operator precedence handling
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor expression content".to_string())
        })?;
        Self::parse_logical_or_expr(inner, registry)
    }

    fn parse_logical_or_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("logical or expression".to_string())
        })?;
        let mut expr = Self::parse_logical_and_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            // Skip the or_op rule and get the right operand
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand after ||".to_string())
            })?;
            let right_expr = Self::parse_logical_and_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op: BinaryOp::Or,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_logical_and_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("logical and expression".to_string())
        })?;
        let mut expr = Self::parse_equality_expr(first, registry)?;

        while let Some(_op_pair) = pairs.next() {
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand after &&".to_string())
            })?;
            let right_expr = Self::parse_equality_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op: BinaryOp::And,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_equality_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("equality expression".to_string())
        })?;
        let mut expr = Self::parse_comparison_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_str() {
                "==" => BinaryOp::Eq,
                "!=" => BinaryOp::Ne,
                _ => return Err(ParseError::InvalidValue(format!("unknown equality operator: {}", op_pair.as_str()))),
            };
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right_expr = Self::parse_comparison_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_comparison_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("comparison expression".to_string())
        })?;
        let mut expr = Self::parse_additive_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_str() {
                "<" => BinaryOp::Lt,
                "<=" => BinaryOp::Le,
                ">" => BinaryOp::Gt,
                ">=" => BinaryOp::Ge,
                _ => return Err(ParseError::InvalidValue(format!("unknown comparison operator: {}", op_pair.as_str()))),
            };
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right_expr = Self::parse_additive_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_additive_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("additive expression".to_string())
        })?;
        let mut expr = Self::parse_multiplicative_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_str() {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                _ => return Err(ParseError::InvalidValue(format!("unknown additive operator: {}", op_pair.as_str()))),
            };
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right_expr = Self::parse_multiplicative_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_multiplicative_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("multiplicative expression".to_string())
        })?;
        let mut expr = Self::parse_power_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_str() {
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "%" => BinaryOp::Mod,
                "@" => BinaryOp::MatMul,
                "⊙" => BinaryOp::Hadamard,
                _ => return Err(ParseError::InvalidValue(format!("unknown multiplicative operator: {}", op_pair.as_str()))),
            };
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right_expr = Self::parse_power_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_power_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("power expression".to_string())
        })?;
        let mut expr = Self::parse_unary_expr(first, registry)?;

        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_str() {
                "**" => BinaryOp::Power,
                "⊗" => BinaryOp::TensorProd,
                _ => return Err(ParseError::InvalidValue(format!("unknown power operator: {}", op_pair.as_str()))),
            };
            let right = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right operand".to_string())
            })?;
            let right_expr = Self::parse_unary_expr(right, registry)?;
            expr = TensorExpr::BinaryOp {
                op,
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        Ok(expr)
    }

    fn parse_unary_expr(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().ok_or_else(|| {
            ParseError::MissingField("unary expression".to_string())
        })?;

        match first.as_rule() {
            Rule::unary_op => {
                let op = Self::parse_unary_op(first)?;
                let operand = pairs.next().ok_or_else(|| {
                    ParseError::MissingField("unary operand".to_string())
                })?;
                let operand_expr = Self::parse_unary_expr(operand, registry)?;
                Ok(TensorExpr::UnaryOp {
                    op,
                    operand: Box::new(operand_expr),
                })
            }
            Rule::tensor_term => Self::parse_tensor_term(first, registry),
            _ => Err(ParseError::InvalidValue(format!("unexpected rule in unary expression: {:?}", first.as_rule()))),
        }
    }

    fn parse_binary_op(pair: pest::iterators::Pair<Rule>) -> Result<BinaryOp, ParseError> {
        match pair.as_str() {
            "+" => Ok(BinaryOp::Add),
            "-" => Ok(BinaryOp::Sub),
            "*" => Ok(BinaryOp::Mul),
            "/" => Ok(BinaryOp::Div),
            "%" => Ok(BinaryOp::Mod),
            "@" => Ok(BinaryOp::MatMul),
            "**" => Ok(BinaryOp::Power),
            "⊗" => Ok(BinaryOp::TensorProd),
            "⊙" => Ok(BinaryOp::Hadamard),
            "==" => Ok(BinaryOp::Eq),
            "!=" => Ok(BinaryOp::Ne),
            "<" => Ok(BinaryOp::Lt),
            "<=" => Ok(BinaryOp::Le),
            ">" => Ok(BinaryOp::Gt),
            ">=" => Ok(BinaryOp::Ge),
            "&&" => Ok(BinaryOp::And),
            "||" => Ok(BinaryOp::Or),
            _ => Err(ParseError::InvalidValue(format!("unknown binary operator: {}", pair.as_str()))),
        }
    }

    fn parse_unary_op(pair: pest::iterators::Pair<Rule>) -> Result<UnaryOp, ParseError> {
        match pair.as_str() {
            "-" => Ok(UnaryOp::Neg),
            "!" => Ok(UnaryOp::Not),
            "inv" => Ok(UnaryOp::Inverse),
            "det" => Ok(UnaryOp::Determinant),
            _ => Err(ParseError::InvalidValue(format!("unknown unary operator: {}", pair.as_str()))),
        }
    }

    fn parse_tensor_term(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor term content".to_string())
        })?;

        match inner.as_rule() {
            Rule::identifier => Ok(TensorExpr::Variable(Self::parse_identifier(inner)?)),
            Rule::tensor_literal => {
                let lit = Self::parse_tensor_literal(inner, registry)?;
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
                let tensors = Self::parse_tensor_list(tensor_list, registry)?;

                Ok(TensorExpr::EinSum { spec, tensors })
            }
            Rule::function_call => Self::parse_function_call(inner, registry),
            Rule::postfix_expr => {
                let mut inner_pairs = inner.into_inner();

                // Parse primary expression (function_call or identifier)
                let primary = inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("primary expression".to_string())
                })?;

                let mut expr = match primary.as_rule() {
                    Rule::primary_expr => {
                        // primary_expr contains function_call or identifier
                        let inner_primary = primary.into_inner().next().ok_or_else(|| {
                            ParseError::MissingField("primary_expr content".to_string())
                        })?;
                        match inner_primary.as_rule() {
                            Rule::associated_call => Self::parse_associated_call(inner_primary, registry)?,
                            Rule::function_call => Self::parse_function_call(inner_primary, registry)?,
                            Rule::identifier => TensorExpr::Variable(Self::parse_identifier(inner_primary)?),
                            _ => return Err(ParseError::UnexpectedRule {
                                expected: "associated_call, function_call or identifier".to_string(),
                                found: format!("{:?}", inner_primary.as_rule()),
                            }),
                        }
                    }
                    Rule::function_call => Self::parse_function_call(primary, registry)?,
                    Rule::identifier => TensorExpr::Variable(Self::parse_identifier(primary)?),
                    _ => return Err(ParseError::UnexpectedRule {
                        expected: "primary_expr, function_call, or identifier".to_string(),
                        found: format!("{:?}", primary.as_rule()),
                    }),
                };

                // Apply postfix operations (indexing, property access, method calls)
                for postfix_op in inner_pairs {
                    match postfix_op.as_rule() {
                        Rule::postfix_op => {
                            // postfix_op contains method_call, property_access, or index_access
                            let inner_op = postfix_op.into_inner().next().ok_or_else(|| {
                                ParseError::MissingField("postfix operation content".to_string())
                            })?;

                            match inner_op.as_rule() {
                                Rule::method_call => {
                                    // Parse: .identifier(args)
                                    let mut method_inner = inner_op.into_inner();
                                    let method_name = Self::parse_identifier(method_inner.next().ok_or_else(|| {
                                        ParseError::MissingField("method name".to_string())
                                    })?)?;

                                    // Parse optional argument list
                                    let args = if let Some(arg_list) = method_inner.next() {
                                        Self::parse_tensor_list(arg_list, registry)?
                                    } else {
                                        vec![]
                                    };

                                    expr = TensorExpr::MethodCall {
                                        object: Box::new(expr),
                                        method: method_name,
                                        args,
                                    };
                                }
                                Rule::property_access => {
                                    // Parse: .identifier
                                    let mut prop_inner = inner_op.into_inner();
                                    let property_name = Self::parse_identifier(prop_inner.next().ok_or_else(|| {
                                        ParseError::MissingField("property name".to_string())
                                    })?)?;

                                    expr = TensorExpr::PropertyAccess {
                                        object: Box::new(expr),
                                        property: property_name,
                                    };
                                }
                                Rule::index_access => {
                                    // Parse: [indices]
                                    let index_list = inner_op.into_inner().next().ok_or_else(|| {
                                        ParseError::MissingField("index list".to_string())
                                    })?;
                                    let indices = Self::parse_index_list(index_list)?;
                                    expr = TensorExpr::TensorIndex {
                                        tensor: Box::new(expr),
                                        indices,
                                    };
                                }
                                _ => return Err(ParseError::UnexpectedRule {
                                    expected: "method_call, property_access, or index_access".to_string(),
                                    found: format!("{:?}", inner_op.as_rule()),
                                }),
                            }
                        }
                        _ => return Err(ParseError::UnexpectedRule {
                            expected: "postfix_op".to_string(),
                            found: format!("{:?}", postfix_op.as_rule()),
                        }),
                    }
                }

                Ok(expr)
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
            Rule::python_call => {
                Self::parse_python_call(inner, registry)
            }
            Rule::struct_literal => {
                Self::parse_struct_literal(inner, registry)
            }
            Rule::string_literal => {
                // String literals in expressions (e.g., for save/load filenames)
                let s = Self::parse_string_literal(inner)?;
                Ok(TensorExpr::Literal(TensorLiteral::Scalar(ScalarLiteral::String(s))))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "tensor term".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_function_call(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut inner_pairs = pair.into_inner();

        // First pair can be type_namespace or identifier
        let first_pair = inner_pairs.next().ok_or_else(|| {
            ParseError::MissingField("function name or type namespace".to_string())
        })?;

        let (type_namespace, name) = match first_pair.as_rule() {
            Rule::type_namespace => {
                // type_namespace::function_name format
                let type_ns = first_pair.as_str().to_string();
                let name_pair = inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("function name after type namespace".to_string())
                })?;
                let name = Self::parse_identifier(name_pair)?;
                (Some(type_ns), name)
            }
            Rule::identifier => {
                // function_name only
                let name = Self::parse_identifier(first_pair)?;
                (None, name)
            }
            _ => return Err(ParseError::UnexpectedRule {
                expected: "type_namespace or identifier".to_string(),
                found: format!("{:?}", first_pair.as_rule()),
            })
        };

        let args = if let Some(tensor_list) = inner_pairs.next() {
            Self::parse_tensor_list(tensor_list, registry)?
        } else {
            Vec::new()
        };

        Ok(TensorExpr::FunctionCall { type_namespace, name, args, resolved: None })
    }

    fn parse_tensor_list(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Vec<TensorExpr>, ParseError> {
        pair.into_inner()
            .map(|expr_pair| Self::parse_tensor_expr(expr_pair, registry))
            .collect()
    }

    fn parse_tensor_literal(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorLiteral, ParseError> {
        // Handle empty arrays: [] has no inner elements
        let mut inner_iter = pair.into_inner();
        let inner = match inner_iter.next() {
            Some(inner) => inner,
            None => {
                // Empty array - return empty Array
                return Ok(TensorLiteral::Array(vec![]));
            }
        };

        match inner.as_rule() {
            Rule::scalar_literal => Ok(TensorLiteral::Scalar(Self::parse_scalar_literal(inner)?)),
            Rule::tensor_elements => {
                let elements = inner
                    .into_inner()
                    .map(|elem| Self::parse_tensor_element(elem, registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TensorLiteral::Array(elements))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "scalar or array".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_element(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ArrayElement, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor element content".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_literal => {
                let lit = Self::parse_tensor_literal(inner, registry)?;
                Ok(ArrayElement::Literal(lit))
            }
            Rule::tensor_expr => {
                // Parse tensor_expr (supports variables like seq_len, d_model)
                let expr = Self::parse_tensor_expr(inner, registry)?;
                Ok(ArrayElement::Expression(expr))
            }
            Rule::number => {
                let s = inner.as_str();
                // Check if it's an integer (no decimal point) or float
                let scalar = if s.contains('.') || s.contains('e') || s.contains('E') {
                    ScalarLiteral::Float(Self::parse_number(inner)?)
                } else {
                    let val = s.parse::<i64>()
                        .map_err(|e| ParseError::InvalidValue(format!("Invalid integer: {}", e)))?;
                    ScalarLiteral::Integer(val)
                };
                Ok(ArrayElement::Literal(TensorLiteral::Scalar(scalar)))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "tensor literal, number, or tensor expression".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_scalar_literal(pair: pest::iterators::Pair<Rule>) -> Result<ScalarLiteral, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("scalar value".to_string())
        })?;

        match inner.as_rule() {
            Rule::number => {
                let s = inner.as_str();
                // Check if it's an integer (no decimal point) or float
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(ScalarLiteral::Float(Self::parse_number(inner)?))
                } else {
                    let val = s.parse::<i64>()
                        .map_err(|e| ParseError::InvalidValue(format!("Invalid integer: {}", e)))?;
                    Ok(ScalarLiteral::Integer(val))
                }
            }
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

    fn parse_constraint(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Constraint, ParseError> {
        // constraint = { constraint_term ~ (logical_op ~ constraint_term)* }
        let mut pairs = pair.into_inner();

        let first_term = pairs.next().ok_or_else(|| {
            ParseError::MissingField("constraint term".to_string())
        })?;
        let mut constraint = Self::parse_constraint_term(first_term, registry)?;

        // Parse logical operators (and, or)
        while let Some(op_pair) = pairs.next() {
            let op = op_pair.as_str();
            let right_term = pairs.next().ok_or_else(|| {
                ParseError::MissingField("right constraint term".to_string())
            })?;
            let right = Self::parse_constraint_term(right_term, registry)?;

            constraint = match op {
                "and" => Constraint::And(Box::new(constraint), Box::new(right)),
                "or" => Constraint::Or(Box::new(constraint), Box::new(right)),
                _ => return Err(ParseError::InvalidValue(format!("unknown logical operator: {}", op))),
            };
        }

        Ok(constraint)
    }

    fn parse_constraint_term(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Constraint, ParseError> {
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
                    Ok(Constraint::Not(Box::new(Self::parse_constraint_term(negated, registry)?)))
                } else {
                    Self::parse_constraint_term(first, registry)
                }
            }
            Rule::constraint => {
                // Parenthesized constraint
                Self::parse_constraint(inner, registry)
            }
            Rule::tensor_constraint => {
                Self::parse_tensor_constraint(inner, registry)
            }
            Rule::comparison => {
                Self::parse_comparison(inner, registry)
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "constraint term".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_tensor_constraint(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Constraint, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("tensor constraint content".to_string())
        })?;

        let constraint_str = inner.as_str();

        if constraint_str.starts_with("shape") {
            // shape(tensor) == shape_spec
            let mut parts = inner.into_inner();
            let tensor_expr = Self::parse_tensor_expr(parts.next().ok_or_else(|| {
                ParseError::MissingField("tensor in shape constraint".to_string())
            })?, registry)?;

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
            })?, registry)?;

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
            })?, registry)?;

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

    fn parse_comparison(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Constraint, ParseError> {
        let mut inner = pair.into_inner();

        let left = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("left side of comparison".to_string())
        })?, registry)?;

        let op_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("comparison operator".to_string())
        })?;
        let op = Self::parse_comp_op(op_pair)?;

        let right = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("right side of comparison".to_string())
        })?, registry)?;

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

    fn parse_tensor_equation(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorEquation, ParseError> {
        let mut inner = pair.into_inner();

        let left = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation left side".to_string())
        })?, registry)?;

        let eq_type = Self::parse_eq_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation type".to_string())
        })?)?;

        let right = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("equation right side".to_string())
        })?, registry)?;

        Ok(TensorEquation { left, right, eq_type })
    }

    fn parse_eq_type(pair: pest::iterators::Pair<Rule>) -> Result<EquationType, ParseError> {
        match pair.as_str() {
            "~" => Ok(EquationType::Approx),
            s => Err(ParseError::InvalidValue(format!("Unknown equation type: {}", s))),
        }
    }

    fn parse_statement(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("statement content".to_string())
        })?;

        match inner.as_rule() {
            Rule::let_statement => {
                let mut inner_pairs = inner.into_inner();
                let target = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("let target".to_string())
                })?)?;
                let value = Self::parse_tensor_expr(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("let value".to_string())
                })?, registry)?;
                Ok(Statement::Let { target, value })
            }
            Rule::break_statement => {
                Ok(Statement::Break)
            }
            Rule::return_statement => {
                let mut inner_pairs = inner.into_inner();
                let value = if let Some(expr_pair) = inner_pairs.next() {
                    Some(Self::parse_tensor_expr(expr_pair, registry)?)
                } else {
                    None
                };
                Ok(Statement::Return { value })
            }
            Rule::panic_statement => {
                let format_args = inner.into_inner().next().ok_or_else(|| {
                    ParseError::MissingField("panic format args".to_string())
                })?;

                let mut format = String::new();
                let mut args = Vec::new();

                for arg_pair in format_args.into_inner() {
                    match arg_pair.as_rule() {
                        Rule::string_literal => {
                            format = Self::parse_string_literal(arg_pair)?;
                        }
                        Rule::tensor_expr => {
                            args.push(Self::parse_tensor_expr(arg_pair, registry)?);
                        }
                        _ => {}
                    }
                }

                Ok(Statement::Panic { format, args })
            }
            Rule::tensor_decl => {
                Ok(Statement::TensorDecl(Self::parse_tensor_decl(inner, registry)?))
            }
            Rule::assignment => {
                let mut inner_pairs = inner.into_inner();
                let target = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("assignment target".to_string())
                })?)?;
                let value = Self::parse_tensor_expr(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("assignment value".to_string())
                })?, registry)?;
                Ok(Statement::Assignment { target, value })
            }
            Rule::tensor_equation => {
                Ok(Statement::Equation(Self::parse_tensor_equation(inner, registry)?))
            }
            Rule::python_import => {
                Self::parse_python_import(inner)
            }
            Rule::function_call => {
                let pest_span = inner.as_span();
                let mut inner_pairs = inner.into_inner();
                let name = Self::parse_identifier(inner_pairs.next().ok_or_else(|| {
                    ParseError::MissingField("function name".to_string())
                })?)?;

                let args = if let Some(tensor_list) = inner_pairs.next() {
                    Self::parse_tensor_list(tensor_list, registry)?
                } else {
                    Vec::new()
                };

                // Convert pest::Span to our Span type
                let (start_line, start_col) = pest_span.start_pos().line_col();
                let (end_line, end_col) = pest_span.end_pos().line_col();
                let start = Position {
                    line: start_line,
                    column: start_col,
                    offset: pest_span.start(),
                };
                let end = Position {
                    line: end_line,
                    column: end_col,
                    offset: pest_span.end(),
                };
                let span = Span::new(start, end);

                Ok(Statement::FunctionCall { name, args, resolved: None, span })
            }
            Rule::fact_assertion => {
                Self::parse_fact_assertion(inner, registry)
            }
            Rule::query => {
                Self::parse_query(inner, registry)
            }
            Rule::inference_call => {
                Self::parse_inference_call(inner, registry)
            }
            Rule::learning_call => {
                Self::parse_learning_call(inner, registry)
            }
            Rule::with_block => {
                Self::parse_with_block(inner, registry)
            }
            Rule::control_flow => {
                Self::parse_control_flow(inner, registry).map(Statement::ControlFlow)
            }
            Rule::block_statement => {
                let statements: Result<Vec<_>, _> = inner
                    .into_inner()
                    .map(|p| Self::parse_statement(p, registry))
                    .collect();
                Ok(Statement::Block {
                    statements: statements?,
                })
            }
            Rule::tensor_expr => {
                // Parse expression as an expression statement (for method calls like cache.set(...))
                let expr = Self::parse_tensor_expr(inner, registry)?;
                Ok(Statement::Expr { expr })
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "statement type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_fact_assertion(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let atom = Self::parse_atom(inner.next().ok_or_else(|| {
            ParseError::MissingField("fact assertion atom".to_string())
        })?, registry)?;

        Ok(Statement::FactAssertion { atom })
    }

    fn parse_query(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let atom = Self::parse_atom(inner.next().ok_or_else(|| {
            ParseError::MissingField("query atom".to_string())
        })?, registry)?;

        let constraints = if let Some(constraint_list) = inner.next() {
            Self::parse_constraint_list(constraint_list, registry)?
        } else {
            Vec::new()
        };

        Ok(Statement::Query { atom, constraints })
    }

    fn parse_constraint_list(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Vec<Constraint>, ParseError> {
        pair.into_inner()
            .map(|constraint_pair| Self::parse_constraint(constraint_pair, registry))
            .collect()
    }

    fn parse_inference_call(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let first = inner.next().ok_or_else(|| {
            ParseError::MissingField("inference method or block".to_string())
        })?;

        match first.as_rule() {
            Rule::inference_block => {
                // Block syntax: infer { ... }
                Self::parse_inference_block(first, registry)
            }
            Rule::inference_method => {
                // Single inference: infer method query
                let method = Self::parse_inference_method(first)?;
                let query = Self::parse_query(inner.next().ok_or_else(|| {
                    ParseError::MissingField("query in inference call".to_string())
                })?, registry)?;

                Ok(Statement::Inference {
                    method,
                    query: Box::new(query),
                })
            }
            _ => Err(ParseError::InvalidValue(
                "Expected inference method or block".to_string()
            )),
        }
    }

    fn parse_inference_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let mut items = Vec::new();

        for item_pair in pair.into_inner() {
            if item_pair.as_rule() == Rule::inference_item {
                let mut item_inner = item_pair.into_inner();

                let method = Self::parse_inference_method(item_inner.next().ok_or_else(|| {
                    ParseError::MissingField("inference method in block item".to_string())
                })?)?;

                let query = Self::parse_query(item_inner.next().ok_or_else(|| {
                    ParseError::MissingField("query in block item".to_string())
                })?, registry)?;

                items.push((method, Box::new(query)));
            }
        }

        Ok(Statement::InferenceBlock { items })
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

    fn parse_learning_call(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("learning spec".to_string())
        })?;

        let learning_spec = Self::parse_learning_spec(inner, registry)?;
        Ok(Statement::Learning(learning_spec))
    }

    fn parse_learning_spec(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<LearningSpec, ParseError> {
        let inner = pair.into_inner();

        // Parse optional statements
        let mut statements = Vec::new();
        let mut objective_expr = None;
        let mut optimizer_spec = None;
        let mut epochs_val = None;
        let mut scheduler_spec = None;

        for pair in inner {
            match pair.as_rule() {
                Rule::statement => {
                    statements.push(Self::parse_statement(pair, registry)?);
                }
                Rule::tensor_expr => {
                    objective_expr = Some(pair);
                }
                Rule::optimizer_spec => {
                    optimizer_spec = Some(pair);
                }
                Rule::integer => {
                    epochs_val = Some(pair);
                }
                Rule::scheduler_spec => {
                    scheduler_spec = Some(pair);
                }
                _ => {}
            }
        }

        let objective = Self::parse_tensor_expr(objective_expr.ok_or_else(|| {
            ParseError::MissingField("objective expression".to_string())
        })?, registry)?;

        let optimizer = Self::parse_optimizer_spec(optimizer_spec.ok_or_else(|| {
            ParseError::MissingField("optimizer spec".to_string())
        })?)?;

        let epochs = Self::parse_number(epochs_val.ok_or_else(|| {
            ParseError::MissingField("epochs value".to_string())
        })?)? as usize;

        let scheduler = if let Some(s) = scheduler_spec {
            Some(Self::parse_scheduler_spec(s)?)
        } else {
            None
        };

        Ok(LearningSpec {
            statements,
            objective,
            optimizer,
            epochs,
            scheduler,
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

    fn parse_scheduler_spec(pair: pest::iterators::Pair<Rule>) -> Result<SchedulerSpec, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("scheduler name".to_string())
        })?)?.as_str().to_string();

        let params = if let Some(params_pair) = inner.next() {
            Self::parse_scheduler_params(params_pair)?
        } else {
            Vec::new()
        };

        Ok(SchedulerSpec { name, params })
    }

    fn parse_scheduler_params(pair: pest::iterators::Pair<Rule>) -> Result<Vec<(String, f64)>, ParseError> {
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

    fn parse_with_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        let entity_type = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("entity type".to_string())
        })?)?;

        let statements = inner
            .map(|p| Self::parse_statement(p, registry))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Statement::WithBlock {
            entity_type,
            statements,
        })
    }

    fn parse_control_flow(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ControlFlow, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("control flow statement".to_string())
        })?;

        match inner.as_rule() {
            Rule::if_statement => Self::parse_if_statement(inner, registry),
            Rule::for_statement => Self::parse_for_statement(inner, registry),
            Rule::while_statement => Self::parse_while_statement(inner, registry),
            Rule::loop_statement => Self::parse_loop_statement(inner, registry),
            _ => Err(ParseError::InvalidValue(format!("Invalid control flow: {}", inner.as_str()))),
        }
    }

    fn parse_if_statement(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ControlFlow, ParseError> {
        // Get the input string and position before consuming pair
        let input_str = pair.as_str();
        let if_start_pos = pair.as_span().start();
        let has_else = input_str.contains("} else {");

        let mut inner = pair.into_inner();

        // Parse condition (always first)
        let condition_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("if condition".to_string())
        })?;
        let condition = Self::parse_condition(condition_pair, registry)?;

        // Find the absolute position of "} else {" to determine then/else boundary
        let else_pos = input_str.find("} else {").map(|p| if_start_pos + p + 1); // +1 for the closing }

        // Collect all statement pairs
        let remaining: Vec<_> = inner
            .filter(|p| p.as_rule() == Rule::statement)
            .collect();

        let mut then_block = Vec::new();
        let mut else_block = None;

        if has_else {
            // Split statements by position: before else_pos goes to then block, after goes to else block
            let boundary_pos = else_pos.unwrap();

            // Parse then block (statements before the else)
            for stmt_pair in remaining.iter() {
                if stmt_pair.as_span().start() < boundary_pos {
                    then_block.push(Self::parse_statement(stmt_pair.clone(), registry)?);
                }
            }

            // Parse else block (statements after the else)
            let mut else_stmts = Vec::new();
            for stmt_pair in remaining.iter() {
                if stmt_pair.as_span().start() > boundary_pos {
                    else_stmts.push(Self::parse_statement(stmt_pair.clone(), registry)?);
                }
            }

            if !else_stmts.is_empty() {
                else_block = Some(else_stmts);
            }
        } else {
            // No else block - all statements go to then block
            for stmt_pair in remaining {
                then_block.push(Self::parse_statement(stmt_pair, registry)?);
            }
        }
        
        Ok(ControlFlow::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_for_statement(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ControlFlow, ParseError> {
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
        let iterable = Self::parse_iterable(iterable_pair, registry)?;
        
        // Parse body
        let mut body = Vec::new();
        for stmt_pair in inner {
            if stmt_pair.as_rule() == Rule::statement {
                body.push(Self::parse_statement(stmt_pair, registry)?);
            }
        }
        
        Ok(ControlFlow::For {
            variable,
            iterable,
            body,
        })
    }

    fn parse_while_statement(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ControlFlow, ParseError> {
        let mut inner = pair.into_inner();
        
        // Parse condition
        let condition_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("while condition".to_string())
        })?;
        let condition = Self::parse_condition(condition_pair, registry)?;
        
        // Parse body
        let mut body = Vec::new();
        for stmt_pair in inner {
            if stmt_pair.as_rule() == Rule::statement {
                body.push(Self::parse_statement(stmt_pair, registry)?);
            }
        }
        
        Ok(ControlFlow::While {
            condition,
            body,
        })
    }

    fn parse_loop_statement(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ControlFlow, ParseError> {
        let inner = pair.into_inner();

        // Parse body
        let mut body = Vec::new();
        for stmt_pair in inner {
            if stmt_pair.as_rule() == Rule::statement {
                body.push(Self::parse_statement(stmt_pair, registry)?);
            }
        }

        Ok(ControlFlow::Loop { body })
    }

    fn parse_condition(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Condition, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("condition value".to_string())
        })?;

        match inner.as_rule() {
            Rule::constraint => Ok(Condition::Constraint(Self::parse_constraint(inner, registry)?)),
            Rule::tensor_expr => Ok(Condition::Tensor(Self::parse_tensor_expr(inner, registry)?)),
            _ => Err(ParseError::InvalidValue(format!("Invalid condition: {}", inner.as_str()))),
        }
    }

    fn parse_iterable(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<Iterable, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("iterable value".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_expr => Ok(Iterable::Tensor(Self::parse_tensor_expr(inner, registry)?)),
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
            Rule::number => {
                let s = inner.as_str();
                // Check if it's an integer (no decimal point) or float
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(Constant::Float(Self::parse_number(inner)?))
                } else {
                    let val = s.parse::<i64>()
                        .map_err(|e| ParseError::InvalidValue(format!("Invalid integer: {}", e)))?;
                    Ok(Constant::Integer(val))
                }
            }
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
            let content = &s[1..s.len() - 1];
            // Process escape sequences
            let processed = content
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            Ok(processed)
        } else {
            Err(ParseError::InvalidValue(format!("Invalid string literal: {}", s)))
        }
    }

    fn parse_index_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<IndexExpr>, ParseError> {
        pair.into_inner()
            .map(|index_pair| Self::parse_index_expr(index_pair))
            .collect()
    }

    fn parse_index_expr(pair: pest::iterators::Pair<Rule>) -> Result<IndexExpr, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("index expression content".to_string())
        })?;

        match inner.as_rule() {
            Rule::integer => {
                let val = inner.as_str().parse::<i64>()
                    .map_err(|e| ParseError::InvalidValue(format!("Invalid integer: {}", e)))?;
                Ok(IndexExpr::Int(val))
            }
            Rule::identifier => {
                Ok(IndexExpr::Var(Self::parse_identifier(inner)?))
            }
            _ => {
                // Check if it's a colon (slice)
                if inner.as_str() == ":" {
                    Ok(IndexExpr::Slice)
                } else {
                    Err(ParseError::UnexpectedRule {
                        expected: "index expression".to_string(),
                        found: format!("{:?}", inner.as_rule()),
                    })
                }
            }
        }
    }

    // ========================================================================
    // Python Integration Parsing
    // ========================================================================

    fn parse_python_import(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
        let mut inner = pair.into_inner();

        // Parse module name
        let module = inner.next()
            .ok_or_else(|| ParseError::MissingField("python module".to_string()))?
            .as_str()
            .to_string();

        // Parse optional alias
        let alias = inner.next().map(|alias_pair| alias_pair.as_str().to_string());

        Ok(Statement::PythonImport { module, alias })
    }

    fn parse_python_call(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut inner = pair.into_inner();

        // Parse function name (string literal)
        let function = Self::parse_string_literal(inner.next().ok_or_else(|| {
            ParseError::MissingField("python function name".to_string())
        })?)?;

        // Parse optional arguments
        let args = if let Some(tensor_list) = inner.next() {
            Self::parse_tensor_list(tensor_list, registry)?
        } else {
            Vec::new()
        };

        Ok(TensorExpr::PythonCall { function, args })
    }

    // ========================================================================
    // Struct and Impl Parsing
    // ========================================================================

    fn parse_struct_decl(pair: pest::iterators::Pair<Rule>) -> Result<StructDecl, ParseError> {
        let mut inner = pair.into_inner();

        // Parse struct name
        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("struct name".to_string())
        })?)?;

        let mut type_params = Vec::new();
        let mut fields = Vec::new();

        // Parse type parameters if present, then fields
        for item in inner {
            match item.as_rule() {
                Rule::type_params => {
                    type_params = Self::parse_type_params(item)?;
                }
                Rule::field_list => {
                    fields = Self::parse_field_list(item)?;
                }
                _ => {}
            }
        }

        Ok(StructDecl {
            name,
            type_params,
            fields,
        })
    }

    fn parse_impl_block(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<ImplBlock, ParseError> {
        let mut inner = pair.into_inner();

        let mut type_params = Vec::new();
        let mut trait_name = None;
        let mut struct_type = None;
        let mut methods = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::type_params => {
                    type_params = Self::parse_type_params(item)?;
                }
                Rule::trait_impl_for => {
                    // Parse "TraitName for" part
                    let trait_id = item.into_inner().next().ok_or_else(|| {
                        ParseError::MissingField("trait name in impl for".to_string())
                    })?;
                    let trait_str = Self::parse_identifier(trait_id)?;

                    // Currently only "Drop" trait is supported
                    if trait_str.as_str() != "Drop" {
                        return Err(ParseError::InvalidValue(format!(
                            "Trait '{}' is not yet supported. Only 'Drop' trait is currently implemented.",
                            trait_str.as_str()
                        )));
                    }

                    trait_name = Some(trait_str);
                }
                Rule::struct_type => {
                    struct_type = Some(Self::parse_struct_type(item)?);
                }
                Rule::method_decl => {
                    methods.push(Self::parse_method_decl(item, registry)?);
                }
                _ => {}
            }
        }

        let struct_type = struct_type.ok_or_else(|| {
            ParseError::MissingField("struct type in impl block".to_string())
        })?;

        Ok(ImplBlock {
            type_params,
            trait_name,
            struct_type,
            methods,
        })
    }

    fn parse_type_params(pair: pest::iterators::Pair<Rule>) -> Result<Vec<TypeParam>, ParseError> {
        let mut type_params = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::type_param_list {
                for param in inner.into_inner() {
                    if param.as_rule() == Rule::identifier {
                        type_params.push(TypeParam {
                            name: Self::parse_identifier(param)?,
                        });
                    }
                }
            }
        }

        Ok(type_params)
    }

    fn parse_field_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<StructField>, ParseError> {
        let mut fields = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::field {
                fields.push(Self::parse_struct_field(inner)?);
            }
        }

        Ok(fields)
    }

    fn parse_struct_field(pair: pest::iterators::Pair<Rule>) -> Result<StructField, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("field name".to_string())
        })?)?;

        let field_type_pair = inner.next().ok_or_else(|| {
            ParseError::MissingField("field type".to_string())
        })?;

        let mut field_type = Self::parse_field_type(field_type_pair)?;

        // Check for learnable modifier
        if let Some(learnable_pair) = inner.next() {
            if learnable_pair.as_rule() == Rule::learnable {
                // Apply learnable to tensor type
                if let FieldType::Tensor(ref mut tensor_type) = field_type {
                    let learnable_str = learnable_pair.as_str();
                    tensor_type.learnable = match learnable_str {
                        "learnable" => LearnableStatus::Learnable,
                        "frozen" => LearnableStatus::Frozen,
                        _ => LearnableStatus::Default,
                    };
                }
            }
        }

        Ok(StructField { name, field_type })
    }

    fn parse_field_type(pair: pest::iterators::Pair<Rule>) -> Result<FieldType, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("field type content".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_type => Ok(FieldType::Tensor(Self::parse_tensor_type(inner)?)),
            Rule::scalar_type => Ok(FieldType::Scalar(Self::parse_scalar_type(inner)?)),
            Rule::struct_type => Ok(FieldType::Struct(Self::parse_struct_type(inner)?)),
            Rule::identifier => {
                // This is a type parameter reference
                Ok(FieldType::TypeParam(Self::parse_identifier(inner)?))
            }
            _ => Err(ParseError::UnexpectedRule {
                expected: "field type".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_struct_type(pair: pest::iterators::Pair<Rule>) -> Result<StructType, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("struct type name".to_string())
        })?)?;

        let mut type_args = Vec::new();

        // Parse type arguments if present
        if let Some(type_arg_list) = inner.next() {
            if type_arg_list.as_rule() == Rule::type_arg_list {
                for arg in type_arg_list.into_inner() {
                    if arg.as_rule() == Rule::type_arg {
                        type_args.push(Self::parse_type_arg(arg)?);
                    }
                }
            }
        }

        Ok(StructType { name, type_args })
    }

    fn parse_type_arg(pair: pest::iterators::Pair<Rule>) -> Result<TypeArg, ParseError> {
        let inner = pair.into_inner().next().ok_or_else(|| {
            ParseError::MissingField("type argument content".to_string())
        })?;

        match inner.as_rule() {
            Rule::tensor_type => Ok(TypeArg::Tensor(Self::parse_tensor_type(inner)?)),
            Rule::scalar_type => Ok(TypeArg::Scalar(Self::parse_scalar_type(inner)?)),
            Rule::struct_type => Ok(TypeArg::Struct(Box::new(Self::parse_struct_type(inner)?))),
            _ => Err(ParseError::UnexpectedRule {
                expected: "type argument".to_string(),
                found: format!("{:?}", inner.as_rule()),
            }),
        }
    }

    fn parse_method_decl(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<MethodDecl, ParseError> {
        let mut inner = pair.into_inner();

        // Parse method name
        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("method name".to_string())
        })?)?;

        let mut params = Vec::new();
        let mut return_type = ReturnType::Void;
        let mut body = Vec::new();

        for item in inner {
            match item.as_rule() {
                Rule::method_param_list => {
                    params = Self::parse_method_param_list(item)?;
                }
                Rule::return_type => {
                    return_type = Self::parse_return_type(item)?;
                }
                Rule::statement => {
                    body.push(Self::parse_statement(item, registry)?);
                }
                _ => {}
            }
        }

        Ok(MethodDecl {
            name,
            params,
            return_type,
            body,
        })
    }

    fn parse_method_param_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<MethodParam>, ParseError> {
        let mut params = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::method_param => {
                    let param_inner = inner.into_inner().next().ok_or_else(|| {
                        ParseError::MissingField("method param content".to_string())
                    })?;

                    if param_inner.as_str() == "self" {
                        params.push(MethodParam::SelfParam);
                    } else {
                        params.push(MethodParam::Regular(Self::parse_param(param_inner)?));
                    }
                }
                _ => {}
            }
        }

        Ok(params)
    }

    fn parse_struct_literal(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut inner = pair.into_inner();

        let struct_type = Self::parse_struct_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("struct type in literal".to_string())
        })?)?;

        let mut fields = Vec::new();

        if let Some(field_init_list) = inner.next() {
            if field_init_list.as_rule() == Rule::field_init_list {
                for field_init in field_init_list.into_inner() {
                    if field_init.as_rule() == Rule::field_init {
                        fields.push(Self::parse_field_init(field_init, registry)?);
                    }
                }
            }
        }

        Ok(TensorExpr::StructLiteral {
            struct_type,
            fields,
        })
    }

    fn parse_associated_call(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<TensorExpr, ParseError> {
        let mut inner = pair.into_inner();

        let struct_type = Self::parse_struct_type(inner.next().ok_or_else(|| {
            ParseError::MissingField("struct type in associated call".to_string())
        })?)?;

        let function = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("function name in associated call".to_string())
        })?)?;

        let args = if let Some(tensor_list) = inner.next() {
            Self::parse_tensor_list(tensor_list, registry)?
        } else {
            Vec::new()
        };

        Ok(TensorExpr::AssociatedCall {
            struct_type,
            function,
            args,
        })
    }

    fn parse_field_init(pair: pest::iterators::Pair<Rule>, registry: &FunctionRegistry) -> Result<FieldInit, ParseError> {
        let mut inner = pair.into_inner();

        let name = Self::parse_identifier(inner.next().ok_or_else(|| {
            ParseError::MissingField("field name in field init".to_string())
        })?)?;

        let value = Self::parse_tensor_expr(inner.next().ok_or_else(|| {
            ParseError::MissingField("field value in field init".to_string())
        })?, registry)?;

        Ok(FieldInit { name, value })
    }
}

#[cfg(test)]
mod tests;
