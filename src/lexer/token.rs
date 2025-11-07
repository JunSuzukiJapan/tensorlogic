//! Token definitions for the TensorLogic lexer

use std::fmt;

/// Position in source code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Token with position information
#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithPos {
    pub token: Token,
    pub start: Position,
    pub end: Position,
}

impl TokenWithPos {
    pub fn new(token: Token, start: Position, end: Position) -> Self {
        Self { token, start, end }
    }
}

/// Token types
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Number(f64),
    String(String),
    Identifier(String),
    
    // Keywords
    Tensor,
    Relation,
    Rule,
    Embedding,
    Fn,
    Main,
    Learnable,
    Frozen,
    Entity,
    Concept,
    Embed,
    Einsum,
    Infer,
    Learn,
    Forward,
    Backward,
    Gradient,
    Symbolic,
    If,
    Else,
    For,
    While,
    Loop,
    Break,
    Return,
    In,
    Range,
    Let,
    True,
    False,
    Not,
    And,
    Or,
    Shape,
    Rank,
    Norm,
    Inv,
    Det,
    Objective,
    Optimizer,
    Epochs,
    Auto,
    Random,
    Xavier,
    He,
    Zeros,
    Ones,
    Void,
    Python,
    Import,
    As,
    Match,

    // Type keywords
    Float16,
    Int16,
    Int32,
    Int64,
    Bool,
    Complex16,
    
    // Operators
    Plus,           // +
    Minus,          // -
    Star,           // *
    Slash,          // /
    Power,          // **
    At,             // @
    TensorProduct,  // ⊗
    Hadamard,       // ⊙
    
    // Comparison
    Eq,             // ==
    Ne,             // !=
    Lt,             // <
    Le,             // <=
    Gt,             // >
    Ge,             // >=
    
    // Assignment
    Assign,         // =
    ColonEq,        // :=
    
    // Logical
    Bang,           // !
    Pipe,           // |
    DoublePipe,     // ||

    // Delimiters
    LParen,         // (
    RParen,         // )
    LBrace,         // {
    RBrace,         // }
    LBracket,       // [
    RBracket,       // ]
    
    // Punctuation
    Comma,          // ,
    Colon,          // :
    Semicolon,      // ;
    Dot,            // .
    Question,       // ?
    Arrow,          // ->
    
    // Special
    Eof,
    Newline,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Number(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Identifier(id) => write!(f, "{}", id),
            Token::Tensor => write!(f, "tensor"),
            Token::Fn => write!(f, "fn"),
            Token::Main => write!(f, "main"),
            Token::Let => write!(f, "let"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::Return => write!(f, "return"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Power => write!(f, "**"),
            Token::At => write!(f, "@"),
            Token::Eq => write!(f, "=="),
            Token::Ne => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Le => write!(f, "<="),
            Token::Gt => write!(f, ">"),
            Token::Ge => write!(f, ">="),
            Token::Assign => write!(f, "="),
            Token::ColonEq => write!(f, ":="),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Arrow => write!(f, "->"),
            Token::Eof => write!(f, "EOF"),
            _ => write!(f, "{:?}", self),
        }
    }
}
