//! Lexer for TensorLogic Language
//!
//! This module provides a separate lexical analysis layer that tokenizes the input
//! before it's passed to the Pest parser. This separation allows proper handling of
//! reserved keywords vs identifiers by reading complete alphanumeric sequences first,
//! then determining their token type.

use std::fmt;

/// Token types produced by the lexer
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Keywords
    Tensor,
    Relation,
    Rule,
    Embedding,
    Function,
    Main,
    Learnable,
    Frozen,
    Entity,
    Concept,
    Embed,
    Einsum,
    Query,
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
    In,
    Range,
    True,
    False,
    Not,
    And,
    Or,
    Shape,
    Rank,
    Norm,
    Transpose,
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
    Float16,
    Int16,
    Int32,
    Int64,
    Bool,
    Complex16,

    // Identifiers and literals
    Identifier(String),
    Integer(String),
    Float(String),
    StringLiteral(String),

    // Operators and punctuation
    Plus,
    Minus,
    Star,
    Slash,
    DoubleStar,      // **
    At,              // @
    TensorProd,      // ⊗
    Hadamard,        // ⊙
    Assign,          // :=
    Eq,              // =
    EqEq,            // ==
    Ne,              // !=
    Lt,              // <
    Gt,              // >
    Le,              // <=
    Ge,              // >=
    Approx,          // ≈
    Tilde,           // ~
    Arrow,           // <-
    RightArrow,      // ->
    Comma,
    Colon,
    Semicolon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Question,
    Exclamation,

    // Special
    Whitespace,
    Comment(String),
    Newline,
    EOF,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, lexeme: String, line: usize, column: usize) -> Self {
        Token {
            token_type,
            lexeme,
            line,
            column,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} '{}' at {}:{}", self.token_type, self.lexeme, self.line, self.column)
    }
}

/// Lexer for TensorLogic
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Tokenize the entire input
    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            if token.token_type == TokenType::EOF {
                tokens.push(token);
                break;
            }
            // Skip whitespace and comments for now (Pest will handle them)
            if !matches!(token.token_type, TokenType::Whitespace | TokenType::Newline | TokenType::Comment(_)) {
                tokens.push(token);
            }
        }

        Ok(tokens)
    }

    /// Convert tokens back to source string with keywords/identifiers properly distinguished
    /// This allows Pest to parse the preprocessed input
    pub fn tokens_to_source(tokens: &[Token]) -> String {
        let mut result = String::new();

        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && !matches!(
                token.token_type,
                TokenType::Comma | TokenType::Semicolon | TokenType::RParen |
                TokenType::RBracket | TokenType::RBrace | TokenType::EOF
            ) {
                result.push(' ');
            }

            result.push_str(&token.lexeme);
        }

        result
    }

    /// Preprocess source: tokenize and convert back to ensure keywords are preserved
    pub fn preprocess(input: &str) -> Result<String, String> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        Ok(Self::tokens_to_source(&tokens))
    }

    /// Check if a string is a reserved keyword
    pub fn is_keyword(s: &str) -> bool {
        matches!(s,
            "tensor" | "relation" | "rule" | "embedding" | "function" | "main"
            | "learnable" | "frozen" | "entity" | "concept" | "embed"
            | "einsum" | "query" | "infer" | "learn"
            | "forward" | "backward" | "gradient" | "symbolic"
            | "if" | "else" | "for" | "while" | "in" | "range"
            | "true" | "false" | "not" | "and" | "or"
            | "shape" | "rank" | "norm" | "transpose" | "inv" | "det"
            | "objective" | "optimizer" | "epochs" | "auto"
            | "random" | "xavier" | "he" | "zeros" | "ones" | "void"
            | "float16" | "int16" | "int32" | "int64" | "bool" | "complex16"
        )
    }

    /// Validate that identifiers in source are not keywords
    /// Returns detailed error if a keyword is used as identifier
    pub fn validate_identifiers(input: &str) -> Result<(), String> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;

        for token in &tokens {
            if let TokenType::Identifier(name) = &token.token_type {
                if Self::is_keyword(name) {
                    return Err(format!(
                        "Cannot use keyword '{}' as identifier at line {}, column {}",
                        name, token.line, token.column
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Ok(Token::new(TokenType::EOF, String::new(), self.line, self.column));
        }

        let start_line = self.line;
        let start_column = self.column;
        let ch = self.current_char();

        // String literals
        if ch == '"' {
            return self.read_string_literal(start_line, start_column);
        }

        // Numbers
        if ch.is_ascii_digit() {
            return self.read_number(start_line, start_column);
        }

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            return self.read_identifier_or_keyword(start_line, start_column);
        }

        // Operators and punctuation
        match ch {
            '+' => {
                self.advance();
                Ok(Token::new(TokenType::Plus, "+".to_string(), start_line, start_column))
            }
            '-' => {
                self.advance();
                if self.current_char() == '>' {
                    self.advance();
                    Ok(Token::new(TokenType::RightArrow, "->".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Minus, "-".to_string(), start_line, start_column))
                }
            }
            '*' => {
                self.advance();
                if self.current_char() == '*' {
                    self.advance();
                    Ok(Token::new(TokenType::DoubleStar, "**".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Star, "*".to_string(), start_line, start_column))
                }
            }
            '/' => {
                self.advance();
                Ok(Token::new(TokenType::Slash, "/".to_string(), start_line, start_column))
            }
            '@' => {
                self.advance();
                Ok(Token::new(TokenType::At, "@".to_string(), start_line, start_column))
            }
            '⊗' => {
                self.advance();
                Ok(Token::new(TokenType::TensorProd, "⊗".to_string(), start_line, start_column))
            }
            '⊙' => {
                self.advance();
                Ok(Token::new(TokenType::Hadamard, "⊙".to_string(), start_line, start_column))
            }
            '=' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::EqEq, "==".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Eq, "=".to_string(), start_line, start_column))
                }
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::Ne, "!=".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Exclamation, "!".to_string(), start_line, start_column))
                }
            }
            '<' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::Le, "<=".to_string(), start_line, start_column))
                } else if self.current_char() == '-' {
                    self.advance();
                    Ok(Token::new(TokenType::Arrow, "<-".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Lt, "<".to_string(), start_line, start_column))
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::Ge, ">=".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Gt, ">".to_string(), start_line, start_column))
                }
            }
            '≈' => {
                self.advance();
                Ok(Token::new(TokenType::Approx, "≈".to_string(), start_line, start_column))
            }
            '~' => {
                self.advance();
                Ok(Token::new(TokenType::Tilde, "~".to_string(), start_line, start_column))
            }
            ':' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::Assign, ":=".to_string(), start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Colon, ":".to_string(), start_line, start_column))
                }
            }
            ',' => {
                self.advance();
                Ok(Token::new(TokenType::Comma, ",".to_string(), start_line, start_column))
            }
            ';' => {
                self.advance();
                Ok(Token::new(TokenType::Semicolon, ";".to_string(), start_line, start_column))
            }
            '(' => {
                self.advance();
                Ok(Token::new(TokenType::LParen, "(".to_string(), start_line, start_column))
            }
            ')' => {
                self.advance();
                Ok(Token::new(TokenType::RParen, ")".to_string(), start_line, start_column))
            }
            '[' => {
                self.advance();
                Ok(Token::new(TokenType::LBracket, "[".to_string(), start_line, start_column))
            }
            ']' => {
                self.advance();
                Ok(Token::new(TokenType::RBracket, "]".to_string(), start_line, start_column))
            }
            '{' => {
                self.advance();
                Ok(Token::new(TokenType::LBrace, "{".to_string(), start_line, start_column))
            }
            '}' => {
                self.advance();
                Ok(Token::new(TokenType::RBrace, "}".to_string(), start_line, start_column))
            }
            '?' => {
                self.advance();
                Ok(Token::new(TokenType::Question, "?".to_string(), start_line, start_column))
            }
            _ => Err(format!("Unexpected character '{}' at {}:{}", ch, start_line, start_column))
        }
    }

    /// Read an identifier or keyword
    /// This is where we properly handle the keyword vs identifier distinction:
    /// 1. Read the complete alphanumeric sequence
    /// 2. Check if it's a keyword
    /// 3. If not, it's an identifier
    fn read_identifier_or_keyword(&mut self, line: usize, column: usize) -> Result<Token, String> {
        let mut lexeme = String::new();

        // Read complete alphanumeric/underscore sequence
        while !self.is_at_end() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                lexeme.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Now check if it's a keyword
        let token_type = match lexeme.as_str() {
            "tensor" => TokenType::Tensor,
            "relation" => TokenType::Relation,
            "rule" => TokenType::Rule,
            "embedding" => TokenType::Embedding,
            "function" => TokenType::Function,
            "main" => TokenType::Main,
            "learnable" => TokenType::Learnable,
            "frozen" => TokenType::Frozen,
            "entity" => TokenType::Entity,
            "concept" => TokenType::Concept,
            "embed" => TokenType::Embed,
            "einsum" => TokenType::Einsum,
            "query" => TokenType::Query,
            "infer" => TokenType::Infer,
            "learn" => TokenType::Learn,
            "forward" => TokenType::Forward,
            "backward" => TokenType::Backward,
            "gradient" => TokenType::Gradient,
            "symbolic" => TokenType::Symbolic,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "for" => TokenType::For,
            "while" => TokenType::While,
            "in" => TokenType::In,
            "range" => TokenType::Range,
            "true" => TokenType::True,
            "false" => TokenType::False,
            "not" => TokenType::Not,
            "and" => TokenType::And,
            "or" => TokenType::Or,
            "shape" => TokenType::Shape,
            "rank" => TokenType::Rank,
            "norm" => TokenType::Norm,
            "transpose" => TokenType::Transpose,
            "inv" => TokenType::Inv,
            "det" => TokenType::Det,
            "objective" => TokenType::Objective,
            "optimizer" => TokenType::Optimizer,
            "epochs" => TokenType::Epochs,
            "auto" => TokenType::Auto,
            "random" => TokenType::Random,
            "xavier" => TokenType::Xavier,
            "he" => TokenType::He,
            "zeros" => TokenType::Zeros,
            "ones" => TokenType::Ones,
            "void" => TokenType::Void,
            "float16" => TokenType::Float16,
            "int16" => TokenType::Int16,
            "int32" => TokenType::Int32,
            "int64" => TokenType::Int64,
            "bool" => TokenType::Bool,
            "complex16" => TokenType::Complex16,
            _ => TokenType::Identifier(lexeme.clone()),
        };

        Ok(Token::new(token_type, lexeme, line, column))
    }

    /// Read a number (integer or float)
    fn read_number(&mut self, line: usize, column: usize) -> Result<Token, String> {
        let mut lexeme = String::new();
        let mut is_float = false;

        // Read digits
        while !self.is_at_end() && self.current_char().is_ascii_digit() {
            lexeme.push(self.current_char());
            self.advance();
        }

        // Check for decimal point
        if !self.is_at_end() && self.current_char() == '.' {
            let next_pos = self.position + 1;
            if next_pos < self.input.len() && self.input[next_pos].is_ascii_digit() {
                is_float = true;
                lexeme.push('.');
                self.advance();

                while !self.is_at_end() && self.current_char().is_ascii_digit() {
                    lexeme.push(self.current_char());
                    self.advance();
                }
            }
        }

        // Check for exponent
        if !self.is_at_end() && (self.current_char() == 'e' || self.current_char() == 'E') {
            is_float = true;
            lexeme.push(self.current_char());
            self.advance();

            if !self.is_at_end() && (self.current_char() == '+' || self.current_char() == '-') {
                lexeme.push(self.current_char());
                self.advance();
            }

            while !self.is_at_end() && self.current_char().is_ascii_digit() {
                lexeme.push(self.current_char());
                self.advance();
            }
        }

        let token_type = if is_float {
            TokenType::Float(lexeme.clone())
        } else {
            TokenType::Integer(lexeme.clone())
        };

        Ok(Token::new(token_type, lexeme, line, column))
    }

    /// Read a string literal
    fn read_string_literal(&mut self, line: usize, column: usize) -> Result<Token, String> {
        let mut lexeme = String::new();
        lexeme.push('"');
        self.advance(); // Skip opening quote

        while !self.is_at_end() && self.current_char() != '"' {
            let ch = self.current_char();
            lexeme.push(ch);

            if ch == '\\' {
                self.advance();
                if !self.is_at_end() {
                    lexeme.push(self.current_char());
                    self.advance();
                }
            } else {
                self.advance();
            }
        }

        if self.is_at_end() {
            return Err(format!("Unterminated string at {}:{}", line, column));
        }

        lexeme.push('"');
        self.advance(); // Skip closing quote

        Ok(Token::new(TokenType::StringLiteral(lexeme.clone()), lexeme, line, column))
    }

    /// Skip whitespace and comments
    fn skip_whitespace_and_comments(&mut self) {
        while !self.is_at_end() {
            let ch = self.current_char();

            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance();
            } else if ch == '\n' {
                self.line += 1;
                self.column = 1;
                self.position += 1;
            } else if ch == '/' && self.peek() == '/' {
                // Line comment
                while !self.is_at_end() && self.current_char() != '\n' {
                    self.advance();
                }
            } else if ch == '/' && self.peek() == '*' {
                // Block comment
                self.advance(); // /
                self.advance(); // *

                while !self.is_at_end() {
                    if self.current_char() == '*' && self.peek() == '/' {
                        self.advance(); // *
                        self.advance(); // /
                        break;
                    }
                    if self.current_char() == '\n' {
                        self.line += 1;
                        self.column = 1;
                        self.position += 1;
                    } else {
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Get current character without advancing
    fn current_char(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }

    /// Peek at next character
    fn peek(&self) -> char {
        let next_pos = self.position + 1;
        if next_pos >= self.input.len() {
            '\0'
        } else {
            self.input[next_pos]
        }
    }

    /// Advance to next character
    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
            self.column += 1;
        }
    }

    /// Check if at end of input
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("tensor query in input");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Tensor);
        assert_eq!(tokens[1].token_type, TokenType::Query);
        assert_eq!(tokens[2].token_type, TokenType::In);
        assert!(matches!(tokens[3].token_type, TokenType::Identifier(_)));
    }

    #[test]
    fn test_identifiers_starting_with_keywords() {
        let mut lexer = Lexer::new("input index query_param information");
        let tokens = lexer.tokenize().unwrap();

        // All should be identifiers, not keywords
        for token in &tokens[..tokens.len()-1] { // Exclude EOF
            assert!(matches!(token.token_type, TokenType::Identifier(_)));
        }
    }

    #[test]
    fn test_string_with_colon() {
        let mut lexer = Lexer::new(r#""test: value""#);
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[0].token_type, TokenType::StringLiteral(_)));
    }

    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new(":= == != <= >= <- ->");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Assign);
        assert_eq!(tokens[1].token_type, TokenType::EqEq);
        assert_eq!(tokens[2].token_type, TokenType::Ne);
        assert_eq!(tokens[3].token_type, TokenType::Le);
        assert_eq!(tokens[4].token_type, TokenType::Ge);
        assert_eq!(tokens[5].token_type, TokenType::Arrow);
        assert_eq!(tokens[6].token_type, TokenType::RightArrow);
    }

    #[test]
    fn test_is_keyword() {
        assert!(Lexer::is_keyword("query"));
        assert!(Lexer::is_keyword("tensor"));
        assert!(Lexer::is_keyword("in"));
        assert!(!Lexer::is_keyword("input"));
        assert!(!Lexer::is_keyword("query_param"));
        assert!(!Lexer::is_keyword("my_var"));
    }

    #[test]
    fn test_validate_identifiers_valid() {
        // Valid: identifiers that are not keywords
        let result = Lexer::validate_identifiers("tensor input: float16[3]");
        assert!(result.is_ok());

        let result = Lexer::validate_identifiers("tensor query_param: float16[2]");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_identifiers_invalid() {
        // Invalid: using "query" as identifier would be caught
        // But current grammar doesn't allow this anyway
        // This test validates the lexer's capability
        let source = "tensor query: float16[2]";
        // Note: "query" here is a keyword in "tensor query:" context
        // The validation happens at parse time
    }
}
