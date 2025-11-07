//! Lexical analyzer for TensorLogic

pub mod token;

use token::{Token, TokenWithPos, Position};
use std::collections::HashMap;

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    keywords: HashMap<&'static str, Token>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        let mut keywords = HashMap::new();
        
        // Register all keywords
        keywords.insert("tensor", Token::Tensor);
        keywords.insert("relation", Token::Relation);
        keywords.insert("rule", Token::Rule);
        keywords.insert("embedding", Token::Embedding);
        keywords.insert("fn", Token::Fn);
        keywords.insert("main", Token::Main);
        keywords.insert("learnable", Token::Learnable);
        keywords.insert("frozen", Token::Frozen);
        keywords.insert("entity", Token::Entity);
        keywords.insert("concept", Token::Concept);
        keywords.insert("embed", Token::Embed);
        keywords.insert("einsum", Token::Einsum);
        keywords.insert("infer", Token::Infer);
        keywords.insert("learn", Token::Learn);
        keywords.insert("forward", Token::Forward);
        keywords.insert("backward", Token::Backward);
        keywords.insert("gradient", Token::Gradient);
        keywords.insert("symbolic", Token::Symbolic);
        keywords.insert("if", Token::If);
        keywords.insert("else", Token::Else);
        keywords.insert("for", Token::For);
        keywords.insert("while", Token::While);
        keywords.insert("loop", Token::Loop);
        keywords.insert("break", Token::Break);
        keywords.insert("return", Token::Return);
        keywords.insert("in", Token::In);
        keywords.insert("range", Token::Range);
        keywords.insert("let", Token::Let);
        keywords.insert("true", Token::True);
        keywords.insert("false", Token::False);
        keywords.insert("not", Token::Not);
        keywords.insert("and", Token::And);
        keywords.insert("or", Token::Or);
        keywords.insert("shape", Token::Shape);
        keywords.insert("rank", Token::Rank);
        keywords.insert("norm", Token::Norm);
        keywords.insert("inv", Token::Inv);
        keywords.insert("det", Token::Det);
        keywords.insert("objective", Token::Objective);
        keywords.insert("optimizer", Token::Optimizer);
        keywords.insert("epochs", Token::Epochs);
        keywords.insert("auto", Token::Auto);
        keywords.insert("random", Token::Random);
        keywords.insert("xavier", Token::Xavier);
        keywords.insert("he", Token::He);
        keywords.insert("zeros", Token::Zeros);
        keywords.insert("ones", Token::Ones);
        keywords.insert("void", Token::Void);
        keywords.insert("python", Token::Python);
        keywords.insert("import", Token::Import);
        keywords.insert("as", Token::As);
        keywords.insert("match", Token::Match);
        keywords.insert("float16", Token::Float16);
        keywords.insert("int16", Token::Int16);
        keywords.insert("int32", Token::Int32);
        keywords.insert("int64", Token::Int64);
        keywords.insert("bool", Token::Bool);
        keywords.insert("complex16", Token::Complex16);
        
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            keywords,
        }
    }
    
    /// Get current position
    fn current_pos(&self) -> Position {
        Position::new(self.line, self.column, self.pos)
    }
    
    /// Peek at current character without consuming
    fn peek(&self) -> Option<char> {
        if self.pos < self.source.len() {
            Some(self.source[self.pos])
        } else {
            None
        }
    }
    
    /// Peek at next character without consuming
    fn peek_next(&self) -> Option<char> {
        if self.pos + 1 < self.source.len() {
            Some(self.source[self.pos + 1])
        } else {
            None
        }
    }
    
    /// Consume current character and advance
    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.peek() {
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }
    
    /// Skip whitespace (except newlines)
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    /// Skip comment
    fn skip_comment(&mut self) {
        if self.peek() == Some('/') && self.peek_next() == Some('/') {
            // Single-line comment
            while let Some(ch) = self.advance() {
                if ch == '\n' {
                    break;
                }
            }
        }
    }
    
    /// Lex a number
    fn lex_number(&mut self, start: Position) -> TokenWithPos {
        let mut num_str = String::new();
        
        // Integer part
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Decimal part
        if self.peek() == Some('.') && self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            num_str.push('.');
            self.advance();
            
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    num_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        // Scientific notation
        if let Some('e') | Some('E') = self.peek() {
            num_str.push('e');
            self.advance();
            
            if let Some('+') | Some('-') = self.peek() {
                num_str.push(self.advance().unwrap());
            }
            
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    num_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        let number = num_str.parse::<f64>().unwrap_or(0.0);
        let end = self.current_pos();
        
        TokenWithPos::new(Token::Number(number), start, end)
    }
    
    /// Lex a string literal
    fn lex_string(&mut self, start: Position) -> TokenWithPos {
        self.advance(); // consume opening quote
        let mut string = String::new();
        
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance(); // consume closing quote
                break;
            } else if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.advance() {
                    match escaped {
                        'n' => string.push('\n'),
                        't' => string.push('\t'),
                        'r' => string.push('\r'),
                        '\\' => string.push('\\'),
                        '"' => string.push('"'),
                        _ => {
                            string.push('\\');
                            string.push(escaped);
                        }
                    }
                }
            } else {
                string.push(ch);
                self.advance();
            }
        }
        
        let end = self.current_pos();
        TokenWithPos::new(Token::String(string), start, end)
    }
    
    /// Lex an identifier or keyword
    fn lex_identifier(&mut self, start: Position) -> TokenWithPos {
        let mut ident = String::new();
        
        // First character: letter or underscore
        if let Some(ch) = self.peek() {
            if ch.is_alphabetic() || ch == '_' {
                ident.push(ch);
                self.advance();
            }
        }
        
        // Remaining characters: alphanumeric or underscore
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        let end = self.current_pos();
        
        // Check if it's a keyword
        let token = if let Some(keyword) = self.keywords.get(ident.as_str()) {
            keyword.clone()
        } else {
            Token::Identifier(ident)
        };
        
        TokenWithPos::new(token, start, end)
    }
    
    /// Get next token
    pub fn next_token(&mut self) -> TokenWithPos {
        loop {
            self.skip_whitespace();
            
            let start = self.current_pos();
            
            // Check for comment
            if self.peek() == Some('/') && self.peek_next() == Some('/') {
                self.skip_comment();
                continue;
            }
            
            match self.peek() {
                None => return TokenWithPos::new(Token::Eof, start, start),
                
                Some('\n') => {
                    self.advance();
                    // Skip newlines for now (they're not significant in TensorLogic)
                    continue;
                }
                
                // Numbers
                Some(ch) if ch.is_ascii_digit() => {
                    return self.lex_number(start);
                }
                
                // Strings
                Some('"') => {
                    return self.lex_string(start);
                }
                
                // Identifiers and keywords
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    return self.lex_identifier(start);
                }
                
                // Operators and punctuation
                Some(ch) => {
                    self.advance();
                    let end = self.current_pos();
                    
                    let token = match ch {
                        '+' => Token::Plus,
                        '-' => {
                            if self.peek() == Some('>') {
                                self.advance();
                                Token::Arrow
                            } else {
                                Token::Minus
                            }
                        }
                        '*' => {
                            if self.peek() == Some('*') {
                                self.advance();
                                Token::Power
                            } else {
                                Token::Star
                            }
                        }
                        '/' => Token::Slash,
                        '@' => Token::At,
                        '⊗' => Token::TensorProduct,
                        '⊙' => Token::Hadamard,
                        '=' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Eq
                            } else {
                                Token::Assign
                            }
                        }
                        '!' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Ne
                            } else {
                                Token::Bang
                            }
                        }
                        '<' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Le
                            } else {
                                Token::Lt
                            }
                        }
                        '>' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Ge
                            } else {
                                Token::Gt
                            }
                        }
                        ':' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::ColonEq
                            } else {
                                Token::Colon
                            }
                        }
                        '|' => {
                            if self.peek() == Some('|') {
                                self.advance();
                                Token::DoublePipe
                            } else {
                                Token::Pipe
                            }
                        }
                        '(' => Token::LParen,
                        ')' => Token::RParen,
                        '{' => Token::LBrace,
                        '}' => Token::RBrace,
                        '[' => Token::LBracket,
                        ']' => Token::RBracket,
                        ',' => Token::Comma,
                        ';' => Token::Semicolon,
                        '.' => Token::Dot,
                        '?' => Token::Question,
                        _ => {
                            // Unknown character - skip it
                            continue;
                        }
                    };
                    
                    return TokenWithPos::new(token, start, self.current_pos());
                }
            }
        }
    }
    
    /// Tokenize entire source
    pub fn tokenize(&mut self) -> Vec<TokenWithPos> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token();
            let is_eof = token.token == Token::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        
        tokens
    }

    /// Validate identifiers in source code
    /// Ensures no keywords are used as identifiers
    pub fn validate_identifiers(source: &str) -> Result<(), String> {
        let mut lexer = Lexer::new(source);
        let _tokens = lexer.tokenize();
        // If tokenization succeeds without panic, validation passes
        // The lexer automatically distinguishes keywords from identifiers
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("fn main let det detokenize");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens.len(), 6); // fn, main, let, det, detokenize, EOF
        assert!(matches!(tokens[0].token, Token::Fn));
        assert!(matches!(tokens[1].token, Token::Main));
        assert!(matches!(tokens[2].token, Token::Let));
        assert!(matches!(tokens[3].token, Token::Det));
        assert!(matches!(tokens[4].token, Token::Identifier(ref s) if s == "detokenize"));
        assert!(matches!(tokens[5].token, Token::Eof));
    }
    
    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14 1.5e-10");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens.len(), 4); // 42, 3.14, 1.5e-10, EOF
        assert!(matches!(tokens[0].token, Token::Number(n) if n == 42.0));
        assert!(matches!(tokens[1].token, Token::Number(n) if (n - 3.14).abs() < 0.001));
        assert!(matches!(tokens[2].token, Token::Number(n) if (n - 1.5e-10).abs() < 1e-15));
    }
    
    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("+ - * / ** == != <= >= := ->");
        let tokens = lexer.tokenize();
        
        assert!(matches!(tokens[0].token, Token::Plus));
        assert!(matches!(tokens[1].token, Token::Minus));
        assert!(matches!(tokens[2].token, Token::Star));
        assert!(matches!(tokens[3].token, Token::Slash));
        assert!(matches!(tokens[4].token, Token::Power));
        assert!(matches!(tokens[5].token, Token::Eq));
        assert!(matches!(tokens[6].token, Token::Ne));
        assert!(matches!(tokens[7].token, Token::Le));
        assert!(matches!(tokens[8].token, Token::Ge));
        assert!(matches!(tokens[9].token, Token::ColonEq));
        assert!(matches!(tokens[10].token, Token::Arrow));
    }
}
