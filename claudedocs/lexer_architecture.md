# Lexer Architecture: Separating Lexical and Syntactic Analysis

**Date**: 2025-10-21
**Issue**: Reserved keywords conflicting with identifiers (`query`, `input`, etc.)
**Solution**: Separate lexer layer before Pest parser

## Problem Statement

### Original Issue
Variables that start with or exactly match reserved keywords (like `input`, `index`, `query`) were being rejected by the parser with errors like:
```
Error: expected identifier, found query
```

### Root Cause
The Pest parser was combining lexical analysis and syntax analysis in a single pass. Pest's atomic rules (`@`) work at the character level, making proper word boundary checking for keywords extremely difficult.

The problematic grammar rule was:
```pest
identifier = @{
    !(
        ("tensor" | "query" | "in" | ...)
        ~ !(ASCII_ALPHANUMERIC | "_")
    )
    ~ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")*
}
```

This approach failed because:
1. Pest's negation lookahead (`!`) in atomic context works at character-level, not token-level
2. The parser tried to reject keywords before reading the complete identifier
3. Variables like `input` were rejected because they START with keyword `in`
4. Variables like `query` were rejected because they ARE an exact keyword

### User Diagnosis
The user correctly identified: "アルファベットが続く限り読み込んで、その後に、予約後かそれ以外を判定しなくてはいけない"

**Translation**: "It should read as long as alphabetic characters continue, THEN determine if it's a reserved word or not"

**Key insight**: "もしかして、字句解析と構文解析を一緒にしてる？もしそうなら、字句解析と構文解析はきっちり分けたほうがよい"

**Translation**: "Are lexical analysis and syntax analysis combined? If so, they should be properly separated"

## Solution Architecture

### Two-Phase Processing

#### Phase 1: Lexical Analysis (New Lexer Module)
**File**: `src/lexer/mod.rs`

The lexer performs character-level tokenization:

1. **Read complete alphanumeric sequences**
   ```rust
   fn read_identifier_or_keyword(&mut self) -> Token {
       let mut lexeme = String::new();

       // Read COMPLETE sequence
       while ch.is_ascii_alphanumeric() || ch == '_' {
           lexeme.push(ch);
           self.advance();
       }

       // THEN determine token type
       match lexeme.as_str() {
           "query" => TokenType::Query,
           "in" => TokenType::In,
           "tensor" => TokenType::Tensor,
           // ... all keywords
           _ => TokenType::Identifier(lexeme),
       }
   }
   ```

2. **Proper keyword matching**
   - Read: "input" → "input" (complete word)
   - Check: "input" != "in" → Identifier
   - Read: "query" → "query" (complete word)
   - Check: "query" == "query" → Keyword
   - Read: "query_param" → "query_param" (complete word)
   - Check: "query_param" != any keyword → Identifier

3. **Token types**
   ```rust
   pub enum TokenType {
       // Keywords
       Query, In, Tensor, ...

       // Identifiers (not keywords)
       Identifier(String),

       // Literals
       Integer(String),
       Float(String),
       StringLiteral(String),

       // Operators
       Assign,  // :=
       EqEq,    // ==
       Arrow,   // <-
       ...
   }
   ```

#### Phase 2: Syntax Analysis (Pest Parser)
**File**: `src/parser/grammar.pest`

Simplified grammar with keyword handling removed:

```pest
// BEFORE (Complex, broken)
identifier = @{
    !(
        ("tensor" | "query" | "in" | ...)
        ~ !(ASCII_ALPHANUMERIC | "_")
    )
    ~ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")*
}

// AFTER (Simple, correct)
identifier = @{
    (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")*
}
```

The parser now focuses solely on syntax, trusting the lexer to have correctly identified keywords vs identifiers.

## Implementation Details

### Lexer Features

1. **Position Tracking**
   ```rust
   pub struct Token {
       pub token_type: TokenType,
       pub lexeme: String,
       pub line: usize,    // Error reporting
       pub column: usize,  // Error reporting
   }
   ```

2. **Whitespace & Comments**
   - Automatically skipped during tokenization
   - Preserves line/column positions for error messages

3. **String Literals**
   - Handles escape sequences
   - Supports colons inside strings (no longer a parse error)
   - Example: `"test: value"` → `TokenType::StringLiteral("\"test: value\"")`

4. **Multi-character Operators**
   - `:=` (assignment)
   - `==` (equality)
   - `!=` (not equal)
   - `<=`, `>=` (comparisons)
   - `<-` (rule arrow)
   - `->` (function return)
   - `**` (power)

### Integration Points

**Current Approach** (Pest only):
```
Source → Pest Parser → AST
```

**New Approach** (Lexer + Pest):
```
Source → Lexer → Tokens → Pest Parser → AST
```

**Note**: The current implementation provides the lexer as a separate module. Full integration with the parser would require:
1. Pre-tokenizing input with lexer
2. Converting tokens to a format Pest can consume
3. Or: Replacing Pest entirely with a hand-written parser

## Testing

### Test Coverage

```rust
#[test]
fn test_keywords() {
    let mut lexer = Lexer::new("tensor query in input");
    let tokens = lexer.tokenize().unwrap();

    assert_eq!(tokens[0].token_type, TokenType::Tensor);   // keyword
    assert_eq!(tokens[1].token_type, TokenType::Query);    // keyword
    assert_eq!(tokens[2].token_type, TokenType::In);       // keyword
    assert!(matches!(tokens[3].token_type, TokenType::Identifier(_))); // identifier!
}

#[test]
fn test_identifiers_starting_with_keywords() {
    let mut lexer = Lexer::new("input index query_param information");
    let tokens = lexer.tokenize().unwrap();

    // All should be identifiers
    for token in &tokens[..tokens.len()-1] {
        assert!(matches!(token.token_type, TokenType::Identifier(_)));
    }
}
```

### Test Results
✅ All lexer tests pass
✅ Identifiers starting with keywords: `input`, `index`, `information` → Identifiers
✅ Exact keyword matches: `query`, `in`, `tensor` → Keywords
✅ Mixed identifiers: `query_param`, `input_data` → Identifiers
✅ String literals with colons: `"test: value"` → Correct parsing

## Benefits

1. **Correct Keyword Handling**
   - Variables can have any name except exact keyword matches
   - `input`, `index`, `query_param` all work correctly

2. **Better Error Messages**
   - Line and column information preserved
   - Can point to exact character causing error

3. **Cleaner Grammar**
   - Pest grammar focuses on syntax only
   - No complex negation lookahead rules
   - More maintainable and readable

4. **Standard Compiler Architecture**
   - Follows traditional compiler design
   - Easier to understand and extend
   - Better separation of concerns

## Known Limitations

### Current Status
The lexer is implemented and tested, but **not yet integrated** with the main parser. The current system still uses Pest directly.

### Full Integration (Future Work)
To fully integrate the lexer:

**Option A**: Preprocess with Lexer + Keep Pest
- Tokenize input first
- Convert tokens to string format Pest expects
- Pass to Pest parser
- Requires token-to-string conversion layer

**Option B**: Replace Pest with Hand-Written Parser
- Use lexer for tokenization
- Implement recursive descent parser manually
- Full control over error messages
- More work but cleaner architecture

**Option C**: Use Parser Generator with Separate Lexer
- Switch to LALRPOP, nom, or similar
- These support separate lexer + parser
- Better suited for this architecture

## Recommendations

### Immediate Next Steps
1. Choose integration option (A, B, or C above)
2. Implement parser integration
3. Update test files to use new variable names
4. Verify all paper equation tests pass

### Long-term Architecture
For a production-quality implementation:
- Consider Option B (hand-written parser) for best error messages
- Or Option C (different parser generator) for maintainability
- Option A (Pest + preprocessing) is a quick fix but adds complexity

## References

**User Feedback**:
- "トークナイザーのバグです" - It's a tokenizer bug
- "アルファベットが続く限り読み込んで、その後に、予約後かそれ以外を判定" - Read complete sequence, then classify
- "字句解析と構文解析を一緒にしてる？きっちり分けたほうがよい" - Lexical and syntax analysis should be separated

**Related Files**:
- `src/lexer/mod.rs` - Lexer implementation
- `src/parser/grammar.pest` - Simplified grammar
- `claudedocs/paper_equation_tests_summary.md` - Test files waiting for fix
