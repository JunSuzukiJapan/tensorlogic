use tower_lsp::lsp_types::*;

/// Get definition location for a symbol at the given position
pub fn get_definition(text: &str, position: Position, uri: &str) -> Option<Location> {
    let lines: Vec<&str> = text.lines().collect();
    if position.line as usize >= lines.len() {
        return None;
    }

    let line = lines[position.line as usize];
    let word = extract_word_at_position(line, position.character as usize)?;

    // Search for definition in the document
    find_definition_in_document(text, &word, uri)
}

/// Extract word at cursor position
fn extract_word_at_position(line: &str, column: usize) -> Option<String> {
    if column > line.len() {
        return None;
    }

    // Find word boundaries
    let start = line[..column]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);

    let end = line[column..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| column + i)
        .unwrap_or(line.len());

    if start >= end {
        return None;
    }

    Some(line[start..end].to_string())
}

/// Find definition of a symbol in the document
fn find_definition_in_document(text: &str, symbol: &str, uri: &str) -> Option<Location> {
    for (line_num, line) in text.lines().enumerate() {
        // Function definition: fn symbol(...)
        if let Some(fn_pos) = line.find("fn ") {
            if let Some(name_start) = line[fn_pos + 3..].find(symbol) {
                let name_start = fn_pos + 3 + name_start;
                // Verify it's exactly the symbol (not a substring)
                let before = if name_start > 0 {
                    line.chars().nth(name_start - 1)
                } else {
                    Some(' ')
                };
                let after = line.chars().nth(name_start + symbol.len());

                if is_word_boundary(before) && is_word_boundary(after) {
                    return Some(Location {
                        uri: Url::parse(uri).ok()?,
                        range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: name_start as u32,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: (name_start + symbol.len()) as u32,
                            },
                        },
                    });
                }
            }
        }

        // Tensor declaration: tensor symbol: type
        if let Some(tensor_pos) = line.find("tensor ") {
            if let Some(name_start) = line[tensor_pos + 7..].find(symbol) {
                let name_start = tensor_pos + 7 + name_start;
                let before = if name_start > 0 {
                    line.chars().nth(name_start - 1)
                } else {
                    Some(' ')
                };
                let after = line.chars().nth(name_start + symbol.len());

                if is_word_boundary(before) && is_word_boundary(after) {
                    return Some(Location {
                        uri: Url::parse(uri).ok()?,
                        range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: name_start as u32,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: (name_start + symbol.len()) as u32,
                            },
                        },
                    });
                }
            }
        }

        // Let binding: let symbol =
        if let Some(let_pos) = line.find("let ") {
            if let Some(name_start) = line[let_pos + 4..].find(symbol) {
                let name_start = let_pos + 4 + name_start;
                let before = if name_start > 0 {
                    line.chars().nth(name_start - 1)
                } else {
                    Some(' ')
                };
                let after = line.chars().nth(name_start + symbol.len());

                if is_word_boundary(before) && is_word_boundary(after) {
                    return Some(Location {
                        uri: Url::parse(uri).ok()?,
                        range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: name_start as u32,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: (name_start + symbol.len()) as u32,
                            },
                        },
                    });
                }
            }
        }

        // Relation declaration: relation symbol(...)
        if let Some(rel_pos) = line.find("relation ") {
            if let Some(name_start) = line[rel_pos + 9..].find(symbol) {
                let name_start = rel_pos + 9 + name_start;
                let before = if name_start > 0 {
                    line.chars().nth(name_start - 1)
                } else {
                    Some(' ')
                };
                let after = line.chars().nth(name_start + symbol.len());

                if is_word_boundary(before) && is_word_boundary(after) {
                    return Some(Location {
                        uri: Url::parse(uri).ok()?,
                        range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: name_start as u32,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: (name_start + symbol.len()) as u32,
                            },
                        },
                    });
                }
            }
        }
    }

    None
}

/// Check if a character is a word boundary
fn is_word_boundary(ch: Option<char>) -> bool {
    match ch {
        None => true,
        Some(c) => !c.is_alphanumeric() && c != '_',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_word() {
        assert_eq!(extract_word_at_position("tensor x", 3), Some("tensor".to_string()));
        assert_eq!(extract_word_at_position("fn foo()", 4), Some("foo".to_string()));
    }

    #[test]
    fn test_find_function_definition() {
        let text = r#"
fn add(a: int, b: int) -> int {
    return a + b
}

main {
    let result = add(1, 2)
}
"#;
        let location = find_definition_in_document(text, "add", "file:///test.tl");
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.range.start.line, 1); // Second line (0-indexed)
    }

    #[test]
    fn test_find_tensor_definition() {
        let text = r#"
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y = x + x
}
"#;
        let location = find_definition_in_document(text, "x", "file:///test.tl");
        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.range.start.line, 2); // Third line
    }
}
