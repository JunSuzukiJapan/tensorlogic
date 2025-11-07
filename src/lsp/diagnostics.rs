use tower_lsp::lsp_types::*;
use crate::parser::TensorLogicParser;
use pest::Parser;

/// Generate diagnostics (errors, warnings) for a document
pub fn generate_diagnostics(text: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Try to parse the document
    match TensorLogicParser::parse_program(text) {
        Ok(_) => {
            // Parsing successful - no syntax errors
        }
        Err(e) => {
            // Parse error - convert to diagnostic
            let error_msg = format!("{}", e);

            // Extract line and column from error message if possible
            let (line, column) = extract_position_from_error(&error_msg);

            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: column.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: (column + 10) as u32, // Approximate end position
                    },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("tensorlogic".to_string()),
                message: error_msg,
                related_information: None,
                tags: None,
                data: None,
            });
        }
    }

    // Add semantic analysis diagnostics here in the future
    // For example:
    // - Type mismatches
    // - Undefined variables
    // - Unused variables warnings

    diagnostics
}

/// Extract line and column from error message
fn extract_position_from_error(error: &str) -> (usize, usize) {
    // Try to extract position from pest error format
    // Pest errors typically contain "(line X, col Y)" or similar

    // Default position if we can't extract
    let mut line = 1;
    let mut col = 0;

    // Look for pattern like "(1, 5)"
    if let Some(start) = error.find('(') {
        if let Some(end) = error[start..].find(')') {
            let coords = &error[start + 1..start + end];
            let parts: Vec<&str> = coords.split(',').collect();
            if parts.len() == 2 {
                if let Ok(l) = parts[0].trim().parse() {
                    line = l;
                }
                if let Ok(c) = parts[1].trim().parse() {
                    col = c;
                }
            }
        }
    }

    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_program() {
        let text = r#"
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]
}
"#;
        let diagnostics = generate_diagnostics(text);
        assert_eq!(diagnostics.len(), 0);
    }

    #[test]
    fn test_invalid_syntax() {
        let text = r#"
main {
    tensor x: float16[3] =
}
"#;
        let diagnostics = generate_diagnostics(text);
        assert!(diagnostics.len() > 0);
        assert_eq!(diagnostics[0].severity, Some(DiagnosticSeverity::ERROR));
    }
}
