//! Enhanced error reporting with source location information
//!
//! Provides user-friendly error messages with line/column information,
//! context snippets, and suggestions for common mistakes.

use crate::ast::span::Span;
use std::fmt;

use super::stack_trace::StackTrace;

#[cfg(test)]
use crate::ast::span::Position;

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
        }
    }
}

/// Diagnostic message with source location
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
    pub suggestions: Vec<String>,
    pub stack_trace: Option<StackTrace>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            suggestions: Vec::new(),
            stack_trace: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            suggestions: Vec::new(),
            stack_trace: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    pub fn with_stack_trace(mut self, stack_trace: StackTrace) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }

    /// Format the diagnostic for display
    pub fn format(&self, source: Option<&str>) -> String {
        let mut output = String::new();

        // Header: error/warning at location
        if let Some(ref span) = self.span {
            output.push_str(&format!("{}: {}\n", self.severity, self.message));
            output.push_str(&format!("  --> {}:{}\n", span.start.line, span.start.column));

            // Show source code context if available
            if let Some(src) = source {
                let lines: Vec<&str> = src.lines().collect();
                let line_idx = span.start.line.saturating_sub(1);

                if line_idx < lines.len() {
                    let line_num = span.start.line;
                    let line = lines[line_idx];

                    // Line number padding
                    let num_width = line_num.to_string().len();

                    // Show the line
                    output.push_str(&format!("{:>width$} | {}\n", line_num, line, width = num_width));

                    // Show the caret (^) under the error location
                    let col = span.start.column.saturating_sub(1);
                    let spaces = " ".repeat(col + num_width + 3); // +3 for " | "
                    let carets = "^".repeat((span.end.column - span.start.column).max(1));
                    output.push_str(&format!("{}{}--- {}\n", spaces, carets, self.severity));
                }
            }
        } else {
            output.push_str(&format!("{}: {}\n", self.severity, self.message));
        }

        // Add notes
        for note in &self.notes {
            output.push_str(&format!("  = note: {}\n", note));
        }

        // Add suggestions
        for suggestion in &self.suggestions {
            output.push_str(&format!("  = help: {}\n", suggestion));
        }

        // Add stack trace if available
        if let Some(stack_trace) = &self.stack_trace {
            if !stack_trace.is_empty() {
                output.push_str("\n");
                output.push_str(&stack_trace.format());
            }
        }

        output
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format(None))
    }
}

/// Error reporter that accumulates diagnostics
#[derive(Debug, Default)]
pub struct ErrorReporter {
    diagnostics: Vec<Diagnostic>,
    source: Option<String>,
}

impl ErrorReporter {
    pub fn new() -> Self {
        ErrorReporter {
            diagnostics: Vec::new(),
            source: None,
        }
    }

    pub fn with_source(source: String) -> Self {
        ErrorReporter {
            diagnostics: Vec::new(),
            source: Some(source),
        }
    }

    pub fn set_source(&mut self, source: String) {
        self.source = Some(source);
    }

    pub fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn error(&mut self, message: impl Into<String>, span: Option<Span>) {
        let mut diag = Diagnostic::error(message);
        if let Some(s) = span {
            diag = diag.with_span(s);
        }
        self.diagnostics.push(diag);
    }

    pub fn warning(&mut self, message: impl Into<String>, span: Option<Span>) {
        let mut diag = Diagnostic::warning(message);
        if let Some(s) = span {
            diag = diag.with_span(s);
        }
        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    /// Format all diagnostics for display
    pub fn format_all(&self) -> String {
        let mut output = String::new();

        for diag in &self.diagnostics {
            output.push_str(&diag.format(self.source.as_deref()));
            output.push('\n');
        }

        // Summary
        let errors = self.error_count();
        let warnings = self.warning_count();

        if errors > 0 || warnings > 0 {
            output.push_str(&format!(
                "Found {} error(s) and {} warning(s)\n",
                errors, warnings
            ));
        }

        output
    }
}

impl fmt::Display for ErrorReporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_all())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let diag = Diagnostic::error("Type mismatch")
            .with_span(Span::new(
                Position::new(1, 5, 4),
                Position::new(1, 10, 9),
            ))
            .with_note("Expected type: int")
            .with_suggestion("Try using an integer literal");

        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.message, "Type mismatch");
        assert_eq!(diag.notes.len(), 1);
        assert_eq!(diag.suggestions.len(), 1);
    }

    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new();

        reporter.error("Undefined variable: x", None);
        reporter.warning("Unused variable: y", None);

        assert_eq!(reporter.error_count(), 1);
        assert_eq!(reporter.warning_count(), 1);
        assert!(reporter.has_errors());
    }

    #[test]
    fn test_diagnostic_formatting_with_source() {
        let source = "tensor w: float32[10]\nlet x = w + 5";
        let mut reporter = ErrorReporter::with_source(source.to_string());

        let diag = Diagnostic::error("Type mismatch: cannot add tensor and scalar")
            .with_span(Span::new(
                Position::new(2, 9, 23),
                Position::new(2, 14, 28),
            ))
            .with_note("Left operand has type: Tensor<float32[10]>")
            .with_note("Right operand has type: int")
            .with_suggestion("Use broadcasting: w + Tensor::from(5)");

        reporter.report(diag);

        let formatted = reporter.format_all();
        assert!(formatted.contains("error:"));
        assert!(formatted.contains("Type mismatch"));
        assert!(formatted.contains("2:9"));
        assert!(formatted.contains("note:"));
        assert!(formatted.contains("help:"));
    }

    #[test]
    fn test_diagnostic_without_source() {
        let diag = Diagnostic::error("Parse error")
            .with_span(Span::new(
                Position::new(1, 1, 0),
                Position::new(1, 5, 4),
            ));

        let formatted = diag.format(None);
        assert!(formatted.contains("error: Parse error"));
        assert!(formatted.contains("1:1"));
    }

    #[test]
    fn test_clear_diagnostics() {
        let mut reporter = ErrorReporter::new();
        reporter.error("Error 1", None);
        reporter.error("Error 2", None);

        assert_eq!(reporter.error_count(), 2);

        reporter.clear();
        assert_eq!(reporter.error_count(), 0);
        assert!(!reporter.has_errors());
    }
}
