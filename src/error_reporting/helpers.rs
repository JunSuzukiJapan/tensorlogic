//! Helper functions for creating common error diagnostics

use super::{Diagnostic, StackTrace, StackFrame, FrameType};
use crate::ast::span::Span;
use crate::typecheck::TypeError;

/// Convert TypeError to Diagnostic with user-friendly messages
pub fn type_error_to_diagnostic(error: &TypeError) -> Diagnostic {
    match error {
        TypeError::UndefinedVariable(name) => {
            Diagnostic::error(format!("Undefined variable '{}'", name))
                .with_note("Variable must be declared before use")
                .with_suggestion(format!("Did you mean to declare: 'let {} = ...'?", name))
        }
        TypeError::TypeMismatch { expected, found } => {
            Diagnostic::error(format!("Type mismatch: expected {}, found {}", expected, found))
                .with_note("Types must be compatible for this operation")
                .with_suggestion("Check the types of all operands")
        }
        TypeError::DimensionMismatch { left, right } => {
            Diagnostic::error(format!(
                "Incompatible tensor dimensions: {:?} vs {:?}",
                left, right
            ))
            .with_note("Tensors must have compatible shapes for element-wise operations")
            .with_suggestion("Use broadcasting or reshape operations to match dimensions")
        }
        TypeError::BaseTypeMismatch { left, right } => {
            Diagnostic::error(format!(
                "Incompatible base types: {:?} vs {:?}",
                left, right
            ))
            .with_note("All operands must have the same base type (e.g., float32, int32)")
            .with_suggestion("Convert operands to the same type using explicit casts")
        }
        TypeError::InvalidOperation { op, left, right } => {
            Diagnostic::error(format!(
                "Invalid operation '{}' for types {} and {}",
                op, left, right
            ))
            .with_note(format!("Operation '{}' is not defined for these types", op))
            .with_suggestion("Check the operator and operand types")
        }
        TypeError::DuplicateDeclaration(name) => {
            Diagnostic::error(format!("Duplicate declaration of '{}'", name))
                .with_note("Each variable, function, or type can only be declared once in a scope")
                .with_suggestion(format!("Use a different name or remove the duplicate declaration of '{}'", name))
        }
        TypeError::UndefinedRelation(name) => {
            Diagnostic::error(format!("Relation '{}' not found", name))
                .with_note("Relations must be declared before use")
                .with_suggestion(format!("Declare the relation: 'relation {}(...)'", name))
        }
        TypeError::UndefinedFunction(name) => {
            Diagnostic::error(format!("Function '{}' not found", name))
                .with_note("Functions must be declared before use")
                .with_suggestion(format!("Declare the function: 'function {}(...) -> type {{ ... }}'", name))
        }
        TypeError::ArgumentCountMismatch { expected, found } => {
            Diagnostic::error(format!(
                "Wrong number of arguments: expected {}, found {}",
                expected, found
            ))
            .with_note("Function calls must provide the exact number of arguments")
            .with_suggestion("Add or remove arguments to match the function signature")
        }
        TypeError::CannotInferType => {
            Diagnostic::error("Cannot infer type for expression")
                .with_note("The type system needs more information to determine the type")
                .with_suggestion("Add explicit type annotations to help type inference")
        }
        TypeError::UndefinedDimensionVariable(name) => {
            Diagnostic::error(format!("Dimension variable '{}' not in scope", name))
                .with_note("Dimension variables must be declared in the tensor type")
                .with_suggestion(format!("Add '{}' to the dimension variables in scope", name))
        }
        TypeError::UndefinedStruct(name) => {
            Diagnostic::error(format!("Struct '{}' not found", name))
                .with_note("Structs must be declared before use")
                .with_suggestion(format!("Declare the struct: 'struct {} {{ ... }}'", name))
        }
        TypeError::UndefinedMethod(struct_name, method_name) => {
            Diagnostic::error(format!("Method '{}::{}' not found", struct_name, method_name))
                .with_note("Methods must be declared in an impl block")
                .with_suggestion(format!("Add method to impl block: 'impl {} {{ fn {}(...) {{ ... }} }}'", struct_name, method_name))
        }
        TypeError::UndefinedField(field_name, struct_name) => {
            Diagnostic::error(format!("Field '{}' not found in struct '{}'", field_name, struct_name))
                .with_note("Struct fields must be declared in the struct definition")
                .with_suggestion(format!("Add field to struct: 'struct {} {{ {}: type, ... }}'", struct_name, field_name))
        }
        TypeError::TypeArgumentCountMismatch { expected, found } => {
            Diagnostic::error(format!(
                "Wrong number of type arguments: expected {}, found {}",
                expected, found
            ))
            .with_note("Type constructors must provide the exact number of type arguments")
            .with_suggestion("Add or remove type arguments to match the type signature")
        }
        TypeError::UndefinedTypeParameter(name) => {
            Diagnostic::error(format!("Type parameter '{}' not in scope", name))
                .with_note("Type parameters must be declared in the type signature")
                .with_suggestion(format!("Add '{}' to the type parameters", name))
        }
    }
}

/// Create a diagnostic for a parse error
pub fn parse_error_diagnostic(message: impl Into<String>, span: Option<Span>) -> Diagnostic {
    let mut diag = Diagnostic::error(message)
        .with_note("Syntax error in source code");

    if let Some(s) = span {
        diag = diag.with_span(s);
    }

    diag.with_suggestion("Check the syntax and fix any typos")
}

/// Create a diagnostic for a runtime error
pub fn runtime_error_diagnostic(message: impl Into<String>, span: Option<Span>) -> Diagnostic {
    let mut diag = Diagnostic::error(message)
        .with_note("Runtime error during execution");

    if let Some(s) = span {
        diag = diag.with_span(s);
    }

    diag
}

/// Create a warning diagnostic
pub fn warning_diagnostic(message: impl Into<String>, span: Option<Span>) -> Diagnostic {
    let mut diag = Diagnostic::warning(message);

    if let Some(s) = span {
        diag = diag.with_span(s);
    }

    diag
}

/// Create a runtime error diagnostic with stack trace
pub fn runtime_error_with_trace(
    message: impl Into<String>,
    span: Option<Span>,
    stack_trace: StackTrace,
) -> Diagnostic {
    let mut diag = runtime_error_diagnostic(message, span);
    diag = diag.with_stack_trace(stack_trace);
    diag
}

/// Create a simple stack trace from error context
pub fn simple_stack_trace(context: &str, file: Option<&str>) -> StackTrace {
    let mut trace = StackTrace::new();

    if let Some(f) = file {
        trace.push(StackFrame::with_location(
            context.to_string(),
            FrameType::Statement,
            f.to_string(),
            0,
        ));
    } else {
        trace.push(StackFrame::new(
            context.to_string(),
            FrameType::Statement,
        ));
    }

    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::error_reporting::Severity;

    #[test]
    fn test_type_error_to_diagnostic() {
        let error = TypeError::UndefinedVariable("x".to_string());
        let diag = type_error_to_diagnostic(&error);

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("x"));
        assert!(!diag.notes.is_empty());
        assert!(!diag.suggestions.is_empty());
    }

    #[test]
    fn test_type_mismatch_diagnostic() {
        let error = TypeError::TypeMismatch {
            expected: "int".to_string(),
            found: "float".to_string(),
        };
        let diag = type_error_to_diagnostic(&error);

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("int"));
        assert!(diag.message.contains("float"));
    }

    #[test]
    fn test_parse_error_diagnostic() {
        let span = Span::new(
            crate::ast::span::Position::new(1, 1, 0),
            crate::ast::span::Position::new(1, 5, 4),
        );

        let diag = parse_error_diagnostic("Unexpected token", Some(span));

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.span.is_some());
    }

    #[test]
    fn test_runtime_error_diagnostic() {
        let diag = runtime_error_diagnostic("Division by zero", None);

        assert_eq!(diag.severity, Severity::Error);
        assert!(diag.message.contains("Division by zero"));
    }

    #[test]
    fn test_warning_diagnostic() {
        let diag = warning_diagnostic("Unused variable", None);

        assert_eq!(diag.severity, Severity::Warning);
    }
}
