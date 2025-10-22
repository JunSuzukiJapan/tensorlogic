//! Formatting utilities for displaying logic programming terms
//!
//! Provides functions for formatting atoms, terms, and constants for display.

use crate::ast::*;

/// Format an atom for display
pub(super) fn format_atom(atom: &Atom) -> String {
    let terms: Vec<String> = atom.terms.iter().map(format_term).collect();
    format!("{}({})", atom.predicate.as_str(), terms.join(", "))
}

/// Format a term for display
pub(super) fn format_term(term: &Term) -> String {
    match term {
        Term::Variable(v) => v.as_str().to_string(),
        Term::Constant(c) => format_constant(c),
        Term::Tensor(_) => "<tensor>".to_string(),
    }
}

/// Format a constant for display
fn format_constant(constant: &Constant) -> String {
    match constant {
        Constant::Integer(n) => n.to_string(),
        Constant::Float(n) => n.to_string(),
        Constant::String(s) => s.clone(),
        Constant::Boolean(b) => b.to_string(),
    }
}
