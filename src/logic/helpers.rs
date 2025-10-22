//! Helper utilities for logic programming
//!
//! Provides utility functions for working with atoms and terms.

use crate::ast::*;
use std::collections::HashSet;

/// Collect all variable names from an atom
pub(super) fn collect_variables_from_atom(atom: &Atom) -> HashSet<String> {
    let mut vars = HashSet::new();
    for term in &atom.terms {
        if let Term::Variable(v) = term {
            vars.insert(v.as_str().to_string());
        }
    }
    vars
}
