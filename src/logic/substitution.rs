//! Substitution operations for logic programming
//!
//! Handles variable substitutions during unification and query resolution.

use crate::ast::*;
use super::Substitution;

/// Apply substitution to an atom
pub(super) fn apply_substitution_to_atom(atom: &Atom, sub: &Substitution) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom.terms.iter().map(|term| apply_substitution_to_term(term, sub)).collect(),
    }
}

/// Apply substitution to a term
pub(super) fn apply_substitution_to_term(term: &Term, sub: &Substitution) -> Term {
    match term {
        Term::Variable(v) => {
            if let Some(bound) = sub.get(v.as_str()) {
                // Recursively apply substitution
                apply_substitution_to_term(bound, sub)
            } else {
                term.clone()
            }
        }
        _ => term.clone(),
    }
}
