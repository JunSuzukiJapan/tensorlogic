//! Unification algorithm for logic programming
//!
//! Implements the core unification algorithm for matching atoms and terms.

use crate::ast::*;
use super::Substitution;
use super::substitution::apply_substitution_to_term;

/// Unify two atoms
pub(super) fn unify_atoms(atom1: &Atom, atom2: &Atom, sub: &Substitution) -> Option<Substitution> {
    if atom1.predicate.as_str() != atom2.predicate.as_str() {
        return None;
    }

    if atom1.terms.len() != atom2.terms.len() {
        return None;
    }

    let mut new_sub = sub.clone();

    for (t1, t2) in atom1.terms.iter().zip(atom2.terms.iter()) {
        if !unify_terms(t1, t2, &mut new_sub) {
            return None;
        }
    }

    Some(new_sub)
}

/// Unify two terms
fn unify_terms(term1: &Term, term2: &Term, sub: &mut Substitution) -> bool {
    // Apply existing substitutions
    let t1 = apply_substitution_to_term(term1, sub);
    let t2 = apply_substitution_to_term(term2, sub);

    match (&t1, &t2) {
        (Term::Variable(v1), Term::Variable(v2)) => {
            if v1.as_str() == v2.as_str() {
                // Same variable
                true
            } else {
                // Bind v1 to v2
                sub.insert(v1.as_str().to_string(), t2.clone());
                true
            }
        }
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            // Bind variable to term
            sub.insert(v.as_str().to_string(), t.clone());
            true
        }
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        _ => false,
    }
}
