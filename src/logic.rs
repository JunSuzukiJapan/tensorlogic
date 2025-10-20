//! Logic programming engine for TensorLogic
//!
//! Provides basic unification and query resolution for logic programming features.

use crate::ast::*;
use crate::interpreter::RuntimeResult;
use std::collections::HashMap;

/// Substitution map for variables
pub type Substitution = HashMap<String, Term>;

/// Logic engine for query resolution
pub struct LogicEngine {
    /// Known facts (ground atoms)
    facts: Vec<Atom>,
    /// Rules
    rules: Vec<RuleDecl>,
}

impl LogicEngine {
    /// Create a new logic engine
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&mut self, rule: RuleDecl) {
        self.rules.push(rule);
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&mut self, atom: Atom) {
        self.facts.push(atom);
    }

    /// Query the knowledge base
    pub fn query(&self, atom: &Atom) -> RuntimeResult<Vec<Substitution>> {
        let mut results = Vec::new();

        // Check facts
        for fact in &self.facts {
            if let Some(sub) = unify_atoms(atom, fact, &Substitution::new()) {
                results.push(sub);
            }
        }

        // Check rules (simplified: only direct matching)
        for rule in &self.rules {
            if let RuleHead::Atom(rule_head) = &rule.head {
                if let Some(mut sub) = unify_atoms(atom, rule_head, &Substitution::new()) {
                    // Simplified: assume rule body is satisfied if it's empty or simple
                    if rule.body.is_empty() {
                        results.push(sub);
                    } else {
                        // Try to satisfy body (simplified version)
                        if self.try_satisfy_body(&rule.body, &mut sub) {
                            results.push(sub);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Try to satisfy rule body (simplified)
    fn try_satisfy_body(&self, body: &[BodyTerm], sub: &mut Substitution) -> bool {
        for term in body {
            match term {
                BodyTerm::Atom(atom) => {
                    // Try to find a matching fact
                    let mut found = false;
                    for fact in &self.facts {
                        if unify_atoms(atom, fact, sub).is_some() {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return false;
                    }
                }
                _ => {
                    // Simplified: skip other types
                    continue;
                }
            }
        }
        true
    }
}

/// Unify two atoms
fn unify_atoms(atom1: &Atom, atom2: &Atom, sub: &Substitution) -> Option<Substitution> {
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
    match (term1, term2) {
        (Term::Variable(v1), Term::Variable(v2)) => {
            // Check if already bound
            let t1_opt = sub.get(v1.as_str()).cloned();
            let t2_opt = sub.get(v2.as_str()).cloned();

            match (t1_opt, t2_opt) {
                (Some(t1), Some(t2)) => unify_terms(&t1, &t2, sub),
                (Some(t), None) => {
                    sub.insert(v2.as_str().to_string(), t);
                    true
                }
                (None, Some(t)) => {
                    sub.insert(v1.as_str().to_string(), t);
                    true
                }
                (None, None) => {
                    // Bind first to second
                    sub.insert(v1.as_str().to_string(), Term::Variable(v2.clone()));
                    true
                }
            }
        }
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            if let Some(bound) = sub.get(v.as_str()).cloned() {
                unify_terms(&bound, t, sub)
            } else {
                sub.insert(v.as_str().to_string(), t.clone());
                true
            }
        }
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fact_query() {
        let mut engine = LogicEngine::new();

        // Add fact: parent(alice, bob)
        let fact = Atom {
            predicate: Identifier::new("parent"),
            terms: vec![
                Term::Constant(Constant::String("alice".to_string())),
                Term::Constant(Constant::String("bob".to_string())),
            ],
        };
        engine.add_fact(fact);

        // Query: parent(alice, bob)?
        let query = Atom {
            predicate: Identifier::new("parent"),
            terms: vec![
                Term::Constant(Constant::String("alice".to_string())),
                Term::Constant(Constant::String("bob".to_string())),
            ],
        };

        let results = engine.query(&query).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_variable_query() {
        let mut engine = LogicEngine::new();

        // Add fact: parent(alice, bob)
        let fact = Atom {
            predicate: Identifier::new("parent"),
            terms: vec![
                Term::Constant(Constant::String("alice".to_string())),
                Term::Constant(Constant::String("bob".to_string())),
            ],
        };
        engine.add_fact(fact);

        // Query: parent(alice, X)?
        let query = Atom {
            predicate: Identifier::new("parent"),
            terms: vec![
                Term::Constant(Constant::String("alice".to_string())),
                Term::Variable(Identifier::new("X")),
            ],
        };

        let results = engine.query(&query).unwrap();
        assert_eq!(results.len(), 1);

        // X should be bound to "bob"
        let sub = &results[0];
        assert!(sub.contains_key("X"));
    }
}
