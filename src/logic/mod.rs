//! Logic programming engine for TensorLogic
//!
//! Provides unification and query resolution for logic programming features.

use crate::ast::*;
use crate::interpreter::RuntimeResult;
use std::collections::HashMap;

mod substitution;
mod unification;
mod helpers;

use substitution::{apply_substitution_to_atom, apply_substitution_to_term};
use unification::unify_atoms;
use helpers::collect_variables_from_atom;

/// Substitution map for variables
pub type Substitution = HashMap<String, Term>;

/// Maximum depth for recursive query resolution (prevents infinite loops)
const MAX_DEPTH: usize = 100;

/// Logic engine for query resolution
pub struct LogicEngine {
    /// Known facts (ground atoms)
    facts: Vec<Atom>,
    /// Rules
    rules: Vec<RuleDecl>,
    /// Counter for variable renaming
    rename_counter: std::cell::Cell<usize>,
}

impl LogicEngine {
    /// Create a new logic engine
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            rules: Vec::new(),
            rename_counter: std::cell::Cell::new(0),
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
        // Collect variables from the query
        let query_vars = collect_variables_from_atom(atom);

        let results = self.query_with_depth(atom, &Substitution::new(), 0);

        // Filter results to only show bindings for query variables,
        // and fully apply substitutions to get ground terms
        let filtered_results: Vec<Substitution> = results
            .into_iter()
            .map(|sub| {
                query_vars.iter()
                    .filter_map(|var| {
                        if let Some(term) = sub.get(var) {
                            // Fully apply substitution to resolve to ground term
                            let resolved = apply_substitution_to_term(term, &sub);
                            Some((var.clone(), resolved))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Remove duplicate results
        let mut unique_results = Vec::new();
        for result in filtered_results {
            if !unique_results.contains(&result) {
                unique_results.push(result);
            }
        }

        Ok(unique_results)
    }

    /// Query with depth tracking (internal)
    fn query_with_depth(&self, atom: &Atom, sub: &Substitution, depth: usize) -> Vec<Substitution> {
        if depth > MAX_DEPTH {
            // Depth limit reached - this is normal for transitive closures
            return vec![];
        }

        let mut results = Vec::new();

        // Apply current substitution to the query atom
        let atom_with_sub = apply_substitution_to_atom(atom, sub);

        // 1. Try to match against facts
        for fact in &self.facts {
            if let Some(new_sub) = unify_atoms(&atom_with_sub, fact, sub) {
                results.push(new_sub);
            }
        }

        // 2. Try to match against rules
        for rule in &self.rules {
            if let RuleHead::Atom(_rule_head) = &rule.head {
                // Rename variables in the rule to avoid conflicts
                let renamed_rule = self.rename_rule_variables(rule);

                if let RuleHead::Atom(renamed_head) = &renamed_rule.head {
                    // Try to unify query with rule head
                    if let Some(head_sub) = unify_atoms(&atom_with_sub, renamed_head, sub) {
                        // Try to satisfy the rule body
                        let body_results = self.satisfy_body(&renamed_rule.body, &head_sub, depth + 1);
                        results.extend(body_results);
                    }
                }
            }
        }

        results
    }

    /// Try to satisfy a rule body (list of body terms)
    fn satisfy_body(&self, body: &[BodyTerm], sub: &Substitution, depth: usize) -> Vec<Substitution> {
        if body.is_empty() {
            // Empty body is always satisfied
            return vec![sub.clone()];
        }

        // Process first term and recursively process rest
        let first = &body[0];
        let rest = &body[1..];

        match first {
            BodyTerm::Atom(atom) => {
                // Find all ways to satisfy this atom
                let atom_results = self.query_with_depth(atom, sub, depth);

                let mut results = Vec::new();
                for atom_sub in atom_results {
                    // For each way to satisfy the first atom,
                    // try to satisfy the rest of the body
                    let rest_results = self.satisfy_body(rest, &atom_sub, depth);
                    results.extend(rest_results);
                }
                results
            }
            _ => {
                // For other body terms (constraints, equations), skip for now
                // and try to satisfy the rest
                self.satisfy_body(rest, sub, depth)
            }
        }
    }

    /// Rename all variables in a rule to make them unique
    fn rename_rule_variables(&self, rule: &RuleDecl) -> RuleDecl {
        let suffix = self.rename_counter.get();
        self.rename_counter.set(suffix + 1);

        let renamed_head = match &rule.head {
            RuleHead::Atom(atom) => RuleHead::Atom(rename_atom_variables(atom, suffix)),
            RuleHead::Equation(eq) => RuleHead::Equation(eq.clone()),
        };

        let renamed_body: Vec<BodyTerm> = rule.body.iter().map(|term| {
            match term {
                BodyTerm::Atom(atom) => BodyTerm::Atom(rename_atom_variables(atom, suffix)),
                _ => term.clone(),
            }
        }).collect();

        RuleDecl {
            head: renamed_head,
            body: renamed_body,
        }
    }
}

/// Rename all variables in an atom
fn rename_atom_variables(atom: &Atom, suffix: usize) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom.terms.iter().map(|term| rename_term_variables(term, suffix)).collect(),
    }
}

/// Rename all variables in a term
fn rename_term_variables(term: &Term, suffix: usize) -> Term {
    match term {
        Term::Variable(v) => {
            let new_name = format!("{}_{}", v.as_str(), suffix);
            Term::Variable(Identifier::new(&new_name))
        }
        _ => term.clone(),
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
            predicate: Identifier::new("Parent"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("bob")),
            ],
        };
        engine.add_fact(fact);

        // Query: parent(alice, bob)?
        let query = Atom {
            predicate: Identifier::new("Parent"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("bob")),
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
            predicate: Identifier::new("Parent"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("bob")),
            ],
        };
        engine.add_fact(fact);

        // Query: parent(alice, X)?
        let query = Atom {
            predicate: Identifier::new("Parent"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("X")),
            ],
        };

        let results = engine.query(&query).unwrap();
        assert_eq!(results.len(), 1);

        // X should be bound to bob
        let sub = &results[0];
        assert!(sub.contains_key("X"));
    }

    #[test]
    fn test_rule_application() {
        let mut engine = LogicEngine::new();

        // Add fact: Parent(alice, bob)
        engine.add_fact(Atom {
            predicate: Identifier::new("Parent"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("bob")),
            ],
        });

        // Add rule: Ancestor(X, Y) <- Parent(X, Y)
        engine.add_rule(RuleDecl {
            head: RuleHead::Atom(Atom {
                predicate: Identifier::new("Ancestor"),
                terms: vec![
                    Term::Variable(Identifier::new("X")),
                    Term::Variable(Identifier::new("Y")),
                ],
            }),
            body: vec![BodyTerm::Atom(Atom {
                predicate: Identifier::new("Parent"),
                terms: vec![
                    Term::Variable(Identifier::new("X")),
                    Term::Variable(Identifier::new("Y")),
                ],
            })],
        });

        // Query: Ancestor(alice, X)?
        let query = Atom {
            predicate: Identifier::new("Ancestor"),
            terms: vec![
                Term::Variable(Identifier::new("alice")),
                Term::Variable(Identifier::new("X")),
            ],
        };

        let results = engine.query(&query).unwrap();
        assert_eq!(results.len(), 1);

        let sub = &results[0];
        assert!(sub.contains_key("X"));
    }
}
