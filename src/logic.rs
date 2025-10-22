//! Logic programming engine for TensorLogic
//!
//! Provides unification and query resolution for logic programming features.

use crate::ast::*;
use crate::interpreter::RuntimeResult;
use std::collections::HashMap;

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

/// Collect all variable names from an atom
fn collect_variables_from_atom(atom: &Atom) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    for term in &atom.terms {
        if let Term::Variable(v) = term {
            vars.insert(v.as_str().to_string());
        }
    }
    vars
}

/// Apply substitution to an atom
fn apply_substitution_to_atom(atom: &Atom, sub: &Substitution) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom.terms.iter().map(|term| apply_substitution_to_term(term, sub)).collect(),
    }
}

/// Apply substitution to a term
fn apply_substitution_to_term(term: &Term, sub: &Substitution) -> Term {
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
