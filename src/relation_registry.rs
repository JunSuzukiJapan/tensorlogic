//! Relation Registry
//!
//! Manages relation types and their instances for knowledge graph embeddings.
//! Provides relation-to-index and index-to-relation mappings for tensor operations.

use std::collections::HashMap;
use crate::ast::RelationDecl;

/// Relation declaration type
#[derive(Debug, Clone, PartialEq)]
pub enum RelationDeclType {
    /// Relations collected from declarations
    FromDeclarations,
}

/// Information about a single relation type
#[derive(Debug, Clone)]
pub struct RelationTypeInfo {
    /// Relation name
    pub name: String,
    /// Relation name -> index mapping (for tensor operations)
    relation_to_index: HashMap<String, usize>,
    /// Index -> relation name mapping
    index_to_relation: Vec<String>,
}

impl RelationTypeInfo {
    /// Create a new relation info
    pub fn new(name: String) -> Self {
        let mut relation_to_index = HashMap::new();
        relation_to_index.insert(name.clone(), 0);

        Self {
            name: name.clone(),
            relation_to_index,
            index_to_relation: vec![name],
        }
    }

    /// Get the index for a relation name
    pub fn get_relation_index(&self, relation_name: &str) -> Option<usize> {
        self.relation_to_index.get(relation_name).copied()
    }

    /// Get the relation name for an index
    pub fn get_relation_name(&self, index: usize) -> Option<&str> {
        self.index_to_relation.get(index).map(|s| s.as_str())
    }

    /// Get the total number of relations
    pub fn relation_count(&self) -> usize {
        self.index_to_relation.len()
    }

    /// Get all relation names
    pub fn all_relations(&self) -> &[String] {
        &self.index_to_relation
    }

    /// Check if a relation exists
    pub fn has_relation(&self, relation_name: &str) -> bool {
        self.relation_to_index.contains_key(relation_name)
    }
}

/// Relation Registry
///
/// Central registry for managing relation types.
#[derive(Debug, Clone)]
pub struct RelationRegistry {
    /// Relation name -> RelationTypeInfo mapping
    relations: HashMap<String, RelationTypeInfo>,
    /// All relation names (for "all relations" queries)
    all_relation_names: Vec<String>,
}

impl RelationRegistry {
    /// Create a new empty relation registry
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
            all_relation_names: Vec::new(),
        }
    }

    /// Register a relation from declaration
    pub fn register_from_decl(&mut self, decl: &RelationDecl) {
        let relation_name = decl.name.as_str().to_string();

        if !self.relations.contains_key(&relation_name) {
            let info = RelationTypeInfo::new(relation_name.clone());
            self.relations.insert(relation_name.clone(), info);
            self.all_relation_names.push(relation_name);
        }
    }

    /// Register a relation by name
    pub fn register(&mut self, relation_name: String) {
        if !self.relations.contains_key(&relation_name) {
            let info = RelationTypeInfo::new(relation_name.clone());
            self.relations.insert(relation_name.clone(), info);
            self.all_relation_names.push(relation_name);
        }
    }

    /// Get relation type information
    pub fn get_relation_info(&self, relation_name: &str) -> Option<&RelationTypeInfo> {
        self.relations.get(relation_name)
    }

    /// Get the index for a relation
    pub fn get_relation_index(&self, relation_name: &str) -> Option<usize> {
        self.all_relation_names.iter()
            .position(|r| r == relation_name)
    }

    /// Get the relation name for an index
    pub fn get_relation_name(&self, index: usize) -> Option<&str> {
        self.all_relation_names.get(index).map(|s| s.as_str())
    }

    /// Get the number of registered relations
    pub fn get_relation_count(&self) -> usize {
        self.all_relation_names.len()
    }

    /// Check if a relation exists
    pub fn has_relation(&self, relation_name: &str) -> bool {
        self.relations.contains_key(relation_name)
    }

    /// Get all registered relation names
    pub fn all_relation_names(&self) -> &[String] {
        &self.all_relation_names
    }

    /// Get relation indices for a list of relation names
    pub fn get_relation_indices(&self, relation_names: &[String]) -> Option<Vec<usize>> {
        let mut indices = Vec::new();

        for name in relation_names {
            if let Some(idx) = self.get_relation_index(name) {
                indices.push(idx);
            } else {
                return None; // Relation not found
            }
        }

        Some(indices)
    }

    /// Create a one-hot encoding vector for a relation
    pub fn relation_to_onehot(&self, relation_name: &str) -> Option<Vec<f32>> {
        let relation_count = self.get_relation_count();
        let relation_idx = self.get_relation_index(relation_name)?;

        let mut onehot = vec![0.0_f32; relation_count];
        onehot[relation_idx] = 1.0;

        Some(onehot)
    }

    /// Get the relation dimension (number of relations)
    pub fn get_relation_dimension(&self) -> usize {
        self.get_relation_count()
    }
}

impl Default for RelationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_registry() {
        let mut registry = RelationRegistry::new();

        registry.register("lives_in".to_string());
        registry.register("owns".to_string());
        registry.register("friend_of".to_string());

        assert!(registry.has_relation("lives_in"));
        assert_eq!(registry.get_relation_count(), 3);

        assert_eq!(registry.get_relation_index("lives_in"), Some(0));
        assert_eq!(registry.get_relation_index("owns"), Some(1));
        assert_eq!(registry.get_relation_index("friend_of"), Some(2));

        assert_eq!(registry.get_relation_name(0), Some("lives_in"));
        assert_eq!(registry.get_relation_name(1), Some("owns"));
        assert_eq!(registry.get_relation_name(2), Some("friend_of"));
    }

    #[test]
    fn test_relation_indices() {
        let mut registry = RelationRegistry::new();

        registry.register("parent_of".to_string());
        registry.register("sibling_of".to_string());

        let names = vec!["parent_of".to_string(), "sibling_of".to_string()];
        let indices = registry.get_relation_indices(&names);

        assert_eq!(indices, Some(vec![0, 1]));
    }

    #[test]
    fn test_relation_to_onehot() {
        let mut registry = RelationRegistry::new();

        registry.register("red".to_string());
        registry.register("green".to_string());
        registry.register("blue".to_string());

        // Test one-hot encoding for "green" (index 1)
        let onehot = registry.relation_to_onehot("green");
        assert_eq!(onehot, Some(vec![0.0, 1.0, 0.0]));

        // Test one-hot encoding for "blue" (index 2)
        let onehot = registry.relation_to_onehot("blue");
        assert_eq!(onehot, Some(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_relation_dimension() {
        let mut registry = RelationRegistry::new();

        registry.register("has".to_string());
        registry.register("is".to_string());

        assert_eq!(registry.get_relation_dimension(), 2);
    }
}
