//! Entity Registry
//!
//! Manages entity types and their instances for logic programming.
//! Provides entity-to-index and index-to-entity mappings for tensor operations.

use std::collections::HashMap;
use crate::ast::EntityDecl;

/// Entity declaration type
#[derive(Debug, Clone, PartialEq)]
pub enum EntityDeclType {
    /// Explicit enumeration: entity T = {e1, e2, e3}
    Explicit(Vec<String>),
    /// Data-driven construction: entity T
    FromData,
}

/// Information about a single entity type
#[derive(Debug, Clone)]
pub struct EntityTypeInfo {
    /// Entity type name
    pub name: String,
    /// Declaration type
    pub declaration_type: EntityDeclType,
    /// Entity name -> index mapping (for tensor operations)
    entity_to_index: HashMap<String, usize>,
    /// Index -> entity name mapping
    index_to_entity: Vec<String>,
}

impl EntityTypeInfo {
    /// Create a new entity type with explicit enumeration
    pub fn explicit(name: String, entities: Vec<String>) -> Self {
        let mut entity_to_index = HashMap::new();
        let mut index_to_entity = Vec::new();

        for (idx, entity_name) in entities.iter().enumerate() {
            entity_to_index.insert(entity_name.clone(), idx);
            index_to_entity.push(entity_name.clone());
        }

        Self {
            name,
            declaration_type: EntityDeclType::Explicit(entities),
            entity_to_index,
            index_to_entity,
        }
    }

    /// Create a new entity type for data-driven construction
    pub fn from_data(name: String) -> Self {
        Self {
            name,
            declaration_type: EntityDeclType::FromData,
            entity_to_index: HashMap::new(),
            index_to_entity: Vec::new(),
        }
    }

    /// Add a new entity instance (for data-driven types)
    pub fn add_entity(&mut self, entity_name: String) -> usize {
        if let Some(&idx) = self.entity_to_index.get(&entity_name) {
            // Entity already exists, return existing index
            idx
        } else {
            // Add new entity
            let idx = self.index_to_entity.len();
            self.entity_to_index.insert(entity_name.clone(), idx);
            self.index_to_entity.push(entity_name);
            idx
        }
    }

    /// Get the index for an entity name
    pub fn get_entity_index(&self, entity_name: &str) -> Option<usize> {
        self.entity_to_index.get(entity_name).copied()
    }

    /// Get the entity name for an index
    pub fn get_entity_name(&self, index: usize) -> Option<&str> {
        self.index_to_entity.get(index).map(|s| s.as_str())
    }

    /// Get the total number of entities
    pub fn entity_count(&self) -> usize {
        self.index_to_entity.len()
    }

    /// Get all entity names
    pub fn all_entities(&self) -> &[String] {
        &self.index_to_entity
    }

    /// Check if an entity exists
    pub fn has_entity(&self, entity_name: &str) -> bool {
        self.entity_to_index.contains_key(entity_name)
    }
}

/// Entity Registry
///
/// Central registry for managing entity types and their instances.
#[derive(Debug, Clone)]
pub struct EntityRegistry {
    /// Entity type name -> EntityTypeInfo mapping
    entity_types: HashMap<String, EntityTypeInfo>,
}

impl EntityRegistry {
    /// Create a new empty entity registry
    pub fn new() -> Self {
        Self {
            entity_types: HashMap::new(),
        }
    }

    /// Register an entity type from declaration
    pub fn register_from_decl(&mut self, decl: &EntityDecl) {
        let type_info = match decl {
            EntityDecl::Explicit { name, entities } => {
                let entity_names: Vec<String> = entities
                    .iter()
                    .map(|id| id.as_str().to_string())
                    .collect();
                EntityTypeInfo::explicit(name.as_str().to_string(), entity_names)
            }
            EntityDecl::FromData { name } => {
                EntityTypeInfo::from_data(name.as_str().to_string())
            }
        };

        self.entity_types.insert(type_info.name.clone(), type_info);
    }

    /// Register an entity type with explicit entities
    pub fn register_explicit(&mut self, type_name: String, entities: Vec<String>) {
        let type_info = EntityTypeInfo::explicit(type_name.clone(), entities);
        self.entity_types.insert(type_name, type_info);
    }

    /// Register an entity type for data-driven construction
    pub fn register_from_data(&mut self, type_name: String) {
        let type_info = EntityTypeInfo::from_data(type_name.clone());
        self.entity_types.insert(type_name, type_info);
    }

    /// Add an entity instance to a type (for data-driven types)
    pub fn add_entity(&mut self, type_name: &str, entity_name: String) -> Result<usize, String> {
        if let Some(type_info) = self.entity_types.get_mut(type_name) {
            Ok(type_info.add_entity(entity_name))
        } else {
            Err(format!("Entity type '{}' not registered", type_name))
        }
    }

    /// Get entity type information
    pub fn get_type_info(&self, type_name: &str) -> Option<&EntityTypeInfo> {
        self.entity_types.get(type_name)
    }

    /// Get entity type information (mutable)
    pub fn get_type_info_mut(&mut self, type_name: &str) -> Option<&mut EntityTypeInfo> {
        self.entity_types.get_mut(type_name)
    }

    /// Get the index for an entity in its type
    pub fn get_entity_index(&self, type_name: &str, entity_name: &str) -> Option<usize> {
        self.entity_types
            .get(type_name)
            .and_then(|info| info.get_entity_index(entity_name))
    }

    /// Get the entity name for an index in a type
    pub fn get_entity_name(&self, type_name: &str, index: usize) -> Option<&str> {
        self.entity_types
            .get(type_name)
            .and_then(|info| info.get_entity_name(index))
    }

    /// Get the number of entities in a type
    pub fn get_entity_count(&self, type_name: &str) -> Option<usize> {
        self.entity_types.get(type_name).map(|info| info.entity_count())
    }

    /// Check if an entity type exists
    pub fn has_type(&self, type_name: &str) -> bool {
        self.entity_types.contains_key(type_name)
    }

    /// Check if an entity exists in a type
    pub fn has_entity(&self, type_name: &str, entity_name: &str) -> bool {
        self.entity_types
            .get(type_name)
            .map(|info| info.has_entity(entity_name))
            .unwrap_or(false)
    }

    /// Get all registered entity type names
    pub fn all_type_names(&self) -> Vec<&str> {
        self.entity_types.keys().map(|s| s.as_str()).collect()
    }

    /// Collect entities from a fact's atom
    ///
    /// For data-driven entity types, automatically add entity instances
    /// encountered in facts to the registry.
    pub fn collect_entities_from_atom(&mut self, predicate: &str, terms: &[String]) {
        // This is a placeholder - in a full implementation, we would:
        // 1. Look up the relation signature to determine parameter types
        // 2. For each parameter with an entity type that's data-driven
        // 3. Add the corresponding term value to that entity type

        // For now, we'll implement this when we integrate with the interpreter
        let _ = (predicate, terms);
    }

    /// Get entity indices for a list of entity names
    ///
    /// Returns a vector of indices for the given entity names in the specified type.
    /// Returns None if any entity is not found.
    pub fn get_entity_indices(&self, type_name: &str, entity_names: &[String]) -> Option<Vec<usize>> {
        let type_info = self.entity_types.get(type_name)?;
        let mut indices = Vec::new();

        for name in entity_names {
            if let Some(idx) = type_info.get_entity_index(name) {
                indices.push(idx);
            } else {
                return None; // Entity not found
            }
        }

        Some(indices)
    }

    /// Create a one-hot encoding vector for an entity
    ///
    /// Returns a vector of 0s with a single 1 at the entity's index position.
    /// Vector length equals the total number of entities in the type.
    pub fn entity_to_onehot(&self, type_name: &str, entity_name: &str) -> Option<Vec<f32>> {
        let type_info = self.entity_types.get(type_name)?;
        let entity_count = type_info.entity_count();
        let entity_idx = type_info.get_entity_index(entity_name)?;

        let mut onehot = vec![0.0_f32; entity_count];
        onehot[entity_idx] = 1.0;

        Some(onehot)
    }

    /// Get the entity dimension (number of entities in a type)
    ///
    /// Useful for creating appropriately-sized tensors for entity operations.
    pub fn get_entity_dimension(&self, type_name: &str) -> Option<usize> {
        self.get_entity_count(type_name)
    }
}

impl Default for EntityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_entity_type() {
        let mut registry = EntityRegistry::new();

        registry.register_explicit(
            "Person".to_string(),
            vec!["alice".to_string(), "bob".to_string(), "charlie".to_string()],
        );

        assert!(registry.has_type("Person"));
        assert_eq!(registry.get_entity_count("Person"), Some(3));

        assert_eq!(registry.get_entity_index("Person", "alice"), Some(0));
        assert_eq!(registry.get_entity_index("Person", "bob"), Some(1));
        assert_eq!(registry.get_entity_index("Person", "charlie"), Some(2));

        assert_eq!(registry.get_entity_name("Person", 0), Some("alice"));
        assert_eq!(registry.get_entity_name("Person", 1), Some("bob"));
        assert_eq!(registry.get_entity_name("Person", 2), Some("charlie"));
    }

    #[test]
    fn test_data_driven_entity_type() {
        let mut registry = EntityRegistry::new();

        registry.register_from_data("City".to_string());

        assert!(registry.has_type("City"));
        assert_eq!(registry.get_entity_count("City"), Some(0));

        // Add entities dynamically
        registry.add_entity("City", "tokyo".to_string()).unwrap();
        registry.add_entity("City", "london".to_string()).unwrap();
        registry.add_entity("City", "paris".to_string()).unwrap();

        assert_eq!(registry.get_entity_count("City"), Some(3));
        assert_eq!(registry.get_entity_index("City", "tokyo"), Some(0));
        assert_eq!(registry.get_entity_index("City", "london"), Some(1));
        assert_eq!(registry.get_entity_index("City", "paris"), Some(2));
    }

    #[test]
    fn test_duplicate_entity_in_data_driven() {
        let mut registry = EntityRegistry::new();

        registry.register_from_data("City".to_string());

        let idx1 = registry.add_entity("City", "tokyo".to_string()).unwrap();
        let idx2 = registry.add_entity("City", "tokyo".to_string()).unwrap();

        // Should return same index for duplicate
        assert_eq!(idx1, idx2);
        assert_eq!(registry.get_entity_count("City"), Some(1));
    }

    #[test]
    fn test_entity_indices() {
        let mut registry = EntityRegistry::new();

        registry.register_explicit(
            "Person".to_string(),
            vec!["alice".to_string(), "bob".to_string(), "charlie".to_string()],
        );

        let names = vec!["alice".to_string(), "charlie".to_string()];
        let indices = registry.get_entity_indices("Person", &names);

        assert_eq!(indices, Some(vec![0, 2]));
    }

    #[test]
    fn test_entity_to_onehot() {
        let mut registry = EntityRegistry::new();

        registry.register_explicit(
            "Color".to_string(),
            vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        );

        // Test one-hot encoding for "green" (index 1)
        let onehot = registry.entity_to_onehot("Color", "green");
        assert_eq!(onehot, Some(vec![0.0, 1.0, 0.0]));

        // Test one-hot encoding for "blue" (index 2)
        let onehot = registry.entity_to_onehot("Color", "blue");
        assert_eq!(onehot, Some(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_entity_dimension() {
        let mut registry = EntityRegistry::new();

        registry.register_explicit(
            "Animal".to_string(),
            vec!["cat".to_string(), "dog".to_string()],
        );

        assert_eq!(registry.get_entity_dimension("Animal"), Some(2));

        // Add more entities to a data-driven type
        registry.register_from_data("Plant".to_string());
        registry.add_entity("Plant", "tree".to_string()).unwrap();
        registry.add_entity("Plant", "flower".to_string()).unwrap();

        assert_eq!(registry.get_entity_dimension("Plant"), Some(2));
    }
}
