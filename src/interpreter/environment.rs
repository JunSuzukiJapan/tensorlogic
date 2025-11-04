//! Runtime environment and scope management for TensorLogic interpreter
//!
//! This module implements a scope stack architecture for proper variable lifetime management.
//! Variables are automatically dropped when their scope ends (block, loop, function).

use std::collections::HashMap;
use crate::device::MetalDevice;
use super::value::Value;
use super::{RuntimeError, RuntimeResult};

/// Types of scopes in the interpreter
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    /// Global scope (program lifetime)
    Global,
    /// Function scope (function name)
    Function(String),
    /// Block scope (generic { } block)
    Block,
    /// Loop scope (for/while loop body)
    Loop,
}

/// A single scope containing variables
/// When a scope is dropped, all variables in it are automatically dropped
#[derive(Debug)]
pub struct Scope {
    /// Type of this scope
    pub scope_type: ScopeType,
    /// Variables defined in this scope
    /// When the HashMap is dropped, all Value instances are dropped (Arc ref_count decremented)
    pub variables: HashMap<String, Value>,
}

impl Scope {
    /// Create a new empty scope
    pub fn new(scope_type: ScopeType) -> Self {
        Self {
            scope_type,
            variables: HashMap::new(),
        }
    }

    /// Check if a variable exists in this scope
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Declare a variable in this scope
    /// If the variable already exists in this scope, the old value is dropped
    pub fn declare_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Get a variable from this scope (read-only)
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Set a variable in this scope (if it exists)
    pub fn set_variable(&mut self, name: &str, value: Value) -> bool {
        if self.variables.contains_key(name) {
            self.variables.insert(name.to_string(), value);
            true
        } else {
            false
        }
    }
}

/// Runtime environment with scope stack management
#[derive(Debug)]
pub struct RuntimeEnvironment {
    /// Scope stack (bottom to top: Global -> Function -> Block -> Loop)
    /// Variables are resolved by searching from top to bottom
    /// When a scope is popped, all its variables are automatically dropped
    pub(super) scope_stack: Vec<Scope>,

    /// Current Metal device for tensor operations
    metal_device: MetalDevice,
}

impl RuntimeEnvironment {
    /// Create a new runtime environment with global scope
    pub fn new() -> Self {
        let mut scope_stack = Vec::new();
        // Initialize with global scope
        scope_stack.push(Scope::new(ScopeType::Global));

        Self {
            scope_stack,
            metal_device: MetalDevice::new().unwrap(),
        }
    }

    /// Push a new scope onto the stack
    pub fn push_scope(&mut self, scope_type: ScopeType) {
        self.scope_stack.push(Scope::new(scope_type));
    }

    /// Pop the current scope from the stack
    /// All variables in the popped scope are automatically dropped
    /// Returns the popped scope, or None if trying to pop global scope
    pub fn pop_scope(&mut self) -> Option<Scope> {
        // Never pop the global scope (keep at least one scope)
        if self.scope_stack.len() > 1 {
            self.scope_stack.pop()
        } else {
            None
        }
    }

    /// Get the current (top) scope
    fn current_scope(&self) -> &Scope {
        self.scope_stack.last().expect("Scope stack should never be empty")
    }

    /// Get the current (top) scope mutably
    fn current_scope_mut(&mut self) -> &mut Scope {
        self.scope_stack.last_mut().expect("Scope stack should never be empty")
    }

    /// Declare a variable in the current scope
    /// If the variable already exists in the current scope, the old value is dropped
    pub fn declare_variable(&mut self, name: String, value: Value) -> RuntimeResult<()> {
        self.current_scope_mut().declare_variable(name, value);
        Ok(())
    }

    /// Get a variable by searching the scope stack (from top to bottom)
    /// This implements lexical scoping - inner scopes can access outer scopes
    pub fn get_variable(&self, name: &str) -> RuntimeResult<Value> {
        // Search from innermost scope to outermost
        for scope in self.scope_stack.iter().rev() {
            if let Some(value) = scope.get_variable(name) {
                return Ok(value.clone());
            }
        }

        Err(RuntimeError::UndefinedVariable(name.to_string()))
    }

    /// Set a variable by searching the scope stack (from top to bottom)
    /// Updates the first occurrence found
    /// Returns error if variable is not defined in any scope
    pub fn set_variable(&mut self, name: &str, value: Value) -> RuntimeResult<()> {
        // Search from innermost scope to outermost
        for scope in self.scope_stack.iter_mut().rev() {
            if scope.set_variable(name, value.clone()) {
                return Ok(());
            }
        }

        Err(RuntimeError::UndefinedVariable(name.to_string()))
    }

    /// Check if a variable exists in any scope
    pub fn has_variable(&self, name: &str) -> bool {
        self.scope_stack.iter().rev().any(|scope| scope.has_variable(name))
    }

    /// List all variable names across all scopes
    pub fn list_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for scope in &self.scope_stack {
            vars.extend(scope.variables.keys().cloned());
        }
        vars
    }

    /// Get current Metal device
    pub fn metal_device(&self) -> &MetalDevice {
        &self.metal_device
    }

    /// Get the current scope depth (0 = global scope only)
    pub fn scope_depth(&self) -> usize {
        self.scope_stack.len() - 1
    }
}

impl Default for RuntimeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Function call frame for stack trace (no longer holds variables)
/// Variables are now managed by the scope stack
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Name of the function being executed
    pub function_name: String,
}

impl CallFrame {
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_stack_basic() {
        let mut env = RuntimeEnvironment::new();

        // Global scope
        env.declare_variable("x".to_string(), Value::Float(10.0)).unwrap();
        assert_eq!(env.scope_depth(), 0);

        // Push block scope
        env.push_scope(ScopeType::Block);
        assert_eq!(env.scope_depth(), 1);

        // Variable from outer scope should be accessible
        assert!(env.has_variable("x"));

        // Declare variable in inner scope
        env.declare_variable("y".to_string(), Value::Float(20.0)).unwrap();
        assert!(env.has_variable("y"));

        // Pop scope - y should be dropped
        env.pop_scope();
        assert_eq!(env.scope_depth(), 0);
        assert!(env.has_variable("x"));
        assert!(!env.has_variable("y"));
    }

    #[test]
    fn test_variable_shadowing() {
        let mut env = RuntimeEnvironment::new();

        // Global x = 10
        env.declare_variable("x".to_string(), Value::Float(10.0)).unwrap();

        // Push scope, shadow x = 20
        env.push_scope(ScopeType::Block);
        env.declare_variable("x".to_string(), Value::Float(20.0)).unwrap();

        // Should get shadowed value
        match env.get_variable("x").unwrap() {
            Value::Float(v) => assert_eq!(v, 20.0),
            _ => panic!("Expected Float"),
        }

        // Pop scope - should restore original x
        env.pop_scope();
        match env.get_variable("x").unwrap() {
            Value::Float(v) => assert_eq!(v, 10.0),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_cannot_pop_global_scope() {
        let mut env = RuntimeEnvironment::new();

        assert_eq!(env.scope_depth(), 0);
        assert!(env.pop_scope().is_none());
        assert_eq!(env.scope_depth(), 0);
    }
}
