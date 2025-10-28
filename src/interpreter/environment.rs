//! Runtime environment and scope management for TensorLogic interpreter

use std::collections::HashMap;
use crate::device::MetalDevice;
use super::value::Value;
use super::{RuntimeError, RuntimeResult};

/// Function call frame for local scope management
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Name of the function being executed
    pub function_name: String,
    /// Local variables in this scope
    pub local_vars: HashMap<String, Value>,
}

impl CallFrame {
    pub fn new(function_name: String) -> Self {
        Self {
            function_name,
            local_vars: HashMap::new(),
        }
    }
}

/// Runtime environment
#[derive(Debug)]
pub struct RuntimeEnvironment {
    /// Variable name → value
    pub(super) variables: HashMap<String, Value>,
    /// Current Metal device for tensor operations
    metal_device: MetalDevice,
}

impl RuntimeEnvironment {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            metal_device: MetalDevice::new().unwrap(),
        }
    }

    /// Check if a variable exists
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Declare a new variable (allows shadowing of existing variables)
    pub fn declare_variable(&mut self, name: String, value: Value) -> RuntimeResult<()> {
        // Allow shadowing - just insert/overwrite the variable
        // The execute_block() function will handle restoration of shadowed values
        self.variables.insert(name, value);
        Ok(())
    }

    /// Set a variable (update existing or error if not defined)
    pub fn set_variable(&mut self, name: String, value: Value) -> RuntimeResult<()> {
        if !self.variables.contains_key(&name) {
            return Err(RuntimeError::UndefinedVariable(name));
        }
        self.variables.insert(name, value);
        Ok(())
    }

    /// Get a variable
    pub fn get_variable(&self, name: &str) -> RuntimeResult<&Value> {
        self.variables
            .get(name)
            .ok_or_else(|| RuntimeError::UndefinedVariable(name.to_string()))
    }

    /// List all variable names
    pub fn list_variables(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Clear all variables except the ones specified (for memory cleanup)
    pub fn clear_except(&mut self, keep: &[String]) {
        self.variables.retain(|k, _| keep.contains(k));
    }

    /// Get current Metal device
    pub fn metal_device(&self) -> &MetalDevice {
        &self.metal_device
    }
}

impl Default for RuntimeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}
