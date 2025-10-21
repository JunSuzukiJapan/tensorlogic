//! Python execution environment for TensorLogic
//!
//! This module provides a wrapper around PyO3 to manage Python execution,
//! module imports, and function calls from TensorLogic code.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use half::f16;

use crate::tensor::Tensor;

/// Python execution environment
pub struct PythonEnvironment {
    /// Imported Python modules
    modules: HashMap<String, Py<PyAny>>,
}

impl PythonEnvironment {
    /// Create a new Python environment
    pub fn new() -> Self {
        PythonEnvironment {
            modules: HashMap::new(),
        }
    }

    /// Import a Python module
    ///
    /// # Arguments
    /// * `module_path` - The module to import (e.g., "numpy", "torch")
    /// * `alias` - Optional alias for the module (e.g., "np" for numpy)
    ///
    /// # Example
    /// ```ignore
    /// env.import_module("numpy", Some("np"))?;
    /// ```
    pub fn import_module(&mut self, module_path: &str, alias: Option<&str>) -> Result<(), String> {
        Python::with_gil(|py| {
            // Import the module
            let module = py.import_bound(module_path)
                .map_err(|e| format!("Failed to import module '{}': {}", module_path, e))?;

            // Store with alias or original name
            let name = alias.unwrap_or(module_path).to_string();
            // Convert PyModule to PyAny for storage
            let module_any: Py<PyAny> = module.into_any().unbind();
            self.modules.insert(name, module_any);

            Ok(())
        })
    }

    /// Call a Python function with tensor arguments
    ///
    /// # Arguments
    /// * `function_path` - The function to call (e.g., "np.sum", "torch.mean")
    /// * `args` - Tensor arguments to pass to the function
    ///
    /// # Returns
    /// A tensor containing the result
    ///
    /// # Example
    /// ```ignore
    /// let result = env.call_function("np.sum", vec![x])?;
    /// ```
    pub fn call_function(&self, function_path: &str, args: Vec<&Tensor>) -> Result<Tensor, String> {
        Python::with_gil(|py| {
            // Parse function path (e.g., "np.sum" -> module="np", function="sum")
            let parts: Vec<&str> = function_path.split('.').collect();
            if parts.len() < 2 {
                return Err(format!(
                    "Invalid function path '{}'. Expected format: 'module.function'",
                    function_path
                ));
            }

            let module_name = parts[0];
            let func_name = parts[1..].join(".");

            // Get the module
            let module = self.modules.get(module_name)
                .ok_or_else(|| format!("Module '{}' not imported", module_name))?;

            // Get the function
            let func = module.bind(py).getattr(func_name.as_str())
                .map_err(|e| format!("Failed to get function '{}': {}", func_name, e))?;

            // Convert tensors to NumPy arrays
            let py_args: Result<Vec<PyObject>, String> = args.iter()
                .map(|tensor| tensor_to_numpy(py, tensor))
                .collect();
            let py_args = py_args?;

            // Call the function
            let result = func.call1(PyTuple::new_bound(py, py_args))
                .map_err(|e| format!("Failed to call function '{}': {}", function_path, e))?;

            // Convert result back to tensor
            numpy_to_tensor(py, &result)
        })
    }

    /// Check if a module is imported
    pub fn has_module(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Get list of imported modules
    pub fn imported_modules(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }
}

/// Convert a TensorLogic Tensor to a NumPy array
fn tensor_to_numpy(py: Python<'_>, tensor: &Tensor) -> Result<PyObject, String> {
    use numpy::{PyArray, PyArrayMethods};

    // Get tensor data as CPU tensor
    let cpu_tensor = tensor.to_cpu()
        .map_err(|e| format!("Failed to move tensor to CPU: {}", e))?;

    // Convert f16 to f32 for NumPy compatibility
    let f16_data = cpu_tensor.to_vec();
    let f32_data: Vec<f32> = f16_data.iter().map(|&v| v.to_f32()).collect();
    let shape = tensor.dims();

    // Create NumPy array and reshape
    let array = PyArray::from_vec_bound(py, f32_data);
    let reshaped = PyArrayMethods::reshape(&array, shape)
        .map_err(|e| format!("Failed to reshape array: {}", e))?;

    Ok(reshaped.unbind().into())
}

/// Convert a NumPy array to a TensorLogic Tensor
fn numpy_to_tensor(_py: Python<'_>, array: &Bound<'_, PyAny>) -> Result<Tensor, String> {
    use numpy::{PyReadonlyArray, PyUntypedArrayMethods, ndarray};

    // Try to extract as float32 array
    if let Ok(arr) = array.extract::<PyReadonlyArray<f32, ndarray::IxDyn>>() {
        let data: Vec<f16> = arr.as_slice()
            .map_err(|e| format!("Failed to get array slice: {}", e))?
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let shape = arr.shape().to_vec();

        let tensor = Tensor::from_vec(data, shape)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        return Ok(tensor);
    }

    // Try float64
    if let Ok(arr) = array.extract::<PyReadonlyArray<f64, ndarray::IxDyn>>() {
        let data: Vec<f16> = arr.as_slice()
            .map_err(|e| format!("Failed to get array slice: {}", e))?
            .iter()
            .map(|&v| f16::from_f64(v))
            .collect();
        let shape = arr.shape().to_vec();

        let tensor = Tensor::from_vec(data, shape)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        return Ok(tensor);
    }

    // Try to extract as scalar
    if let Ok(val) = array.extract::<f64>() {
        let tensor = Tensor::from_vec(vec![f16::from_f64(val)], vec![1])
            .map_err(|e| format!("Failed to create scalar tensor: {}", e))?;
        return Ok(tensor);
    }

    if let Ok(val) = array.extract::<f32>() {
        let tensor = Tensor::from_vec(vec![f16::from_f32(val)], vec![1])
            .map_err(|e| format!("Failed to create scalar tensor: {}", e))?;
        return Ok(tensor);
    }

    Err("Failed to convert Python object to tensor: unsupported type".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_environment() {
        let env = PythonEnvironment::new();
        assert_eq!(env.imported_modules().len(), 0);
    }

    #[test]
    fn test_import_numpy() {
        let mut env = PythonEnvironment::new();
        env.import_module("numpy", Some("np")).unwrap();
        assert!(env.has_module("np"));
        assert_eq!(env.imported_modules(), vec!["np"]);
    }

    #[test]
    fn test_numpy_sum() {
        let mut env = PythonEnvironment::new();
        env.import_module("numpy", Some("np")).unwrap();

        // Create a test tensor
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let tensor = Tensor::from_vec(data, vec![3]).unwrap();

        // Call np.sum
        let result = env.call_function("np.sum", vec![&tensor]).unwrap();

        // Result should be [6.0] with shape [1]
        let result_data = result.to_vec();
        assert_eq!(result_data.len(), 1);
        assert!((result_data[0].to_f32() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_numpy_add() {
        let mut env = PythonEnvironment::new();
        env.import_module("numpy", Some("np")).unwrap();

        // Create test tensors
        let x = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
            vec![3]
        ).unwrap();

        let y = Tensor::from_vec(
            vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
            vec![3]
        ).unwrap();

        // Call np.add
        let result = env.call_function("np.add", vec![&x, &y]).unwrap();

        // Result should be [5.0, 7.0, 9.0]
        let result_data = result.to_vec();
        assert_eq!(result_data.len(), 3);
        assert!((result_data[0].to_f32() - 5.0).abs() < 0.01);
        assert!((result_data[1].to_f32() - 7.0).abs() < 0.01);
        assert!((result_data[2].to_f32() - 9.0).abs() < 0.01);
    }
}
