//! Python bindings for TensorLogic Interpreter

use pyo3::prelude::*;

use crate::interpreter::Interpreter;
use crate::parser::TensorLogicParser;
use super::tensor::PyTensor;

/// Python wrapper for TensorLogic Interpreter
#[pyclass(name = "Interpreter")]
pub struct PyInterpreter {
    inner: Interpreter,
}

#[pymethods]
impl PyInterpreter {
    /// Create a new interpreter
    #[new]
    fn new() -> Self {
        PyInterpreter {
            inner: Interpreter::new(),
        }
    }

    /// Execute TensorLogic code
    ///
    /// # Arguments
    /// * `code` - TensorLogic source code as a string
    ///
    /// # Returns
    /// * Execution result (if any)
    ///
    /// # Example
    /// ```python
    /// interp = Interpreter()
    /// result = interp.execute("""
    ///     main {
    ///         tensor x: float16[3] = [1.0, 2.0, 3.0]
    ///         print("x:", x)
    ///     }
    /// """)
    /// ```
    fn execute(&mut self, code: &str) -> PyResult<Option<String>> {
        // Parse code
        let program = TensorLogicParser::parse_program(code)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PySyntaxError, _>(e.to_string()))?;

        // Execute
        self.inner.execute(&program)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // TODO: Return actual result instead of None
        Ok(Some("Executed successfully".to_string()))
    }

    /// Get a variable from the interpreter's environment
    ///
    /// Returns None if variable doesn't exist
    fn get_variable(&self, py: Python<'_>, name: &str) -> PyResult<Option<PyObject>> {
        if let Some(value) = self.inner.get_variable(name) {
            match value {
                crate::interpreter::Value::Tensor(tensor) => {
                    let py_tensor = PyTensor::from_tensor(tensor.clone());
                    Ok(Some(py_tensor.into_py(py)))
                }
                crate::interpreter::Value::Boolean(b) => Ok(Some(b.into_py(py))),
                crate::interpreter::Value::Integer(i) => Ok(Some(i.into_py(py))),
                crate::interpreter::Value::Float(f) => Ok(Some(f.into_py(py))),
                crate::interpreter::Value::String(s) => Ok(Some(s.into_py(py))),
                crate::interpreter::Value::Void => Ok(Some(py.None())),
            }
        } else {
            Ok(None)
        }
    }

    /// Set a variable in the interpreter's environment
    fn set_variable(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(py_tensor) = value.extract::<PyTensor>() {
            let tensor = py_tensor.into_tensor();
            self.inner.set_variable(
                name.to_string(),
                crate::interpreter::Value::Tensor(tensor),
            );
            Ok(())
        } else if let Ok(b) = value.extract::<bool>() {
            self.inner.set_variable(
                name.to_string(),
                crate::interpreter::Value::Boolean(b),
            );
            Ok(())
        } else if let Ok(i) = value.extract::<i64>() {
            self.inner.set_variable(
                name.to_string(),
                crate::interpreter::Value::Integer(i),
            );
            Ok(())
        } else if let Ok(f) = value.extract::<f64>() {
            self.inner.set_variable(
                name.to_string(),
                crate::interpreter::Value::Float(f),
            );
            Ok(())
        } else if let Ok(s) = value.extract::<String>() {
            self.inner.set_variable(
                name.to_string(),
                crate::interpreter::Value::String(s),
            );
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Value must be Tensor, bool, int, float, or str"
            ))
        }
    }

    /// List all variables in the environment
    fn list_variables(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.list_variables())
    }

    /// Reset the interpreter (clear all variables)
    fn reset(&mut self) {
        self.inner = Interpreter::new();
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Interpreter()".to_string()
    }
}

impl PyInterpreter {
    /// Get reference to internal interpreter
    pub fn as_interpreter(&self) -> &Interpreter {
        &self.inner
    }

    /// Get mutable reference to internal interpreter
    pub fn as_interpreter_mut(&mut self) -> &mut Interpreter {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpreter_creation() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            let interp = PyInterpreter::new();
            assert_eq!(interp.__repr__(), "Interpreter()");
        });
    }

    #[test]
    fn test_simple_execution() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|_py| {
            let mut interp = PyInterpreter::new();

            let code = r#"
                main {
                    tensor x: float16[2] = [1.0, 2.0]
                }
            "#;

            let result = interp.execute(code);
            assert!(result.is_ok());
        });
    }
}
