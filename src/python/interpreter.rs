//! Python bindings for TensorLogic Interpreter

use pyo3::prelude::*;

use crate::interpreter::Interpreter;
use crate::parser::TensorLogicParser;

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

    // TODO: Implement variable access methods in Phase 2
    // /// Get a variable from the interpreter's environment
    // fn get_variable(&self, name: &str) -> PyResult<Option<PyTensor>> { ... }
    //
    // /// Set a variable in the interpreter's environment
    // fn set_variable(&mut self, name: &str, value: PyTensor) -> PyResult<()> { ... }

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
