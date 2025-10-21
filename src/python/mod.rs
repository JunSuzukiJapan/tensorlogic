//! Python bindings for TensorLogic using PyO3
//!
//! This module provides Python FFI for TensorLogic, enabling:
//! - Tensor ↔ NumPy array conversion
//! - TensorLogic interpreter access from Python
//! - Python function calls from TensorLogic

#[cfg(any(feature = "python", feature = "python-extension"))]
pub mod tensor;

#[cfg(any(feature = "python", feature = "python-extension"))]
pub mod interpreter;

#[cfg(any(feature = "python", feature = "python-extension"))]
pub mod environment;

#[cfg(any(feature = "python", feature = "python-extension"))]
use pyo3::prelude::*;

/// TensorLogic Python module (internal _native module)
#[cfg(any(feature = "python", feature = "python-extension"))]
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<interpreter::PyInterpreter>()?;
    Ok(())
}
