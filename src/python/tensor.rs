//! Tensor ↔ NumPy conversion
//!
//! This module handles conversion between TensorLogic Tensors and NumPy arrays,
//! maintaining f16 precision and shape information.

use pyo3::prelude::*;
use numpy::{PyArray, PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods, ndarray};
use half::f16;

use crate::tensor::Tensor;

/// Python wrapper for TensorLogic Tensor
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from a Python list and shape
    #[new]
    #[pyo3(signature = (data, shape))]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        // Convert f32 to f16
        let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();

        let tensor = Tensor::from_vec(f16_data, shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(PyTensor { inner: tensor })
    }

    /// Create tensor from NumPy array
    #[staticmethod]
    fn from_numpy(_py: Python, array: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try float32 first (most common)
        if let Ok(arr) = array.extract::<PyReadonlyArray<f32, ndarray::IxDyn>>() {
            let data: Vec<f16> = arr.as_slice()?
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect();
            let shape = arr.shape().to_vec();

            let tensor = Tensor::from_vec(data, shape)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            return Ok(PyTensor { inner: tensor });
        }

        // Try float64 and convert
        if let Ok(arr) = array.extract::<PyReadonlyArray<f64, ndarray::IxDyn>>() {
            let data: Vec<f16> = arr.as_slice()?
                .iter()
                .map(|&v| f16::from_f64(v))
                .collect();
            let shape = arr.shape().to_vec();

            let tensor = Tensor::from_vec(data, shape)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            return Ok(PyTensor { inner: tensor });
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected NumPy array with dtype float32 or float64"
        ))
    }

    /// Convert tensor to NumPy array (as f32 for compatibility)
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray<f32, ndarray::IxDyn>>> {
        // Get data from CPU (copy from Metal if necessary)
        let cpu_tensor = self.inner.to_cpu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Convert f16 to f32 for NumPy compatibility
        let f16_data = cpu_tensor.to_vec();
        let f32_data: Vec<f32> = f16_data.iter().map(|&v| v.to_f32()).collect();
        let shape = self.inner.dims();

        // Create NumPy array
        let array = PyArray::from_vec_bound(py, f32_data);
        let reshaped = array.reshape(shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(reshaped.unbind())
    }

    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    /// Get tensor rank (number of dimensions)
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.dims().len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, dtype=float16)", self.shape())
    }

    /// Convert to Python list (for small tensors)
    fn tolist(&self) -> PyResult<Vec<f32>> {
        let cpu_tensor = self.inner.to_cpu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let data: Vec<f32> = cpu_tensor.to_vec()
            .iter()
            .map(|&v| v.to_f32())
            .collect();

        Ok(data)
    }

    /// Get requires_grad flag
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Set requires_grad flag
    #[setter]
    fn set_requires_grad(&mut self, requires: bool) {
        self.inner.set_requires_grad(requires);
    }

    /// Get gradient tensor (if available)
    #[getter]
    fn grad(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(grad_tensor) = self.inner.grad() {
            let py_grad = PyTensor::from_tensor(grad_tensor.clone());
            Ok(Some(py_grad.into_py(py)))
        } else {
            Ok(None)
        }
    }

    /// Zero the gradient
    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    /// Perform backward pass
    fn backward(&mut self) -> PyResult<()> {
        self.inner.backward()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Detach tensor from computation graph
    fn detach(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.detach())
    }

    /// Element-wise addition
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.inner + &other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Element-wise subtraction
    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.inner - &other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Element-wise multiplication
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.inner * &other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.matmul(&other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// ReLU activation
    fn relu(&self) -> PyResult<PyTensor> {
        let result = self.inner.relu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Softmax activation
    fn softmax(&self, dim: isize) -> PyResult<PyTensor> {
        let result = self.inner.softmax(dim)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Sum reduction
    fn sum(&self) -> PyResult<PyTensor> {
        let result = self.inner.sum()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Mean reduction
    fn mean(&self) -> PyResult<PyTensor> {
        let result = self.inner.mean()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyTensor::from_tensor(result))
    }
}

impl PyTensor {
    /// Create PyTensor from internal Tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        PyTensor { inner: tensor }
    }

    /// Get reference to internal Tensor
    pub fn as_tensor(&self) -> &Tensor {
        &self.inner
    }

    /// Get mutable reference to internal Tensor
    pub fn as_tensor_mut(&mut self) -> &mut Tensor {
        &mut self.inner
    }

    /// Take ownership of internal Tensor
    pub fn into_tensor(self) -> Tensor {
        self.inner
    }
}

// Enable Python → Rust conversion
impl<'py> FromPyObject<'py> for PyTensor {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try direct extraction
        if let Ok(tensor) = ob.extract::<PyRef<PyTensor>>() {
            return Ok(PyTensor {
                inner: tensor.inner.clone(),
            });
        }

        // Try NumPy array conversion
        PyTensor::from_numpy(ob.py(), ob)
    }
}

// Enable Rust → Python conversion
impl ToPyObject for PyTensor {
    fn to_object(&self, py: Python) -> PyObject {
        match self.to_numpy(py) {
            Ok(array) => array.to_object(py),
            Err(e) => {
                // If conversion fails, return error
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    .to_object(py)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];

            let tensor = PyTensor::new(data, shape).unwrap();
            assert_eq!(tensor.shape(), vec![2, 2]);
            assert_eq!(tensor.ndim(), 2);
        });
    }

    #[test]
    fn test_numpy_conversion() {
        use numpy::PyUntypedArrayMethods;
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0];
            let shape = vec![3];

            let tensor = PyTensor::new(data, shape).unwrap();
            let numpy_array = tensor.to_numpy(py).unwrap();

            // numpy_arrayをbindしてshape()を呼び出す
            assert_eq!(numpy_array.bind(py).shape(), &[3]);
        });
    }
}
