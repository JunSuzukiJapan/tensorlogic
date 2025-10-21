"""
Integration tests for TensorLogic Tensor Python bindings
"""

import pytest
import numpy as np
import tensorlogic as tl


class TestTensorCreation:
    """Test Tensor creation and basic operations"""

    def test_create_from_list(self):
        """Test creating tensor from Python list"""
        data = [1.0, 2.0, 3.0, 4.0]
        shape = [2, 2]
        tensor = tl.Tensor(data, shape)
        assert tensor is not None
        assert tensor.shape() == shape

    def test_create_from_numpy_f32(self):
        """Test creating tensor from NumPy float32 array"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = tl.Tensor.from_numpy(arr)
        assert tensor is not None
        assert tensor.shape() == [2, 2]

    def test_create_from_numpy_f64(self):
        """Test creating tensor from NumPy float64 array"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tensor = tl.Tensor.from_numpy(arr)
        assert tensor is not None
        assert tensor.shape() == [2, 2]

    def test_create_from_numpy_f16(self):
        """Test creating tensor from NumPy float16 array"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        tensor = tl.Tensor.from_numpy(arr)
        assert tensor is not None
        assert tensor.shape() == [2, 2]


class TestTensorConversion:
    """Test Tensor to NumPy conversion"""

    def test_to_numpy(self):
        """Test converting tensor to NumPy array"""
        data = [1.0, 2.0, 3.0, 4.0]
        shape = [2, 2]
        tensor = tl.Tensor(data, shape)
        arr = tensor.to_numpy()
        assert arr is not None
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float16

    def test_roundtrip_conversion(self):
        """Test NumPy -> Tensor -> NumPy roundtrip"""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = tl.Tensor.from_numpy(original)
        result = tensor.to_numpy()

        # Allow small precision loss due to f16 conversion
        np.testing.assert_allclose(
            result.astype(np.float32),
            original,
            rtol=1e-3,
            atol=1e-3
        )


class TestTensorOperations:
    """Test Tensor operations"""

    def test_addition(self):
        """Test tensor addition"""
        a = tl.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = tl.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
        c = a.add(b)
        assert c is not None

        result = c.to_numpy()
        expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_multiplication(self):
        """Test tensor multiplication"""
        a = tl.Tensor([2.0, 3.0], [2])
        b = tl.Tensor([4.0, 5.0], [2])
        c = a.mul(b)
        assert c is not None

        result = c.to_numpy()
        expected = np.array([8.0, 15.0], dtype=np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-3)


class TestTensorProperties:
    """Test Tensor properties and metadata"""

    def test_shape(self):
        """Test getting tensor shape"""
        tensor = tl.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        shape = tensor.shape()
        assert shape == [2, 3]

    def test_device(self):
        """Test getting tensor device"""
        tensor = tl.Tensor([1.0, 2.0], [2])
        device = tensor.device()
        assert device in ["cpu", "metal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
