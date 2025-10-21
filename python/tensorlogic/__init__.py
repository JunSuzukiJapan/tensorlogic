"""
TensorLogic: A unified tensor algebra and logic programming language

This package provides Python bindings for TensorLogic, enabling:
- High-performance f16 tensor operations on Apple Silicon
- Metal GPU acceleration via Apple's Neural Engine
- Integration with NumPy for data exchange
- TensorLogic interpreter for logic programming

Note: This package requires macOS with Apple Silicon (M1/M2/M3/M4) chips.
"""

__version__ = "0.1.0"

try:
    from ._native import Tensor, Interpreter
    __all__ = ["Tensor", "Interpreter", "__version__"]
except ImportError as e:
    raise ImportError(
        "Failed to import TensorLogic native module. "
        "Make sure you have installed the package with: pip install -e ."
    ) from e

# Optional Jupyter kernel support
try:
    from . import kernel
    __all__.append("kernel")
except ImportError:
    # ipykernel not installed, skip kernel support
    pass
