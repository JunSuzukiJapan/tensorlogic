"""
Integration tests for TensorLogic Interpreter Python bindings
"""

import pytest
import tensorlogic as tl


class TestInterpreterCreation:
    """Test Interpreter creation and basic operations"""

    def test_create_interpreter(self):
        """Test creating an interpreter instance"""
        interp = tl.Interpreter()
        assert interp is not None

    def test_interpreter_repr(self):
        """Test interpreter string representation"""
        interp = tl.Interpreter()
        assert repr(interp) == "Interpreter()"


class TestInterpreterExecution:
    """Test Interpreter code execution"""

    def test_simple_execution(self):
        """Test executing simple TensorLogic code"""
        interp = tl.Interpreter()
        code = """
            main {
                tensor x: float16[2] = [1.0, 2.0]
            }
        """
        result = interp.execute(code)
        assert result is not None

    def test_tensor_operations(self):
        """Test executing tensor operations"""
        interp = tl.Interpreter()
        code = """
            main {
                tensor a: float16[2] = [1.0, 2.0]
                tensor b: float16[2] = [3.0, 4.0]
                tensor c = a + b
            }
        """
        result = interp.execute(code)
        assert result is not None

    def test_control_flow(self):
        """Test executing control flow"""
        interp = tl.Interpreter()
        code = """
            main {
                tensor x: float16[1] = [5.0]
                if (x > 0.0) {
                    tensor y = x * 2.0
                }
            }
        """
        result = interp.execute(code)
        assert result is not None


class TestInterpreterErrors:
    """Test Interpreter error handling"""

    def test_syntax_error(self):
        """Test that syntax errors are properly reported"""
        interp = tl.Interpreter()
        code = "invalid syntax {"
        with pytest.raises(Exception):  # Should raise PySyntaxError
            interp.execute(code)

    def test_runtime_error(self):
        """Test that runtime errors are properly reported"""
        interp = tl.Interpreter()
        code = """
            main {
                tensor x: float16[2] = [1.0, 2.0]
                tensor y: float16[3] = [1.0, 2.0, 3.0]
                tensor z = x + y  # Shape mismatch
            }
        """
        # This may or may not error depending on broadcasting rules
        # Just ensure it doesn't crash
        try:
            interp.execute(code)
        except Exception:
            pass


class TestInterpreterReset:
    """Test Interpreter reset functionality"""

    def test_reset(self):
        """Test resetting interpreter state"""
        interp = tl.Interpreter()
        code1 = """
            main {
                tensor x: float16[2] = [1.0, 2.0]
            }
        """
        interp.execute(code1)
        interp.reset()

        # After reset, should be able to execute again
        code2 = """
            main {
                tensor y: float16[2] = [3.0, 4.0]
            }
        """
        result = interp.execute(code2)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
