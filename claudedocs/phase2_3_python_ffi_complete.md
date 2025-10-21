# Phase 2 & 3: Python FFI Integration - Complete ✅

**Date**: 2025-10-21
**Status**: ✅ Complete
**Build**: ✅ Successful (tensorlogic-0.1.0-cp38-abi3-macosx_11_0_arm64.whl)
**Tests**: ✅ All integration tests passing

---

## Summary

Successfully implemented Phase 2 (AST & Parser Layer) and Phase 3 (Python Execution Engine) of the Jupyter Lab integration plan. TensorLogic can now:

- ✅ Import Python modules (e.g., `python import numpy as np`)
- ✅ Call Python functions with tensor arguments (e.g., `python.call("np.sum", x)`)
- ✅ Convert tensors between TensorLogic (f16) and NumPy (f32/f64)
- ✅ Execute Python code seamlessly from TensorLogic
- ✅ Build as both CLI binary and Python module

---

## Phase 2: AST & Parser Layer ✅

### AST Extensions

**[src/ast/mod.rs](../src/ast/mod.rs)**

Added two new AST node types:

```rust
pub enum Statement {
    // ... existing statements

    /// Python module import: python import module [as alias]
    PythonImport {
        module: String,
        alias: Option<String>,
    },
}

pub enum TensorExpr {
    // ... existing expressions

    /// Python function call: python.call("function", args...)
    PythonCall {
        function: String,
        args: Vec<TensorExpr>,
    },
}
```

### Parser Grammar

**[src/parser/grammar.pest](../src/parser/grammar.pest)**

Added Python-specific grammar rules:

```pest
// Python import: python import module [as alias]
python_import = { "python" ~ "import" ~ python_module ~ ("as" ~ identifier)? }
python_module = @{ identifier ~ ("." ~ identifier)* }

// Python function call: python.call("function_name", args)
python_call = { "python" ~ "." ~ "call" ~ "(" ~ string_literal ~ ("," ~ tensor_list)? ~ ")" }

// Updated statement to include python_import
statement = {
    tensor_decl
    | assignment
    | tensor_equation
    | python_import  // NEW
    | function_call
    // ...
}

// Updated tensor_term to include python_call
tensor_term = {
    "(" ~ tensor_expr ~ ")"
    | unary_op ~ tensor_term
    | tensor_index
    | embedding_lookup
    | einstein_sum
    | python_call  // NEW
    | function_call
    // ...
}

// Added keywords
reserved_keyword = {
    // ...
    | "python" | "import" | "as"
}
```

### Parser Implementation

**[src/parser/mod.rs](../src/parser/mod.rs)**

Added parsing functions:

```rust
fn parse_python_import(pair: pest::iterators::Pair<Rule>) -> Result<Statement, ParseError> {
    let mut inner = pair.into_inner();
    let module = inner.next()
        .ok_or_else(|| ParseError::MissingField("python module".to_string()))?
        .as_str()
        .to_string();
    let alias = inner.next().map(|alias_pair| alias_pair.as_str().to_string());
    Ok(Statement::PythonImport { module, alias })
}

fn parse_python_call(pair: pest::iterators::Pair<Rule>) -> Result<TensorExpr, ParseError> {
    let mut inner = pair.into_inner();
    let function = Self::parse_string_literal(inner.next().ok_or_else(|| {
        ParseError::MissingField("python function name".to_string())
    })?)?;
    let args = if let Some(tensor_list) = inner.next() {
        Self::parse_tensor_list(tensor_list)?
    } else {
        Vec::new()
    };
    Ok(TensorExpr::PythonCall { function, args })
}
```

### Lexer Updates

**[src/lexer/mod.rs](../src/lexer/mod.rs)**

Added new token types:

```rust
pub enum TokenType {
    // ... existing tokens
    Python,   // "python"
    Import,   // "import"
    As,       // "as"
    Dot,      // "."
}

// Keyword matching
"python" => TokenType::Python,
"import" => TokenType::Import,
"as" => TokenType::As,

// Dot operator
'.' => {
    self.advance();
    Ok(Token::new(TokenType::Dot, ".".to_string(), start_line, start_column))
}
```

### Visitor Pattern Support

**[src/ast/visitor.rs](../src/ast/visitor.rs)**

Added visitor implementations:

```rust
// In walk_tensor_expr()
TensorExpr::PythonCall { args, .. } => {
    for arg in args {
        visitor.visit_tensor_expr(arg)?;
    }
    Ok(())
}

// In walk_statement()
Statement::PythonImport { .. } => {
    Ok(())
}
```

### Type Inference

**[src/typecheck/mod.rs](../src/typecheck/mod.rs)**

Added type inference for Python calls:

```rust
TensorExpr::PythonCall { .. } => {
    // Dynamic type inference from Python function signature
    // Returns f32 with dynamic dimensions for now
    Ok(TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Dynamic],
    ))
}
```

### Parser Tests

**[tests/python_parser_test.rs](../tests/python_parser_test.rs)** (124 lines)

Created comprehensive test suite:

```rust
#[test]
fn test_parse_python_import() { /* ... */ }

#[test]
fn test_parse_python_import_without_alias() { /* ... */ }

#[test]
fn test_parse_python_call() { /* ... */ }

#[test]
fn test_parse_python_call_multiple_args() { /* ... */ }

#[test]
fn test_parse_combined_python_integration() { /* ... */ }
```

**Result**: All 5 tests passing ✅

---

## Phase 3: Python Execution Engine ✅

### Python Environment Wrapper

**[src/python/environment.rs](../src/python/environment.rs)** (234 lines)

Created complete Python execution environment:

```rust
pub struct PythonEnvironment {
    modules: HashMap<String, Py<PyAny>>,
}

impl PythonEnvironment {
    pub fn new() -> Self {
        PythonEnvironment {
            modules: HashMap::new(),
        }
    }

    pub fn import_module(&mut self, module_path: &str, alias: Option<&str>) -> Result<(), String> {
        Python::with_gil(|py| {
            let module = py.import_bound(module_path)
                .map_err(|e| format!("Failed to import module '{}': {}", module_path, e))?;
            let name = alias.unwrap_or(module_path).to_string();
            let module_any: Py<PyAny> = module.into_any().unbind();
            self.modules.insert(name, module_any);
            Ok(())
        })
    }

    pub fn call_function(&self, function_path: &str, args: Vec<&Tensor>) -> Result<Tensor, String> {
        Python::with_gil(|py| {
            // Parse function path (e.g., "np.sum" -> module="np", function="sum")
            let parts: Vec<&str> = function_path.split('.').collect();
            let module_name = parts[0];
            let func_name = parts[1..].join(".");

            // Get module and function
            let module = self.modules.get(module_name)?;
            let func = module.bind(py).getattr(func_name.as_str())?;

            // Convert tensors to NumPy arrays
            let py_args: Result<Vec<PyObject>, String> = args.iter()
                .map(|tensor| tensor_to_numpy(py, tensor))
                .collect();

            // Call function
            let result = func.call1(PyTuple::new_bound(py, py_args?))?;

            // Convert result back to tensor
            numpy_to_tensor(py, &result)
        })
    }
}
```

**Conversion Functions**:

```rust
fn tensor_to_numpy(py: Python<'_>, tensor: &Tensor) -> Result<PyObject, String> {
    use numpy::{PyArray, PyArrayMethods};

    // f16 → f32 → NumPy
    let cpu_tensor = tensor.to_cpu()?;
    let f16_data = cpu_tensor.to_vec();
    let f32_data: Vec<f32> = f16_data.iter().map(|&v| v.to_f32()).collect();
    let shape = tensor.dims();

    let array = PyArray::from_vec_bound(py, f32_data);
    let reshaped = PyArrayMethods::reshape(&array, shape)?;
    Ok(reshaped.unbind().into())
}

fn numpy_to_tensor(_py: Python<'_>, array: &Bound<'_, PyAny>) -> Result<Tensor, String> {
    use numpy::{PyReadonlyArray, PyUntypedArrayMethods, ndarray};

    // NumPy → f32/f64 → f16
    // Try float32, float64, and scalars
    if let Ok(arr) = array.extract::<PyReadonlyArray<f32, ndarray::IxDyn>>() {
        let data: Vec<f16> = arr.as_slice()?.iter().map(|&v| f16::from_f32(v)).collect();
        let shape = arr.shape().to_vec();
        return Tensor::from_vec(data, shape).map_err(|e| e.to_string());
    }
    // ... similar for f64 and scalars
}
```

### Interpreter Integration

**[src/interpreter/mod.rs](../src/interpreter/mod.rs)**

Integrated Python environment into Interpreter:

```rust
pub struct Interpreter {
    env: RuntimeEnvironment,
    logic_engine: LogicEngine,
    embeddings: HashMap<String, (HashMap<String, usize>, Tensor)>,
    #[cfg(any(feature = "python", feature = "python-extension"))]
    python_env: Option<crate::python::environment::PythonEnvironment>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: RuntimeEnvironment::new(),
            logic_engine: LogicEngine::new(),
            embeddings: HashMap::new(),
            #[cfg(any(feature = "python", feature = "python-extension"))]
            python_env: None,
        }
    }
}
```

**Statement Execution**:

```rust
Statement::PythonImport { module, alias } => {
    #[cfg(any(feature = "python", feature = "python-extension"))]
    {
        if self.python_env.is_none() {
            self.python_env = Some(crate::python::environment::PythonEnvironment::new());
        }
        let name = alias.as_deref();
        self.python_env.as_mut().unwrap()
            .import_module(module, name)
            .map_err(|e| RuntimeError::InvalidOperation(e))?;
        let display_name = alias.as_ref().unwrap_or(module);
        println!("✓ Python import: {} (as {})", module, display_name);
        Ok(())
    }
    #[cfg(not(any(feature = "python", feature = "python-extension")))]
    {
        Err(RuntimeError::NotImplemented(
            "Python integration not enabled".to_string()
        ))
    }
}
```

**Expression Evaluation**:

```rust
TensorExpr::PythonCall { function, args } => {
    #[cfg(any(feature = "python", feature = "python-extension"))]
    {
        if self.python_env.is_none() {
            return Err(RuntimeError::InvalidOperation(
                "Python environment not initialized".to_string()
            ));
        }

        // Evaluate all arguments
        let tensor_args: Result<Vec<_>, _> = args.iter()
            .map(|arg| {
                let val = self.eval_expr(arg)?;
                val.as_tensor().map(|t| t.clone())
            })
            .collect();

        let tensor_args = tensor_args?;
        let tensor_refs: Vec<&Tensor> = tensor_args.iter().collect();

        // Call Python function
        let result = self.python_env.as_ref().unwrap()
            .call_function(function, tensor_refs)
            .map_err(|e| RuntimeError::InvalidOperation(e))?;

        println!("✓ Python call: {}({} args)", function, args.len());
        Ok(Value::Tensor(result))
    }
    #[cfg(not(any(feature = "python", feature = "python-extension")))]
    {
        Err(RuntimeError::NotImplemented(
            "Python integration not enabled".to_string()
        ))
    }
}
```

### Build Configuration

**[Cargo.toml](../Cargo.toml)**

Added dual feature flags:

```toml
# Python統合
pyo3 = { version = "0.21", features = ["auto-initialize"], optional = true }
numpy = { version = "0.21", optional = true }

[features]
default = []
# python: CLIバイナリ用 (auto-initialize)
python = ["pyo3", "numpy"]
# python-extension: Pythonモジュール(.so)用 (extension-module)
python-extension = ["pyo3/extension-module", "pyo3/abi3-py38", "numpy"]
```

**[pyproject.toml](../pyproject.toml)**

maturin configuration:

```toml
[tool.maturin]
features = ["python-extension"]
python-source = "python"
module-name = "tensorlogic._native"
bindings = "pyo3"
```

---

## Testing & Validation ✅

### Integration Tests

**Test 1: Python Import**

```python
import tensorlogic as tl

code = """
main {
    python import numpy as np
    print("Python module imported successfully!")
}
"""

interp = tl.Interpreter()
interp.execute(code)
# Output: ✓ Python import: numpy (as np)
#         Python module imported successfully!
```

**Test 2: Simple Function Call**

```python
code = """
main {
    python import numpy as np

    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor sum_x: float16[1] = python.call("np.sum", x)
    print("Result:", sum_x)
}
"""
interp.execute(code)
# Output: ✓ Python call: np.sum(1 args)
#         Result: [6.0000]
```

**Test 3: Multiple Arguments**

```python
code = """
main {
    python import numpy as np

    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[3] = [4.0, 5.0, 6.0]

    tensor sum_result: float16[3] = python.call("np.add", x, y)
    print("np.add(x, y):", sum_result)
}
"""
interp.execute(code)
# Output: ✓ Python call: np.add(2 args)
#         np.add(x, y): [5.0000, 7.0000, 9.0000]
```

**Test 4: Various NumPy Functions**

All tested successfully ✅:

| Function | Input | Output | Status |
|----------|-------|--------|--------|
| `np.sum()` | `[1, 2, 3]` | `[6]` | ✅ |
| `np.add()` | `[1,2,3], [4,5,6]` | `[5,7,9]` | ✅ |
| `np.mean()` | `[1, 2, 3]` | `[2]` | ✅ |
| `np.max()` | `[4, 5, 6]` | `[6]` | ✅ |

### Parser Tests

**[tests/python_parser_test.rs](../tests/python_parser_test.rs)**

All 5 parser tests passing:

```bash
$ cargo test python_parser
running 5 tests
test test_parse_python_import ... ok
test test_parse_python_import_without_alias ... ok
test test_parse_python_call ... ok
test test_parse_python_call_multiple_args ... ok
test test_parse_combined_python_integration ... ok

test result: ok. 5 passed
```

---

## Example Usage

**[examples/python_integration_test.tl](../examples/python_integration_test.tl)**

```tensorlogic
// Python Integration Test Example
main {
    // Import Python modules
    python import numpy as np
    python import torch

    // Create tensor
    tensor x: float16[3] = [1.0, 2.0, 3.0]

    // Call Python functions
    tensor sum_x: float16[1] = python.call("np.sum", x)
    tensor mean_x: float16[1] = python.call("np.mean", x)

    print("Sum:", sum_x)
    print("Mean:", mean_x)
}
```

---

## Technical Implementation Details

### f16 ↔ NumPy Conversion Strategy

**Challenge**: numpy 0.21 doesn't support f16 as an Element type

**Solution**:
- **TensorLogic → NumPy**: f16 → f32 → NumPy array (float32)
- **NumPy → TensorLogic**: NumPy array (f32/f64) → f16 → Tensor
- Precision: Small precision loss acceptable for most ML operations
- GPU Support: Tensors on Metal GPU are moved to CPU for conversion

### Feature Flag Architecture

Two separate feature flags enable dual compilation modes:

1. **`python`**: For standalone CLI binary
   - Uses `auto-initialize` feature
   - Automatically initializes Python interpreter
   - Links to system Python
   - Use: `cargo build --features python`

2. **`python-extension`**: For Python module (.so)
   - Uses `extension-module` + `abi3-py38` features
   - No auto-initialization (Python already running)
   - Stable ABI for Python 3.8+
   - Use: `maturin build --features python-extension`

### Error Handling

Comprehensive error handling at each layer:

1. **Module Import Errors**: Missing modules, import failures
2. **Function Call Errors**: Invalid function names, wrong arguments
3. **Type Conversion Errors**: Unsupported NumPy types, shape mismatches
4. **Python Exceptions**: Propagated to TensorLogic as RuntimeError

---

## Known Limitations (Phase 2 & 3)

### Not Yet Implemented

❌ **Variable access from Python**:
```python
# TODO: Future enhancement
interp.get_variable("x")  # Not implemented
interp.set_variable("x", tensor)  # Not implemented
```

❌ **Advanced type inference**:
- Python function return types are currently inferred as `float32[Dynamic]`
- Future: Parse Python function signatures for accurate type inference

❌ **PyTorch integration**:
- Parser supports `python import torch` syntax
- Execution not yet tested with PyTorch tensors
- Future: Add PyTorch tensor conversion

---

## Files Modified/Created

### New Files (9 files, ~1100 lines)

| File | Lines | Purpose |
|------|-------|---------|
| [src/python/environment.rs](../src/python/environment.rs) | 234 | Python execution environment |
| [src/python/interpreter.rs](../src/python/interpreter.rs) | 105 | PyInterpreter bindings |
| [src/python/tensor.rs](../src/python/tensor.rs) | 214 | Tensor ↔ NumPy conversion |
| [src/python/mod.rs](../src/python/mod.rs) | 28 | Python module entry |
| [tests/python_parser_test.rs](../tests/python_parser_test.rs) | 124 | Parser integration tests |
| [examples/python_integration_test.tl](../examples/python_integration_test.tl) | 17 | Usage example |
| [python/tensorlogic/__init__.py](../python/tensorlogic/__init__.py) | 19 | Python package entry |
| [python/tests/test_interpreter.py](../python/tests/test_interpreter.py) | 104 | Interpreter tests |
| [python/tests/test_tensor.py](../python/tests/test_tensor.py) | 121 | Tensor tests |

### Modified Files (8 files)

| File | Changes |
|------|---------|
| [src/ast/mod.rs](../src/ast/mod.rs) | Added PythonImport, PythonCall nodes |
| [src/parser/grammar.pest](../src/parser/grammar.pest) | Added python_import, python_call rules |
| [src/parser/mod.rs](../src/parser/mod.rs) | Added parsing functions |
| [src/lexer/mod.rs](../src/lexer/mod.rs) | Added Python, Import, As, Dot tokens |
| [src/interpreter/mod.rs](../src/interpreter/mod.rs) | Integrated Python environment |
| [src/ast/visitor.rs](../src/ast/visitor.rs) | Added visitor handlers |
| [src/typecheck/mod.rs](../src/typecheck/mod.rs) | Added type inference |
| [Cargo.toml](../Cargo.toml) | Added pyo3, numpy dependencies |

---

## Build Instructions

### CLI Binary (with Python support)

```bash
# Build
cargo build --features python --release

# Run
cargo run --features python -- examples/python_integration_test.tl
```

**Note**: Requires system Python library linked. May need `DYLD_LIBRARY_PATH` on macOS.

### Python Module

```bash
# Install maturin
pip install maturin

# Build wheel
maturin build --features python-extension --release

# Install
pip install target/wheels/tensorlogic-0.1.0-cp38-abi3-macosx_11_0_arm64.whl

# Use
python3
>>> import tensorlogic as tl
>>> interp = tl.Interpreter()
>>> interp.execute("main { print(\"Hello from TensorLogic!\") }")
```

---

## Next Steps (Future Phases)

### Phase 4: Advanced Python Integration (Optional)

1. **Variable Sharing**:
   - `interp.get_variable("x")` → Extract TensorLogic variables to Python
   - `interp.set_variable("x", tensor)` → Inject Python tensors into TensorLogic

2. **PyTorch Support**:
   - Test PyTorch tensor conversion
   - Add torch.Tensor ↔ TensorLogic Tensor conversion
   - Support GPU tensors (Metal ↔ CUDA)

3. **Advanced Type Inference**:
   - Parse Python function signatures
   - Infer accurate return types
   - Support multi-output functions

4. **Error Handling**:
   - Better Python exception messages
   - Stack trace propagation
   - Debugging support

### Phase 5: Jupyter Kernel (Future)

From [jupyter_integration_plan.md](jupyter_integration_plan.md):

1. **Jupyter Protocol**:
   - Implement ipykernel protocol
   - Message handling (execute, complete, inspect)
   - Rich output (HTML, LaTeX, images)

2. **Kernel Installation**:
   - kernelspec.json
   - Kernel installation script
   - Icon and branding

3. **Interactive Features**:
   - Tab completion
   - Syntax highlighting
   - Inline documentation
   - Interactive debugging

---

## Success Metrics ✅

### Phase 2 & 3 Completion Criteria

- ✅ AST extended with PythonImport and PythonCall nodes
- ✅ Parser handles python import and python.call() syntax
- ✅ Lexer recognizes Python keywords
- ✅ Parser tests passing (5/5)
- ✅ Python environment wrapper implemented
- ✅ Tensor ↔ NumPy conversion working
- ✅ Python function calls execute successfully
- ✅ NumPy integration validated (sum, add, mean, max)
- ✅ Feature flags working (python, python-extension)
- ✅ Wheel builds successfully
- ✅ Integration tests passing

### Build Quality

- ✅ Release build optimized (opt-level=3, LTO)
- ✅ abi3 support for Python 3.8+ compatibility
- ✅ Only minor warnings (unused variables)
- ✅ No compilation errors
- ✅ Both features compile correctly

---

## Achievements

### Technical Wins

1. **Dual Feature System**: Successfully implemented separate builds for CLI and Python module
2. **f16 NumPy Bridge**: Seamless conversion between f16-only TensorLogic and f32/f64 NumPy
3. **PyO3 0.21 Bound API**: Used modern API instead of deprecated GIL Refs
4. **Clean Integration**: Python FFI feels native to TensorLogic syntax
5. **Comprehensive Testing**: End-to-end validation from AST to execution

### Development Velocity

- **Phase 2 (AST & Parser)**: ~2 hours (7 files modified, 5 tests)
- **Phase 3 (Execution Engine)**: ~3 hours (3 new files, integration tests)
- **Total**: ~5 hours (under 1 week target)

---

**Status**: ✅ Phase 2 & 3 Complete - Python FFI fully functional
**Deliverable**: `tensorlogic-0.1.0-cp38-abi3-macosx_11_0_arm64.whl`
**Next Phase**: Advanced Python integration or Jupyter kernel (optional)
