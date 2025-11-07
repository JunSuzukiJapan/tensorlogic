# Struct Feature Test Suite

This directory contains comprehensive tests for TensorLogic's struct and generic support.

## Test Files

### 1. `test_struct_basic.tl`
Tests basic struct functionality:
- Struct literal creation
- Field access
- Multiple struct instances
- Structs with multiple fields

**Example:**
```tensorlogic
struct Point {
    x: float32,
    y: float32,
}

let p = Point { x: 3.0, y: 4.0 }
```

### 2. `test_struct_generic.tl`
Tests generic struct definitions:
- Single type parameter `Container<T>`
- Multiple type parameters `Pair<T, U>`, `Triple<A, B, C>`
- Different type instantiations

**Example:**
```tensorlogic
struct Container<T> {
    value: T,
    count: int32,
}

let c = Container<float32> { value: 42.0, count: 1 }
```

### 3. `test_struct_tensors.tl`
Tests structs with tensor fields:
- 1D tensor fields
- 2D tensor fields
- Learnable tensor parameters
- Neural network layer structure

**Example:**
```tensorlogic
struct NeuralLayer {
    weights: float16[?, ?] learnable,
    bias: float16[?] learnable,
    input_dim: int32,
    output_dim: int32,
}
```

### 4. `test_struct_methods.tl`
Tests impl blocks and associated functions:
- Constructor patterns (`new()`)
- Multiple associated functions
- Different struct types with methods

**Example:**
```tensorlogic
struct Point {
    x: float32,
    y: float32,
}

impl Point {
    fn new(x: float32, y: float32) -> Point {
        return Point { x: x, y: y }
    }

    fn origin() -> Point {
        return Point { x: 0.0, y: 0.0 }
    }
}

let p = Point::new(10.0, 20.0)
```

### 5. `test_struct_generic_impl.tl`
Tests generic impl blocks:
- Generic constructors `Box<T>::new()`
- Multiple type parameters in impl
- Different type instantiations
- Mix of generic and non-generic methods

**Example:**
```tensorlogic
struct Box<T> {
    value: T,
}

impl<T> Box<T> {
    fn new(value: T) -> Box<T> {
        return Box<T> { value: value }
    }
}

let b1 = Box<float32>::new(3.14)
let b2 = Box<int32>::new(42)
```

### 6. `test_struct_nn_integration.tl`
Integration test with real neural network:
- Generic layer structure
- Forward pass implementation
- Multi-layer network
- End-to-end workflow

**Example:**
```tensorlogic
struct Layer<W, B> {
    weights: W learnable,
    bias: B learnable,
    input_dim: int32,
    output_dim: int32,
    name: string,
}

impl Layer<float16[?, ?], float16[?]> {
    fn new(name: string, input_dim: int32, output_dim: int32) -> Layer<float16[?, ?], float16[?]> {
        // Initialize weights and bias
        tensor w: float16[input_dim, output_dim] learnable = random_normal(0.0, 0.1)
        tensor b: float16[output_dim] learnable = zeros()

        return Layer<float16[?, ?], float16[?]> {
            weights: w,
            bias: b,
            input_dim: input_dim,
            output_dim: output_dim,
            name: name,
        }
    }

    fn forward(self, input: float16[?, ?]) -> float16[?, ?] {
        result := (input @ self.weights) + self.bias
    }
}

// Usage
let layer = Layer<float16[?, ?], float16[?]>::new("hidden", 784, 128)
let output = layer.forward(input)
```

## Running Tests

To run all struct tests:

```bash
# Run individual tests
cargo run -- tests/test_struct_basic.tl
cargo run -- tests/test_struct_generic.tl
cargo run -- tests/test_struct_tensors.tl
cargo run -- tests/test_struct_methods.tl
cargo run -- tests/test_struct_generic_impl.tl
cargo run -- tests/test_struct_nn_integration.tl
```

## Test Coverage

| Feature | Test File |
|---------|-----------|
| Basic struct syntax | test_struct_basic.tl |
| Generic structs | test_struct_generic.tl |
| Tensor fields | test_struct_tensors.tl |
| Impl blocks | test_struct_methods.tl |
| Generic impl | test_struct_generic_impl.tl |
| Integration | test_struct_nn_integration.tl |

## Expected Output

Each test should print progress messages and complete with:
```
All [test type] tests passed!
```

For the integration test:
```
=== Integration test completed successfully! ===
Successfully created a 2-layer neural network using structs
```

## Implementation Status

✅ **Completed Features:**
- Struct declarations with fields
- Generic type parameters
- Impl blocks with methods
- Associated functions (constructors)
- Struct literals
- Field access
- Tensor-typed fields with learnable modifier
- Runtime struct instantiation
- Method calls

## Notes

- These tests verify both compile-time (parser, type checker) and runtime (interpreter) functionality
- Tests are designed to be incremental, starting from basic features to complex integration
- The integration test demonstrates real-world usage with neural network layers
