# TensorLogic Language Reference

**Version**: 0.2.0-alpha
**Last Updated**: 2025-10-20

## Table of Contents

1. [Introduction](#introduction)
2. [Program Structure](#program-structure)
3. [Data Types](#data-types)
4. [Declarations](#declarations)
5. [Expressions](#expressions)
6. [Statements](#statements)
7. [Operators](#operators)
8. [Built-in Functions](#built-in-functions)
9. [Learning System](#learning-system)
10. [Logic Programming](#logic-programming)

---

## 1. Introduction

TensorLogic is a programming language that unifies tensor algebra with logic programming, enabling neural-symbolic AI. It combines differentiable tensor operations with logical reasoning for next-generation AI systems.

### Key Features

- **Tensor Operations**: High-performance GPU-accelerated computations
- **Automatic Differentiation**: Built-in gradient computation
- **Learning System**: Gradient descent with multiple optimizers
- **Logic Programming**: Relations, rules, and queries
- **Neural-Symbolic Integration**: Embeddings for entities and relations

---

## 2. Program Structure

### 2.1 Basic Structure

```tensorlogic
// Declarations
tensor w: float32[10] learnable = [...]
relation Parent(x: entity, y: entity)

// Main execution block
main {
    // Statements
    result := w * w

    // Learning
    learn {
        objective: result,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

### 2.2 Importing External Files

TensorLogic supports importing declarations from external files:

```tensorlogic
// Import declarations from another file
import "path/to/module.tl"
import "../lib/constants.tl"

main {
    // Use imported tensors and functions
    result := imported_tensor * 2
}
```

**Features**:
- Relative path resolution (relative to the importing file)
- Circular dependency detection (prevents infinite loops)
- Duplicate import prevention (same file won't be imported twice)
- Only declarations are imported (main blocks are not executed)

**Example**:

File: `lib/constants.tl`
```tensorlogic
tensor pi: float16[1] = [3.14159]
tensor e: float16[1] = [2.71828]
```

File: `main.tl`
```tensorlogic
import "lib/constants.tl"

main {
    tensor circumference: float16[1] = [2.0]
    result := circumference * pi  // Uses imported pi
    print("Result:", result)
}
```

### 2.3 Comments

```tensorlogic
// Single-line comment

/* Multi-line comment
   Not yet implemented */
```

---

## 3. Data Types

### 3.1 Base Types

| Type | Description | Precision |
|------|-------------|-----------|
| `float16` | 16-bit floating point (f16) | Half precision (Apple Silicon optimized) |
| `float32` | 32-bit floating point | Single precision |
| `float64` | 64-bit floating point | Double precision |
| `int32` | 32-bit integer | Signed integer |
| `int64` | 64-bit integer | Signed long integer |
| `bool` | Boolean | true/false |
| `complex64` | 64-bit complex number | Complex float32 |

**Note**: TensorLogic primarily uses `float16` for optimal performance on Apple Silicon (Metal GPU and Neural Engine).

### 3.1.1 Numeric Literals

TensorLogic supports positive and negative numeric literals:

```tensorlogic
tensor positive: float16[1] = [3.14]
tensor negative: float16[1] = [-2.71]
tensor zero: float16[1] = [0.0]
tensor neg_int: float16[1] = [-42.0]
```

### 3.2 Tensor Types

```tensorlogic
tensor x: float32[10]           // 1D tensor
tensor W: float32[3, 4]         // 2D matrix
tensor T: float32[2, 3, 4]      // 3D tensor
tensor D: float32[n, m]         // Variable dimensions
tensor F: float32[?, ?]         // Dynamic dimensions
```

### 3.3 Entity Types

```tensorlogic
entity      // Logic programming entity
concept     // High-level concept
```

---

## 4. Declarations

### 4.1 Tensor Declarations

#### Basic Syntax

```tensorlogic
tensor name: type[dimensions] attributes = initial_value
```

#### Examples

```tensorlogic
// Simple declaration
tensor x: float32[5]

// With initialization
tensor w: float32[3] = [1.0, 2.0, 3.0]

// Learnable parameter
tensor w: float32[10] learnable

// With initial value and learnable
tensor b: float32[1] learnable = [0.0]

// Frozen (non-trainable)
tensor const: float32[5] frozen = [1.0, 1.0, 1.0, 1.0, 1.0]
```

### 4.2 Relation Declarations

```tensorlogic
// Basic relation
relation Parent(x: entity, y: entity)

// Relation with embedding
relation Friend(x: entity, y: entity) embed float32[64]

// Multi-parameter relation
relation WorksAt(person: entity, company: entity, role: concept)
```

### 4.3 Embedding Declarations

```tensorlogic
// Explicit entity set
embedding person_embed {
    entities: {alice, bob, charlie}
    dimension: 64
    init: xavier
}

// Auto entity set (dynamic)
embedding word_embed {
    entities: auto
    dimension: 128
    init: random
}
```

#### Initialization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `random` | Uniform(-0.1, 0.1) | General purpose |
| `xavier` | Xavier/Glorot initialization | Deep networks |
| `he` | He initialization | ReLU networks |
| `zeros` | All zeros | Bias initialization |
| `ones` | All ones | Special cases |

### 4.4 Function Declarations

```tensorlogic
// Function declaration (future feature)
fn sigmoid(x: float32[?]) -> float32[?] {
    // Implementation
}
```

---

## 5. Expressions

### 5.1 Tensor Literals

```tensorlogic
[1.0, 2.0, 3.0]                // 1D array
[[1.0, 2.0], [3.0, 4.0]]       // 2D matrix (future)
```

### 5.2 Variables

```tensorlogic
x                               // Variable reference
w                               // Tensor variable
```

### 5.3 Binary Operations

```tensorlogic
a + b                          // Element-wise addition
a - b                          // Element-wise subtraction
a * b                          // Element-wise multiplication
a / b                          // Element-wise division
a @ b                          // Matrix multiplication
a ** b                         // Power
```

### 5.4 Unary Operations

```tensorlogic
-x                             // Negation
!x                             // Logical NOT
```

### 5.5 Embedding Lookup

```tensorlogic
person_embed["alice"]          // Literal entity lookup
person_embed[entity_var]       // Variable entity lookup
```

### 5.6 Einstein Summation

```tensorlogic
einsum("ij,jk->ik", A, B)      // Matrix multiplication
einsum("ii->", A)              // Trace
einsum("ij->ji", A)            // Transpose
```

---

## 6. Statements

### 6.1 Assignment

```tensorlogic
x := 10
result := w * w + b
```

### 6.2 Tensor Equations

```tensorlogic
y = w @ x + b                  // Exact equation
y ~ w @ x + b                  // Approximate equation
```

### 6.3 Control Flow

#### If Statement

```tensorlogic
if x > 0 {
    result := x
} else {
    result := 0
}
```

#### For Loop

```tensorlogic
for i in range(10) {
    sum := sum + i
}
```

#### While Loop

```tensorlogic
while x > 0 {
    x := x - 1
}
```

### 6.4 Query Statement

```tensorlogic
query Parent(alice, X)
query Parent(X, Y) where X != Y
```

### 6.5 Inference Statement

```tensorlogic
infer forward query Ancestor(alice, X)
infer backward query HasProperty(X, color)
infer gradient query Relation(X, Y)
infer symbolic query Rule(X)
```

### 6.6 Learning Statement

```tensorlogic
learn {
    objective: loss_expression,
    optimizer: sgd(lr: 0.1),
    epochs: 100
}
```

---

## 7. Operators

### 7.1 Arithmetic Operators

| Operator | Name | Description | Example |
|----------|------|-------------|---------|
| `+` | Addition | Element-wise add | `a + b` |
| `-` | Subtraction | Element-wise subtract | `a - b` |
| `*` | Multiplication | Element-wise multiply (Hadamard) | `a * b` |
| `/` | Division | Element-wise divide | `a / b` |
| `@` | Matrix Multiplication | Tensor contraction | `A @ B` |
| `**` | Power | Element-wise power | `a ** 2` |

### 7.2 Comparison Operators

| Operator | Name | Example |
|----------|------|---------|
| `==` | Equal | `x == y` |
| `!=` | Not equal | `x != y` |
| `<` | Less than | `x < y` |
| `>` | Greater than | `x > y` |
| `<=` | Less or equal | `x <= y` |
| `>=` | Greater or equal | `x >= y` |
| `≈` | Approximately equal | `x ≈ y` |

### 7.3 Logical Operators

| Operator | Name | Example |
|----------|------|---------|
| `and` | Logical AND | `a and b` |
| `or` | Logical OR | `a or b` |
| `not` | Logical NOT | `not a` |

### 7.4 Operator Precedence

Highest to lowest precedence:

1. `()` - Parentheses
2. `**` - Power
3. `-` (unary), `!` - Unary operators
4. `@` - Matrix multiplication
5. `*`, `/` - Multiplication, division
6. `+`, `-` - Addition, subtraction
7. `<`, `>`, `<=`, `>=`, `==`, `!=`, `≈` - Comparisons
8. `not` - Logical NOT
9. `and` - Logical AND
10. `or` - Logical OR

---

## 8. Built-in Functions

### 8.1 Tensor Operations

#### Activation Functions

```tensorlogic
relu(x)                        // ReLU activation
gelu(x)                        // GELU activation
sigmoid(x)                     // Sigmoid (future)
tanh(x)                        // Tanh (future)
```

#### Matrix Operations

```tensorlogic
transpose(A)                   // Transpose (future)
inverse(A)                     // Matrix inverse (future)
determinant(A)                 // Determinant (future)
```

#### Reduction Operations

```tensorlogic
sum(x)                         // Sum all elements
mean(x)                        // Mean value (future)
max(x)                         // Maximum value (future)
min(x)                         // Minimum value (future)
```

### 8.2 Shape Operations

```tensorlogic
shape(x)                       // Get tensor shape (constraint)
rank(x)                        // Get tensor rank (constraint)
```

### 8.3 Einstein Summation

```tensorlogic
einsum(equation, tensors...)   // Einstein summation notation
```

**Examples**:

```tensorlogic
// Matrix multiplication
C := einsum("ij,jk->ik", A, B)

// Trace
trace := einsum("ii->", A)

// Transpose
B := einsum("ij->ji", A)

// Batch matrix multiplication
C := einsum("bij,bjk->bik", A, B)
```

---

## 9. Learning System

### 9.1 Learnable Parameters

```tensorlogic
tensor w: float32[10] learnable = [...]
tensor b: float32[1] learnable = [0.0]
```

### 9.2 Optimizers

#### SGD (Stochastic Gradient Descent)

```tensorlogic
optimizer: sgd(lr: 0.1)
```

**Parameters**:
- `lr`: Learning rate (default: 0.01)

#### Adam

```tensorlogic
optimizer: adam(lr: 0.001)
```

**Parameters**:
- `lr`: Learning rate (default: 0.001)
- `beta1`: First moment decay (default: 0.9)
- `beta2`: Second moment decay (default: 0.999)
- `epsilon`: Small constant (default: 1e-8)

#### AdamW

```tensorlogic
optimizer: adamw(lr: 0.001, weight_decay: 0.01)
```

**Parameters**:
- `lr`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay coefficient (default: 0.01)

### 9.3 Learning Specification

```tensorlogic
learn {
    // Optional: Local variable declarations for intermediate computations
    intermediate := some_expression
    another_var := other_expression

    objective: loss_expression,
    optimizer: optimizer_spec,
    epochs: number
}
```

**Requirements**:
- `objective` must be a scalar tensor expression
- All tensors marked `learnable` will be optimized
- Gradients computed via automatic differentiation
- Local variables (`:=`) can be used before `objective` for intermediate computations
- Local variables are re-computed at each epoch

**Example - Basic Learning**:

```tensorlogic
tensor w: float32[10] learnable = [...]
tensor x: float32[10] = [...]

main {
    pred := w * x
    loss := pred * pred

    learn {
        objective: loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }
}
```

**Example - With Local Variables**:

```tensorlogic
tensor W: float16[1] learnable = [0.5]
tensor x1: float16[1] = [1.0]
tensor y1: float16[1] = [3.0]
tensor x2: float16[1] = [-2.0]  // Negative numbers supported
tensor y2: float16[1] = [-6.0]

main {
    learn {
        // Local variables for intermediate computations
        pred1 := x1 * W
        pred2 := x2 * W

        // Compute errors
        err1 := pred1 - y1
        err2 := pred2 - y2

        // Sum of squared errors
        total_loss := err1 * err1 + err2 * err2

        objective: total_loss,
        optimizer: sgd(lr: 0.01),
        epochs: 100
    }

    print("Learned W:", W)  // Should be close to 3.0
}
```

**Note**: Only tensors explicitly declared with the `learnable` keyword are optimized. Local variables computed within the `learn` block are not treated as learnable parameters.

---

## 10. Logic Programming

### 10.1 Relations

```tensorlogic
relation Parent(x: entity, y: entity)
relation Sibling(x: entity, y: entity)
```

### 10.2 Rules (Future)

```tensorlogic
rule Grandparent(X, Z) <- Parent(X, Y), Parent(Y, Z)
rule Ancestor(X, Y) <- Parent(X, Y)
rule Ancestor(X, Z) <- Parent(X, Y), Ancestor(Y, Z)
```

### 10.3 Queries (Partial)

```tensorlogic
query Parent(alice, X)
query Parent(X, Y) where X != Y
```

### 10.4 Constraints

```tensorlogic
shape(w) == [10, 20]           // Shape constraint
rank(A) == 2                   // Rank constraint
norm(w) < 1.0                  // Norm constraint
```

### 10.5 Inference Methods

```tensorlogic
infer forward query Q          // Logic → Tensor
infer backward query Q         // Tensor → Logic
infer gradient query Q         // Differential reasoning
infer symbolic query Q         // Symbolic manipulation
```

---

## Appendix A: Reserved Keywords

```
and, auto, bool, complex64, concept, dimension, einsum, else, embed, embedding,
entities, entity, epochs, float32, float64, for, frozen, function, gelu, if,
in, init, int32, int64, infer, learn, learnable, main, norm, not, objective,
ones, optimizer, or, query, random, range, rank, relation, relu, rule, shape,
tensor, where, while, xavier, zeros
```

## Appendix B: Grammar Summary

See [TensorLogic Grammar](../../Papers/実装/tensorlogic_grammar.md) for complete BNF specification.

## Appendix C: Examples

See [Tutorial 01: Linear Regression](../../claudedocs/tutorial_01_linear_regression.md) and other tutorials for working examples.

---

**End of Language Reference**

For questions or contributions, visit: https://github.com/JunSuzukiJapan/tensorlogic
