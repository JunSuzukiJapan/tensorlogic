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

### 2.2 Comments

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
| `float32` | 32-bit floating point | Single precision |
| `float64` | 64-bit floating point | Double precision |
| `int32` | 32-bit integer | Signed integer |
| `int64` | 64-bit integer | Signed long integer |
| `bool` | Boolean | true/false |
| `complex64` | 64-bit complex number | Complex float32 |

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
function sigmoid(x: float32[?]) -> float32[?] {
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
    objective: loss_expression,
    optimizer: optimizer_spec,
    epochs: number
}
```

**Requirements**:
- `objective` must be a scalar tensor expression
- All tensors marked `learnable` will be optimized
- Gradients computed via automatic differentiation

**Example**:

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
