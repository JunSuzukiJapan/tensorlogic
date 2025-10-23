# Logic Programming and Tensor Unification Design

**Target Audience**: Developers and researchers who want to understand TensorLogic's design philosophy
**Last Updated**: 2025-10-23

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Correspondence Between Logic Programming and Tensor Elements](#2-correspondence-between-logic-programming-and-tensor-elements)
3. [Why Einstein Summation is Important](#3-why-einstein-summation-is-important)
4. [Unified Grammar Proposals](#4-unified-grammar-proposals)
5. [Entity Type Design](#5-entity-type-design)
6. [Learning and Inference Integration Methods](#6-learning-and-inference-integration-methods)
7. [Implementation Considerations](#7-implementation-considerations)
8. [Summary](#8-summary)

---

## 1. Overview and Objectives

### 1.1 Background

TensorLogic is a neuro-symbolic AI language that integrates logic programming with deep learning. Traditionally, these have been treated as separate paradigms:

| Paradigm | Strengths | Weaknesses |
|----------|-----------|------------|
| **Logic Programming** | Precise inference, explainable | Cannot learn, weak with uncertainty |
| **Deep Learning** | High learning capability, flexible | Black box, poor at logical reasoning |

### 1.2 Purpose of Unification

By unifying both through the common foundation of **tensor operations**:

- ✅ Precision and explainability of logical reasoning
- ✅ Learning capability and flexibility of deep learning
- ✅ High-speed computation via GPU acceleration
- ✅ Unified, concise syntax

are simultaneously achieved.

### 1.3 Design Principles

Design principles in this document:

- **Conciseness Priority**: Syntax that can be written as briefly as possible
- **Type Safety**: Error detection at compile time
- **Automatic Optimization**: Automatic generation of optimal operations from type information
- **Internal Unification**: Represent all logical operations as tensors

---

## 2. Correspondence Between Logic Programming and Tensor Elements

### 2.1 Basic Correspondence

Correspondence between the three elements of logic programming and tensor representations:

| Logic Programming | Tensor Representation | Description |
|------------------|---------------------|-------------|
| **Facts** | Tensor elements | Represent truth values as tensor elements |
| **Rules** | Einstein summation | Represent logical conjunction as tensor products |
| **Queries** | Index/slicing | Represent variables as tensor dimensions |

### 2.2 Facts → Tensor Elements

```tensorlogic
// Logic programming
Parent(alice, bob)       // true
Parent(alice, charlie)   // true
Parent(bob, diana)       // true

// Tensor representation
tensor parent: float16[4, 4] = [
    //        alice  bob  charlie diana
    /* alice */  [0,    1,    1,      0],
    /* bob   */  [0,    0,    0,      1],
    /* charlie*/ [0,    0,    0,      0],
    /* diana */  [0,    0,    0,      0]
]
```

**Correspondence**:
- Fact `Parent(alice, bob)` → Tensor element `parent[0, 1] = 1.0`
- Fact `Parent(alice, charlie)` → Tensor element `parent[0, 2] = 1.0`
- Entity names are mapped to integer indices

### 2.3 Rules → Einstein Summation

```tensorlogic
// Logic programming
Grandparent(X, Z) :- Parent(X, Y), Parent(Y, Z)

// Tensor representation (Einstein summation)
grandparent := einsum('xy,yz->xz', parent, parent)
//              ↑       ↑   ↑   ↑
//              |       X   Y   Z
//              |       Common variable Y is joined
//              operation
```

**Correspondence**:
- Variables `X, Y, Z` → Tensor indices `x, y, z`
- Common variable `Y` → Contracted dimension `y`
- Comma `,` → Tensor product
- `:-` → Assignment `:=`

### 2.4 Queries → Index/Slicing

```tensorlogic
// Logic programming
Parent(alice, X)?        // Who are alice's children?

// Tensor representation
result := parent[alice_id, :]
// → [0, 1, 1, 0]
// → bob (index 1) and charlie (index 2) are solutions
```

**Correspondence**:
- Constant `alice` → Fixed index `alice_id`
- Variable `X` → Slice `:`
- Result → Indices of non-zero elements

---

## 3. Why Einstein Summation is Important

### 3.1 Correspondence with Variable Binding

Variable binding in logic programming **completely corresponds** to index sharing in Einstein summation.

```
Logic programming:
  Ancestor(X, Z) :- Parent(X, Y), Parent(Y, Z)
                           ↑         ↑
                           Y is common variable (join point)

Einstein summation:
  einsum('xy, yz -> xz', parent, parent)
              ↑   ↑
              y is common index (contracted)
```

### 3.2 Concrete Example: Grandparent Relationship Calculation

#### Data

```tensorlogic
tensor parent: float16[4, 4] = [
    [0, 1, 1, 0],  // alice → bob, charlie
    [0, 0, 0, 1],  // bob → diana
    [0, 0, 0, 1],  // charlie → diana
    [0, 0, 0, 0]   // diana → (none)
]
```

#### Rule Application

```tensorlogic
// Grandparent(X, Z) :- Parent(X, Y), Parent(Y, Z)
grandparent := einsum('xy,yz->xz', parent, parent)
```

#### Calculation Process (example: alice, diana)

```
grandparent[alice, diana] = Σ(parent[alice, Y] * parent[Y, diana])
                          = parent[alice,alice]*parent[alice,diana]
                          + parent[alice,bob]*parent[bob,diana]      ← 1*1=1
                          + parent[alice,charlie]*parent[charlie,diana] ← 1*1=1
                          + parent[alice,diana]*parent[diana,diana]
                          = 0*0 + 1*1 + 1*1 + 0*0
                          = 2  ← Two paths exist!
```

#### Result

```tensorlogic
grandparent = [
    [0, 0, 0, 2],  // alice → diana (via bob and via charlie)
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
```

### 3.3 Power in Complex Rules

#### Combining Three or More Relations

```tensorlogic
// Friend(X, Y), Likes(Y, Z), Recommends(Z, W) → Suggestion(X, W)

suggestion := einsum('xy,yz,zw->xw', friend, likes, recommends)
//                     ↑   ↑   ↑
//                     y   z are automatically joined
```

**Advantages**:
- Process multiple common variables at once
- Leverage GPU-optimized einsum operations
- Easy parallel computation

#### Integration of Multiple Logical Paths

```tensorlogic
// Rule 1: Direct friends
// Rule 2: Friends of friends
// Rule 3: Shared hobbies

rule Connected(x, y) :-
    Friend(x, y)

rule Connected(x, z) :-
    Friend(x, y), Friend(y, z)

rule Connected(x, y) :-
    SharedHobby(x, y)

// Tensor computation
connected := Friend
           + einsum('xy,yz->xz', Friend, Friend)  // transitivity
           + SharedHobby                          // alternative evidence
```

### 3.4 Summary of Einstein Summation Advantages

| Advantage | Description |
|-----------|-------------|
| **Natural Correspondence** | Logic variables ↔ Tensor indices |
| **Automatic Optimization** | Already optimized in GPU kernels |
| **Parallel Computation** | Process multiple joins simultaneously |
| **Differentiable** | Easy integration with learning |
| **Concise Expression** | Even complex rules written briefly |

---

## 4. Unified Grammar Proposals

### 4.1 Evaluation Criteria

Each proposal is evaluated based on:

- **Conciseness**: Code brevity, ease of writing
- **Clarity**: Ease of understanding intent
- **Type Safety**: Compile-time error detection
- **Automatic Optimization**: Possibility of compiler optimization
- **Implementation Ease**: Parser and compiler implementation difficulty

---

### 4.2 Proposal 1: Explicit Tensor Rule Syntax

```tensorlogic
// Declare relation as tensor
relation Parent(x: entity, y: entity) as tensor float16[N, N]

// Describe rule with Einstein summation
rule Grandparent(x: entity, z: entity) {
    // Logical definition
    Parent(x, y), Parent(y, z)

    // Tensor computation (can be auto-generated)
    compute: einsum('xy,yz->xz', Parent, Parent)
}

// Query
main {
    // Normal query
    Grandparent(alice, X)?

    // Access as tensor
    result := Grandparent[alice_id, :]
}
```

**Evaluation**:

| Criteria | Rating | Reason |
|----------|--------|--------|
| Conciseness | ⭐⭐⭐ | Somewhat verbose |
| Clarity | ⭐⭐⭐⭐⭐ | Logic and tensor clearly separated |
| Type Safety | ⭐⭐⭐⭐ | Types explicit |
| Automatic Optimization | ⭐⭐⭐⭐ | Provides optimization hints |
| Implementation Ease | ⭐⭐⭐⭐ | Relatively simple |

**Pros**:
- ✅ Separate logical definition from tensor computation
- ✅ Maintain existing logic programming syntax
- ✅ Can explicitly provide optimization hints

**Cons**:
- ❌ Verbose (requires compute block)
- ❌ Write same information twice (logic and tensor)

---

### 4.3 Proposal 2: Implicit Conversion Syntax

```tensorlogic
// Relation declaration (tensor-backed)
relation Parent(x: entity, y: entity) tensor

// Rule: Writing logically automatically converts to einsum
rule Grandparent(x: entity, z: entity) :-
    Parent(x, y), Parent(y, z)
    // Automatically: einsum('xy,yz->xz', Parent, Parent)

rule Ancestor(x: entity, z: entity) :-
    Parent(x, z)  // Base case
rule Ancestor(x: entity, z: entity) :-
    Parent(x, y), Ancestor(y, z)  // Recursive case
    // Automatically iterative computation

// Query as is
main {
    Ancestor(alice, X)?
    // Internally: ancestor[alice_id, :] as tensor slice
}
```

**Automatic Mapping from Variables to Dimensions**:
```
Parent(x, y)          → Index 'xy'
Parent(y, z)          → Index 'yz'
Common variable y     → Contracted dimension
Result Grandparent(x,z) → Index 'xz'
```

**Evaluation**:

| Criteria | Rating | Reason |
|----------|--------|--------|
| Conciseness | ⭐⭐⭐⭐⭐ | Most concise |
| Clarity | ⭐⭐⭐⭐ | Logic programming as is |
| Type Safety | ⭐⭐⭐ | Depends on type inference |
| Automatic Optimization | ⭐⭐⭐⭐⭐ | Compiler has full control |
| Implementation Ease | ⭐⭐ | Requires sophisticated compiler |

**Pros**:
- ✅ Existing logic programming syntax as is
- ✅ Most natural integration
- ✅ Compiler automatically generates einsum

**Cons**:
- ❌ Requires sophisticated compiler
- ❌ Difficult to provide optimization hints

---

### 4.4 Proposal 3: Hybrid Syntax

```tensorlogic
// Basic relation
relation Parent(x: entity, y: entity) tensor

// Simple rule: automatic conversion
rule Grandparent(x, z) :- Parent(x, y), Parent(y, z)

// Complex rule: explicit specification
rule WeightedInfluence(x: entity, w: entity) {
    logic: Friend(x, y), Influence(y, z), Recommends(z, w)

    // Custom computation (weighted)
    compute {
        friend_weight := Friend * 0.5
        influence_weight := Influence * 0.3
        recommend_weight := Recommends * 0.2

        result := einsum('xy,yz,zw->xw',
                        friend_weight,
                        influence_weight,
                        recommend_weight)
    }
}

// Fuzzy rule (with score)
rule Similar(x: entity, y: entity) score {
    // Compute combined score from multiple evidences
    let friend_score = Friend(x, y)              // 0.8
    let hobby_score = SharedHobby(x, y)          // 0.6
    let location_score = SameLocation(x, y)      // 0.9

    // Weighted average
    return 0.4 * friend_score + 0.3 * hobby_score + 0.3 * location_score
}
```

**Evaluation**:

| Criteria | Rating | Reason |
|----------|--------|--------|
| Conciseness | ⭐⭐⭐⭐ | Concise for simple cases |
| Clarity | ⭐⭐⭐⭐ | Intent is clear |
| Type Safety | ⭐⭐⭐⭐ | Type checking possible |
| Automatic Optimization | ⭐⭐⭐ | Simple cases only |
| Implementation Ease | ⭐⭐⭐ | Medium difficulty |

**Pros**:
- ✅ Simple cases are concise
- ✅ Complex cases have explicit control
- ✅ Integration with fuzzy logic

**Cons**:
- ❌ Need to learn two modes
- ❌ Consistency somewhat lower

---

### 4.5 Proposal 4: Typed Einstein Syntax (Recommended)

```tensorlogic
// Treat entity types as dimensions
entity Person = {alice, bob, charlie, diana}
entity Item = {book, movie, game}

// Relation type signature automatically determines tensor shape
relation Parent(x: Person, y: Person)
// Automatically: tensor float16[|Person|, |Person|]
//                           ↑          ↑
//                           4          4

relation Likes(x: Person, y: Item)
// Automatically: tensor float16[|Person|, |Item|]
//                           ↑        ↑
//                           4        3

// Rule: Automatically generate einsum from variable types
rule Recommends(x: Person, z: Item) :-
    Friend(x, y),      // Person × Person
    Likes(y, z)        // Person × Item
    // Type inference:
    // Friend: [Person, Person] → Index 'ab'
    // Likes:  [Person, Item]   → Index 'bc'
    // Result: [Person, Item]   → Index 'ac'
    // einsum: 'ab,bc->ac' (Person's 2nd dimension is common)

// Type-safe query
main {
    Recommends(alice, X)?
    // X: Item type → only book, movie, or game returned
}
```

**Type Inference Flow**:
```
1. Friend(x, y) where x: Person, y: Person
   → Friend tensor shape: [Person, Person]

2. Likes(y, z) where y: Person, z: Item
   → Likes tensor shape: [Person, Item]

3. Common variable y: Person
   → Join Friend's 2nd dimension with Likes' 1st dimension

4. Result Recommends(x, z) where x: Person, z: Item
   → Recommends tensor shape: [Person, Item]

5. Generate einsum: 'ab,bc->ac'
```

**Evaluation**:

| Criteria | Rating | Reason |
|----------|--------|--------|
| Conciseness | ⭐⭐⭐⭐⭐ | Shape determined by entity definition only |
| Clarity | ⭐⭐⭐⭐⭐ | Types make intent explicit |
| Type Safety | ⭐⭐⭐⭐⭐ | Compile-time error detection |
| Automatic Optimization | ⭐⭐⭐⭐⭐ | Generate optimal einsum from type info |
| Implementation Ease | ⭐⭐⭐ | Requires type inference engine |

**Pros**:
- ✅ **Most concise**: Tensor shape determined by entity definition only
- ✅ **Type safe**: Compile-time error detection
- ✅ **Automatic optimization**: einsum auto-generation from type info
- ✅ **GPU efficient**: Pre-allocate memory, optimal operation order

**Cons**:
- ⚠️ Requires type inference engine implementation
- ⚠️ Need to learn entity types

---

### 4.6 Proposal 5: Declarative Query with einsum

```tensorlogic
relation Parent(x: entity, y: entity) tensor
relation Friend(x: entity, y: entity) tensor

main {
    // Standard query
    infer forward Parent(alice, X)?

    // einsum query (complex inference)
    infer {
        // Compute "alice's friends' parents" in one shot
        result := query einsum('xy,yz->xz', Friend, Parent)[alice_id, :]
    }

    // Recursive query (reachability)
    infer {
        // Compute transitive closure
        ancestor := Parent
        for i in 1..10 {
            ancestor := ancestor + einsum('xy,yz->xz', ancestor, Parent)
        }

        // All nodes reachable from alice
        result := ancestor[alice_id, :]
    }
}
```

**Evaluation**:

| Criteria | Rating | Reason |
|----------|--------|--------|
| Conciseness | ⭐⭐⭐ | For complex queries |
| Clarity | ⭐⭐⭐ | Too procedural |
| Type Safety | ⭐⭐ | Difficult to check |
| Automatic Optimization | ⭐⭐ | User controls |
| Implementation Ease | ⭐⭐⭐⭐ | Relatively easy |

**Pros**:
- ✅ Efficiently express complex queries
- ✅ Integration with iterative computation
- ✅ High flexibility

**Cons**:
- ❌ Loses logic programming feel
- ❌ Too procedural

---

### 4.7 Proposal Comparison Table

| Proposal | Conciseness | Clarity | Type Safety | Auto-Opt | Impl | Overall |
|----------|------------|---------|-------------|----------|------|---------|
| 1. Explicit Tensor Rule | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 2. Implicit Conversion | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 3. Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **4. Typed Einstein** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐** | **⭐⭐⭐⭐⭐** |
| 5. Declarative Query | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 4.8 Recommended Proposal

We recommend **Proposal 4: Typed Einstein Syntax**.

Reasons:
1. **Most concise**: Tensor shape automatically determined by entity definition only
2. **Type safe**: Compile-time error detection
3. **Automatic optimization**: Generate optimal einsum from type information
4. **Aligns with design principles**: Satisfies all of conciseness, type safety, and automatic optimization

---

## 5. Entity Type Design

### 5.1 What are Entity Types

Entity types are a concept that unifies **logic programming domains** with **tensor dimensions**.

```tensorlogic
entity Person = {alice, bob, charlie, diana}
//     ↑       ↑
//     Type name  Value set (4 elements)

// This definition results in:
// - Person type variables are one of alice, bob, charlie, diana
// - Tensor dimension using Person type is 4
```

### 5.2 Similarities and Differences with Enums

#### Similarities

```rust
// Rust enum
enum Person {
    Alice,
    Bob,
    Charlie,
    Diana,
}

// TensorLogic entity type
entity Person = {alice, bob, charlie, diana}
```

Both are:
- ✅ Finite set of values
- ✅ Determinable at compile time
- ✅ Internally mapped to integer indices

#### Important Differences

| Aspect | Enum | Entity Type |
|--------|------|-------------|
| **Purpose** | State/category | **Tensor dimension definition** |
| **Operations** | Comparison/pattern match | **Tensor operation shape determination** |
| **Size** | Fixed in type definition | Determined by type definition or data |

**Entity Type = Enum + Tensor Dimension Definition**

### 5.3 Static vs Dynamic Entity Types

#### Static Entity Type (Recommended)

```tensorlogic
entity Person = {alice, bob, charlie, diana}
// Fixed at compile time, size 4
```

**Pros**:
- ✅ Pre-allocate memory (fast)
- ✅ Easy GPU optimization
- ✅ Fewer bugs

**Cons**:
- ❌ Cannot add entities at runtime

#### Dynamic Entity Type (Not Recommended)

```tensorlogic
entity Person  // Size undefined

main {
    // Add at runtime
    Person.add("alice")
    Person.add("eve")

    // Tensors dynamically resized
}
```

**Pros**:
- ✅ High flexibility

**Cons**:
- ❌ Tensor reallocation required (slow)
- ❌ Complex GPU memory management
- ❌ Source of bugs

**Conclusion**: Dynamic types unnecessary. Support static types only.

### 5.4 Automatic Construction from Data

To increase static type flexibility, support **automatic construction from data**.

When explicit value enumeration is omitted in entity definitions, they are automatically constructed from data.

```tensorlogic
entity Person
//     ↑
//     Automatically construct entity set from data (= is omitted)

main {
    // Extract entities from facts
    Friend(alice, bob)      // Add alice, bob to Person
    Friend(bob, charlie)    // Add charlie to Person
    Friend(charlie, diana)  // Add diana to Person

    // At this point Person = {alice, bob, charlie, diana} is fixed
}
```

**Syntax**:
- `entity T = {...}` → Explicit enumeration
- `entity T` → Auto-construction from data (distinguished by presence of `=`)

**Pros**:
- ✅ No need to manually enumerate entities
- ✅ Can automatically construct from data files
- ✅ Fixed before execution, retaining static type advantages
- ✅ Concise syntax (distinguished by presence of `=` only)

### 5.5 `with` Block Syntax

**Problem**: With `entity T` alone, unclear where entities are fixed

**Solution**: Provide explicit scope with `with` block

```tensorlogic
entity Person

main {
    // Place fact definitions within with block
    with Person {
        // Automatically extract entities from facts
        Friend(alice, bob)      // Add alice, bob to Person
        Friend(bob, charlie)    // Add charlie to Person
        Friend(charlie, diana)  // Add diana to Person
    }
    // ← Person fixed here: {alice, bob, charlie, diana}

    // Cannot add entities beyond this point

    Friend(eve, alice)  // ❌ Compile error!
    // Error: Cannot add entity 'eve' to Person outside 'with' block

    learn {
        // Person already fixed
        // tensor Friend: [4, 4]
    }
}
```

#### Advantages of `with` Block

| Advantage | Description |
|-----------|-------------|
| **Clear Scope** | Entity collection phase visually clear |
| **Safety** | Prohibit entity addition outside block |
| **Multiple Entities** | Independently manage multiple entity types |
| **Error Detection** | Detect entity usage outside block at compile time |

#### Managing Multiple Entity Types

```tensorlogic
entity Person
entity Location
entity Item

main {
    // Phase 1: Person entity collection
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }
    // Person fixed: {alice, bob, charlie}

    // Phase 2: Location entity collection
    with Location {
        LivesIn(alice, tokyo)
        LivesIn(bob, osaka)
    }
    // Location fixed: {tokyo, osaka}

    // Phase 3: Item entity collection
    with Item {
        Likes(alice, book)
        Likes(bob, movie)
    }
    // Item fixed: {book, movie}

    // All entities fixed
    // Person: [3], Location: [2], Item: [2]
}
```

### 5.6 Index Mapping

Mapping from entity names to integer indices:

```tensorlogic
entity Person

main {
    with Person {
        Friend(alice, bob)
        Friend(charlie, diana)
    }
}
```

**Internal Processing**:
```rust
// Entity ID mapping
let person_ids: HashMap<&str, usize> = hashmap! {
    "alice" => 0,
    "bob" => 1,
    "charlie" => 2,
    "diana" => 3,
};

// Fact insertion
// Friend(alice, bob) → friend_tensor[0, 1] = 1.0
// Friend(charlie, diana) → friend_tensor[2, 3] = 1.0
```

---

## 6. Learning and Inference Integration Methods

We propose three methods for using logical operations in learning and inference.

### 6.1 Method 1: Embedding-Based (TransE Style)

#### Overview

Represent relations as vector embeddings and learn with a scoring function.

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) embed float16[64] learnable

main {
    with Person {
        Friend(alice, bob)      // Positive example
        Friend(bob, charlie)    // Positive example
    }

    learn {
        // Scoring function (TransE)
        for each (s, r, o) in positive_facts {
            pos_score := -norm(embed[s] + rel_embed[r] - embed[o])
        }

        // Negative sampling
        for each negative in sample_negatives() {
            neg_score := -norm(embed[negative.s] + rel_embed[negative.r] - embed[negative.o])
        }

        // Ranking loss
        loss := sum(max(0, margin - pos_score + neg_score))

        objective: loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    infer {
        // Query in embedding space
        forward Friend(alice, X)?

        // Internal processing:
        // 1. Get alice's embedding
        // 2. Get Friend relation embedding
        // 3. Search for alice_embed + friend_embed ≈ ?
        // 4. Compute distance to all entities
        // 5. Return entities with close distance (high score)
    }
}
```

#### Evaluation

| Aspect | Rating |
|--------|--------|
| **Implementation Ease** | ⭐⭐⭐⭐ |
| **Existing Research** | ⭐⭐⭐⭐⭐ (TransE, RotatE, ComplEx, etc.) |
| **Expressiveness** | ⭐⭐⭐ (Simple patterns only) |
| **Explainability** | ⭐⭐ (Black box-like) |

**Pros**:
- ✅ Relatively easy implementation
- ✅ Abundant existing research (TransE, RotatE, ComplEx, etc.)
- ✅ Can handle large-scale data

**Cons**:
- ❌ Difficult to handle complex logical rules
- ❌ Low explainability

---

### 6.2 Method 2: Tensor Rule-Based

#### Overview

Learn tensor values themselves and directly apply logical rules.

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) learnable

rule Connected(x: Person, z: Person) :-
    Friend(x, y), Friend(y, z)

main {
    with Person {
        Friend(alice, bob)      // Observed
        Friend(bob, charlie)    // Observed
        // Friend(alice, charlie) = ? (Unknown, to be learned)
    }

    learn {
        // Compute prediction from rules
        predicted := Connected(alice, charlie)
        // = einsum('xy,yz->xz', Friend, Friend)[alice, charlie]
        // = Friend[alice,bob] * Friend[bob,charlie]
        // = 1.0 * 1.0 = 1.0

        // Observed value (sparse tensor)
        observed := Friend[alice, charlie]  // Unknown (0 or learning target)

        // Loss
        loss := (predicted - observed) ** 2

        // Regularization (constrain tensor values to 0-1 range)
        regularization := sum((Friend - sigmoid(Friend)) ** 2)

        total_loss := loss + 0.1 * regularization

        objective: total_loss,
        optimizer: sgd(lr: 0.1),
        epochs: 100
    }

    infer {
        // Inference by applying rules
        forward Connected(alice, X)?
        // Compute based on rules
    }
}
```

#### Evaluation

| Aspect | Rating |
|--------|--------|
| **Implementation Ease** | ⭐⭐⭐ |
| **Existing Research** | ⭐⭐ (Limited) |
| **Expressiveness** | ⭐⭐⭐⭐ (Direct integration with logical rules) |
| **Explainability** | ⭐⭐⭐⭐ (Rule-based) |

**Pros**:
- ✅ Natural integration with logical rules
- ✅ High explainability
- ✅ Rule-based inference possible

**Cons**:
- ❌ Complex implementation
- ❌ Limited existing research
- ❌ Scalability challenges

---

### 6.3 Method 3: Hybrid (Recommended)

#### Overview

Combine embedding-based learning with logical rule constraints.

```tensorlogic
entity Person

relation Friend(x: Person, y: Person) embed float16[64] learnable

// Rules as soft constraints
rule Symmetric constraint :-
    Friend(x, y) <-> Friend(y, x)
    // Symmetry: friendship is bidirectional

rule Transitive constraint :-
    Friend(x, y), Friend(y, z) -> Friend(x, z)
    // Transitivity: friend of friend may be friend

main {
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }

    learn {
        // ========================================
        // Data Loss (Embedding-Based)
        // ========================================
        data_loss := ranking_loss(positive_facts, negative_facts)

        // ========================================
        // Constraint Loss (Rule-Based)
        // ========================================

        // Symmetry constraint
        symmetric_loss := sum(
            (score(x, Friend, y) - score(y, Friend, x)) ** 2
            for all (x,y) pairs
        )

        // Transitivity constraint
        transitive_loss := sum(
            max(0, min(score(x,y), score(y,z)) - score(x,z))
            for all (x,y,z) triples
        )

        // ========================================
        // Integrated Loss
        // ========================================
        total_loss := data_loss
                    + 0.1 * symmetric_loss
                    + 0.1 * transitive_loss

        objective: total_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    infer {
        // Inference using both embeddings and rules
        forward Friend(alice, X)?

        // Internal processing:
        // 1. Embedding-based score computation
        // 2. Rule-based inference
        // 3. Combine both to determine final score
    }
}
```

#### Evaluation

| Aspect | Rating |
|--------|--------|
| **Implementation Ease** | ⭐⭐ |
| **Existing Research** | ⭐⭐⭐ (Neural-Symbolic AI) |
| **Expressiveness** | ⭐⭐⭐⭐⭐ (Most flexible) |
| **Explainability** | ⭐⭐⭐⭐ (Explainable with rules) |

**Pros**:
- ✅ Most flexible
- ✅ High expressiveness
- ✅ Leverage both data-driven and knowledge-driven approaches
- ✅ Maintain explainability

**Cons**:
- ❌ Most complex
- ❌ Requires hyperparameter tuning

---

### 6.4 Recommended Approach

#### Phased Implementation

**Phase 1**: Start with Method 1 (Embedding-Based)
- Relatively easy implementation
- Leverage existing research insights
- Establish basic learning functionality

**Phase 2**: Extend to Method 3 (Hybrid)
- Add rule constraints
- Improve explainability
- Support more complex reasoning

**Phase 3** (Future): Research Method 2 (Tensor Rule-Based)
- Aim for complete integration
- New research area

---

## 7. Implementation Considerations

### 7.1 Parsing

#### PEG Grammar Extension

```pest
// Entity type declaration
entity_decl = { "entity" ~ IDENT ~ ("=" ~ "{" ~ entity_list ~ "}")? }
// When "=" is omitted, automatically construct from data
// entity Person = {alice, bob}  → Explicit enumeration
// entity Person                 → Auto-construction from data

// with block
with_block = { "with" ~ entity_type ~ "{" ~ statement* ~ "}" }

// Relation declaration
relation_decl = { "relation" ~ IDENT ~ "(" ~ param_list ~ ")" ~ relation_spec? }
relation_spec = {
    ("embed" ~ tensor_type ~ "learnable"?)   // Embedding
    | "learnable"                             // Learnable as tensor
}

// Rule definition
rule_def = { "rule" ~ IDENT ~ "(" ~ param_list ~ ")" ~ rule_body }
rule_body = {
    ":-" ~ rule_conditions                    // Logical rule
    | "constraint" ~ ":-" ~ rule_conditions   // Constraint rule
}
```

### 7.2 Entity Registry

#### Data Structure

```rust
struct EntityRegistry {
    types: HashMap<String, EntityType>,
}

struct EntityType {
    name: String,
    entities: HashSet<String>,     // {"alice", "bob", ...}
    frozen: bool,                   // true after with block
    entity_to_id: HashMap<String, usize>,  // "alice" -> 0
    id_to_entity: Vec<String>,      // [0 -> "alice", 1 -> "bob", ...]
}

impl EntityRegistry {
    fn add_entity(&mut self, type_name: &str, entity: &str) -> Result<usize> {
        let entity_type = self.types.get_mut(type_name)?;

        if entity_type.frozen {
            return Err(Error::EntityTypeFrozen(
                format!("Cannot add '{}' to frozen type '{}'", entity, type_name)
            ));
        }

        if !entity_type.entities.contains(entity) {
            let id = entity_type.entities.len();
            entity_type.entities.insert(entity.to_string());
            entity_type.entity_to_id.insert(entity.to_string(), id);
            entity_type.id_to_entity.push(entity.to_string());
        }

        Ok(entity_type.entity_to_id[entity])
    }

    fn freeze(&mut self, type_name: &str) {
        if let Some(entity_type) = self.types.get_mut(type_name) {
            entity_type.frozen = true;
            println!("Entity type '{}' frozen with {} entities",
                     type_name, entity_type.entities.len());
        }
    }

    fn get_id(&self, type_name: &str, entity: &str) -> Result<usize> {
        let entity_type = self.types.get(type_name)?;
        entity_type.entity_to_id.get(entity)
            .copied()
            .ok_or(Error::UnknownEntity(entity.to_string()))
    }
}
```

### 7.3 Type Inference Engine

#### Type Inference Flow

```rust
struct TypeInference {
    entity_types: HashMap<String, EntityType>,
    relation_types: HashMap<String, RelationType>,
}

struct RelationType {
    params: Vec<(String, String)>,  // [(x, Person), (y, Person)]
    shape: Vec<usize>,               // [4, 4]
}

impl TypeInference {
    fn infer_rule(&self, rule: &Rule) -> Result<EinsumSpec> {
        // 1. Infer variable types from rule conditions
        let mut var_types: HashMap<String, String> = HashMap::new();

        for condition in &rule.conditions {
            let relation = self.relation_types.get(&condition.name)?;

            for (arg, (param_name, param_type)) in
                condition.args.iter().zip(&relation.params) {

                if let Some(var_name) = arg.as_variable() {
                    // Record variable type
                    var_types.insert(var_name.clone(), param_type.clone());
                }
            }
        }

        // 2. Generate einsum string
        let einsum_spec = self.generate_einsum(&rule.conditions, &var_types)?;

        Ok(einsum_spec)
    }

    fn generate_einsum(
        &self,
        conditions: &[Condition],
        var_types: &HashMap<String, String>
    ) -> Result<EinsumSpec> {
        // Mapping from variable names to index characters
        let mut var_to_index: HashMap<String, char> = HashMap::new();
        let mut next_index = 'a';

        let mut input_specs = Vec::new();
        let mut output_vars = Vec::new();

        for condition in conditions {
            let mut indices = String::new();

            for arg in &condition.args {
                if let Some(var_name) = arg.as_variable() {
                    let index = *var_to_index.entry(var_name.clone())
                        .or_insert_with(|| {
                            let idx = next_index;
                            next_index = (next_index as u8 + 1) as char;
                            idx
                        });
                    indices.push(index);
                }
            }

            input_specs.push((condition.name.clone(), indices));
        }

        // Determine output specification
        // (Only variables in rule conclusion)

        Ok(EinsumSpec {
            inputs: input_specs,
            output: output_spec,
        })
    }
}
```

### 7.4 Tensor Allocation and Memory Management

#### Processing at End of with Block

```rust
impl Interpreter {
    fn execute_with_block(
        &mut self,
        entity_type: &str,
        statements: &[Statement]
    ) -> Result<()> {
        // 1. Set entity type to "collection mode"
        self.entity_registry.set_collecting_mode(entity_type, true);

        // 2. Execute statements in block (collect entities)
        for stmt in statements {
            self.execute_statement(stmt)?;
        }

        // 3. Freeze entity type
        self.entity_registry.freeze(entity_type);

        // 4. Allocate tensors for all relations using this type
        self.allocate_tensors_for_entity(entity_type)?;

        Ok(())
    }

    fn allocate_tensors_for_entity(&mut self, entity_type: &str) -> Result<()> {
        let entity_count = self.entity_registry.get_count(entity_type)?;

        // Find all relations using this type
        for (rel_name, rel_type) in &self.relation_types {
            let mut shape = Vec::new();

            for (_, param_type) in &rel_type.params {
                if param_type == entity_type {
                    shape.push(entity_count);
                } else {
                    // Size of other entity types
                    let size = self.entity_registry.get_count(param_type)?;
                    shape.push(size);
                }
            }

            // Allocate tensor
            let device = self.metal_device();
            let tensor = Tensor::zeros(device, &shape)?;
            self.relation_tensors.insert(rel_name.clone(), tensor);

            println!("Allocated tensor for '{}' with shape {:?}", rel_name, shape);
        }

        Ok(())
    }
}
```

### 7.5 GPU Optimization

#### einsum Implementation in Metal GPU

```rust
impl Tensor {
    pub fn einsum(
        spec: &str,
        tensors: &[&Tensor]
    ) -> TensorResult<Tensor> {
        // 1. Parse einsum specification
        let (inputs, output) = parse_einsum_spec(spec)?;

        // 2. Determine optimal computation order
        let plan = optimize_einsum_plan(&inputs, &output, tensors)?;

        // 3. Execute on Metal GPU
        match plan {
            EinsumPlan::MatrixMultiply(a, b) => {
                // Execute as matrix multiplication (fastest)
                Self::matmul(tensors[a], tensors[b])
            }
            EinsumPlan::Contraction(dims) => {
                // General tensor contraction
                Self::contract_gpu(tensors, &dims)
            }
        }
    }

    fn contract_gpu(
        tensors: &[&Tensor],
        contraction: &ContractionSpec
    ) -> TensorResult<Tensor> {
        // Implementation using Metal Compute Shader
        let device = tensors[0].device();
        let command_queue = device.new_command_queue();

        // Execute kernel
        // ...
    }
}
```

---

## 8. Summary

### 8.1 Core of the Design

TensorLogic's unification of logic programming and tensors is based on three core ideas:

1. **Typed Entities**: Entity types determine tensor dimensions
2. **Einstein Summation**: Automatically convert logical rules to tensor operations
3. **with Block**: Balance static type advantages with flexibility

### 8.2 Recommended Syntax

```tensorlogic
// Entity type definition
entity Person
entity Item

// Relation definition
relation Friend(x: Person, y: Person)
relation Likes(x: Person, y: Item)

// Rule definition
rule Recommends(x: Person, z: Item) :-
    Friend(x, y),
    Likes(y, z)

main {
    // Entity collection
    with Person {
        Friend(alice, bob)
        Friend(bob, charlie)
    }

    with Item {
        Likes(alice, book)
        Likes(bob, movie)
    }

    // Learning
    learn {
        // Hybrid approach
    }

    // Inference
    infer {
        forward Recommends(alice, X)?
    }
}
```

### 8.3 Implementation Roadmap

**Phase 1: Foundation** (Priority: Highest)
- ✅ Parse `entity T` (explicit enumeration and auto-construction from data)
- ✅ Parse `with T { ... }` block
- ✅ Implement entity registry
- ✅ Tensor allocation

**Phase 2: Rules** (Priority: High)
- ✅ Parse rule definitions
- ✅ Type inference engine
- ✅ einsum auto-generation

**Phase 3: Learning Integration** (Priority: Medium)
- ✅ Embedding-based learning (Method 1)
- ✅ Support constraint rules (Part of Method 3)

**Phase 4: Advanced Features** (Priority: Low)
- ✅ Complete hybrid approach (Method 3)
- ✅ Tensor rule-based (Method 2, research)

### 8.4 Next Steps

1. **Document Review**: Examine and improve this design
2. **Prototype Implementation**: Phase 1 foundation implementation
3. **Create Examples**: Tutorials and use cases
4. **Community Feedback**: Collect user opinions

---

**References**:
- [Einstein Summation Convention](https://en.wikipedia.org/wiki/Einstein_notation)
- [TransE: Translating Embeddings for Knowledge Graphs](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [Neural-Symbolic Computing: An Effective Methodology](https://arxiv.org/abs/1905.06088)
