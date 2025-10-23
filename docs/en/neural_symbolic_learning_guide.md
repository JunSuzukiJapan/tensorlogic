# Neural-Symbolic Learning Basics Guide

**Target Audience**: Developers without deep learning background
**Last Updated**: 2025-10-23

## Table of Contents

1. [Required Data for Learning](#1-required-data-for-learning)
2. [Role of Functions in Learning](#2-role-of-functions-in-learning)
3. [Overall Learning Flow](#3-overall-learning-flow)
4. [Learn Block Internals](#4-learn-block-internals)
5. [Learnable Keyword Details](#5-learnable-keyword-details)
6. [Summary](#6-summary)

---

## 1. Required Data for Learning

### 1.1 Existing Tensor Learning (Currently Implemented)

First, let's review the data used in current `learn` blocks:

```tensorlogic
// Parameters (learning targets)
tensor w: float16[10] learnable = [...]

// Data
tensor x: float16[10] = [1.0, 2.0, ...]
tensor y_true: float16[1] = [5.0]

main {
    learn {
        // Compute prediction
        y_pred := w * x

        // Compute error
        error := y_pred - y_true
        loss := error * error

        objective: loss,      // ← Value to minimize
        optimizer: sgd(lr: 0.1),
        epochs: 100
    }
}
```

**Data Types**:
- **Learning Parameters** (`learnable`): Values adjusted during learning (weights `w`)
- **Input Data**: Fixed values (`x`)
- **Ground Truth Labels**: Target values (`y_true`)
- **Loss Function**: Difference between prediction and truth (`loss`)

---

### 1.2 Learning Data in Logic Programming

When integrating with logic programming, the data format changes:

#### Knowledge Graph Embedding Learning Example

```tensorlogic
// Relation definition (with learnable embeddings)
relation Friend(x: entity, y: entity) embed float16[64] learnable

// Entity embeddings (learning targets)
embedding person_embed {
    entities: {alice, bob, charlie, diana}
    dimension: 64
    init: xavier
} learnable

main {
    // ========================================
    // Data 1: Observed Facts (Positive Examples)
    // ========================================
    Friend(alice, bob)      // alice and bob are friends
    Friend(bob, charlie)    // bob and charlie are friends
    Friend(charlie, diana)  // charlie and diana are friends

    // ========================================
    // Data 2: False Facts (Negative Examples)
    // ========================================
    // These are typically auto-generated
    // e.g., Friend(alice, diana) is false

    // ========================================
    // Data 3: Logic Constraints
    // ========================================
    // Symmetry: Friendship is bidirectional
    // Friend(X, Y) -> Friend(Y, X)

    // Transitivity: Friend of a friend may be a friend
    // Friend(X, Y), Friend(Y, Z) -> Friend(X, Z)

    learn {
        // Learning goals:
        // 1. Assign high scores to positive facts
        // 2. Assign low scores to negative facts
        // 3. Adjust to satisfy logic constraints

        objective: ranking_loss + constraint_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }
}
```

#### Detailed Data Types

**1. Positive Facts**
```tensorlogic
Friend(alice, bob)
Friend(bob, charlie)
```
- Observed true facts
- Information that "alice and bob are actually friends"
- Want to assign high scores (confidence) to these

**2. Negative Facts**
```tensorlogic
Friend(alice, diana) is false
Friend(eve, frank) is false
```
- Facts that don't hold
- Typically auto-generated (random sampling)
- Want to assign low scores to these

**3. Entity Embeddings**
```
alice   → [0.1, -0.3, 0.5, ..., 0.2]  (64-dimensional vector)
bob     → [0.2, -0.1, 0.4, ..., 0.3]
charlie → [0.3, 0.0, 0.6, ..., 0.1]
```
- Represent each person as a numeric vector
- Optimized during learning

**4. Relation Embeddings**
```
Friend → [0.5, 0.3, -0.2, ..., 0.4]  (64-dimensional vector)
```
- Relations are also represented as numeric vectors
- Optimized during learning

**5. Logic Constraints**
- Rules like symmetry, transitivity
- Added to the loss as "soft constraints"

---

## 2. Role of Functions in Learning

### 2.1 Basic Function Flow

```
Input Data → Score Computation → Loss Computation → Gradient Computation → Parameter Update
```

Let's explain each function in detail.

---

### 2.2 Scoring Function

**Role**: Evaluate numerically how "correct" a fact seems

**Example**: TransE (representative knowledge graph embedding method)

```tensorlogic
// Compute score for Friend(alice, bob)
fn score_fact(subject: entity, relation: entity, object: entity) -> float16 {
    // Get embeddings for entity and relation
    let h = person_embed[subject]      // alice's embedding [64-dim]
    let r = relation_embed[relation]   // Friend's embedding [64-dim]
    let t = person_embed[object]       // bob's embedding [64-dim]

    // TransE score: h + r ≈ t means high score
    // i.e., "alice + Friend relation = bob" relationship
    let diff = h + r - t
    let score = -norm(diff)  // Smaller distance = higher score

    return score
}

// Usage
score1 := score_fact(alice, Friend, bob)      // Positive: expect high score
score2 := score_fact(alice, Friend, diana)    // Negative: expect low score
```

**Intuition**:
```
Positive Friend(alice, bob):
  alice_vec + friend_vec ≈ bob_vec
  [0.1, 0.2] + [0.5, 0.3] ≈ [0.6, 0.5]  ← Close! High score

Negative Friend(alice, diana):
  alice_vec + friend_vec ≠ diana_vec
  [0.1, 0.2] + [0.5, 0.3] ≠ [0.9, 0.1]  ← Far! Low score
```

---

### 2.3 Loss Function

**Role**: Quantify how "wrong" current predictions are

#### 2.3.1 Margin Ranking Loss

```tensorlogic
fn margin_ranking_loss(
    positive_score: float16,
    negative_score: float16,
    margin: float16
) -> float16 {
    // Positive score should be at least margin higher than negative
    // positive_score > negative_score + margin

    let loss = max(0, margin - positive_score + negative_score)
    return loss
}

// Example
positive_score := score_fact(alice, Friend, bob)       // 0.8
negative_score := score_fact(alice, Friend, diana)     // 0.3
margin := 1.0

loss := margin_ranking_loss(0.8, 0.3, 1.0)
// = max(0, 1.0 - 0.8 + 0.3)
// = max(0, 0.5)
// = 0.5  ← Still not enough gap, need more learning
```

**Intuition**:
```
Goal: positive_score - negative_score >= margin

Current: 0.8 - 0.3 = 0.5  ← Smaller than margin (1.0)
→ loss = 0.5 (room for improvement)

Better: 0.9 - 0.1 = 0.8  ← Smaller than margin (1.0) but closer
→ loss = 0.2 (improved)

Perfect: 1.5 - 0.3 = 1.2  ← Exceeds margin (1.0)
→ loss = 0.0 (perfect!)
```

#### 2.3.2 Logic Constraint Loss

```tensorlogic
fn symmetry_constraint_loss() -> float16 {
    // Symmetry: If Friend(X, Y) then Friend(Y, X) should also hold

    let score_forward = score_fact(alice, Friend, bob)   // 0.8
    let score_backward = score_fact(bob, Friend, alice)  // 0.6

    // Minimize difference between both directions
    let diff = score_forward - score_backward
    let loss = diff * diff  // Squared error

    return loss
    // = (0.8 - 0.6)^2 = 0.04
}

fn transitivity_constraint_loss() -> float16 {
    // Transitivity: If Friend(X,Y) and Friend(Y,Z), then Friend(X,Z) should hold

    let score_xy = score_fact(alice, Friend, bob)      // 0.8
    let score_yz = score_fact(bob, Friend, charlie)    // 0.9
    let score_xz = score_fact(alice, Friend, charlie)  // 0.4  ← Low!

    // Bring Friend(alice, charlie) score close to
    // min(score_xy, score_yz) (fuzzy logic)
    let expected = min(score_xy, score_yz)  // min(0.8, 0.9) = 0.8
    let diff = score_xz - expected
    let loss = diff * diff

    return loss
    // = (0.4 - 0.8)^2 = 0.16  ← Large! Needs improvement
}
```

---

### 2.4 Similarity Function

**Role**: Measure how similar two vectors are

```tensorlogic
fn cosine_similarity(vec1: float16[?], vec2: float16[?]) -> float16 {
    // Cosine similarity: -1 (opposite) ~ 1 (same direction)
    let dot_product = sum(vec1 * vec2)
    let norm1 = sqrt(sum(vec1 * vec1))
    let norm2 = sqrt(sum(vec2 * vec2))

    return dot_product / (norm1 * norm2)
}

// Usage
alice_vec := person_embed[alice]
bob_vec := person_embed[bob]
similarity := cosine_similarity(alice_vec, bob_vec)
// 0.8 → Very similar (maybe friends)
// 0.2 → Not very similar (probably not friends)
```

---

### 2.5 Embedding Retrieval Function

```tensorlogic
fn get_embedding(entity_name: entity) -> float16[64] {
    // Get numeric vector from entity name
    return person_embed[entity_name]
}

// Example
alice_vec := get_embedding(alice)
// → [0.1, -0.3, 0.5, 0.2, ..., 0.4]
```

---

## 3. Overall Learning Flow

Let's walk through the actual learning process:

```tensorlogic
relation Friend(x: entity, y: entity) embed float16[64] learnable

embedding person_embed {
    entities: {alice, bob, charlie, diana}
    dimension: 64
    init: random  // Random initialization
} learnable

main {
    // ========================================
    // Step 1: Data Preparation
    // ========================================

    // Positive examples
    Friend(alice, bob)
    Friend(bob, charlie)

    // ========================================
    // Step 2: Learning Loop
    // ========================================

    learn {
        // --- Epoch 1 ---

        // 2.1 Score computation
        pos_score_1 := score_fact(alice, Friend, bob)     // 0.3 (initially low)
        pos_score_2 := score_fact(bob, Friend, charlie)   // 0.4

        // 2.2 Negative score computation (auto-generated)
        neg_score_1 := score_fact(alice, Friend, diana)   // 0.6 (initially high!)
        neg_score_2 := score_fact(bob, Friend, diana)     // 0.5

        // 2.3 Loss computation
        ranking_loss_1 := margin_ranking_loss(0.3, 0.6, 1.0)  // 1.3
        ranking_loss_2 := margin_ranking_loss(0.4, 0.5, 1.0)  // 1.1

        // 2.4 Constraint loss
        symmetry_loss := symmetry_constraint_loss()  // 0.1

        // 2.5 Total loss
        total_loss := ranking_loss_1 + ranking_loss_2 + symmetry_loss
        // = 1.3 + 1.1 + 0.1 = 2.5

        // --- Epoch 50 (after learning progresses) ---

        pos_score_1 := 0.8  // Increased!
        neg_score_1 := 0.2  // Decreased!

        ranking_loss_1 := margin_ranking_loss(0.8, 0.2, 1.0)  // 0.4
        total_loss := 0.6  // Reduced!

        // --- Epoch 100 (learning completed) ---

        pos_score_1 := 0.95  // Further increased!
        neg_score_1 := 0.05  // Further decreased!

        total_loss := 0.05  // Almost zero!

        objective: total_loss,
        optimizer: adam(lr: 0.001),
        epochs: 100
    }

    // ========================================
    // Step 3: Inference After Learning
    // ========================================

    // New question: Are alice and charlie friends?
    infer {
        forward Friend(alice, charlie)?
    }
    // → Score 0.75 is computed
    // → Can infer "probably friends"
}
```

### Data Flow

```
1. Input
   ├─ Positive facts: Friend(alice, bob)
   ├─ Negative facts: Friend(alice, diana) (auto-generated)
   └─ Entities: {alice, bob, charlie, diana}

2. Embeddings (learning parameters)
   ├─ alice  → [0.1, -0.3, ..., 0.2]
   ├─ bob    → [0.2, -0.1, ..., 0.3]
   └─ Friend → [0.5, 0.3, ..., 0.4]

3. Score computation
   ├─ score(alice, Friend, bob) = 0.8  (high)
   └─ score(alice, Friend, diana) = 0.3  (low)

4. Loss computation
   ├─ ranking_loss = 0.5
   └─ constraint_loss = 0.1

5. Optimization
   └─ Update embeddings to reduce loss
```

### Function Role Table

| Function | Input | Output | Role |
|----------|-------|--------|------|
| **score_fact** | (alice, Friend, bob) | 0.8 | Fact confidence |
| **margin_ranking_loss** | (pos=0.8, neg=0.3) | 0.5 | Pos-neg gap evaluation |
| **symmetry_loss** | (forward/backward scores) | 0.1 | Symmetry constraint violation |
| **cosine_similarity** | (vec1, vec2) | 0.75 | Vector similarity |
| **get_embedding** | alice | [0.1, ...] | Embedding retrieval |

---

## 4. Learn Block Internals

### 4.1 Hidden Loop Implementation

**Important**: `learn` blocks contain a hidden loop.

**Implementation location**: [src/interpreter/mod.rs:4498-4510](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/interpreter/mod.rs#L4498-L4510)

```rust
// Training loop with detailed progress display
println!("\n--- Training Progress ---");
for epoch in 0..spec.epochs {           // ← Hidden loop!
    // Zero gradients before computing loss
    if epoch > 0 {
        opt.zero_grad();
    }

    // Re-execute statements for each epoch (recompute intermediate variables)
    for stmt in &spec.statements {      // ← Inner loop!
        self.execute_statement(stmt)?;
    }

    // Compute loss
    let loss_val = self.eval_expr(&spec.objective)?;
    // ...gradient computation and parameter update
}
```

### 4.2 User-Written Code vs Actual Execution

**User-written code**:
```tensorlogic
learn {
    pred := x * w
    loss := (pred - y) * (pred - y)

    objective: loss,
    epochs: 100          // ← This is the loop count
}
```

**Actual execution** (expanded):
```rust
for epoch in 0..100 {
    // Re-execute statements each epoch
    pred := x * w                    // Computed every time
    loss := (pred - y) * (pred - y)  // Computed every time

    // Compute gradients from loss
    backward(loss)

    // Update parameters
    w = w - lr * grad_w
}
```

---

## 5. Learnable Keyword Details

### 5.1 Syntax Position (Prefix vs Postfix)

#### Current Syntax (Postfix)
```tensorlogic
tensor w: float16[10] learnable
relation Friend(x: entity, y: entity) embed float16[64] learnable
```

#### Proposed Syntax (Prefix)
```tensorlogic
learnable tensor w: float16[10]
learnable relation Friend(x: entity, y: entity) embed float16[64]
```

#### Comparison of Pros and Cons

| Aspect | Current (Postfix) | Proposed (Prefix) |
|--------|------------------|-------------------|
| **Declaration type identification** | ⭕ Immediate (`tensor`, `relation`) | ❌ Need to read 2 tokens ahead |
| **Optionality** | ⭕ Can express with `learnable?` | ❌ Need 3 patterns<br>(`learnable tensor`, `frozen tensor`, `tensor`) |
| **Grammar complexity** | ⭕ Simple | ❌ More choices |
| **Search for learning-related** | ❌ Need regex | ⭕ Can search with `^learnable` |

#### Advantages of Current Syntax (Postfix)

1. **Declaration type comes first**
   ```
   tensor ...     ← Immediately know "it's a tensor declaration"
   relation ...   ← Immediately know "it's a relation declaration"
   ```

2. **Good compatibility with PEG grammar**
   ```pest
   declaration = {
       tensor_decl      // starts with "tensor"
       | relation_decl   // starts with "relation"
       | rule_decl       // starts with "rule"
   }
   ```

3. **Optionality is clear**
   ```pest
   tensor_type = { base_type ~ "[" ~ dimensions ~ "]" ~ learnable? }
   ```

#### Recommendation

**Recommend maintaining current syntax (postfix)**

Reasons:
- Compatibility with PEG grammar
- Expression of optionality
- Parser implementation is simpler

---

### 5.2 Internal System Benefits

The `learnable` keyword is a **very important optimization hint**.

#### 5.2.1 Memory Efficiency

```rust
// learnable tensor
tensor w: float16[1000, 1000] learnable
→ requires_grad = true
→ Memory allocation:
   - Data: 1000×1000×2 bytes = 2MB
   - Gradient: 1000×1000×2 bytes = 2MB  ← Additional memory for gradients
   - Computation graph node
   Total: ~4MB

// non-learnable tensor
tensor x: float16[1000, 1000]
→ requires_grad = false
→ Memory allocation:
   - Data: 1000×1000×2 bytes = 2MB only
   Total: 2MB  ← Half the memory!
```

**Example**:
```tensorlogic
// Learning parameters (few)
tensor w: float16[10] learnable      // Allocate gradient buffer
tensor b: float16[1] learnable       // Allocate gradient buffer

// Input data (large)
tensor x_train: float16[10000, 10]   // No gradient buffer needed!
tensor y_train: float16[10000, 1]    // No gradient buffer needed!
```

**Memory comparison**:
```
With learnable: 22 bytes + 22 bytes (gradient) = 44 bytes
Without learnable: 220KB + 220KB (gradient) = 440KB ← 10,000x larger!
```

---

#### 5.2.2 Computation Graph Construction Efficiency

**Implementation location**: [src/tensor/tensor.rs:171-176](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/tensor/tensor.rs#L171-L176)

```rust
pub fn set_requires_grad(&mut self, requires: bool) {
    self.requires_grad = requires;

    // Allocate a node ID for this tensor if requires_grad is true
    if requires && self.grad_node.is_none() {
        let node_id = AutogradContext::allocate_id();
        self.grad_node = Some(node_id);
    }
}
```

**Computation graph example**:
```
learnable tensor w
    ↓
  w * x  ← Record this operation (for backpropagation)
    ↓
  loss
    ↓
backward() ← Compute gradient for w
```

**Non-learnable tensors are excluded from computation graph**:
```
non-learnable tensor x
    ↓
  w * x  ← Don't compute gradient for x (not needed)
    ↓
  loss
```

---

#### 5.2.3 GPU Memory Allocation Optimization

**All tensors are created on Metal GPU, but treated differently**

```rust
// All tensors are created on Metal GPU
let device = self.env.metal_device();
let tensor = Tensor::new(device, shape, data)?;

// But learnable tensors get special treatment
if learnable == LearnableStatus::Learnable {
    tensor.set_requires_grad(true);  // Enable gradient computation
    // → Also allocate gradient buffer on GPU
}
```

**Memory layout (on GPU)**:

```
GPU Memory:

[learnable tensor w]
  ├─ Data buffer (Metal Buffer)
  └─ Gradient buffer (Metal Buffer)  ← Only for learnable

[regular tensor x]
  └─ Data buffer (Metal Buffer) only
```

---

#### 5.2.4 Optimizer Registration

**Implementation location**: [src/interpreter/mod.rs:4439](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/interpreter/mod.rs#L4439)

```rust
// Collect parameter tensors
let params: Vec<Tensor> = learnable_params.iter()
    .map(|(_, t)| t.clone())
    .collect();

// Create optimizer based on spec
let mut opt: Box<dyn Optimizer> = match spec.optimizer.name.as_str() {
    "sgd" => Box::new(SGD::new(params.clone(), lr)),
    "adam" => Box::new(Adam::new(params.clone(), lr)),
    //...
};
```

**Optimizer manages only learnable parameters**:
```
Optimizer (Adam)
  ├─ w: Manage learning rate, momentum, second moment
  ├─ b: Manage learning rate, momentum, second moment
  └─ (x is NOT included!)
```

---

#### 5.2.5 Backpropagation Efficiency

**Implementation location**: [src/tensor/tensor.rs:373-377](https://github.com/JunSuzukiJapan/tensorlogic/blob/main/src/tensor/tensor.rs#L373-L377)

```rust
pub fn backward(&mut self) -> TensorResult<()> {
    if !self.requires_grad {
        return Err(TensorError::InvalidOperation(
            "Cannot call backward on tensor with requires_grad=False".to_string(),
        ));
    }
    // Execute gradient computation...
}
```

**Computation example**:
```tensorlogic
tensor w: float16[10] learnable
tensor x: float16[10]

main {
    pred := w * x
    loss := sum(pred * pred)

    // When backward is called:
    // - Gradient for w IS computed ✅
    // - Gradient for x is NOT computed ❌ (not needed)
}
```

---

### 5.3 Summary Table

| Aspect | With learnable | Without learnable |
|--------|---------------|------------------|
| **Memory** | Data + gradient buffer | Data only |
| **Computation graph** | Node created | Node not created |
| **GPU allocation** | Both data + gradient buffers | Data buffer only |
| **Optimizer** | Registered | Not registered |
| **Backpropagation** | Gradient computed | Gradient not computed |
| **Update** | Parameters updated | Fixed value |

---

## 6. Summary

### 6.1 Types of Learning Data

**Tensor Learning**:
- Learning parameters (`learnable`)
- Input data
- Ground truth labels
- Loss function

**Neural-Symbolic Learning**:
- Positive facts
- Negative facts
- Entity embeddings
- Relation embeddings
- Logic constraints

### 6.2 Important Functions

| Function | Role |
|----------|------|
| **Scoring function** | Compute fact confidence |
| **Ranking loss** | Evaluate positive-negative gap |
| **Constraint loss** | Evaluate logic constraint violations |
| **Similarity function** | Measure vector similarity |

### 6.3 Learn Block Features

- ✅ Hidden loop (`epochs` iterations)
- ✅ Re-execute statements each epoch
- ✅ Automatic gradient computation and parameter update

### 6.4 Importance of Learnable Keyword

**Optimization effects**:
1. Reduce memory usage by up to 50%
2. Minimize computation graph
3. Efficient GPU memory utilization
4. Optimize optimizer management
5. Minimize backpropagation computation

**Recommendations**:
- ✅ Always add `learnable` to learning parameters
- ✅ Don't add to input data
- ✅ Maintain postfix syntax (current implementation)

---

**Next Steps**:
- [Logic Programming Integration Grammar Design](https://github.com/JunSuzukiJapan/tensorlogic) (under development)
- [Language Reference](language_reference.md)
