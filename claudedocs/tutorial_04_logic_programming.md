# Tutorial 04: Logic Programming with TensorLogic

**Difficulty**: Beginner
**Time**: 5 minutes
**Topics**: Relations, Entity Types, Neural-Symbolic Integration

## Overview

Learn how to declare relations for logic programming in TensorLogic, setting the foundation for neural-symbolic AI.

## Complete Code

See [examples/tutorial_04_logic_programming.tl](../examples/tutorial_04_logic_programming.tl)

## Key Concepts

- **Relation Declarations**: Define predicates with typed parameters
- **Entity Types**: Use `entity` type for logic programming entities
- **Embeddings**: Attach neural embeddings to relations for learning
- **Neural-Symbolic**: Combine logical reasoning with neural networks

## Relation Syntax

```tensorlogic
relation Name(param1: type, param2: type)
relation Name(param1: type, param2: type) embed float32[dimension]
```

## Future Features

- **Rules**: Define inference rules (e.g., `rule Grandparent(X,Z) <- Parent(X,Y), Parent(Y,Z)`)
- **Queries**: Execute logical queries (e.g., `query Parent(alice, bob)`)
- **Learning**: Train embeddings during gradient descent

---

**Status**: ✅ Verified working (2025-10-20)
