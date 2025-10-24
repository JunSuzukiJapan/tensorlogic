# Interpreter Module Structure

This directory contains the TensorLogic interpreter implementation, organized into focused modules for maintainability.

## Current Structure

### Core Modules
- **mod.rs** (5,751 lines) - Main interpreter implementation
- **value.rs** (102 lines) - Runtime value types and conversions
- **environment.rs** (90 lines) - Runtime environment and scope management
- **eval.rs** (47 lines) - Evaluation logic (migration target)
- **formatter.rs** (30 lines) - Value formatting utilities

### Builtin Function Modules (Category-based)

All builtin modules follow the same pattern:
- `eval_<category>_function(name, args) -> Option<RuntimeResult<Value>>`
- Returns `Some(result)` if function belongs to this category
- Returns `None` if function not in this category

**Created:**
- **builtin_tensor.rs** (20 lines) - Basic tensor operations (15 functions)
- **builtin_math.rs** (19 lines) - Math operations (15 functions)
- **builtin_nn.rs** (20 lines) - Neural network operations (12 functions)
- **builtin_kg.rs** (117 lines) - Knowledge graph embeddings (17 functions)
- **builtin_gnn.rs** (17 lines) - Graph neural networks (4 functions)
- **builtin_model.rs** (19 lines) - Model and I/O operations (10 functions)
- **builtin_sampling.rs** (18 lines) - Sampling and generation (8 functions)
- **builtin_util.rs** (18 lines) - Utility functions (8 functions)

**Total:** 89 builtin functions across 8 modules

## Migration Status

### ✅ Completed
1. Created module structure with clear separation of concerns
2. Extracted Value types to value.rs
3. Extracted environment management to environment.rs
4. Created skeleton for 8 builtin modules with dispatchers
5. Updated eval_function_call() to use category dispatchers

### 🚧 In Progress (Stubs)
- Builtin function implementations (currently return NotImplemented error)
- Evaluation logic still in mod.rs

### 📋 Next Steps

#### Priority 1: Migrate Builtin Function Implementations
Move actual implementations from mod.rs match arms to respective modules:

1. **builtin_tensor.rs** (~300 lines)
   - zeros, ones, reshape, flatten, shape
   - transpose, permute, concat, gather, scatter
   - broadcast_to, chunk, split, squeeze, unsqueeze

2. **builtin_math.rs** (~350 lines)
   - matmul, sum, mean, max, min, pow
   - sigmoid, relu, gelu, tanh, exp, log, sqrt
   - sin, cos, tan

3. **builtin_nn.rs** (~400 lines)
   - layer_norm, rms_norm, batch_norm, dropout
   - embedding, positional_encoding
   - Mask operations, fused operations

4. **builtin_kg.rs** (~700 lines)
   - Currently has stubs for all 17 KG functions
   - Need to extract implementations from mod.rs lines 4163-4856

5. **builtin_gnn.rs** (~200 lines)
   - aggregate_neighbors, relational_aggregate
   - graph_attention, normalize_features

6. **builtin_model.rs** (~500 lines)
   - Model loading, tokenization
   - I/O operations (save, load)
   - Generation and sampling

7. **builtin_sampling.rs** (~350 lines)
   - Sampling strategies
   - Temperature, softmax, argmax/argmin

8. **builtin_util.rs** (~200 lines)
   - Utility functions
   - Optimizer initialization

**Total implementation to migrate:** ~3,000 lines

#### Priority 2: Migrate Evaluation Logic to eval.rs

Move from mod.rs to eval.rs (~1,500 lines):

1. **execute_statement()** (~550 lines)
   - Statement execution logic
   - Control flow (if, for, while, break, return)
   - Variable declarations and assignments

2. **eval_expr()** (~180 lines)
   - Expression evaluation
   - Variable lookup, literals, operations

3. **eval_binary_op()** (~400 lines)
   - Binary operations (arithmetic, comparison, logical)
   - Type coercion and error handling

4. **eval_unary_op()** (~100 lines)
   - Unary operations (negation, not)

5. **eval_embedding_lookup()** (~100 lines)
   - Embedding lookup logic

6. **eval_einsum()** (~170 lines)
   - Einstein summation notation

**Total evaluation logic:** ~1,500 lines

## Target Structure

After full migration:

```
interpreter/
├── mod.rs                   (~1,000 lines) - Core Interpreter struct, public API
├── value.rs                 (~100 lines) - Value types
├── environment.rs           (~90 lines) - Environment management
├── eval.rs                  (~1,500 lines) - All evaluation logic
├── builtin_tensor.rs        (~300 lines) - Tensor operations
├── builtin_math.rs          (~350 lines) - Math operations
├── builtin_nn.rs            (~400 lines) - Neural network ops
├── builtin_kg.rs            (~700 lines) - Knowledge graph embeddings
├── builtin_gnn.rs           (~200 lines) - Graph neural networks
├── builtin_model.rs         (~500 lines) - Model & I/O
├── builtin_sampling.rs      (~350 lines) - Sampling & generation
├── builtin_util.rs          (~200 lines) - Utilities
└── formatter.rs             (~30 lines) - Formatting

Total: ~5,720 lines (same as now, but organized)
```

## Implementation Guidelines

### For Builtin Functions

When migrating a builtin function from mod.rs:

1. Find the function in mod.rs (e.g., `"zeros" =>`)
2. Copy the entire match arm body
3. Create a method in the appropriate builtin_*.rs:
   ```rust
   fn eval_zeros(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
       // Paste implementation here
   }
   ```
4. Update the dispatcher to call it:
   ```rust
   "zeros" => Some(self.eval_zeros(args)),
   ```
5. Remove from mod.rs

### For Evaluation Logic

When migrating to eval.rs:

1. Move the entire method signature and body
2. Keep as `pub(super)` to allow mod.rs to call it
3. Update any private helper methods it depends on

## Benefits of This Structure

1. **Maintainability**: Each file has clear, focused responsibility
2. **Navigability**: Easy to find specific functionality
3. **Testability**: Can test individual categories independently
4. **Scalability**: Easy to add new functions to appropriate category
5. **Code Review**: Smaller, focused changes
6. **Documentation**: Each module can have category-specific docs

## Notes

- All changes maintain backward compatibility
- No functional changes, only organizational
- Tests remain in tests.rs
- Migration can be done incrementally
