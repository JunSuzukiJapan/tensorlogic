# Critical Bug: For-loop corrupts tensor values computed before the loop

## Summary
When a `for` loop is declared after heavy GPU computation (22 transformer layers),
tensor values computed BEFORE the loop become corrupted, producing zero or NaN values.

## Reproduction

### Minimal Test Case
```tl
// Process 22 transformer layers (heavy GPU usage)
let h = transformer_layer(...) // Process all 22 layers
let logits = linear(h, weights)  // Compute logits from h

// ⚠️ BUG: This for-loop declaration corrupts `logits`
for i in range(50) {
    let token = temperature_sample(logits, 0.0)  // ← logits are now ZERO!
    // ... process token ...
    let logits = ...  // Recompute for next iteration (THIS works fine)
}
```

### Test Results

| Scenario | Result |
|----------|---------|
| 22 layers WITHOUT for-loop | ✓ Valid logits (5.932094) |
| 22 layers + for-loop (first iter) | ❌ ZERO logits (0.000000) |
| 22 layers + for-loop (second+ iter) | ✓ Valid logits (5.551721) |

## Observed Pattern

### Chat Demo Output
```
First decode token:  logit=0.000000 (ZERO) → token 给
Second decode token: logit=5.551721 (VALID) → "ered"
Third decode token:  logit=5.570559 (VALID) → "Pfl"
```

### Key Finding
- Variables computed BEFORE the loop are corrupted
- Variables computed INSIDE the loop work correctly
- Corruption only occurs after heavy GPU usage (22 layers × 35 tokens)
- Problem does NOT occur with lighter workloads (3 layers, 10 layers)

## Root Cause Analysis

### Heisenbug Discovery
**CRITICAL**: This is a Heisenbug - adding `eprintln!()` debug logging "fixes" the bug due to timing delays.

The for-loop initialization triggers:
1. Vec allocation for loop items (`Vec<Value>`)
2. Memory pressure on system allocator
3. **Buffer pool premature recycling** - GPU buffers recycled while still referenced
4. Arc reference counting doesn't prevent recycling when GPU operations pending

### Failed Fix Attempts

| Approach | Result | Why it Failed |
|----------|--------|---------------|
| No changes | First iteration zero, rest valid | Original bug |
| GPU sync BEFORE iterable | **ALL iterations zero** | Buffers recycled after sync completes |
| GPU sync BEFORE + AFTER Vec | Alternating zero/valid pattern | Complex interaction, buffers recycled between syncs |
| Debug logging (`eprintln!`) | ✓ Works temporarily | Timing delays mask race condition |

### The Real Problem
**Buffer Lifetime Management**: The BufferPool recycles buffers based on Arc strong_count, but this doesn't account for:
1. Pending GPU command buffers that reference the buffer
2. Memory pressure from Vec allocation triggering premature cleanup
3. Race condition between GPU async execution and CPU memory management

## Related Code

### For-loop evaluation (src/interpreter/eval.rs:143-227)
```rust
ControlFlow::For { variable, iterable, body } => {
    // Evaluate iterable
    let items = match iterable {
        Iterable::Range(n) => {
            (0..*n).map(|i| Value::Integer(i as i64)).collect::<Vec<_>>()
        }
        ...
    };

    for item in items {
        self.env.push_scope(ScopeType::Loop);  // ← Scope management
        self.env.declare_variable(loop_var_name.clone(), item)?;
        let result = self.execute_block(body, true);
        self.env.pop_scope();  // ← Cleanup
        ...
    }
}
```

### Buffer pool (src/device/buffer_pool.rs)
- LRU-based buffer reuse
- Buffers returned to pool on Drop
- Possible premature reuse of buffers?

## Impact

**CRITICAL**: All chat demos and autoregressive generation produce incorrect first tokens,
making the output unreliable and unpredictable.

## Potential Solutions

### Option 1: Disable Buffer Pool Temporarily
- Set a flag to disable buffer recycling during critical operations
- Force `wait_until_completed()` before allowing buffer reuse
- **Pros**: Simple, guaranteed to work
- **Cons**: Performance impact, doesn't fix root cause

### Option 2: Track GPU Command Buffer Dependencies
- Maintain a list of buffers referenced by pending GPU commands
- Only recycle buffers after their command buffers complete
- Use Metal's completion handlers
- **Pros**: Proper fix, maintains performance
- **Cons**: Complex implementation

### Option 3: Retain Buffers Until GPU Sync
- Keep strong Arc references to buffers until explicit sync point
- Add buffer retention list to MetalDevice
- **Pros**: Relatively simple, effective
- **Cons**: May hold onto buffers longer than necessary

### Option 4: Change Buffer Pool Strategy
- Don't recycle buffers based on Arc count alone
- Wait for GPU idle before recycling
- Use time-based or usage-based heuristics
- **Pros**: System-wide fix
- **Cons**: May impact all GPU operations

## Test Files

- `debug/test_combination.tl` - Shows progressive isolation
- `debug/test_forloop_corruption.tl` - Minimal reproduction
- `debug/chat_optimized_transformer.tl` - Real-world failure case
