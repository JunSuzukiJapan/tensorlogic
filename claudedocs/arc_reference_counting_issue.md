# Arc Reference Counting Issue - Root Cause Analysis

## Problem Summary

GPU memory leaks occur due to excessive `tensor.clone()` operations that increase Arc reference counts, preventing buffers from being returned to the buffer pool.

## Architecture

```
Model
 └─> HashMap<String, Tensor<T>>
      └─> BufferHandle::Metal(MetalBuffer<T>)
           └─> Arc<Buffer>  ← GPU memory reference
```

When a Tensor is cloned:
1. `Tensor::clone()` is called (derived Clone trait)
2. This clones `BufferHandle::Metal(MetalBuffer<T>)`
3. MetalBuffer's `#[derive(Clone)]` clones `Arc<Buffer>`
4. **Arc reference count increases by 1**

## Locations of Unnecessary `.clone()` Calls

### 1. **Critical: builtin_model.rs:229, 236**
[src/interpreter/builtin_model.rs:229](src/interpreter/builtin_model.rs#L229)
```rust
fn eval_get_tensor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    // ...
    match model_val {
        Value::ModelF16(ref model) => {
            let tensor = model.get_tensor(&name)  // Returns &Tensor
                .ok_or_else(|| RuntimeError::InvalidOperation(...))?;
            Ok(Value::TensorF16(tensor.clone()))  // ← CLONE HERE
        }
    }
}
```

**Impact**: Every `get_tensor(model, "name")` call clones the tensor
- Increases Arc<Buffer> ref_count from 1 to 2
- When Model is dropped, Arc ref_count = 1 (not 0)
- Buffer NOT returned to pool (ref_count > 1)
- Buffer stays in GPU memory until cloned tensor is also dropped

### 2. **model/mod.rs:102** - build_layer_collection()
[src/model/mod.rs:102](src/model/mod.rs#L102)
```rust
pub fn build_layer_collection(&self) -> Option<ModelLayerCollection<T>> {
    for (name, tensor) in &self.tensors {
        if let Some(rest) = name.strip_prefix("blk.") {
            // ...
            layers.entry(layer_idx)
                .or_insert_with(HashMap::new)
                .insert(feature_path, tensor.clone());  // ← CLONE
        }
    }
}
```

**Impact**: Creates a second copy of every layer tensor
- Each layer tensor now has Arc ref_count = 2

### 3. **model/mod.rs:130, 134** - get_property()
[src/model/mod.rs:130](src/model/mod.rs#L130)
```rust
pub fn get_property(&self, property_name: &str) -> Option<ModelFeature<T>> {
    for (name, tensor) in &self.tensors {
        if let Some(rest) = name.strip_prefix(&prefix) {
            properties.insert(rest.to_string(), tensor.clone());  // ← CLONE
        } else if name == property_name {
            props.insert("".to_string(), tensor.clone());  // ← CLONE
        }
    }
}
```

### 4. **model/gguf_weight_cache.rs:137** - Cache retrieval
[src/model/gguf_weight_cache.rs:137](src/model/gguf_weight_cache.rs#L137)
```rust
if let Some(tensor) = cache.get(name) {
    return Ok(tensor.clone());  // ← CLONE when returning cached tensor
}
```

### 5. **model/gguf_weight_cache.rs:247** - Cache storage
[src/model/gguf_weight_cache.rs:247](src/model/gguf_weight_cache.rs#L247)
```rust
cache.put(name.to_string(), tensor.clone());  // ← CLONE when caching
```

### 6. **model/llama.rs:172-195** - Multiple clones
[src/model/llama.rs:172-195](src/model/llama.rs#L172-L195)
```rust
let wq = layer_tensors.get("attn_q.weight").unwrap().clone();
let wk = layer_tensors.get("attn_k.weight").unwrap().clone();
let wv = layer_tensors.get("attn_v.weight").unwrap().clone();
// ... more clones
```

## Why This Causes Memory Leaks

### Buffer Pool Return Logic
[src/device/metal_buffer.rs:230-275](src/device/metal_buffer.rs#L230-L275)

```rust
impl<T: FloatType> Drop for MetalBuffer<T> {
    fn drop(&mut self) {
        if let (Some(pool), Some(size_class)) = (&self.pool, self.size_class) {
            let ref_count = Arc::strong_count(&self.buffer);

            // Only return to pool if this is the LAST reference
            if ref_count == 1 {
                pool.return_buffer(size_class, Arc::clone(&self.buffer));
            } else {
                // ref_count > 1: Buffer is still referenced elsewhere
                // Don't return to pool, just drop this reference
            }
        }
    }
}
```

**The Problem**:
1. When Model is dropped, its tensors are dropped
2. But cloned tensors still exist (e.g., from `get_tensor()` call)
3. Arc ref_count = 2 (original + clone)
4. Drop checks: `ref_count == 1`? No, it's 2
5. Buffer NOT returned to pool
6. Buffer accumulates in GPU memory

## Test Case Demonstration

[examples/test_model_load_leak.tl](examples/test_model_load_leak.tl):
```tl
main {
    let model = load_model(path)           // Creates tensors with Arc ref_count = 1
    let emb = get_tensor(model, "...")     // Clones tensor → Arc ref_count = 2
    // Model goes out of scope → drop
    // But emb still exists → Arc ref_count = 1
    // Buffer NOT returned to pool
}
```

Result: **1180.12 MB leaked** (122 buffers remain in pool)

## Potential Solutions

### Solution 1: Remove Clones (Requires Design Change)
**Option A**: Use Arc<Tensor> in Value enum
```rust
pub enum Value {
    TensorF16(Arc<Tensor<half::f16>>),  // Instead of Tensor<half::f16>
    // ...
}
```
- Pros: Minimal Arc cloning, shared ownership
- Cons: Large refactoring, changes API

**Option B**: Return references instead of clones
```rust
// Not feasible - Value needs owned data for variable storage
```

**Option C**: Implement Copy-on-Write (CoW) for Tensors
- Use Arc<Tensor> internally
- Clone only when mutation needed
- Cons: Complex implementation

### Solution 2: Lazy Buffer Pool Return
Track all Arc<Buffer> references and return to pool when last reference drops:
```rust
// Add to MetalBuffer
weak_refs: Arc<Mutex<Vec<Weak<Buffer>>>>
```
- Cons: Complex, runtime overhead

### Solution 3: Manual Memory Management
Add explicit `release_buffer()` method:
```rust
tensor.release_buffer()  // Manually return buffer to pool
```
- Cons: Unsafe, easy to misuse

### Solution 4: Keep Current Forced Purge (Recommended for Now)
✅ **Current implementation** - Force purge at program end
- Pros: Simple, effective, no design changes needed
- Cons: Doesn't solve root cause, memory stays high during execution

## Recommended Action

**Short term**: Keep the forced purge mechanism (already implemented)
- Prevents system crashes
- Works at program end when buffers are no longer needed
- Simple and effective

**Long term**: Refactor to use Arc<Tensor> throughout
- Change Value enum to use Arc<Tensor>
- Remove unnecessary clones
- Implement shared ownership pattern
- Requires significant refactoring (~50+ files affected)

## Impact on Memory Usage

Current situation with 2.1GB GGUF model:
- Initial load: 918 MB (Model's HashMap)
- After `get_tensor()` clone: +1180 MB (cloned tensors)
- After `build_layer_collection()`: Additional clones for layer collection
- **Total**: ~2098 MB leaked

With forced purge:
- Final memory: 0.45 MB ✅
- System crash prevented ✅
- But high memory usage during execution (until purge)

With Arc<Tensor> refactor:
- Memory usage: ~918 MB (only Model's HashMap)
- No unnecessary clones
- Buffers returned to pool immediately when Model dropped
- **50%+ memory reduction** during execution

## Files That Would Need Changes for Arc<Tensor> Refactor

1. `src/interpreter/value.rs` - Value enum
2. `src/interpreter/builtin_model.rs` - Remove clones
3. `src/model/mod.rs` - Return Arc<Tensor> from methods
4. `src/model/llama.rs` - Use Arc references
5. `src/model/gguf_weight_cache.rs` - Cache Arc<Tensor>
6. All tensor operation functions (~30+ files)

Estimated refactoring effort: **1-2 weeks** for careful implementation and testing.
