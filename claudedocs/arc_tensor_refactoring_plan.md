# Arc<Tensor> Refactoring Plan

## Executive Summary

**Goal**: Eliminate GPU memory leaks by refactoring `Value` enum to use `Arc<Tensor>` instead of owned `Tensor`, reducing unnecessary clones and enabling immediate buffer pool return.

**Impact**:
- **Memory reduction**: 50%+ during execution (918 MB vs 2098 MB for 2.1GB model)
- **Performance**: Minimal overhead (Arc clone is cheap - just atomic increment)
- **Reliability**: Buffers returned to pool immediately when last reference drops

**Estimated Effort**: 1-2 weeks for careful implementation and testing

## Problem Summary

Current architecture causes Arc<Buffer> reference counts to exceed 1 due to tensor cloning:

```
Current Flow:
1. Model::tensors contains Tensor (Arc ref_count = 1)
2. get_tensor() clones Tensor → ref_count = 2
3. build_layer_collection() clones again → ref_count = 3+
4. Model drops → ref_count still > 1
5. Buffers NOT returned to pool (Drop requires ref_count == 1)
6. GPU memory leaks accumulate

After Refactoring:
1. Model::tensors contains Arc<Tensor> (Arc ref_count = 1)
2. get_tensor() clones Arc → cheap atomic increment, same underlying Tensor
3. Model drops → ref_count decrements
4. When last Arc drops → Tensor drops → Buffer returns to pool
5. No memory leaks
```

## Phase 1: Core Type Changes (Week 1, Days 1-2)

### 1.1 Update Value Enum
**File**: `src/interpreter/value.rs`

```rust
// Before
pub enum Value {
    TensorF16(Tensor<half::f16>),
    TensorF32(Tensor<f32>),
    // ...
}

// After
pub enum Value {
    TensorF16(Arc<Tensor<half::f16>>),
    TensorF32(Arc<Tensor<f32>>),
    // ...
}
```

**Breaking Changes**:
- All code accessing tensors in Value must dereference Arc
- Pattern matching needs adjustment
- Cloning Value becomes cheap (Arc clone only)

### 1.2 Update Model Storage
**File**: `src/model/mod.rs`

```rust
// Before
pub struct Model<T: FloatType = half::f16> {
    pub tensors: HashMap<String, Tensor<T>>,
    pub metadata: ModelMetadata,
}

// After
pub struct Model<T: FloatType = half::f16> {
    pub tensors: HashMap<String, Arc<Tensor<T>>>,
    pub metadata: ModelMetadata,
}
```

**Impact**:
- Model::get_tensor() returns `&Arc<Tensor>` instead of `&Tensor`
- Loading code creates `Arc::new(tensor)` once
- All downstream code gets Arc references

## Phase 2: Interpreter Updates (Week 1, Days 3-4)

### 2.1 Remove Clones in builtin_model.rs
**File**: `src/interpreter/builtin_model.rs`

```rust
// Before (Line 229, 236)
fn eval_get_tensor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    match model_val {
        Value::ModelF16(ref model) => {
            let tensor = model.get_tensor(&name)?;
            Ok(Value::TensorF16(tensor.clone()))  // ← REMOVE CLONE
        }
    }
}

// After
fn eval_get_tensor(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
    match model_val {
        Value::ModelF16(ref model) => {
            let tensor = model.get_tensor(&name)?;  // Returns &Arc<Tensor>
            Ok(Value::TensorF16(Arc::clone(tensor)))  // Arc clone only
        }
    }
}
```

**Impact**: Major memory leak eliminated - ref_count stays at 1

### 2.2 Update Tensor Operations
**Files**:
- `src/interpreter/builtin_tensor.rs`
- `src/interpreter/builtin_nn.rs`

All functions that take `Tensor` parameters must change to `&Tensor` or `Arc<Tensor>`:

```rust
// Before
fn add_tensors(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> { ... }

// After (Option 1: Reference)
fn add_tensors(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> { ... }

// After (Option 2: Arc - if ownership transfer needed)
fn add_tensors(a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Arc<Tensor<T>> { ... }
```

**Decision Point**: Use references where possible for maximum efficiency

## Phase 3: Model Layer Collections (Week 1, Day 5)

### 3.1 Update build_layer_collection()
**File**: `src/model/mod.rs:102`

```rust
// Before
pub fn build_layer_collection(&self, ...) -> Option<ModelLayerCollection<T>> {
    layers.entry(layer_idx)
        .or_insert_with(HashMap::new)
        .insert(feature_path, tensor.clone());  // ← REMOVE CLONE
}

// After
pub fn build_layer_collection(&self, ...) -> Option<ModelLayerCollection<T>> {
    layers.entry(layer_idx)
        .or_insert_with(HashMap::new)
        .insert(feature_path, Arc::clone(tensor));  // Arc clone only
}
```

### 3.2 Update ModelLayerCollection
**File**: `src/interpreter/value.rs` (or wherever ModelLayerCollection is defined)

```rust
// Before
pub struct ModelLayerCollection<T: FloatType> {
    pub layers: HashMap<usize, HashMap<String, Tensor<T>>>,
    pub model_metadata: ModelMetadata,
}

// After
pub struct ModelLayerCollection<T: FloatType> {
    pub layers: HashMap<usize, HashMap<String, Arc<Tensor<T>>>>,
    pub model_metadata: ModelMetadata,
}
```

## Phase 4: Weight Cache Updates (Week 2, Days 1-2)

### 4.1 Update GGUFWeightCache
**File**: `src/model/gguf_weight_cache.rs`

```rust
// Before (Line 137, 247)
if let Some(tensor) = cache.get(name) {
    return Ok(tensor.clone());  // ← REMOVE CLONE
}
cache.put(name.to_string(), tensor.clone());  // ← REMOVE CLONE

// After
if let Some(tensor) = cache.get(name) {
    return Ok(Arc::clone(tensor));  // Arc clone only
}
cache.put(name.to_string(), Arc::new(tensor));  // Store Arc
```

### 4.2 Update Cache Type
```rust
// Before
lru_cache: LruCache<String, Tensor<T>>,

// After
lru_cache: LruCache<String, Arc<Tensor<T>>>,
```

## Phase 5: Llama Model Updates (Week 2, Day 3)

### 5.1 Remove Clones in llama.rs
**File**: `src/model/llama.rs:172-195`

```rust
// Before
let wq = layer_tensors.get("attn_q.weight").unwrap().clone();
let wk = layer_tensors.get("attn_k.weight").unwrap().clone();
// ... many more clones

// After
let wq = layer_tensors.get("attn_q.weight").unwrap();  // &Arc<Tensor>
let wk = layer_tensors.get("attn_k.weight").unwrap();  // &Arc<Tensor>
// Pass as &Tensor to operations: &**wq, &**wk
```

## Phase 6: Tensor Creation Updates (Week 2, Day 4)

### 6.1 Wrap New Tensors in Arc
All tensor creation points that go into Value enum must wrap in Arc:

```rust
// Before
Value::TensorF16(Tensor::zeros(device, shape)?)

// After
Value::TensorF16(Arc::new(Tensor::zeros(device, shape)?))
```

**Affected Functions**:
- `Tensor::from_vec_gpu()`
- `Tensor::zeros()`
- `Tensor::ones()`
- All tensor operations returning new tensors

### 6.2 Consider Arc Pool for Hot Path
For frequently created tensors, consider Arc pooling to reduce allocations:

```rust
// Optional optimization
thread_local! {
    static TENSOR_ARC_POOL: RefCell<Vec<Arc<Tensor<f16>>>> = RefCell::new(Vec::new());
}
```

## Phase 7: Testing & Validation (Week 2, Day 5)

### 7.1 Unit Tests
Update existing tests to work with Arc:

```rust
// Before
fn test_tensor_add() {
    let a = Tensor::zeros(...);
    let b = Tensor::ones(...);
    let c = a + b;
}

// After
fn test_tensor_add() {
    let a = Arc::new(Tensor::zeros(...));
    let b = Arc::new(Tensor::ones(...));
    let c = &*a + &*b;  // Or implement Add for Arc<Tensor>
}
```

### 7.2 Memory Leak Tests
Run test_model_load_leak.tl and verify:
- No memory leak warnings
- Buffer pool return immediately after Model drop
- Arc ref_counts stay at 1 throughout

Expected result:
```bash
$ TL_MEMORY_CHECK=1 ./target/release/tl run examples/test_model_load_leak.tl

=== GPU Memory Check: After Execution ===
GPU memory allocated: 0.61 MB  ← Should be back to baseline
Memory change: +0.00 MB
✅ No memory leaks detected
```

### 7.3 Performance Tests
Measure performance impact:
- Chat demo generation speed
- Memory usage during execution
- Arc clone overhead (should be negligible)

## Implementation Order

### Critical Path (Must be done sequentially):
1. ✅ Phase 1.1: Update Value enum → **All dependent code breaks**
2. ✅ Phase 1.2: Update Model storage → **Required for Value enum**
3. ✅ Phase 2: Update interpreter → **Fix compilation errors**
4. ✅ Phase 6: Tensor creation → **Wrap all new tensors**
5. ✅ Phase 7: Testing → **Verify correctness**

### Parallel Work (Can be done simultaneously):
- Phase 3: Model layer collections
- Phase 4: Weight cache
- Phase 5: Llama model

## Risk Mitigation

### Risk 1: Performance Regression
**Mitigation**:
- Arc clone is ~5ns (atomic increment)
- Compared to tensor clone (~ms for large buffers)
- Net performance gain expected

### Risk 2: Breaking API Changes
**Mitigation**:
- Internal refactoring only
- Public API unchanged (TensorLogic language unchanged)
- Only Rust code affected

### Risk 3: Subtle Reference Counting Bugs
**Mitigation**:
- Comprehensive testing with TL_DEBUG_MEMORY=1
- Monitor Arc ref_counts during execution
- Add assertions for expected ref_counts

### Risk 4: Increased Complexity
**Mitigation**:
- Document Arc usage patterns
- Add helper functions for common operations
- Consider implementing trait methods for Arc<Tensor>

## Helper Traits (Optional)

To reduce verbosity, consider implementing:

```rust
// Add trait for ergonomic Arc<Tensor> operations
impl<T: FloatType> std::ops::Add for &Arc<Tensor<T>> {
    type Output = Arc<Tensor<T>>;

    fn add(self, rhs: &Arc<Tensor<T>>) -> Self::Output {
        Arc::new(&**self + &**rhs)
    }
}

// Usage
let c = &a + &b;  // Instead of Arc::new(&*a + &*b)
```

## Success Metrics

### Memory Metrics
- ✅ **Primary**: No GPU memory leaks in test_model_load_leak.tl
- ✅ **Secondary**: 50%+ memory reduction during chat demo execution
- ✅ **Tertiary**: Buffer pool return rate > 95%

### Performance Metrics
- ✅ Arc clone overhead < 1% of total execution time
- ✅ Chat demo generation speed unchanged (±5%)
- ✅ No increase in GPU synchronization

### Code Quality Metrics
- ✅ All tests passing
- ✅ No new compiler warnings
- ✅ Code coverage maintained or improved

## Rollback Plan

If issues arise:
1. **Immediate**: Revert to main branch (forced purge still works)
2. **Partial**: Cherry-pick specific fixes to production
3. **Complete**: Mark refactor as experimental, keep both paths

## Documentation Updates

After completion:
1. Update [arc_reference_counting_issue.md](arc_reference_counting_issue.md) with "RESOLVED" status
2. Create migration guide for future contributors
3. Document Arc usage patterns in code
4. Update architecture diagrams

## Timeline

### Week 1
- **Day 1-2**: Phase 1 (Core types)
- **Day 3-4**: Phase 2 (Interpreter)
- **Day 5**: Phase 3 (Layer collections)

### Week 2
- **Day 1-2**: Phase 4 (Weight cache)
- **Day 3**: Phase 5 (Llama model)
- **Day 4**: Phase 6 (Tensor creation)
- **Day 5**: Phase 7 (Testing & validation)

**Buffer Days**: Allow 2-3 extra days for unexpected issues

## Next Steps

1. Review this plan with team/maintainers
2. Create detailed task list in issue tracker
3. Set up monitoring for memory leaks during development
4. Begin Phase 1.1: Update Value enum
5. Run continuous integration tests after each phase

## Notes

- This is a **breaking internal API change** but preserves TensorLogic language compatibility
- Focus on correctness first, optimization second
- Use TL_DEBUG_MEMORY=1 extensively during development
- Consider gradual rollout: enable Arc<Tensor> via feature flag initially

## References

- [arc_reference_counting_issue.md](arc_reference_counting_issue.md) - Root cause analysis
- [memory_leak_investigation.md](memory_leak_investigation.md) - Current forced purge solution
- Rust Arc documentation: https://doc.rust-lang.org/std/sync/struct.Arc.html
