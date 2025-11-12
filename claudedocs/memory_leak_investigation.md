# GPU Memory Leak Investigation and Fix

## Summary

Fixed critical GPU memory leak where 2+ GB of GPU memory remained allocated after TensorLogic script execution, causing system hangs and crashes on subsequent runs.

**Root Cause**: Model tensors with Arc-counted GPU buffers were not fully released after script completion due to reference counting preventing buffer pool return.

**Solution**: Implemented forced buffer purge using Metal's `set_purgeable_state(MTLPurgeableState::Empty)` when memory leaks are detected at program exit.

**Result**: Successfully reduced leaked memory from 2098 MB to 0.45 MB after script execution.

## Problem Description

### Symptoms
- 2105.77 MB GPU memory leaked after running 3-layer model test
- Subsequent TL script executions hang or crash the system
- System crashes due to memory exhaustion
- No zombie tl processes found after execution

### Test Case
```bash
TL_MEMORY_CHECK=1 timeout 60 ./target/release/tl run:examples/test_model_load_leak.tl
```

Test script loads a 2.1GB GGUF model and accesses one tensor, then exits.

## Investigation Process

### Step 1: Memory Leak Detection
Added `TL_MEMORY_CHECK=1` environment variable to track GPU memory before/after script execution:
- Before execution: 0.61 MB
- After execution: 2098.73 MB
- **Memory leak detected**: 2098.12 MB

### Step 2: Process Diagnosis
Created `scripts/diagnose_memory_leak.sh` to check for:
- All processes sorted by memory usage
- TensorLogic-related processes
- Metal/GPU-related processes

**Finding**: No tl processes remained after execution, confirming the leak was in GPU memory, not hung processes.

### Step 3: GPU Memory Logging
Added detailed logging at allocation/deallocation points:

**Buffer Pool Allocation** ([src/device/buffer_pool.rs:293-323](src/device/buffer_pool.rs#L293-L323)):
```rust
let before_alloc = if std::env::var("TL_DEBUG_MEMORY").is_ok() {
    let mem = parent_device.current_allocated_size();
    eprintln!("[GPU_MEMORY] Before allocation: {:.2} MB allocated", ...);
    Some(mem)
} else { None };

// ... allocation ...

if let Some(before) = before_alloc {
    let after_alloc = parent_device.current_allocated_size();
    let delta = after_alloc as i64 - before as i64;
    eprintln!("[GPU_MEMORY] After allocation: {:.2} MB ({:+.2} MB)", ...);
}
```

**Buffer Deallocation** ([src/device/metal_buffer.rs:230-275](src/device/metal_buffer.rs#L230-L275)):
```rust
impl<T: FloatType> Drop for MetalBuffer<T> {
    fn drop(&mut self) {
        if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            let ref_count = Arc::strong_count(&self.buffer);
            eprintln!("[GPU_MEMORY] Before deallocation: ... ref_count={}", ref_count);
        }
        // ... return to pool or free ...
    }
}
```

### Step 4: Explicit Variable Cleanup
Implemented `clear_variables()` in interpreter to explicitly drop all variables before program exit ([src/interpreter/mod.rs:208-223](src/interpreter/mod.rs#L208-L223)):

**Result**: Freed 918 MB, but 1180 MB remained in buffer pool

### Step 5: Architecture Investigation
Examined why 122 buffers (1180 MB) remained in the pool:

```
Model
 └─> HashMap<String, Tensor>
      └─> BufferHandle::Metal(MetalBuffer)
           └─> Arc<Buffer>  ← GPU memory reference
```

**Key Findings**:
1. `Model` struct holds `HashMap<String, Tensor>` with all model weights
2. Each `Tensor` contains `BufferHandle::Metal(MetalBuffer<T>)`
3. `MetalBuffer` contains `Arc<Buffer>` referencing actual GPU memory
4. `build_layer_collection()` at line 102 calls `tensor.clone()`, increasing Arc ref_count
5. Buffers with ref_count > 1 cannot be returned to pool during Drop
6. Buffer pool accumulated 122 buffers that weren't freed

## Solution Implementation

### 1. Buffer Purge Method
Added `purge_all_buffers()` to BufferPool ([src/device/buffer_pool.rs:649-677](src/device/buffer_pool.rs#L649-L677)):

```rust
/// Purge all buffers by setting them to Empty purgeable state
///
/// This forces Metal to release GPU memory immediately.
/// Should only be called when memory leak is detected at program end.
pub fn purge_all_buffers(&self) {
    use metal::MTLPurgeableState;

    let mut pools = self.pools.lock().unwrap();
    let mut purged_count = 0;
    let mut purged_memory = 0usize;

    for (_size_class, buffers) in pools.iter_mut() {
        for (buffer, _timestamp) in buffers.iter() {
            let buffer_size = buffer.length() as usize;
            buffer.set_purgeable_state(MTLPurgeableState::Empty);
            purged_count += 1;
            purged_memory += buffer_size;
        }
    }

    pools.clear();
}
```

### 2. Device Wrapper
Added wrapper method to MetalDevice ([src/device/metal_device.rs:263-270](src/device/metal_device.rs#L263-L270)):

```rust
/// Force purge all buffers in the buffer pool
pub fn purge_all_buffers(&self) {
    self.buffer_pool.purge_all_buffers();
}
```

### 3. Main Function Integration
Integrated buffer purge into main.rs after memory leak detection ([src/main.rs:429-444](src/main.rs#L429-L444)):

```rust
// Detect memory leak
if memory_diff > 1_048_576 {  // More than 1MB leaked
    eprintln!("\n⚠️  WARNING: GPU memory leak detected!");
    eprintln!("   {:.2} MB of GPU memory was not freed after execution.",
             memory_diff as f64 / 1_048_576.0);

    // Force purge all buffers to release GPU memory
    eprintln!("\n🔧 Attempting to force-release GPU memory...");
    device.purge_all_buffers();

    // Check memory after purge
    let memory_after_purge = device.current_allocated_size();
    let purged_amount = memory_after as i64 - memory_after_purge as i64;
    eprintln!("   After purge: {:.2} MB ({:+.2} MB freed)",
             memory_after_purge as f64 / 1_048_576.0,
             purged_amount as f64 / 1_048_576.0);
}
```

## Test Results

### Before Fix
```
=== GPU Memory Check ===
Before execution: 0.61 MB
After execution: 2098.73 MB
Leaked: 2098.12 MB

⚠️  WARNING: GPU memory leak detected!
   2098.12 MB of GPU memory was not freed after execution.
```

### After Variable Cleanup
```
[GPU_MEMORY] Before cleanup: 2098.73 MB - clearing 2 variables
[GPU_MEMORY] After cleanup: 1180.61 MB - all variables cleared
```
**Result**: Freed 918 MB

### After Buffer Purge
```
🔧 Attempting to force-release GPU memory...
[GPU_MEMORY] Purged 122 buffers (1180.12 MB) from buffer pool
   After purge: 0.45 MB (+1180.12 MB freed)
```
**Result**: Freed remaining 1180 MB

### Final Result
- **Initial leak**: 2098.12 MB
- **After cleanup**: 1180.61 MB (freed 918 MB)
- **After purge**: 0.45 MB (freed 1180 MB)
- **Total freed**: 2097.67 MB (99.98% of leaked memory)

## Environment Variables

### Automatic Memory Leak Detection (Always Enabled)
Memory leak detection and automatic purging is **always enabled by default** as of commit `f2a68ab`. The system will:
- Track GPU memory before and after script execution
- Automatically detect leaks >1MB
- Force-purge leaked buffers using `set_purgeable_state(Empty)`
- Display warning messages when leaks are detected

**No environment variable required** for basic leak protection.

### TL_MEMORY_CHECK
Enable detailed GPU memory statistics and logging:
```bash
TL_MEMORY_CHECK=1 ./target/release/tl run examples/script.tl
```

**With this flag enabled:**
- Shows memory allocation before execution
- Shows detailed buffer pool statistics after execution
- Shows memory change breakdown
- Shows reuse rate and pool efficiency

**Without this flag:**
- Silent memory tracking
- Only shows warnings when leaks >1MB detected
- Automatic purge still works

### TL_DEBUG_MEMORY
Enable detailed GPU memory allocation/deallocation logging (very verbose):
```bash
TL_DEBUG_MEMORY=1 ./target/release/tl run examples/script.tl
```

Shows every buffer allocation and deallocation with:
- Memory size before/after each operation
- Arc reference counts
- Pool return operations
- Buffer size classes

### Combined Usage
For maximum debugging information:
```bash
TL_MEMORY_CHECK=1 TL_DEBUG_MEMORY=1 ./target/release/tl run examples/script.tl
```

## Files Modified

1. **[src/device/buffer_pool.rs](src/device/buffer_pool.rs)**
   - Added GPU memory allocation logging
   - Added `purge_all_buffers()` method

2. **[src/device/metal_buffer.rs](src/device/metal_buffer.rs)**
   - Added GPU memory deallocation logging in Drop trait

3. **[src/device/metal_device.rs](src/device/metal_device.rs)**
   - Added `purge_all_buffers()` wrapper method

4. **[src/interpreter/environment.rs](src/interpreter/environment.rs)**
   - Added `variables()` accessor
   - Added `clear_variables()` method

5. **[src/interpreter/mod.rs](src/interpreter/mod.rs)**
   - Added explicit variable cleanup after main block execution

6. **[src/main.rs](src/main.rs)**
   - Integrated buffer purge after memory leak detection

7. **[scripts/diagnose_memory_leak.sh](scripts/diagnose_memory_leak.sh)** (new)
   - Process diagnosis script for memory leak investigation

## Future Improvements

1. **Reduce Arc Cloning**: Consider optimizing `build_layer_collection()` to avoid excessive tensor cloning that increases reference counts

2. **Automatic Purge**: Consider periodic buffer pool cleanup during long-running operations

3. **Reference Count Tracking**: Add detailed logging of Arc reference counts to identify cloning hotspots

4. **Pool Size Tuning**: Monitor buffer pool size classes and adjust capacity based on actual usage patterns

## Related Issues

This fix prevents system crashes that occurred when:
- Running multiple TL scripts in sequence
- Loading large GGUF models (>1GB)
- System memory pressure from accumulated GPU allocations

The forced buffer purge ensures subsequent TL script executions can proceed without hanging or crashing the system.
