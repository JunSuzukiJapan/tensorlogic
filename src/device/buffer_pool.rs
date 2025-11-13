//! Buffer pool for Metal buffer reuse and memory optimization

use crate::device::MetalBuffer;
use crate::error::TensorResult;
use crate::tensor::FloatType;
use half::f16;
use metal::{Buffer, Device as MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Statistics about buffer pool usage
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of buffers currently in the pool
    pub total_pooled: usize,
    /// Number of different buffer sizes in the pool
    pub size_classes: usize,
    /// Total memory (in bytes) held by the pool
    pub total_memory: usize,
    /// Number of successful reuses
    pub reuse_count: usize,
    /// Number of new allocations
    pub allocation_count: usize,
    /// Number of buffers evicted by LRU
    pub eviction_count: usize,
}

/// Buffer pool for efficient Metal buffer reuse with LRU eviction
///
/// Reduces memory allocation overhead by maintaining a pool of reusable buffers
/// organized by size. Tracks last access time for LRU-based eviction.
///
/// Buffer pool supporting both f16 and f32 types.
pub struct BufferPool {
    /// Metal device for buffer creation
    /// Store MTLDevice only to avoid infinite recursion during initialization
    device: Arc<metal::Device>,

    /// Pool of buffers organized by size (in elements, not bytes)
    /// HashMap<size, Vec<(Arc<Buffer>, Instant)>>
    /// Each buffer is paired with its last access time
    pools: Arc<Mutex<HashMap<usize, Vec<(Arc<Buffer>, Instant)>>>>,

    /// Maximum number of buffers to keep per size class
    max_buffers_per_size: usize,

    /// Statistics
    stats: Arc<Mutex<PoolStats>>,
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            device: Arc::clone(&self.device),
            pools: Arc::clone(&self.pools),
            max_buffers_per_size: self.max_buffers_per_size,
            stats: Arc::clone(&self.stats),
        }
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("max_buffers_per_size", &self.max_buffers_per_size)
            .field("stats", &self.stats())
            .finish()
    }
}

/// Get the size class for a given length (in f16 elements)
///
/// Uses power-of-2 size classes with a minimum of 1024 elements to reduce fragmentation
/// and improve buffer reuse rates.
fn get_size_class(length: usize) -> usize {
    const SIZE_CLASSES: &[usize] = &[
        1024,      // 2 KB
        2048,      // 4 KB
        4096,      // 8 KB
        8192,      // 16 KB
        16384,     // 32 KB
        32768,     // 64 KB
        65536,     // 128 KB
        131072,    // 256 KB
        262144,    // 512 KB
        524288,    // 1 MB
        1048576,   // 2 MB
        2097152,   // 4 MB
        4194304,   // 8 MB
        8388608,   // 16 MB
    ];

    SIZE_CLASSES
        .iter()
        .find(|&&size| size >= length)
        .copied()
        .unwrap_or(length) // For very large buffers, use exact size
}

/// Public version of get_size_class for use by MetalBuffer
pub fn get_size_class_pub(length: usize) -> usize {
    get_size_class(length)
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: &MTLDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_buffers_per_size: 10,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Create a new buffer pool with custom max buffers per size
    pub fn with_capacity(device: &MTLDevice, max_buffers_per_size: usize) -> Self {
        Self {
            device: Arc::new(device.clone()),
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_buffers_per_size,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Allocate a MetalBuffer from the pool or create a new one
    ///
    /// # Arguments
    /// * `parent_device` - The MetalDevice that owns this pool (for GPU sync)
    /// * `length` - Number of f16 elements (requested size)
    ///
    /// Uses size-class pooling to improve buffer reuse rates.
    /// The actual allocated buffer may be larger than requested.
    pub fn allocate<T: FloatType>(&self, parent_device: &crate::device::MetalDevice, length: usize) -> TensorResult<MetalBuffer<T>> {
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: ENTRY, length={}", length);
        }
        let size_class = get_size_class(length);

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: size_class={}", size_class);
        }

        let mut pools = self.pools.lock().unwrap();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: pools locked");
        }
        let mut stats = self.stats.lock().unwrap();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: stats locked");
        }

        // Debug logging
        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!("[BufferPool::allocate] requested={}, size_class={}", length, size_class);
        }

        // Try to reuse an existing buffer from the size class
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: checking pool for reuse...");
            std::io::stderr().flush().ok();
        }
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: about to calculate total_pooled...");
            std::io::stderr().flush().ok();
        }

        // Check if BufferPool is full
        let total_pooled: usize = pools.values().map(|v| v.len()).sum();

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: total_pooled={}, max_per_size={}",
                     total_pooled, self.max_buffers_per_size);
            std::io::stderr().flush().ok();
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: calling pools.get_mut(size_class={})...", size_class);
            std::io::stderr().flush().ok();
        }

        if let Some(buffers) = pools.get_mut(&size_class) {
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] BufferPool::allocate: size class {} exists, {} buffers available",
                         size_class, buffers.len());
                std::io::stderr().flush().ok();
            }
            if let Some((buffer, _last_used)) = buffers.pop() {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[DEBUG_RS] BufferPool::allocate: found reusable buffer");
                }
                stats.reuse_count += 1;

                // Periodically check and shrink (every 100 allocations)
                let should_check = stats.allocation_count % 100 == 0;
                drop(stats);
                drop(pools);

                if should_check {
                    self.check_and_shrink();
                }

                // DEBUG: Log buffer contents for large buffers (likely logits)
                if std::env::var("TL_BUFFER_DEBUG").is_ok() && size_class > 1_000_000 {
                    let ptr = buffer.contents() as *const f32;
                    if !ptr.is_null() {
                        let slice = unsafe { std::slice::from_raw_parts(ptr, 10.min(length)) };
                        eprintln!(
                            "[BufferPool::allocate] LARGE BUFFER REUSED (logits?) size_class={}, length={}, buffer_ptr={:p}, first_10={:?}",
                            size_class, length, buffer.as_ref(), slice
                        );
                    }
                }

                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!("[BufferPool::allocate] ✓ reused buffer from pool, size_class={}", size_class);
                }

                // NOTE: DO NOT zero out buffers here!
                // Reasons:
                // 1. new_uninit_pooled() expects uninitialized buffers (for performance)
                // 2. CPU write to GPU memory (write_bytes) causes implicit GPU sync,
                //    which hangs when many GPU operations are pending (e.g., Layer 2+ in transformers)
                // 3. Callers who need zeros should use allocate_zeros() instead
                //
                // Previous synchronous zeroing caused 60s+ hangs at Layer 2 in f32 inference.
                // Kernels overwrite all buffer contents anyway, so uninitialized is safe.

                return Ok(MetalBuffer {
                    buffer,
                    length,  // Store requested length, not size_class
                    _phantom: PhantomData,
                    pool: Some(self.clone()),
                    size_class: Some(size_class),
                    device: parent_device.clone(),
                });
            } else {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[DEBUG_RS] BufferPool::allocate: size class {} exists but no buffers available", size_class);
                    std::io::stderr().flush().ok();
                }
            }
        } else {
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] BufferPool::allocate: size class {} does not exist yet", size_class);
                std::io::stderr().flush().ok();
            }
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: no reusable buffer, creating new one");
            std::io::stderr().flush().ok();
        }

        // Create a new buffer with size_class capacity
        stats.allocation_count += 1;

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: allocation_count incremented to {}", stats.allocation_count);
        }

        // Warn if BufferPool is getting full (many allocations with no reuse)
        // Since buffers are not returned to pool on drop, this indicates memory pressure
        const WARN_THRESHOLD: usize = 1000;
        const ERROR_THRESHOLD: usize = 5000;

        if stats.allocation_count > ERROR_THRESHOLD && stats.reuse_count < stats.allocation_count / 10 {
            eprintln!("[BufferPool] CRITICAL: Too many allocations ({}) with minimal reuse ({})",
                     stats.allocation_count, stats.reuse_count);
            eprintln!("[BufferPool] This indicates memory leak - buffers are not being returned to pool");
            eprintln!("[BufferPool] System may run out of memory soon!");
        } else if stats.allocation_count > WARN_THRESHOLD && stats.reuse_count < stats.allocation_count / 5 {
            eprintln!("[BufferPool] WARNING: High allocation count ({}) with low reuse ({})",
                     stats.allocation_count, stats.reuse_count);
            eprintln!("[BufferPool] Buffers may not be returned to pool properly");
        }

        // Periodically check and shrink (every 100 allocations)
        let should_check = stats.allocation_count % 100 == 0;

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: dropping stats lock...");
        }
        drop(stats); // Release stats lock before allocation

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: dropping pools lock...");
        }
        drop(pools); // Release pools lock before check_and_shrink to avoid deadlock

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: calling device.new_buffer...");
        }

        // GPU Memory Allocation Logging
        let before_alloc = if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            let mem = parent_device.current_allocated_size();
            eprintln!("[GPU_MEMORY] Before allocation: {:.2} MB allocated",
                     mem as f64 / 1_048_576.0);
            Some(mem)
        } else {
            None
        };

        // Allocate buffer with size_class capacity (not exact requested length)
        let byte_length = size_class * std::mem::size_of::<T>();
        let buffer = self.device.as_ref().new_buffer(
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: new_buffer returned");
        }

        // GPU Memory Allocation Logging
        if let Some(before) = before_alloc {
            let after_alloc = parent_device.current_allocated_size();
            let delta = after_alloc as i64 - before as i64;
            eprintln!("[GPU_MEMORY] After allocation: {:.2} MB ({:+.2} MB) - size_class={}, bytes={}",
                     after_alloc as f64 / 1_048_576.0,
                     delta as f64 / 1_048_576.0,
                     size_class,
                     byte_length);
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: checking TL_BUFFER_DEBUG...");
        }

        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!("[BufferPool::allocate] ✗ new allocation, size_class={}, bytes={}",
                     size_class, byte_length);
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: TL_BUFFER_DEBUG check done, should_check={}", should_check);
        }

        // Periodically check and shrink after allocation
        if should_check {
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] BufferPool::allocate: calling check_and_shrink...");
            }
            self.check_and_shrink();
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] BufferPool::allocate: check_and_shrink returned");
            }
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] BufferPool::allocate: creating MetalBuffer result...");
        }

        Ok(MetalBuffer {
            buffer: Arc::new(buffer),
            length,  // Store requested length, not size_class
            _phantom: PhantomData,
            pool: Some(self.clone()),
            size_class: Some(size_class),
            device: parent_device.clone(),
        })
    }

    /// Allocate a MetalBuffer filled with zeros
    pub fn allocate_zeros<T: FloatType>(&self, parent_device: &crate::device::MetalDevice, length: usize) -> TensorResult<MetalBuffer<T>> {
        let buffer = self.allocate::<T>(parent_device, length)?;

        // Zero out the buffer
        unsafe {
            let ptr = buffer.buffer.contents() as *mut T;
            std::ptr::write_bytes(ptr, 0, length);
        }

        Ok(buffer)
    }

    /// Try to return a buffer to the pool with current timestamp
    ///
    /// Returns true if the buffer was added to the pool, false otherwise
    /// (e.g., if the pool is full for this size class)
    ///
    /// Uses size-class pooling - the buffer is stored in its size class,
    /// not by its exact length, to improve reuse rates.
    pub fn recycle(&self, buffer: MetalBuffer<half::f16>) -> bool {
        // Get the actual buffer capacity (not the requested length)
        let actual_capacity = (buffer.buffer.length() as usize) / std::mem::size_of::<f16>();
        let size_class = get_size_class(actual_capacity);

        let mut pools = self.pools.lock().unwrap();

        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!("[BufferPool::recycle] buffer.length={}, actual_capacity={}, size_class={}",
                     buffer.length, actual_capacity, size_class);
        }

        let buffers = pools.entry(size_class).or_insert_with(Vec::new);

        // Only add if we haven't reached the limit
        if buffers.len() < self.max_buffers_per_size {
            // Record current time as last access time
            buffers.push((buffer.buffer.clone(), Instant::now()));

            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!("[BufferPool::recycle] ✓ added to pool, size_class={}, pool_size={}",
                         size_class, buffers.len());
            }

            true
        } else {
            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!("[BufferPool::recycle] ✗ pool full, size_class={}, limit={}",
                         size_class, self.max_buffers_per_size);
            }
            false
        }
    }

    /// Try to return a buffer to the pool (called automatically by MetalBuffer::drop)
    ///
    /// Uses try_lock() to avoid deadlock - if the pool is already locked, the buffer
    /// is simply dropped and freed by the OS. This is safe and prevents hangs.
    pub fn try_return_buffer(&self, buffer: &Arc<Buffer>, size_class: usize, length: usize) {
        // Try to lock without blocking - if we can't get the lock, just drop the buffer
        let mut pools = match self.pools.try_lock() {
            Ok(pools) => pools,
            Err(_) => {
                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!(
                        "[BufferPool::try_return_buffer] ✗ pool locked, dropping buffer (size_class={}, length={})",
                        size_class, length
                    );
                }
                return; // Buffer will be dropped and freed
            }
        };

        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!(
                "[BufferPool::try_return_buffer] length={}, size_class={}",
                length, size_class
            );
        }

        // CRITICAL FIX: Check Arc reference count before returning to pool
        // If there are other references (strong_count > 1), don't return to pool
        // to prevent buffer sharing between layers
        let ref_count = Arc::strong_count(buffer);

        // DEBUG: Log buffer contents for large buffers (likely logits)
        if std::env::var("TL_BUFFER_DEBUG").is_ok() && size_class > 1_000_000 {
            let ptr = buffer.contents() as *const f32;
            if !ptr.is_null() {
                let slice = unsafe { std::slice::from_raw_parts(ptr, 10.min(length)) };
                eprintln!(
                    "[BufferPool::try_return_buffer] LARGE BUFFER (logits?) size_class={}, length={}, ref_count={}, buffer_ptr={:p}, first_10={:?}",
                    size_class, length, ref_count, buffer.as_ref(), slice
                );
            }
        }

        if ref_count > 1 {
            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!(
                    "[BufferPool::try_return_buffer] ✗ buffer has {} references, not returning to pool (size_class={})",
                    ref_count, size_class
                );
            }
            return; // Don't return to pool, let it be freed
        }

        let buffers = pools.entry(size_class).or_insert_with(Vec::new);

        // Only add if we haven't reached the limit
        if buffers.len() < self.max_buffers_per_size {
            // Clone the Arc here to add to pool (intentional ref count increase)
            // Record current time as last access time
            buffers.push((buffer.clone(), Instant::now()));

            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!(
                    "[BufferPool::try_return_buffer] ✓ returned to pool, size_class={}, pool_size={}",
                    size_class,
                    buffers.len()
                );
            }
        } else {
            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!(
                    "[BufferPool::try_return_buffer] ✗ pool full, size_class={}, limit={}, dropping buffer",
                    size_class, self.max_buffers_per_size
                );
            }
            // Buffer will be dropped and freed
        }
    }

    /// Clear all buffers from the pool
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
    }

    /// Clear buffers of a specific size from the pool
    pub fn clear_size(&self, length: usize) {
        let mut pools = self.pools.lock().unwrap();
        pools.remove(&length);
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let pools = self.pools.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        let mut total_pooled = 0;
        let mut total_memory = 0;

        for (size, buffers) in pools.iter() {
            total_pooled += buffers.len();
            total_memory += size * std::mem::size_of::<f16>() * buffers.len();
        }

        PoolStats {
            total_pooled,
            size_classes: pools.len(),
            total_memory,
            reuse_count: stats.reuse_count,
            allocation_count: stats.allocation_count,
            eviction_count: stats.eviction_count,
        }
    }

    /// Get the Metal device (MTLDevice)
    pub fn device(&self) -> &MTLDevice {
        self.device.as_ref()
    }

    /// Evict buffers that haven't been used for longer than `max_age`
    ///
    /// This implements LRU (Least Recently Used) eviction policy.
    /// Returns the number of buffers evicted.
    pub fn evict_old_buffers(&self, max_age: Duration) -> usize {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        let mut evicted_count = 0;

        // For each size class
        for (_size, buffers) in pools.iter_mut() {
            // Keep only buffers that were accessed recently
            let original_len = buffers.len();
            buffers.retain(|(_buffer, last_used)| {
                let age = now.duration_since(*last_used);
                age <= max_age
            });
            evicted_count += original_len - buffers.len();
        }

        stats.eviction_count += evicted_count;
        evicted_count
    }

    /// Shrink the pool to fit within memory constraints
    ///
    /// Removes least recently used buffers until total memory is below the limit.
    /// Prioritizes removing buffers from larger size classes first.
    pub fn shrink_to_fit(&self, max_memory_bytes: usize) {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let mut current_memory = 0;
        for (size, buffers) in pools.iter() {
            current_memory += size * std::mem::size_of::<f16>() * buffers.len();
        }

        if current_memory <= max_memory_bytes {
            return;
        }

        // Sort buffers by last access time (oldest first) within each size class
        // Then remove from largest size classes first
        let mut sizes: Vec<usize> = pools.keys().copied().collect();
        sizes.sort_by(|a, b| b.cmp(a)); // Descending order

        for size in sizes {
            if current_memory <= max_memory_bytes {
                break;
            }

            if let Some(buffers) = pools.get_mut(&size) {
                // Sort by last access time (oldest first)
                buffers.sort_by_key(|(_buf, last_used)| *last_used);

                while !buffers.is_empty() && current_memory > max_memory_bytes {
                    buffers.remove(0); // Remove oldest
                    current_memory -= size * std::mem::size_of::<f16>();
                    stats.eviction_count += 1;
                }
            }
        }
    }

    /// Check memory usage and automatically shrink if needed
    ///
    /// Uses environment variable TL_BUFFER_MAX_MB to set memory limit (default: 512 MB)
    /// Returns true if shrinking was performed
    pub fn check_and_shrink(&self) -> bool {
        // Get max memory from environment variable (in MB)
        let max_mb = std::env::var("TL_BUFFER_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512); // Default 512 MB

        let max_bytes = max_mb * 1_048_576; // Convert to bytes

        let current_stats = self.stats();
        if current_stats.total_memory > max_bytes {
            if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                eprintln!("[BufferPool] Memory limit exceeded: {} MB / {} MB, shrinking...",
                    current_stats.total_memory / 1_048_576,
                    max_mb);
            }
            self.shrink_to_fit(max_bytes);
            true
        } else {
            false
        }
    }

    /// Print current buffer pool statistics
    ///
    /// Useful for monitoring during execution
    pub fn print_current_stats(&self, label: &str) {
        let stats = self.stats();
        eprintln!("\n=== Buffer Pool Stats: {} ===", label);
        eprintln!("  Pooled buffers: {}", stats.total_pooled);
        eprintln!("  Size classes: {}", stats.size_classes);
        eprintln!("  Total memory: {:.2} MB", stats.total_memory as f64 / 1_048_576.0);
        eprintln!("  Allocations: {}", stats.allocation_count);
        eprintln!("  Reuses: {}", stats.reuse_count);
        eprintln!("  Evictions: {}", stats.eviction_count);

        let total_ops = stats.allocation_count + stats.reuse_count;
        if total_ops > 0 {
            let reuse_rate = (stats.reuse_count as f64 / total_ops as f64) * 100.0;
            eprintln!("  Reuse rate: {:.1}%", reuse_rate);
        }
        eprintln!("================================\n");
    }

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

        // Clear all pools after purging
        pools.clear();

        if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            eprintln!("[GPU_MEMORY] Purged {} buffers ({:.2} MB) from buffer pool",
                     purged_count,
                     purged_memory as f64 / 1_048_576.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_buffer_pool_creation() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        let stats = pool.stats();
        assert_eq!(stats.total_pooled, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.reuse_count, 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        let buffer = pool.allocate::<half::f16>(&device, 1024).unwrap();
        assert_eq!(buffer.length, 1024);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.reuse_count, 0);
    }

    #[test]
    fn test_buffer_reuse() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        // Allocate and recycle
        let buffer1 = pool.allocate::<half::f16>(&device, 512).unwrap();
        pool.recycle(buffer1);

        // Allocate again - should reuse
        let buffer2 = pool.allocate::<half::f16>(&device, 512).unwrap();
        assert_eq!(buffer2.length, 512);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 1); // Only one allocation
        assert_eq!(stats.reuse_count, 1); // One reuse
    }

    #[test]
    fn test_buffer_pool_capacity() {
        let device = get_test_device();
        let pool = BufferPool::with_capacity(device.metal_device(), 2);

        // Add 3 buffers, only 2 should be kept
        let b1 = pool.allocate::<half::f16>(&device, 256).unwrap();
        let b2 = pool.allocate::<half::f16>(&device, 256).unwrap();
        let b3 = pool.allocate::<half::f16>(&device, 256).unwrap();

        pool.recycle(b1);
        pool.recycle(b2);
        let recycled = pool.recycle(b3);

        assert!(!recycled); // Third buffer should not be recycled

        let stats = pool.stats();
        assert_eq!(stats.total_pooled, 2); // Only 2 buffers in pool
    }

    #[test]
    fn test_pool_clear() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        let buffer = pool.allocate::<half::f16>(&device, 128).unwrap();
        pool.recycle(buffer);

        assert_eq!(pool.stats().total_pooled, 1);

        pool.clear();
        assert_eq!(pool.stats().total_pooled, 0);
    }

    #[test]
    fn test_allocate_zeros() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        let buffer = pool.allocate_zeros::<half::f16>(&device, 10).unwrap();
        let data = buffer.to_vec();

        assert_eq!(data.len(), 10);
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_shrink_to_fit() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        // Allocate and recycle separate buffers
        let buffers: Vec<_> = (0..5).map(|_| pool.allocate::<half::f16>(&device, 1024).unwrap()).collect();
        for buffer in buffers {
            pool.recycle(buffer);
        }

        let stats_before = pool.stats();
        assert_eq!(stats_before.total_pooled, 5);

        // Shrink to fit small memory limit
        let max_memory = 1024 * std::mem::size_of::<f16>() * 2; // Room for 2 buffers
        pool.shrink_to_fit(max_memory);

        let stats_after = pool.stats();
        assert!(stats_after.total_pooled <= 2);
        assert!(stats_after.total_memory <= max_memory);
    }
}
