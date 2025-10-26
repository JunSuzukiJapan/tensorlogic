//! Buffer pool for Metal buffer reuse and memory optimization

use crate::device::MetalBuffer;
use crate::error::TensorResult;
use crate::tensor::FloatType;
use half::f16;
use metal::{Buffer, Device as MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
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
/// Currently optimized for f16 only. For f32, buffers are allocated directly without pooling.
pub struct BufferPool {
    /// Metal device for buffer creation
    device: Arc<MTLDevice>,

    /// Pool of buffers organized by size (in elements, not bytes)
    /// HashMap<size, Vec<(Arc<Buffer>, Instant)>>
    /// Each buffer is paired with its last access time
    pools: Arc<Mutex<HashMap<usize, Vec<(Arc<Buffer>, Instant)>>>>,

    /// Maximum number of buffers to keep per size class
    max_buffers_per_size: usize,

    /// Statistics
    stats: Arc<Mutex<PoolStats>>,
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

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: &MTLDevice) -> Self {
        Self {
            device: Arc::new(device.to_owned()),
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_buffers_per_size: 10,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Create a new buffer pool with custom max buffers per size
    pub fn with_capacity(device: &MTLDevice, max_buffers_per_size: usize) -> Self {
        Self {
            device: Arc::new(device.to_owned()),
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_buffers_per_size,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Allocate a MetalBuffer from the pool or create a new one
    ///
    /// # Arguments
    /// * `length` - Number of f16 elements (requested size)
    ///
    /// Uses size-class pooling to improve buffer reuse rates.
    /// The actual allocated buffer may be larger than requested.
    pub fn allocate(&self, length: usize) -> TensorResult<MetalBuffer<half::f16>> {
        let size_class = get_size_class(length);

        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Debug logging
        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!("[BufferPool::allocate] requested={}, size_class={}", length, size_class);
        }

        // Try to reuse an existing buffer from the size class
        if let Some(buffers) = pools.get_mut(&size_class) {
            if let Some((buffer, _last_used)) = buffers.pop() {
                stats.reuse_count += 1;

                // Periodically check and shrink (every 100 allocations)
                let should_check = stats.allocation_count % 100 == 0;
                drop(stats);
                drop(pools);

                if should_check {
                    self.check_and_shrink();
                }

                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!("[BufferPool::allocate] ✓ reused buffer from pool, size_class={}", size_class);
                }

                // CRITICAL FIX: Zero out the buffer to prevent stale data corruption
                // This fixes non-deterministic behavior where old computation results
                // would leak into new tensors
                unsafe {
                    let ptr = buffer.contents() as *mut f16;
                    std::ptr::write_bytes(ptr, 0, length);
                }

                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!("[BufferPool::allocate] ✓ zeroed reused buffer, length={}", length);
                }

                return Ok(MetalBuffer {
                    buffer,
                    length,  // Store requested length, not size_class
                    _phantom: PhantomData,
                });
            }
        }

        // Create a new buffer with size_class capacity
        stats.allocation_count += 1;

        // Periodically check and shrink (every 100 allocations)
        let should_check = stats.allocation_count % 100 == 0;

        drop(stats); // Release stats lock before allocation

        // Allocate buffer with size_class capacity (not exact requested length)
        let byte_length = size_class * std::mem::size_of::<f16>();
        let buffer = self.device.new_buffer(
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        if std::env::var("TL_BUFFER_DEBUG").is_ok() {
            eprintln!("[BufferPool::allocate] ✗ new allocation, size_class={}, bytes={}",
                     size_class, byte_length);
        }

        // Periodically check and shrink after allocation
        if should_check {
            self.check_and_shrink();
        }

        Ok(MetalBuffer {
            buffer: Arc::new(buffer),
            length,  // Store requested length, not size_class
            _phantom: PhantomData,
        })
    }

    /// Allocate a MetalBuffer filled with zeros
    pub fn allocate_zeros(&self, length: usize) -> TensorResult<MetalBuffer<half::f16>> {
        let buffer = self.allocate(length)?;

        // Zero out the buffer
        unsafe {
            let ptr = buffer.buffer.contents() as *mut f16;
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
            buffers.push((buffer.buffer, Instant::now()));

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

    /// Get the Metal device
    pub fn device(&self) -> &MTLDevice {
        &self.device
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
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            pools: self.pools.clone(),
            max_buffers_per_size: self.max_buffers_per_size,
            stats: self.stats.clone(),
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

        let buffer = pool.allocate(1024).unwrap();
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
        let buffer1 = pool.allocate(512).unwrap();
        pool.recycle(buffer1);

        // Allocate again - should reuse
        let buffer2 = pool.allocate(512).unwrap();
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
        let b1 = pool.allocate(256).unwrap();
        let b2 = pool.allocate(256).unwrap();
        let b3 = pool.allocate(256).unwrap();

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

        let buffer = pool.allocate(128).unwrap();
        pool.recycle(buffer);

        assert_eq!(pool.stats().total_pooled, 1);

        pool.clear();
        assert_eq!(pool.stats().total_pooled, 0);
    }

    #[test]
    fn test_allocate_zeros() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        let buffer = pool.allocate_zeros(10).unwrap();
        let data = buffer.to_vec();

        assert_eq!(data.len(), 10);
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_shrink_to_fit() {
        let device = get_test_device();
        let pool = BufferPool::new(device.metal_device());

        // Allocate and recycle separate buffers
        let buffers: Vec<_> = (0..5).map(|_| pool.allocate(1024).unwrap()).collect();
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
