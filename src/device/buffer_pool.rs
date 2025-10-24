//! Buffer pool for Metal buffer reuse and memory optimization

use crate::device::MetalBuffer;
use crate::error::TensorResult;
use half::f16;
use metal::{Buffer, Device as MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
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
    /// * `length` - Number of f16 elements
    pub fn allocate(&self, length: usize) -> TensorResult<MetalBuffer> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Try to reuse an existing buffer (pop removes the last/most recent one)
        if let Some(buffers) = pools.get_mut(&length) {
            if let Some((buffer, _last_used)) = buffers.pop() {
                stats.reuse_count += 1;

                // Periodically check and shrink (every 100 allocations)
                let should_check = stats.allocation_count % 100 == 0;
                drop(stats);
                drop(pools);

                if should_check {
                    self.check_and_shrink();
                }

                return Ok(MetalBuffer {
                    buffer,
                    length,
                });
            }
        }

        // Create a new buffer
        stats.allocation_count += 1;

        // Periodically check and shrink (every 100 allocations)
        let should_check = stats.allocation_count % 100 == 0;

        drop(stats); // Release stats lock before allocation

        let byte_length = length * std::mem::size_of::<f16>();
        let buffer = self.device.new_buffer(
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Periodically check and shrink after allocation
        if should_check {
            self.check_and_shrink();
        }

        Ok(MetalBuffer {
            buffer: Arc::new(buffer),
            length,
        })
    }

    /// Allocate a MetalBuffer filled with zeros
    pub fn allocate_zeros(&self, length: usize) -> TensorResult<MetalBuffer> {
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
    pub fn recycle(&self, buffer: MetalBuffer) -> bool {
        let mut pools = self.pools.lock().unwrap();

        let buffers = pools.entry(buffer.length).or_insert_with(Vec::new);

        // Only add if we haven't reached the limit
        if buffers.len() < self.max_buffers_per_size {
            // Record current time as last access time
            buffers.push((buffer.buffer, Instant::now()));
            true
        } else {
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
