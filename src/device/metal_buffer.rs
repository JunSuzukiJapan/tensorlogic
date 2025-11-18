//! Metal buffer management for generic floating-point data

use crate::device::{MetalDevice, NeuralEngineBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::FloatType;
use half::f16;
use metal::{Buffer, Device as MTLDevice, NSRange};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Metal buffer wrapper for generic floating-point data
#[derive(Clone)]
pub struct MetalBuffer<T: FloatType> {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) length: usize, // number of T elements
    pub(crate) _phantom: PhantomData<T>,
    /// Reference to MetalDevice for GPU synchronization
    pub(crate) device: MetalDevice,
}

impl<T: FloatType> std::fmt::Debug for MetalBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("length", &self.length)
            .field("device", &self.device)
            .finish()
    }
}

impl<T: FloatType> MetalBuffer<T> {
    /// Create a new Metal buffer from slice
    pub fn from_slice(device: &MetalDevice, data: &[T]) -> TensorResult<Self> {
        let byte_length = data.len() * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        let buffer = device.metal_device().new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length: data.len(),
            _phantom: PhantomData,
            device: device.clone(),
        })
    }

    /// Create a new uninitialized Metal buffer
    pub fn new_uninit(device: &MetalDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        let buffer = device.metal_device().new_buffer(
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            length,
            _phantom: PhantomData,
            device: device.clone(),
        })
    }

    /// Create a new Metal buffer filled with zeros
    pub fn zeros(device: &MetalDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        let zeros = vec![T::zero(); length];
        Self::from_slice(device, &zeros)
    }

    /// Create a new Metal buffer filled with ones
    pub fn ones(device: &MetalDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        let ones = vec![T::one(); length];
        Self::from_slice(device, &ones)
    }

    // Pool methods support both f16 and f32, see impl<T: FloatType> MetalBuffer<T> below

    /// Get the buffer length (number of T elements)
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the buffer byte length
    pub fn byte_length(&self) -> usize {
        self.length * T::size_in_bytes()
    }

    /// Get the underlying Metal buffer
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Read data from buffer to Vec<T>
    ///
    /// CRITICAL: Automatically syncs GPU operations before reading (following Candle's pattern).
    /// This ensures all GPU commands are completed before CPU reads the buffer.
    pub fn to_vec(&self) -> Vec<T> {
        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] to_vec: START reading {} elements", self.length);
        }

        // CRITICAL: Wait for GPU operations to complete before reading (Candle pattern)
        // This fixes the sync bug where to_vec() was reading uninitialized (zero) data
        if std::env::var("TL_DEBUG_SYNC").is_ok() {
            let start = std::time::Instant::now();
            self.device.wait_until_completed().expect("GPU sync failed in MetalBuffer::to_vec");
            eprintln!("[SYNC] to_vec: wait={:?}, length={}", start.elapsed(), self.length);
        } else {
            self.device.wait_until_completed().expect("GPU sync failed in MetalBuffer::to_vec");
        }

        let ptr = self.buffer.contents() as *const T;
        let result = unsafe { std::slice::from_raw_parts(ptr, self.length).to_vec() };
        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[HANG] to_vec: DONE");
        }
        result
    }

    /// Write data to buffer from slice
    ///
    /// For SharedMode buffers, notifies GPU of CPU modifications via didModifyRange.
    /// This is required when writing directly to buffer memory (following Candle's pattern).
    pub fn write_from_slice(&mut self, data: &[T]) -> TensorResult<()> {
        if data.len() != self.length {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.length],
                actual: vec![data.len()],
            });
        }

        let ptr = self.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, self.length);
        }

        // Notify GPU of CPU modification (required for SharedMode buffers)
        // See: https://developer.apple.com/documentation/metal/mtlbuffer/1515396-didmodifyrange
        let byte_range = NSRange::new(0, self.byte_length() as u64);
        self.buffer.did_modify_range(byte_range);

        Ok(())
    }

    /// Get a mutable pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents_mut(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    /// Get a const pointer to the buffer contents
    ///
    /// # Safety
    /// The caller must ensure proper synchronization when accessing the buffer
    pub unsafe fn contents(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    /// Convert to Neural Engine buffer (with data copy, f16 only)
    ///
    /// Note: This performs a data copy. Zero-copy conversion will be implemented in Phase 5.
    pub fn to_neural_engine(&self, shape: &[usize]) -> TensorResult<NeuralEngineBuffer>
    where
        T: 'static,
    {
        // Validate shape matches buffer size
        let total_elements: usize = shape.iter().product();
        if total_elements != self.length {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.length],
                actual: vec![total_elements],
            });
        }

        // For f16, use direct conversion
        if T::is_f16() {
            let data = self.to_vec();
            // Safety: We checked T::is_f16(), so T = f16
            let f16_data: &[f16] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f16, data.len())
            };
            return NeuralEngineBuffer::from_f16_slice(f16_data, shape);
        }

        // For f32, convert to f16 first
        let data = self.to_vec();
        let f16_data: Vec<f16> = data.iter().map(|x| f16::from_f32(x.to_f32())).collect();
        NeuralEngineBuffer::from_f16_slice(&f16_data, shape)
    }
}

// Pool methods supporting both f16 and f32
impl<T: FloatType> MetalBuffer<T> {
    /// Create a new uninitialized Metal buffer from pool
    pub fn new_uninit_pooled(device: &crate::device::MetalDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        Self::allocate_from_pool(device, length)
    }

    /// Create a new Metal buffer filled with zeros from pool
    pub fn zeros_pooled(device: &crate::device::MetalDevice, length: usize) -> TensorResult<Self> {
        let byte_length = length * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        Self::allocate_zeros_from_pool(device, length)
    }

    /// Create a new Metal buffer from slice (same as from_slice but pooled)
    ///
    /// For now, this is identical to the original from_slice implementation
    /// to maintain compatibility while we transition to pooled buffers.
    pub fn from_vec_pooled(device: &crate::device::MetalDevice, data: &[T]) -> TensorResult<Self> {
        let byte_length = data.len() * T::size_in_bytes();

        // Check GPU memory before allocation to prevent hangs
        device.check_gpu_memory(byte_length as u64);

        // Note: For data initialization, we don't use pooling to avoid data corruption
        let buffer = device.metal_device().new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            buffer: std::sync::Arc::new(buffer),
            length: data.len(),
            _phantom: std::marker::PhantomData,
            device: device.clone(),
        })
    }
}

impl<T: FloatType> Drop for MetalBuffer<T> {
    fn drop(&mut self) {
        // GPU Memory Deallocation Logging
        if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            let before_dealloc = self.device.current_allocated_size();
            let ref_count = Arc::strong_count(&self.buffer);
            eprintln!("[GPU_MEMORY] Before deallocation: {:.2} MB - buffer_length={}, ref_count={}",
                     before_dealloc as f64 / 1_048_576.0,
                     self.length,
                     ref_count);
        }

        // Return all buffers to pool automatically
        // Calculate size_class based on buffer length
        let size_class = get_size_class(self.length);

        if std::env::var("TL_DEBUG").is_ok() {
            let ref_count = Arc::strong_count(&self.buffer);
            eprintln!(
                "[DEBUG_RS] MetalBuffer::drop: Returning buffer to pool (size_class={}, length={}, ref_count={})",
                size_class, self.length, ref_count
            );
        }
        // Pass reference to buffer - no clone here!
        // The global pool will clone it if it decides to store it
        Self::try_return_to_pool(&self.buffer, size_class, self.length);

        // GPU Memory Deallocation Logging (after pool return)
        if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            let after_pool_return = self.device.current_allocated_size();
            eprintln!("[GPU_MEMORY] After pool return: {:.2} MB - returned to pool, size_class={}",
                     after_pool_return as f64 / 1_048_576.0,
                     size_class);
        }
    }
}

impl<T: FloatType> PartialEq for MetalBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_create_from_slice() {
        let device = get_test_device();
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let buffer = MetalBuffer::<f16>::from_slice(&device, &data).unwrap();
        assert_eq!(buffer.len(), 3);

        let read_data = buffer.to_vec();
        assert_eq!(read_data.len(), 3);
        assert_eq!(read_data[0], f16::from_f32(1.0));
        assert_eq!(read_data[1], f16::from_f32(2.0));
        assert_eq!(read_data[2], f16::from_f32(3.0));
    }

    #[test]
    fn test_zeros() {
        let device = get_test_device();
        let buffer = MetalBuffer::<f16>::zeros(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ZERO));
    }

    #[test]
    fn test_ones() {
        let device = get_test_device();
        let buffer = MetalBuffer::<f16>::ones(&device, 5).unwrap();

        assert_eq!(buffer.len(), 5);

        let data = buffer.to_vec();
        assert!(data.iter().all(|&x| x == f16::ONE));
    }

    #[test]
    fn test_write_from_slice() {
        let device = get_test_device();
        let mut buffer = MetalBuffer::<f16>::zeros(&device, 3).unwrap();

        let new_data = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        buffer.write_from_slice(&new_data).unwrap();

        let read_data = buffer.to_vec();
        assert_eq!(read_data, new_data);
    }

    #[test]
    fn test_metal_to_neural_engine_conversion() {
        let device = get_test_device();
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let shape = vec![2, 2];

        // Create Metal buffer
        let metal_buffer = MetalBuffer::<f16>::from_slice(&device, &data).unwrap();

        // Convert to Neural Engine
        let ne_buffer = metal_buffer.to_neural_engine(&shape).unwrap();

        // Verify data
        assert_eq!(ne_buffer.shape(), shape);
        assert_eq!(ne_buffer.count(), 4);

        let ne_data = ne_buffer.to_f16_vec();
        assert_eq!(ne_data.len(), 4);
        assert_eq!(ne_data[0].to_f32(), 1.0);
        assert_eq!(ne_data[3].to_f32(), 4.0);
    }

    #[test]
    fn test_metal_neural_engine_roundtrip() {
        let device = get_test_device();
        let original_data = vec![
            f16::from_f32(1.5),
            f16::from_f32(2.5),
            f16::from_f32(3.5),
        ];
        let shape = vec![3];

        // Metal -> Neural Engine -> Metal
        let metal1 = MetalBuffer::<f16>::from_slice(&device, &original_data).unwrap();
        let ne_buffer = metal1.to_neural_engine(&shape).unwrap();
        let metal2 = ne_buffer.to_metal_buffer(&device).unwrap();

        // Verify roundtrip preserves data
        let result = metal2.to_vec();
        assert_eq!(result.len(), original_data.len());
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val.to_f32(), original_data[i].to_f32());
        }
    }
}

// ============================================================================
// Internal Pool Management (migrated from BufferPool)
// ============================================================================

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

/// Internal buffer pool implementation
struct InternalPool {
    /// Metal device for buffer creation
    device: Arc<MTLDevice>,
    /// Pool of buffers organized by size class
    /// HashMap<size_class, Vec<(Arc<Buffer>, Instant)>>
    pools: HashMap<usize, Vec<(Arc<Buffer>, Instant)>>,
    /// Maximum number of buffers to keep per size class
    max_buffers_per_size: usize,
    /// Statistics
    stats: PoolStats,
}

/// Get the size class for a given length (in elements)
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

impl InternalPool {
    fn new(device: &MTLDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
            pools: HashMap::new(),
            max_buffers_per_size: 30,
            stats: PoolStats::default(),
        }
    }

    fn get_stats(&self) -> PoolStats {
        let mut total_pooled = 0;
        let mut total_memory = 0;

        for (size, buffers) in self.pools.iter() {
            total_pooled += buffers.len();
            total_memory += size * std::mem::size_of::<f16>() * buffers.len();
        }

        PoolStats {
            total_pooled,
            size_classes: self.pools.len(),
            total_memory,
            reuse_count: self.stats.reuse_count,
            allocation_count: self.stats.allocation_count,
            eviction_count: self.stats.eviction_count,
        }
    }

    fn shrink_to_fit(&mut self, max_memory_bytes: usize) {
        let mut current_memory = 0;
        for (size, buffers) in self.pools.iter() {
            current_memory += size * std::mem::size_of::<f16>() * buffers.len();
        }

        if current_memory <= max_memory_bytes {
            return;
        }

        // Sort buffers by last access time within each size class
        let mut sizes: Vec<usize> = self.pools.keys().copied().collect();
        sizes.sort_by(|a, b| b.cmp(a)); // Descending order

        for size in sizes {
            if current_memory <= max_memory_bytes {
                break;
            }

            if let Some(buffers) = self.pools.get_mut(&size) {
                buffers.sort_by_key(|(_buf, last_used)| *last_used);

                while !buffers.is_empty() && current_memory > max_memory_bytes {
                    buffers.remove(0); // Remove oldest
                    current_memory -= size * std::mem::size_of::<f16>();
                    self.stats.eviction_count += 1;
                }
            }
        }
    }

    fn purge_all(&mut self) {
        use metal::MTLPurgeableState;

        let mut purged_count = 0;
        let mut purged_memory = 0usize;

        for (_size_class, buffers) in self.pools.iter_mut() {
            for (buffer, _timestamp) in buffers.iter() {
                let buffer_size = buffer.length() as usize;
                buffer.set_purgeable_state(MTLPurgeableState::Empty);
                purged_count += 1;
                purged_memory += buffer_size;
            }
        }

        self.pools.clear();

        if std::env::var("TL_DEBUG_MEMORY").is_ok() {
            eprintln!(
                "[GPU_MEMORY] Purged {} buffers ({:.2} MB) from buffer pool",
                purged_count,
                purged_memory as f64 / 1_048_576.0
            );
        }
    }
}

// ============================================================================
// Global Pool Management for MetalBuffer
// ============================================================================

/// Global buffer pool (singleton, shared across all MetalBuffer instances)
static GLOBAL_POOL: OnceLock<Mutex<InternalPool>> = OnceLock::new();

impl<T: FloatType> MetalBuffer<T> {
    /// Get or initialize the global buffer pool
    fn get_pool(device: &MetalDevice) -> &'static Mutex<InternalPool> {
        GLOBAL_POOL.get_or_init(|| {
            Mutex::new(InternalPool::new(device.metal_device()))
        })
    }

    /// Allocate from global pool with GPU memory check
    ///
    /// CRITICAL: Caller MUST call device.check_gpu_memory() before this method
    fn allocate_from_pool(device: &MetalDevice, length: usize) -> TensorResult<Self> {
        let size_class = get_size_class(length);
        let pool_mutex = Self::get_pool(device);
        let mut pool = pool_mutex.lock().unwrap();

        // Try to reuse from pool
        if let Some(buffers) = pool.pools.get_mut(&size_class) {
            if let Some((buffer, _last_used)) = buffers.pop() {
                pool.stats.reuse_count += 1;

                // Periodically check and shrink
                let should_check = pool.stats.allocation_count % 100 == 0;
                drop(pool);

                if should_check {
                    Self::shrink_pool();
                }

                return Ok(Self {
                    buffer,
                    length,
                    _phantom: PhantomData,
                    device: device.clone(),
                });
            }
        }

        // Create new buffer
        pool.stats.allocation_count += 1;

        let should_check = pool.stats.allocation_count % 100 == 0;
        drop(pool);

        let byte_length = size_class * T::size_in_bytes();
        let buffer = device.metal_device().new_buffer(
            byte_length as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        if should_check {
            Self::shrink_pool();
        }

        Ok(Self {
            buffer: Arc::new(buffer),
            length,
            _phantom: PhantomData,
            device: device.clone(),
        })
    }

    /// Allocate zeros from global pool with GPU memory check
    ///
    /// CRITICAL: Caller MUST call device.check_gpu_memory() before this method
    fn allocate_zeros_from_pool(device: &MetalDevice, length: usize) -> TensorResult<Self> {
        let buffer = Self::allocate_from_pool(device, length)?;

        // Zero out the buffer
        unsafe {
            let ptr = buffer.buffer.contents() as *mut T;
            std::ptr::write_bytes(ptr, 0, length);
        }

        Ok(buffer)
    }

    /// Try to return buffer to global pool (called from drop)
    fn try_return_to_pool(buffer: &Arc<Buffer>, size_class: usize, _length: usize) {
        if let Some(pool_mutex) = GLOBAL_POOL.get() {
            let mut pool = match pool_mutex.try_lock() {
                Ok(p) => p,
                Err(_) => return, // Pool locked, just drop the buffer
            };

            // Check ref count
            if Arc::strong_count(buffer) > 1 {
                return; // Other references exist, don't return to pool
            }

            let max_buffers = pool.max_buffers_per_size;
            let buffers = pool.pools.entry(size_class).or_insert_with(Vec::new);

            if buffers.len() < max_buffers {
                buffers.push((buffer.clone(), Instant::now()));
            }
        }
    }

    /// Get global pool statistics
    pub fn pool_stats() -> PoolStats {
        if let Some(pool_mutex) = GLOBAL_POOL.get() {
            let pool = pool_mutex.lock().unwrap();
            pool.get_stats()
        } else {
            PoolStats::default()
        }
    }

    /// Shrink global pool to fit within memory limit
    pub fn shrink_pool() -> bool {
        if let Some(pool_mutex) = GLOBAL_POOL.get() {
            let max_mb = std::env::var("TL_BUFFER_MAX_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(512);

            let max_bytes = max_mb * 1_048_576;

            let mut pool = pool_mutex.lock().unwrap();
            let current_stats = pool.get_stats();

            if current_stats.total_memory > max_bytes {
                if std::env::var("TL_BUFFER_DEBUG").is_ok() {
                    eprintln!(
                        "[BufferPool] Memory limit exceeded: {} MB / {} MB, shrinking...",
                        current_stats.total_memory / 1_048_576,
                        max_mb
                    );
                }
                pool.shrink_to_fit(max_bytes);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Purge all buffers from global pool
    pub fn purge_all_buffers() {
        if let Some(pool_mutex) = GLOBAL_POOL.get() {
            let mut pool = pool_mutex.lock().unwrap();
            pool.purge_all();
        }
    }
}
