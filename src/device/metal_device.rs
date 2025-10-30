//! Metal device management

use crate::device::{BufferPool, Commands};
use crate::error::{TensorError, TensorResult};
use metal::{Device as MTLDevice, CommandQueue, Library};
use std::sync::{Arc, Mutex, OnceLock};

/// Global Metal device instance (singleton)
static GLOBAL_METAL_DEVICE: OnceLock<Arc<Mutex<MetalDevice>>> = OnceLock::new();

/// Metal GPU device wrapper
#[derive(Clone)]
pub struct MetalDevice {
    device: Arc<MTLDevice>,
    command_queue: Arc<CommandQueue>,
    library: Option<Arc<Library>>,
    buffer_pool: BufferPool,
    /// Command buffer manager for efficient batching
    commands: Arc<Mutex<Commands>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("device", &self.device.name())
            .field("command_queue", &"CommandQueue")
            .field("library", &self.library.is_some())
            .finish()
    }
}

impl MetalDevice {
    /// Get or create the global Metal device (singleton)
    /// All MetalDevice::new() calls return a clone of the same underlying device
    pub fn new() -> TensorResult<Self> {
        let device_arc = GLOBAL_METAL_DEVICE.get_or_init(|| {
            let device = Self::create_device().expect("Failed to initialize global Metal device");
            Arc::new(Mutex::new(device))
        });

        let device = device_arc.lock().unwrap().clone();
        Ok(device)
    }

    /// Create a new Metal device instance (internal use only)
    fn create_device() -> TensorResult<Self> {
        let device = MTLDevice::system_default()
            .ok_or_else(|| TensorError::MetalError("No Metal device found".to_string()))?;

        let command_queue = device.new_command_queue();
        let command_queue = Arc::new(command_queue);

        // Increase buffer pool capacity for deep models (22+ layers)
        let buffer_pool = BufferPool::with_capacity(&device, 100);

        // Create Commands manager for efficient batching
        let commands = Commands::new(command_queue.clone())?;

        let mut metal_device = Self {
            device: Arc::new(device),
            command_queue,
            library: None,
            buffer_pool,
            commands: Arc::new(Mutex::new(commands)),
        };

        // Load unified shader library once during initialization
        let shader_source = include_str!("../../shaders/unified.metal");
        metal_device.load_library(shader_source)?;

        Ok(metal_device)
    }

    /// Create Metal device with specific device
    pub fn with_device(device: MTLDevice) -> TensorResult<Self> {
        let command_queue = device.new_command_queue();
        let command_queue = Arc::new(command_queue);

        // Increase buffer pool capacity for deep models (22+ layers)
        let buffer_pool = BufferPool::with_capacity(&device, 100);

        // Create Commands manager for efficient batching
        let commands = Commands::new(command_queue.clone())?;

        Ok(Self {
            device: Arc::new(device),
            command_queue,
            library: None,
            buffer_pool,
            commands: Arc::new(Mutex::new(commands)),
        })
    }

    /// Get the buffer pool for efficient buffer allocation
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Get buffer pool statistics
    pub fn buffer_pool_stats(&self) -> crate::device::buffer_pool::PoolStats {
        self.buffer_pool.stats()
    }

    /// Print buffer pool statistics to stdout
    pub fn print_buffer_pool_stats(&self, label: &str) {
        let stats = self.buffer_pool.stats();
        println!("\n=== Buffer Pool Stats: {} ===", label);
        println!("  Pooled buffers: {}", stats.total_pooled);
        println!("  Size classes: {}", stats.size_classes);
        println!("  Total memory: {} MB", stats.total_memory / 1_048_576);
        println!("  Allocations: {}", stats.allocation_count);
        println!("  Reuses: {}", stats.reuse_count);
        println!("  Evictions: {}", stats.eviction_count);

        let total_ops = stats.allocation_count + stats.reuse_count;
        if total_ops > 0 {
            let reuse_rate = (stats.reuse_count as f64 / total_ops as f64) * 100.0;
            println!("  Reuse rate: {:.1}%", reuse_rate);
        }
        println!("================================\n");
    }

    /// Print current buffer pool statistics (to stderr for monitoring)
    pub fn print_current_buffer_stats(&self, label: &str) {
        self.buffer_pool.print_current_stats(label);
    }

    /// Check and shrink buffer pool if memory limit exceeded
    pub fn check_buffer_pool_memory(&self) -> bool {
        self.buffer_pool.check_and_shrink()
    }

    /// Get the underlying Metal device
    pub fn metal_device(&self) -> &MTLDevice {
        &self.device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get the next command buffer from the batch
    ///
    /// This is the main entry point for GPU operations.
    /// Returns (flushed, command_buffer) where flushed indicates if a commit happened.
    pub fn command_buffer(&self) -> TensorResult<(bool, crate::device::CommandBuffer)> {
        self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?
            .command_buffer()
    }

    /// Wait for all GPU operations to complete
    ///
    /// This should be called:
    /// - Before reading tensor data from GPU
    /// - At end of operation sequence
    /// - Before deallocating buffers that might be in use
    pub fn wait_until_completed(&self) -> TensorResult<()> {
        // Wait for all Commands-managed buffers
        self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?
            .wait_until_completed()
    }

    /// Load Metal shader library from source
    pub fn load_library(&mut self, source: &str) -> TensorResult<()> {
        let library = self
            .device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| TensorError::MetalError(format!("Failed to compile shaders: {}", e)))?;

        self.library = Some(Arc::new(library));
        Ok(())
    }

    /// Load Metal shader library from default metallib
    pub fn load_default_library(&mut self) -> TensorResult<()> {
        let library = self.device.new_default_library();
        self.library = Some(Arc::new(library));
        Ok(())
    }

    /// Get the shader library
    pub fn library(&self) -> Option<&Library> {
        self.library.as_ref().map(|l| l.as_ref())
    }

    /// Get device name
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Check if device supports f16
    pub fn supports_f16(&self) -> bool {
        // All modern Apple GPUs support f16
        true
    }

    /// Get maximum buffer length
    pub fn max_buffer_length(&self) -> u64 {
        self.device.max_buffer_length()
    }
}

impl PartialEq for MetalDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.device, &other.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_metal_device() {
        let device = MetalDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_device_supports_f16() {
        let device = MetalDevice::new().unwrap();
        assert!(device.supports_f16());
    }

    #[test]
    fn test_device_name() {
        let device = MetalDevice::new().unwrap();
        let name = device.name();
        assert!(!name.is_empty());
        println!("Metal device: {}", name);
    }
}
