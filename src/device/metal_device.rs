//! Metal device management

use crate::device::{BufferPool, Commands};
use crate::error::{TensorError, TensorResult};
use metal::{Device as MTLDevice, CommandQueue, Library};
use std::io::Write;
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

        // Optimize buffer pool capacity for scope management
        // With proper scope cleanup, we only need buffers for active layers
        let buffer_pool = BufferPool::with_capacity(&device, 30);

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

        // Optimize buffer pool capacity for scope management
        // With proper scope cleanup, we only need buffers for active layers
        let buffer_pool = BufferPool::with_capacity(&device, 30);

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
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] MetalDevice::command_buffer: Attempting to lock self.commands...");
            std::io::stderr().flush().ok();
        }

        let mut commands = self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?;

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] MetalDevice::command_buffer: Lock acquired, calling Commands::command_buffer...");
            std::io::stderr().flush().ok();
        }

        let result = commands.command_buffer();

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] MetalDevice::command_buffer: Commands::command_buffer returned, releasing lock...");
            std::io::stderr().flush().ok();
        }

        result
        // commands lock is released here
    }

    /// Flush any pending GPU operations
    ///
    /// This ensures pending operations are committed to the command queue.
    /// Called before sync operations to prevent deadlock from unflushed encoders.
    pub fn flush_if_needed(&self) -> TensorResult<()> {
        self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?
            .flush_if_needed()
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

    /// Get recommended maximum working set size (available GPU memory)
    ///
    /// This returns the recommended maximum working set size in bytes.
    /// On Apple Silicon with unified memory, this is typically the system RAM.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }

    /// Get current allocated size (currently in use)
    ///
    /// This returns the current amount of GPU memory allocated in bytes.
    pub fn current_allocated_size(&self) -> u64 {
        self.device.current_allocated_size()
    }

    /// Check if device has unified memory architecture
    ///
    /// Returns true for Apple Silicon Macs where GPU and CPU share memory.
    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    /// Check if model size fits in available GPU memory
    ///
    /// Returns Ok(()) if model fits, Err with warning message if too large.
    ///
    /// # Arguments
    /// * `model_size_bytes` - Total size of model in bytes
    /// * `model_name` - Name of model for error message
    pub fn check_memory_available(&self, model_size_bytes: u64, model_name: &str) -> TensorResult<()> {
        let recommended_max = self.recommended_max_working_set_size();
        let current_allocated = self.current_allocated_size();
        let available = recommended_max.saturating_sub(current_allocated);

        // Add 20% safety margin for operations and overhead
        let required_with_margin = (model_size_bytes as f64 * 1.2) as u64;

        // Print memory info for debugging
        eprintln!("\n=== GPU Memory Check ===");
        eprintln!("  Model size: {:.2} GB", model_size_bytes as f64 / 1_000_000_000.0);
        eprintln!("  Required (with 20% margin): {:.2} GB", required_with_margin as f64 / 1_000_000_000.0);
        eprintln!("  Recommended max: {:.2} GB", recommended_max as f64 / 1_000_000_000.0);
        eprintln!("  Current allocated: {:.2} GB", current_allocated as f64 / 1_000_000_000.0);
        eprintln!("  Available: {:.2} GB", available as f64 / 1_000_000_000.0);
        eprintln!("========================\n");

        if required_with_margin > available {
            let model_gb = model_size_bytes as f64 / 1_000_000_000.0;
            let available_gb = available as f64 / 1_000_000_000.0;
            let required_gb = required_with_margin as f64 / 1_000_000_000.0;

            return Err(TensorError::InvalidOperation(
                format!(
                    "\n⚠️  GPU MEMORY WARNING ⚠️\n\
                     Model '{}' requires {:.2} GB (with 20% margin: {:.2} GB)\n\
                     Available GPU memory: {:.2} GB\n\
                     \n\
                     The model is too large to load safely.\n\
                     This check prevented your PC from crashing!\n\
                     \n\
                     Consider using:\n\
                     - A smaller model\n\
                     - Lower precision (f16 instead of f32)\n\
                     - Quantized model (Q4_0, Q6_K, etc.)\n",
                    model_name, model_gb, required_gb, available_gb
                )
            ));
        }

        eprintln!("✓ Memory check passed - safe to load model\n");
        Ok(())
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
