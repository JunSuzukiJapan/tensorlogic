//! Metal device management

use crate::device::Commands;
use crate::error::{TensorError, TensorResult};
use metal::{Device as MTLDevice, CommandQueue, Library};
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};

/// Global Metal device instance (singleton)
/// This prevents multiple GPU device initialization within the same process
///
/// ⚠️  CRITICAL: Do NOT run multiple TensorLogic processes simultaneously!
/// Running multiple processes will cause GPU conflicts and system crashes.
/// The singleton pattern protects against in-process conflicts only.
static GLOBAL_METAL_DEVICE: OnceLock<Arc<Mutex<MetalDevice>>> = OnceLock::new();

/// Metal GPU device wrapper
#[derive(Clone)]
pub struct MetalDevice {
    device: Arc<MTLDevice>,
    command_queue: Arc<CommandQueue>,
    library: Option<Arc<Library>>,
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

        // Create Commands manager for efficient batching
        let commands = Commands::new(command_queue.clone())?;

        let mut metal_device = Self {
            device: Arc::new(device),
            command_queue,
            library: None,
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

        // Create Commands manager for efficient batching
        let commands = Commands::new(command_queue.clone())?;

        Ok(Self {
            device: Arc::new(device),
            command_queue,
            library: None,
            commands: Arc::new(Mutex::new(commands)),
        })
    }


    /// Check actual Metal GPU memory availability before allocation
    ///
    /// Panics if insufficient GPU memory to prevent system hangs.
    /// This checks the real Metal device memory using `recommended_max_working_set_size()`
    /// and `current_allocated_size()`.
    ///
    /// # Arguments
    /// * `allocation_size` - Size in bytes to be allocated
    ///
    /// # Panics
    /// Panics if available memory < allocation_size + 100 MB safety margin
    ///
    /// # Example
    /// ```ignore
    /// let size_bytes = length * std::mem::size_of::<f16>();
    /// device.check_gpu_memory(size_bytes as u64); // Panics if insufficient
    /// ```
    pub fn check_gpu_memory(&self, allocation_size: u64) {
        // Get Metal device memory info
        let recommended_max = self.device.recommended_max_working_set_size();
        let current_allocated = self.device.current_allocated_size();

        // Calculate available memory
        let available = recommended_max.saturating_sub(current_allocated);

        // Safety margin: 100 MB extra beyond allocation size
        const SAFETY_MARGIN_MB: u64 = 100;
        const SAFETY_MARGIN_BYTES: u64 = SAFETY_MARGIN_MB * 1_048_576;

        let required = allocation_size.saturating_add(SAFETY_MARGIN_BYTES);

        if available < required {
            // Also try to shrink buffer pool before panicking
            crate::device::MetalBuffer::<half::f16>::shrink_pool();

            // Re-check after shrinking
            let current_allocated_after = self.device.current_allocated_size();
            let available_after = recommended_max.saturating_sub(current_allocated_after);

            if available_after < required {
                panic!(
                    "GPU memory exhausted!\n\
                     Requested: {:.2} MB + {:.2} MB margin = {:.2} MB\n\
                     Available: {:.2} MB\n\
                     Current allocated: {:.2} MB\n\
                     Max recommended: {:.2} MB\n\
                     Deficit: {:.2} MB\n\
                     \n\
                     System will hang if allocation proceeds. Terminating to prevent freeze.",
                    allocation_size as f64 / 1_048_576.0,
                    SAFETY_MARGIN_MB,
                    required as f64 / 1_048_576.0,
                    available_after as f64 / 1_048_576.0,
                    current_allocated_after as f64 / 1_048_576.0,
                    recommended_max as f64 / 1_048_576.0,
                    (required.saturating_sub(available_after)) as f64 / 1_048_576.0
                );
            }
        }
    }

    /// Get the underlying Metal device
    pub fn metal_device(&self) -> &MTLDevice {
        &self.device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Get a command encoder (Candle-style: simple, no semaphore)
    ///
    /// This is the main entry point for GPU operations.
    /// Returns (flushed, encoder) where flushed indicates if a commit happened.
    pub fn command_encoder(&self) -> TensorResult<(bool, metal::ComputeCommandEncoder)> {
        let mut commands = self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?;

        let (flushed, command_encoder) = commands.command_encoder()?;

        Ok((flushed, command_encoder))
    }

    /// Get the next command buffer from the batch
    ///
    /// ⚠️ DEPRECATED: Use command_encoder() instead for proper semaphore state management
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

        let (flushed, buffer) = commands.command_buffer()?;
        // buffer is already CommandBuffer (not a guard anymore)

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] MetalDevice::command_buffer: Commands::command_buffer returned, releasing lock...");
            std::io::stderr().flush().ok();
        }

        Ok((flushed, buffer))
        // commands lock is released here
    }

    /// Wait for all GPU operations to complete
    ///
    /// This should be called:
    /// - Before reading tensor data from GPU
    /// - At end of operation sequence
    /// - Before deallocating buffers that might be in use
    pub fn wait_until_completed(&self) -> TensorResult<()> {
        // Wait for all Commands-managed buffers (per-thread)
        let mut commands = self.commands.lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Commands lock failed: {}", e)))?;
        commands.wait_until_completed()
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

    /// Force purge all buffers in the buffer pool
    ///
    /// This sets all pooled buffers to Empty purgeable state, forcing Metal to
    /// release GPU memory immediately. Should only be called when a memory leak
    /// is detected at program end.
    pub fn purge_all_buffers(&self) {
        crate::device::MetalBuffer::<half::f16>::purge_all_buffers();
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

    /// Create a new tensor buffer with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Buffer capacity in bytes
    ///
    /// # Returns
    /// A TensorBuffer that can be used to allocate tensors without allocation overhead
    ///
    /// # Example
    /// ```
    /// let device = MetalDevice::new()?;
    /// let buf = device.new_tensor_buffer(100 * 1024 * 1024); // 100MB buffer
    /// ```
    pub fn new_tensor_buffer(&self, capacity: usize) -> crate::device::metal::TensorBuffer {
        crate::device::metal::TensorBuffer::new(self.clone(), capacity)
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
