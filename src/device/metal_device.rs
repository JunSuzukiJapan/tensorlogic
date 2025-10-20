//! Metal device management

use crate::device::BufferPool;
use crate::error::{TensorError, TensorResult};
use metal::{Device as MTLDevice, CommandQueue, Library};
use std::sync::Arc;

/// Metal GPU device wrapper
#[derive(Clone)]
pub struct MetalDevice {
    device: Arc<MTLDevice>,
    command_queue: Arc<CommandQueue>,
    library: Option<Arc<Library>>,
    buffer_pool: BufferPool,
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
    /// Create a new Metal device (uses default GPU)
    pub fn new() -> TensorResult<Self> {
        let device = MTLDevice::system_default()
            .ok_or_else(|| TensorError::MetalError("No Metal device found".to_string()))?;

        let command_queue = device.new_command_queue();
        let buffer_pool = BufferPool::new(&device);

        Ok(Self {
            device: Arc::new(device),
            command_queue: Arc::new(command_queue),
            library: None,
            buffer_pool,
        })
    }

    /// Create Metal device with specific device
    pub fn with_device(device: MTLDevice) -> TensorResult<Self> {
        let command_queue = device.new_command_queue();
        let buffer_pool = BufferPool::new(&device);

        Ok(Self {
            device: Arc::new(device),
            command_queue: Arc::new(command_queue),
            library: None,
            buffer_pool,
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

    /// Get the underlying Metal device
    pub fn metal_device(&self) -> &MTLDevice {
        &self.device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
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
