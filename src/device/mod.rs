//! Device management for Metal and Neural Engine

mod metal_device;
mod metal_buffer;
mod kernel_executor;
mod neural_engine_buffer;
mod neural_engine_ops;
mod shared_buffer;
mod buffer_pool;
mod coreml_manager;
mod command_buffer;
mod commands;
mod compute_encoder;
mod encoder_provider;

pub mod metal;

pub use metal_device::MetalDevice;
pub use metal_buffer::MetalBuffer;
pub use kernel_executor::{KernelExecutor, get_kernel_executor};
pub use neural_engine_buffer::NeuralEngineBuffer;
pub use neural_engine_ops::NeuralEngineOps;
pub use shared_buffer::SharedBuffer;
pub use buffer_pool::{BufferPool, PoolStats};
pub use coreml_manager::CoreMLModelManager;
pub use command_buffer::{CommandBuffer, CommandSemaphore, CommandStatus};
pub use commands::Commands;
pub use compute_encoder::ComputeCommandEncoder;
pub use encoder_provider::EncoderProvider;
pub use metal::TensorBuffer;

use crate::error::TensorResult;

/// Compute device types
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    /// Metal GPU device
    Metal(MetalDevice),
    /// Neural Engine (CoreML)
    NeuralEngine,
    /// CPU (control flow only - avoid if possible)
    CPU,
}

impl Device {
    /// Get the default Metal device
    pub fn default_metal() -> TensorResult<Self> {
        Ok(Device::Metal(MetalDevice::new()?))
    }

    /// Check if Neural Engine is available
    pub fn neural_engine_available() -> bool {
        // TODO: Implement CoreML availability check
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }
}
