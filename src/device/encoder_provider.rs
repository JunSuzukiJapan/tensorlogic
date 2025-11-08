//! EncoderProvider trait - Candle-style abstraction for command buffer management
//!
//! This trait abstracts away command buffer lifecycle management from GPU operations,
//! allowing kernel functions to receive encoders without managing command buffers directly.
//!
//! Updated to use semaphore-integrated ComputeCommandEncoder for proper state management.

use super::command_buffer::CommandBuffer;
use super::compute_encoder::ComputeCommandEncoder;

/// Trait for types that can provide compute command encoders with semaphore integration
///
/// This abstraction allows kernel functions to work with command buffers
/// without managing them directly, while ensuring proper semaphore state management.
pub trait EncoderProvider {
    /// Get a compute command encoder with semaphore integration
    ///
    /// For CommandBuffer: creates a new encoder that:
    /// - Integrates with the command buffer's semaphore
    /// - Automatically resets status to Available when dropped
    /// - Ensures proper thread synchronization
    fn encoder(&self) -> ComputeCommandEncoder;
}

/// Implementation for CommandBuffer - creates semaphore-integrated encoder
///
/// This is the primary implementation used throughout the codebase.
/// When a CommandBuffer is passed to a kernel function, calling encoder()
/// creates a new compute command encoder with full semaphore integration.
impl EncoderProvider for CommandBuffer {
    fn encoder(&self) -> ComputeCommandEncoder {
        self.compute_command_encoder()
    }
}

/// Implementation for reference to CommandBuffer
impl EncoderProvider for &CommandBuffer {
    fn encoder(&self) -> ComputeCommandEncoder {
        self.compute_command_encoder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_command_buffer_provides_encoder() {
        let device = MetalDevice::new().unwrap();
        let (_flushed, command_buffer) = device.command_buffer().unwrap();

        // Should be able to get encoder from command buffer
        let encoder = command_buffer.encoder();

        // Encoder should be valid (can set label)
        encoder.as_ref().set_label("test_encoder");
        encoder.end_encoding();
    }
}
