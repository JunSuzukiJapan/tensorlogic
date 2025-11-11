//! EncoderProvider trait - Candle-style abstraction for command buffer management
//!
//! This trait abstracts away command buffer lifecycle management from GPU operations,
//! allowing kernel functions to receive encoders without managing command buffers directly.
//!
//! Matches Candle's simple implementation (no semaphore).

use super::command_buffer::CommandBuffer;

/// Trait for types that can provide compute command encoders
///
/// This abstraction allows kernel functions to work with command buffers
/// without managing them directly.
pub trait EncoderProvider {
    /// Get a compute command encoder (Candle-style: simple, no semaphore)
    fn encoder(&self) -> metal::ComputeCommandEncoder;
}

/// Implementation for CommandBuffer - creates simple encoder
///
/// This is the primary implementation used throughout the codebase.
/// When a CommandBuffer is passed to a kernel function, calling encoder()
/// creates a new compute command encoder.
impl EncoderProvider for CommandBuffer {
    fn encoder(&self) -> metal::ComputeCommandEncoder {
        self.compute_command_encoder()
    }
}

/// Implementation for reference to CommandBuffer
impl EncoderProvider for &CommandBuffer {
    fn encoder(&self) -> metal::ComputeCommandEncoder {
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
        encoder.set_label("test_encoder");
        encoder.end_encoding();
    }
}
