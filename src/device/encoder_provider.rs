//! EncoderProvider trait - Candle-style abstraction for command buffer management
//!
//! This trait abstracts away command buffer lifecycle management from GPU operations,
//! allowing kernel functions to receive encoders without managing command buffers directly.

use super::command_buffer::CommandBuffer;
use metal::ComputeCommandEncoderRef;

/// Trait for types that can provide compute command encoders
///
/// This abstraction allows kernel functions to work with command buffers
/// without managing them directly.
///
/// Based on Candle's EncoderProvider pattern from:
/// `/tmp/candle/candle-metal-kernels/src/utils.rs:160-208`
pub trait EncoderProvider {
    /// Get a compute command encoder reference
    ///
    /// For CommandBuffer: creates a new encoder
    fn encoder(&self) -> &ComputeCommandEncoderRef;
}

/// Implementation for CommandBuffer - creates encoder on demand
///
/// This is the primary implementation used throughout the codebase.
/// When a CommandBuffer is passed to a kernel function, calling encoder()
/// creates a new compute command encoder from it.
impl EncoderProvider for CommandBuffer {
    fn encoder(&self) -> &ComputeCommandEncoderRef {
        self.inner().new_compute_command_encoder()
    }
}

/// Implementation for reference to CommandBuffer
impl EncoderProvider for &CommandBuffer {
    fn encoder(&self) -> &ComputeCommandEncoderRef {
        self.inner().new_compute_command_encoder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    fn test_command_buffer_provides_encoder() {
        let device = MetalDevice::new().unwrap();
        let command_buffer = device.command_buffer().unwrap();

        // Should be able to get encoder from command buffer
        let encoder = command_buffer.encoder();

        // Encoder should be valid (can set label)
        encoder.set_label("test_encoder");
        encoder.end_encoding();
    }
}
