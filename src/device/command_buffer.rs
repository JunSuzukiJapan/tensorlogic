//! Command buffer wrapper for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/command_buffer.rs

use metal::{CommandBuffer as MTLCommandBuffer, CommandBufferRef, MTLCommandBufferStatus};

/// Wrapper around Metal CommandBuffer
/// Matches Candle's simple implementation exactly
#[derive(Clone)]
pub struct CommandBuffer {
    raw: MTLCommandBuffer,
}

impl CommandBuffer {
    pub fn new(raw: MTLCommandBuffer) -> Self {
        Self { raw }
    }

    pub fn as_ref(&self) -> &CommandBufferRef {
        &self.raw
    }

    pub fn commit(&self) {
        self.raw.commit();
    }

    pub fn enqueue(&self) {
        self.raw.enqueue();
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn wait_until_completed(&self) {
        self.raw.wait_until_completed();
    }

    /// Get the inner MTL command buffer reference
    pub fn inner(&self) -> &CommandBufferRef {
        &self.raw
    }

    /// Create a compute command encoder (Candle-style: simple, no semaphore)
    pub fn compute_command_encoder(&self) -> metal::ComputeCommandEncoder {
        self.raw.new_compute_command_encoder().to_owned()
    }
}
