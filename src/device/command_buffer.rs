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

impl Drop for CommandBuffer {
    /// Ensure safe cleanup of GPU resources
    ///
    /// If the command buffer is in-flight (committed but not completed),
    /// we wait for completion to avoid orphaned GPU work and resource leaks.
    fn drop(&mut self) {
        match self.status() {
            MTLCommandBufferStatus::Committed
            | MTLCommandBufferStatus::Scheduled
            | MTLCommandBufferStatus::Enqueued => {
                // Command buffer is in-flight, must wait for completion
                // to ensure GPU resources are properly released
                if std::env::var("TL_DEBUG_HANG").is_ok() {
                    eprintln!(
                        "[DROP] CommandBuffer dropped while in-flight (status={:?}), waiting for completion",
                        self.status()
                    );
                }
                self.wait_until_completed();
                if std::env::var("TL_DEBUG_HANG").is_ok() {
                    eprintln!("[DROP] CommandBuffer completed during drop");
                }
            }
            MTLCommandBufferStatus::NotEnqueued => {
                // Not submitted yet, safe to drop without waiting
                if std::env::var("TL_DEBUG_HANG").is_ok() {
                    eprintln!("[DROP] CommandBuffer dropped before submission (safe)");
                }
            }
            MTLCommandBufferStatus::Completed => {
                // Already completed, no action needed
                if std::env::var("TL_DEBUG_HANG").is_ok() {
                    eprintln!("[DROP] CommandBuffer dropped after completion (safe)");
                }
            }
            MTLCommandBufferStatus::Error => {
                // Error state, don't wait as it may hang
                eprintln!("[WARN] CommandBuffer dropped in error state, skipping wait");
            }
        }
    }
}
