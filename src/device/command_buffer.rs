//! Command buffer wrapper for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/command_buffer.rs

use metal::{CommandBuffer as MTLCommandBuffer, CommandBufferRef, MTLCommandBufferStatus};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Command buffer status for thread synchronization
#[derive(Clone, Debug, PartialEq)]
pub enum CommandStatus {
    /// Command buffer is available for encoding
    Available,
    /// Command buffer is currently being encoded
    Encoding,
    /// Command buffer encoding is done
    Done,
}

/// Semaphore for synchronizing command buffer access across threads
#[derive(Debug)]
pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> Self {
        Self {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    /// Wait until the condition is met
    pub fn wait_until<F>(&self, mut f: F) -> MutexGuard<'_, CommandStatus>
    where
        F: FnMut(&mut CommandStatus) -> bool,
    {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    /// Set the status and notify waiting threads
    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        self.cond.notify_one();
    }
}

/// Wrapper around Metal CommandBuffer with semaphore for thread safety
#[derive(Clone)]
pub struct CommandBuffer {
    raw: MTLCommandBuffer,
    semaphore: Arc<CommandSemaphore>,
}

impl CommandBuffer {
    pub fn new(raw: MTLCommandBuffer, semaphore: Arc<CommandSemaphore>) -> Self {
        Self { raw, semaphore }
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

    /// Get the semaphore
    pub fn semaphore(&self) -> &Arc<CommandSemaphore> {
        &self.semaphore
    }

    /// Create a compute command encoder with semaphore integration
    pub fn compute_command_encoder(&self) -> crate::device::ComputeCommandEncoder {
        // Set status to Encoding before creating encoder (Candle-compatible)
        {
            let mut guard = self.semaphore.wait_until(|s| matches!(s, CommandStatus::Available | CommandStatus::Done));
            *guard = CommandStatus::Encoding;
        }
        self.semaphore.cond.notify_one();

        let raw = self.raw.new_compute_command_encoder().to_owned();
        crate::device::ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore))
    }
}
