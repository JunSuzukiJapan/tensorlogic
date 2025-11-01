//! Command buffer management for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/commands.rs
//!
//! This module provides efficient batching of GPU operations by:
//! 1. Grouping multiple operations into a single CommandBuffer
//! 2. Deferring commit until necessary (batch is full or explicit sync)
//! 3. Managing per-thread CommandBuffers to avoid contention

use crate::device::{CommandBuffer, CommandBufferThreadMap};
use crate::error::{TensorError, TensorResult};
use metal::{CommandQueue, MTLCommandBufferStatus};
use std::sync::{Arc, Mutex};

pub struct Commands {
    /// Single command queue for the entire device
    command_queue: Arc<CommandQueue>,

    /// Thread-local command buffers
    /// Each thread gets its own CommandBuffer to avoid lock contention
    command_buffers: Arc<Mutex<CommandBufferThreadMap>>,

    /// Number of compute operations on the current command buffer
    command_buffer_index: usize,

    /// Maximum number of compute operations per command buffer
    /// Default: 50 (can be configured with TL_COMPUTE_PER_BUFFER env var)
    compute_per_buffer: usize,
}

// SAFETY: Commands uses Arc and Mutex internally for thread safety
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    /// Create a new Commands manager
    pub fn new(command_queue: Arc<CommandQueue>) -> TensorResult<Self> {
        // Create initial command buffer
        let raw_cb = command_queue.new_command_buffer().to_owned();
        let command_buffer = CommandBuffer::new(raw_cb);
        command_buffer.enqueue();

        let mut command_buffers = CommandBufferThreadMap::new();
        command_buffers.insert(command_buffer);

        // Read batch size from environment or use default
        // Increased from 50 to 500 to reduce command buffer commit overhead
        // This reduces flushes from ~20/token to ~2/token, saving 2-6ms/token
        let compute_per_buffer = std::env::var("TL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(500);

        Ok(Self {
            command_queue,
            command_buffers: Arc::new(Mutex::new(command_buffers)),
            command_buffer_index: 0,
            compute_per_buffer,
        })
    }

    /// Get the current command buffer for this thread
    ///
    /// This method:
    /// 1. Gets or creates a command buffer for the current thread
    /// 2. Checks if we've hit the batch size limit
    /// 3. If so, commits the old buffer and creates a new one
    ///
    /// Returns (flushed, command_buffer) where flushed indicates if we committed
    pub fn command_buffer(&mut self) -> TensorResult<(bool, CommandBuffer)> {
        let mut command_buffers = self
            .command_buffers
            .lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Mutex poison: {}", e)))?;

        // Get or create command buffer for this thread
        let command_buffer = match command_buffers.get_mut() {
            Some(cb) => cb,
            None => {
                let raw_cb = self.command_queue.new_command_buffer().to_owned();
                let cb = CommandBuffer::new(raw_cb);
                command_buffers.insert(cb);
                command_buffers.get_mut().unwrap()
            }
        };

        let mut flushed = false;

        // Check if we need to flush (exceeded batch size)
        if self.command_buffer_index > self.compute_per_buffer {
            command_buffer.commit();

            // Create new command buffer
            let raw_cb = self.command_queue.new_command_buffer().to_owned();
            let new_cb = CommandBuffer::new(raw_cb);
            *command_buffer = new_cb;

            self.command_buffer_index = 0;
            flushed = true;
        }

        self.command_buffer_index += 1;
        Ok((flushed, command_buffer.clone()))
    }

    /// Wait for all pending command buffers to complete
    ///
    /// This is called when we need results from the GPU:
    /// - Before reading tensor data
    /// - At end of operation sequence
    ///
    /// The method is smart about waiting:
    /// - Only commits if buffer hasn't been committed yet
    /// - Only waits if buffer hasn't completed yet
    pub fn wait_until_completed(&mut self) -> TensorResult<()> {
        // CRITICAL: Reset command buffer index when replacing buffer
        // This ensures we don't immediately hit the batch size limit after sync
        self.command_buffer_index = 0;

        // Get current command buffer and replace with new one
        let command_buffer = {
            let mut command_buffers = self
                .command_buffers
                .lock()
                .map_err(|e| TensorError::InvalidOperation(format!("Mutex poison: {}", e)))?;

            if let Some(cb) = command_buffers.get_mut() {
                let current = cb.clone();

                // Replace with new buffer
                let raw_cb = self.command_queue.new_command_buffer().to_owned();
                let new_cb = CommandBuffer::new(raw_cb);
                *cb = new_cb;

                Some(current)
            } else {
                None
            }
        };

        if let Some(cb) = command_buffer {
            // Handle different states of the command buffer
            let status = cb.status();
            match status {
                MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                    // Need to commit before waiting
                    cb.commit();
                    cb.wait_until_completed();
                }
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                    // Already committed, just wait
                    cb.wait_until_completed();
                }
                MTLCommandBufferStatus::Completed => {
                    // Already done, no action needed
                }
                MTLCommandBufferStatus::Error => {
                    return Err(TensorError::InvalidOperation(
                        "Command buffer error".to_string(),
                    ));
                }
            }
        } else {
            // No command buffer exists, create one
            let raw_cb = self.command_queue.new_command_buffer().to_owned();
            let cb = CommandBuffer::new(raw_cb);

            let mut command_buffers = self
                .command_buffers
                .lock()
                .map_err(|e| TensorError::InvalidOperation(format!("Mutex poison: {}", e)))?;
            command_buffers.insert(cb);
        }

        Ok(())
    }
}
