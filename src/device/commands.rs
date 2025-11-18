//! Command buffer management for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/commands.rs
//!
//! This module provides efficient batching of GPU operations by:
//! 1. Grouping multiple operations into a single CommandBuffer
//! 2. Deferring commit until batch size limit or explicit sync
//! 3. Managing thread-safe access to command buffers via Mutex

use crate::device::CommandBuffer;
use crate::error::{TensorError, TensorResult};
use metal::{CommandQueue, MTLCommandBufferStatus};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Per-thread command buffer map
/// Matches Candle's implementation exactly
pub struct CommandBufferThreadMap {
    inner: HashMap<thread::ThreadId, CommandBuffer>,
}

impl CommandBufferThreadMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn get(&self) -> Option<&CommandBuffer> {
        self.inner.get(&thread::current().id())
    }

    pub fn get_mut(&mut self) -> Option<&mut CommandBuffer> {
        self.inner.get_mut(&thread::current().id())
    }

    pub fn insert(&mut self, command_buffer: CommandBuffer) -> Option<CommandBuffer> {
        self.inner.insert(thread::current().id(), command_buffer)
    }
}

pub struct Commands {
    /// Single command queue for the entire device
    command_queue: Arc<CommandQueue>,

    /// Per-thread command buffers
    /// Matches Candle's architecture: one buffer per thread
    command_buffers: Arc<Mutex<CommandBufferThreadMap>>,

    /// Keeps track of the current amount of compute command encoders
    /// Single counter for simplicity (matches Candle exactly)
    command_buffer_index: usize,

    /// Maximum number of compute operations per command buffer
    /// Default: 50 (matching Candle, can be configured with TL_COMPUTE_PER_BUFFER env var)
    compute_per_buffer: usize,
}

// SAFETY: Commands uses Arc and Mutex for thread safety
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    /// Create a new Commands manager
    pub fn new(command_queue: Arc<CommandQueue>) -> TensorResult<Self> {
        let command_buffer = Self::create_command_buffer(&command_queue)?;

        // Initialize per-thread buffer map with initial buffer
        let mut command_buffers = CommandBufferThreadMap::new();
        command_buffers.insert(command_buffer);
        let command_buffers = Arc::new(Mutex::new(command_buffers));

        // Fixed batch size matching Candle default
        let compute_per_buffer = 50;

        Ok(Self {
            command_queue,
            command_buffers,
            command_buffer_index: 0,
            compute_per_buffer,
        })
    }

    /// Create a new command buffer
    fn create_command_buffer(
        command_queue: &CommandQueue,
    ) -> TensorResult<CommandBuffer> {
        let raw = command_queue.new_command_buffer().to_owned();
        Ok(CommandBuffer::new(raw))
    }

    /// Get the current command buffer for the calling thread
    ///
    /// This matches Candle's implementation exactly:
    /// - Single counter (not per-thread)
    /// - Simple commit when limit reached
    /// - No in-flight tracking
    ///
    /// Returns (flushed, command_buffer)
    pub fn command_buffer(&mut self) -> TensorResult<(bool, CommandBuffer)> {
        let mut command_buffers = self.command_buffers.lock().map_err(|e| {
            TensorError::InvalidOperation(format!("Mutex poison: {}", e))
        })?;

        let command_buffer = match command_buffers.get_mut() {
            Some(command_buffer) => command_buffer,
            None => {
                let command_buffer = Self::create_command_buffer(&self.command_queue)?;
                command_buffers.insert(command_buffer);
                command_buffers.get_mut().unwrap()
            }
        };

        let mut flushed = false;
        if self.command_buffer_index > self.compute_per_buffer {
            // Commit current buffer
            command_buffer.commit();

            // Replace with new buffer
            *command_buffer = Self::create_command_buffer(&self.command_queue)?;
            self.command_buffer_index = 0;
            flushed = true;
        }

        self.command_buffer_index += 1;
        Ok((flushed, command_buffer.clone()))
    }

    /// Get a command encoder
    ///
    /// This is the main entry point for GPU operations.
    /// Matches Candle's implementation exactly.
    ///
    /// Returns (flushed, encoder) where flushed indicates if a commit happened.
    pub fn command_encoder(
        &mut self,
    ) -> TensorResult<(bool, metal::ComputeCommandEncoder)> {
        // Get command buffer (may trigger flush)
        let (flushed, command_buffer) = self.command_buffer()?;

        // Create encoder directly
        let encoder = command_buffer.compute_command_encoder();

        Ok((flushed, encoder))
    }

    /// Wait for all pending command buffers to complete
    ///
    /// This is called when we need results from the GPU.
    /// Matches Candle's simple implementation.
    pub fn wait_until_completed(&mut self) -> TensorResult<()> {
        // Extract current thread's command buffer, create new in its place
        let command_buffer = {
            let mut command_buffers = self.command_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;

            if let Some(command_buffer) = command_buffers.get_mut() {
                let current = command_buffer.clone();
                *command_buffer = Self::create_command_buffer(&self.command_queue)?;
                Some(current)
            } else {
                None
            }
        };

        // Only commit and wait if we have a buffer (matches Candle exactly)
        if let Some(command_buffer) = command_buffer {
            match command_buffer.status() {
                MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                }
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                    command_buffer.wait_until_completed();
                }
                MTLCommandBufferStatus::Completed => {} // No action needed
                MTLCommandBufferStatus::Error => {
                    return Err(TensorError::InvalidOperation(
                        "Command buffer error".to_string(),
                    ));
                }
            }
        } else {
            // No command buffer to wait for, create one for this thread
            let mut command_buffers = self.command_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            let command_buffer = Self::create_command_buffer(&self.command_queue)?;
            command_buffers.insert(command_buffer);
        }

        Ok(())
    }
}

impl Drop for Commands {
    /// Ensure all command buffers are properly completed
    fn drop(&mut self) {
        // Best effort to wait for completion
        let _ = self.wait_until_completed();
    }
}
