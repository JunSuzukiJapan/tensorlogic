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
use std::io::Write;
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
        // TEMPORARILY REDUCED from 500 to 50 to debug Layer 12 hang
        // This forces more frequent command buffer commits to prevent GPU resource exhaustion
        // Testing hypothesis: accumulated GPU state after 11 layers may be causing hang
        let compute_per_buffer = std::env::var("TL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(50);

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
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: Attempting to lock command_buffers...");
            std::io::stderr().flush().ok();
        }

        let mut command_buffers = self
            .command_buffers
            .lock()
            .map_err(|e| TensorError::InvalidOperation(format!("Mutex poison: {}", e)))?;

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: Lock acquired!");
            std::io::stderr().flush().ok();
        }

        // Get or create command buffer for this thread
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: About to get_mut()...");
            std::io::stderr().flush().ok();
        }

        let command_buffer = match command_buffers.get_mut() {
            Some(cb) => {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[DEBUG_RS] Commands::command_buffer: Reusing existing command buffer");
                    std::io::stderr().flush().ok();
                }
                cb
            }
            None => {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[DEBUG_RS] Commands::command_buffer: Creating NEW command buffer...");
                    std::io::stderr().flush().ok();
                }
                let raw_cb = self.command_queue.new_command_buffer().to_owned();
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[DEBUG_RS] Commands::command_buffer: new_command_buffer() returned");
                    std::io::stderr().flush().ok();
                }
                let cb = CommandBuffer::new(raw_cb);
                command_buffers.insert(cb);
                command_buffers.get_mut().unwrap()
            }
        };

        let mut flushed = false;

        // Check if we need to flush (exceeded batch size)
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: Checking batch size (index={}, limit={})",
                     self.command_buffer_index, self.compute_per_buffer);
            std::io::stderr().flush().ok();
        }

        if self.command_buffer_index > self.compute_per_buffer {
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] Commands::command_buffer: FLUSHING - calling commit()...");
                std::io::stderr().flush().ok();
            }
            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!("[BATCH] Flushing at index {} (limit: {})",
                         self.command_buffer_index, self.compute_per_buffer);
            }
            command_buffer.commit();
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] Commands::command_buffer: commit() returned");
                std::io::stderr().flush().ok();
            }

            // Create new command buffer
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] Commands::command_buffer: Creating replacement command buffer...");
                std::io::stderr().flush().ok();
            }
            let raw_cb = self.command_queue.new_command_buffer().to_owned();
            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!("[DEBUG_RS] Commands::command_buffer: new_command_buffer() returned (replacement)");
                std::io::stderr().flush().ok();
            }
            let new_cb = CommandBuffer::new(raw_cb);
            *command_buffer = new_cb;

            self.command_buffer_index = 0;
            flushed = true;
        }

        self.command_buffer_index += 1;

        if std::env::var("TL_DEBUG_BATCHING").is_ok() && self.command_buffer_index % 100 == 0 {
            eprintln!("[BATCH] Current index: {}/{}",
                     self.command_buffer_index, self.compute_per_buffer);
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: About to clone command_buffer...");
            std::io::stderr().flush().ok();
        }

        let result = command_buffer.clone();

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[DEBUG_RS] Commands::command_buffer: clone() returned, about to release lock (via drop)...");
            std::io::stderr().flush().ok();
        }

        Ok((flushed, result))
        // command_buffers lock is released here when it goes out of scope
    }

    /// Flush pending operations if there are any
    ///
    /// This ensures that any pending GPU operations are committed to the command queue.
    /// Called before sync operations to prevent deadlock from unflushed encoders.
    pub fn flush_if_needed(&mut self) -> TensorResult<()> {
        if self.command_buffer_index > 0 {
            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!("[BATCH] Flushing {} pending operations", self.command_buffer_index);
            }

            let mut command_buffers = self
                .command_buffers
                .lock()
                .map_err(|e| TensorError::InvalidOperation(format!("Mutex poison: {}", e)))?;

            if let Some(cb) = command_buffers.get_mut() {
                // Commit current buffer
                cb.commit();

                // Create new buffer
                let raw_cb = self.command_queue.new_command_buffer().to_owned();
                let new_cb = CommandBuffer::new(raw_cb);
                *cb = new_cb;

                // Reset index
                self.command_buffer_index = 0;
            }
        }
        Ok(())
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
        if std::env::var("TL_DEBUG_BATCHING").is_ok() {
            eprintln!("[BATCH] wait_until_completed called at index {}",
                     self.command_buffer_index);
        }

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
