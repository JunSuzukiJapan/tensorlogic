//! Command buffer management for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/commands.rs
//!
//! This module provides efficient batching of GPU operations by:
//! 1. Grouping multiple operations into a single CommandBuffer
//! 2. Deferring commit until batch size limit or explicit sync
//! 3. Managing thread-safe access to command buffers via semaphore

use crate::device::{CommandBuffer, CommandSemaphore, CommandStatus};
use crate::error::{TensorError, TensorResult};
use metal::{CommandQueue, MTLCommandBufferStatus};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, RwLock,
};

pub struct Commands {
    /// Single command queue for the entire device
    command_queue: Arc<CommandQueue>,

    /// One command buffer at a time
    /// Arc + RwLock for interior mutability and thread safety
    command_buffer: Arc<RwLock<CommandBuffer>>,

    /// Tracks the current amount of compute operations on the current command buffer
    /// AtomicUsize for thread-safe access
    compute_count: AtomicUsize,

    /// Maximum number of compute operations per command buffer
    /// Default: 200 (can be configured with TL_COMPUTE_PER_BUFFER env var)
    compute_per_buffer: usize,

    /// Semaphore for synchronizing command buffer access across threads
    semaphore: Arc<CommandSemaphore>,
}

// SAFETY: Commands uses Arc and atomic operations for thread safety
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    /// Create a new Commands manager
    pub fn new(command_queue: Arc<CommandQueue>) -> TensorResult<Self> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let command_buffer = Self::create_command_buffer(&command_queue, Arc::clone(&semaphore))?;
        let command_buffer = Arc::new(RwLock::new(command_buffer));

        // Read batch size from environment or use default
        // Candle default: 50, but we use 200 for better performance
        let compute_per_buffer = std::env::var("TL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(200);

        Ok(Self {
            command_queue,
            command_buffer,
            compute_count: AtomicUsize::new(0),
            compute_per_buffer,
            semaphore,
        })
    }

    /// Create a new command buffer
    fn create_command_buffer(
        command_queue: &CommandQueue,
        semaphore: Arc<CommandSemaphore>,
    ) -> TensorResult<CommandBuffer> {
        let raw = command_queue.new_command_buffer().to_owned();
        Ok(CommandBuffer::new(raw, semaphore))
    }

    /// Get the current command buffer
    ///
    /// If compute count > compute per buffer, commits current buffer and creates new one.
    /// This is the core batching mechanism from Candle.
    ///
    /// Returns (flushed, command_buffer_guard) where flushed indicates if we committed
    pub fn command_buffer(
        &self,
    ) -> TensorResult<(bool, std::sync::RwLockReadGuard<'_, CommandBuffer>)> {
        // Check if we need to flush (exceeded batch size)
        if self.compute_count.load(Ordering::Relaxed) > self.compute_per_buffer {
            // Acquire write lock to replace command buffer
            let mut command_buffer = self.command_buffer.write().map_err(|e| {
                TensorError::InvalidOperation(format!("RwLock poison: {}", e))
            })?;

            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!(
                    "[BATCH] Flushing at count {} (limit: {})",
                    self.compute_count.load(Ordering::Relaxed),
                    self.compute_per_buffer
                );
            }

            // Commit current buffer
            command_buffer.commit();

            // Create new buffer
            *command_buffer =
                Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;

            // Reset count to 1 (counting this operation)
            self.compute_count.store(1, Ordering::Relaxed);

            // Downgrade to read lock and return
            drop(command_buffer);
            Ok((true, self.command_buffer.read().map_err(|e| {
                TensorError::InvalidOperation(format!("RwLock poison: {}", e))
            })?))
        } else {
            // Increment count and return read lock
            self.compute_count.fetch_add(1, Ordering::Relaxed);
            Ok((false, self.command_buffer.read().map_err(|e| {
                TensorError::InvalidOperation(format!("RwLock poison: {}", e))
            })?))
        }
    }

    /// Get a command encoder with proper semaphore state management
    ///
    /// This is the main entry point for GPU operations.
    /// Sets CommandStatus to Encoding before creating encoder.
    /// Matches Candle's implementation exactly.
    ///
    /// Returns (flushed, encoder) where flushed indicates if a commit happened.
    pub fn command_encoder(
        &mut self,
    ) -> TensorResult<(bool, crate::device::ComputeCommandEncoder)> {
        {
            // Ensure command buffer available, set status to Encoding
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available | CommandStatus::Done));

            // Set status as encoding to block other threads
            *guard = CommandStatus::Encoding;
        }
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        // Get command buffer (may trigger flush)
        let (flushed, command_buffer) = self.command_buffer()?;

        // Create encoder directly without setting status again
        // (status already set above)
        let raw = command_buffer.inner().new_compute_command_encoder().to_owned();
        let command_encoder = crate::device::ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore));

        Ok((flushed, command_encoder))
    }

    /// Wait for all pending command buffers to complete
    ///
    /// This is called when we need results from the GPU:
    /// - Before reading tensor data
    /// - At end of operation sequence
    ///
    /// Matches Candle's implementation exactly
    pub fn wait_until_completed(&mut self) -> TensorResult<()> {
        let start = if std::env::var("TL_DEBUG_SYNC").is_ok() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        if std::env::var("TL_DEBUG_BATCHING").is_ok() {
            eprintln!(
                "[BATCH] wait_until_completed called at count {}",
                self.compute_count.load(Ordering::Relaxed)
            );
        }

        let current = {
            // Ensure command buffer not encoding
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available | CommandStatus::Done));

            // Extract current command buffer, create new in its place
            let current = {
                // Scope drops write lock
                let mut command_buffer = self.command_buffer.write().map_err(|e| {
                    TensorError::InvalidOperation(format!("RwLock poison: {}", e))
                })?;
                let current = command_buffer.clone();
                *command_buffer =
                    Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;
                // Reset compute count
                self.compute_count.store(0, Ordering::Relaxed);
                current
            };

            // After replacing the command buffer it is now safe to continue encoding
            *guard = CommandStatus::Available;
            current
        };
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        // Only commit and wait if needed (matches Candle exactly)
        match current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                current.commit();
                current.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                current.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {} // No action needed
            MTLCommandBufferStatus::Error => {
                return Err(TensorError::InvalidOperation(
                    "Command buffer error".to_string(),
                ));
            }
        }

        if let Some(start) = start {
            eprintln!("[SYNC] wait_until_completed took {:?}", start.elapsed());
        }

        Ok(())
    }
}
