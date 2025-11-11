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

    /// Per-thread indices tracking compute operations count
    /// Each thread has its own index for batching
    per_thread_indices: Arc<Mutex<HashMap<thread::ThreadId, usize>>>,

    /// Maximum number of compute operations per command buffer
    /// Default: 50 (matching Candle, can be configured with TL_COMPUTE_PER_BUFFER env var)
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

        // Initialize per-thread buffer map with initial buffer
        let mut command_buffers = CommandBufferThreadMap::new();
        command_buffers.insert(command_buffer);
        let command_buffers = Arc::new(Mutex::new(command_buffers));

        // Initialize per-thread indices map
        let per_thread_indices = Arc::new(Mutex::new(HashMap::new()));

        // Read batch size from environment or use default
        // Candle default: 50 (matching their proven implementation)
        let compute_per_buffer = std::env::var("TL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(50);

        Ok(Self {
            command_queue,
            command_buffers,
            per_thread_indices,
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

    /// Get the current command buffer for the calling thread
    ///
    /// If compute count > compute per buffer, commits current buffer and creates new one.
    /// This is the core batching mechanism from Candle.
    /// Matches Candle's per-thread buffer architecture exactly.
    ///
    /// Returns (flushed, command_buffer) where flushed indicates if we committed
    pub fn get_or_flush_command_buffer(
        &mut self,
    ) -> TensorResult<(bool, CommandBuffer)> {
        let thread_id = thread::current().id();

        // Lock the per-thread buffer map
        let mut command_buffers = self.command_buffers.lock().map_err(|e| {
            TensorError::InvalidOperation(format!("Mutex poison: {}", e))
        })?;

        // Get or create buffer for current thread
        let command_buffer = match command_buffers.get_mut() {
            Some(command_buffer) => command_buffer,
            None => {
                // Create new buffer for this thread
                let command_buffer = Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;
                command_buffers.insert(command_buffer);
                command_buffers.get_mut().unwrap()
            }
        };

        // Get or initialize index for current thread
        let mut indices = self.per_thread_indices.lock().map_err(|e| {
            TensorError::InvalidOperation(format!("Mutex poison: {}", e))
        })?;
        let current_index = indices.entry(thread_id).or_insert(0);

        let mut flushed = false;
        if *current_index > self.compute_per_buffer {
            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!(
                    "[BATCH] Thread {:?} flushing at index {} (limit: {})",
                    thread_id,
                    *current_index,
                    self.compute_per_buffer
                );
            }

            // Commit current buffer (send to GPU)
            command_buffer.commit();

            // Replace with new buffer
            *command_buffer = Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;

            // Reset index for this thread
            *current_index = 0;
            flushed = true;
        }

        // Increment index for this thread
        *current_index += 1;

        // Return cloned buffer (so we don't hold the mutex locks)
        Ok((flushed, command_buffer.clone()))
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
        let (flushed, command_buffer) = self.get_or_flush_command_buffer()?;

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
    /// Matches Candle's implementation with per-thread buffers
    pub fn wait_until_completed(&mut self) -> TensorResult<()> {
        let start = if std::env::var("TL_DEBUG_SYNC").is_ok() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let thread_id = thread::current().id();

        if std::env::var("TL_DEBUG_BATCHING").is_ok() {
            let indices = self.per_thread_indices.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            let current_index = indices.get(&thread_id).unwrap_or(&0);
            eprintln!(
                "[BATCH] Thread {:?} wait_until_completed called at index {}",
                thread_id,
                current_index
            );
        }

        let current = {
            // Ensure command buffer not encoding
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available | CommandStatus::Done));

            // Extract current thread's command buffer, create new in its place
            let current = {
                let mut command_buffers = self.command_buffers.lock().map_err(|e| {
                    TensorError::InvalidOperation(format!("Mutex poison: {}", e))
                })?;

                if let Some(command_buffer) = command_buffers.get_mut() {
                    let current = command_buffer.clone();
                    *command_buffer =
                        Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;
                    Some(current)
                } else {
                    // No command buffer for this thread yet
                    None
                }
            };

            // After replacing the command buffer it is now safe to continue encoding
            *guard = CommandStatus::Available;
            current
        };
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        // Only commit and wait if we have a buffer (matches Candle exactly)
        if let Some(current) = current {
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

            // Reset this thread's index after creating new buffer
            let mut indices = self.per_thread_indices.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            indices.insert(thread_id, 0);
        } else {
            // No command buffer to wait for, create one for this thread
            let mut command_buffers = self.command_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            let command_buffer = Self::create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;
            command_buffers.insert(command_buffer);

            // Initialize index for this thread
            let mut indices = self.per_thread_indices.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            indices.insert(thread_id, 0);
        }

        if let Some(start) = start {
            eprintln!("[SYNC] wait_until_completed took {:?}", start.elapsed());
        }

        Ok(())
    }
}
