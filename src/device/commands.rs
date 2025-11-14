//! Command buffer management for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/commands.rs
//!
//! This module provides efficient batching of GPU operations by:
//! 1. Grouping multiple operations into a single CommandBuffer
//! 2. Deferring commit until batch size limit or explicit sync
//! 3. Managing thread-safe access to command buffers via Mutex (NO semaphore!)

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

    /// Per-thread indices tracking compute operations count
    /// Each thread has its own index for batching
    per_thread_indices: Arc<Mutex<HashMap<thread::ThreadId, usize>>>,

    /// In-flight command buffers (committed but not yet completed)
    /// Key insight from Candle: track committed buffers without blocking
    /// This allows GPU to work in parallel while CPU prepares next operations
    in_flight_buffers: Arc<Mutex<Vec<CommandBuffer>>>,

    /// Maximum number of compute operations per command buffer
    /// Default: 50 (matching Candle, can be configured with TL_COMPUTE_PER_BUFFER env var)
    compute_per_buffer: usize,
}

// SAFETY: Commands uses Arc and Mutex for thread safety (NO semaphore!)
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

        // Initialize per-thread indices map
        let per_thread_indices = Arc::new(Mutex::new(HashMap::new()));

        // Read batch size from environment or use default
        // Candle default: 50 (matching their proven implementation)
        let compute_per_buffer = std::env::var("TL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(50);

        // Initialize in-flight buffers tracking
        let in_flight_buffers = Arc::new(Mutex::new(Vec::new()));

        Ok(Self {
            command_queue,
            command_buffers,
            per_thread_indices,
            in_flight_buffers,
            compute_per_buffer,
        })
    }

    /// Create a new command buffer (Candle-style: no semaphore)
    fn create_command_buffer(
        command_queue: &CommandQueue,
    ) -> TensorResult<CommandBuffer> {
        let raw = command_queue.new_command_buffer().to_owned();
        Ok(CommandBuffer::new(raw))
    }

    /// Clean up completed buffers from the in-flight queue
    /// This is called periodically to prevent the queue from growing indefinitely
    /// Matches Candle's pattern of non-blocking cleanup
    fn cleanup_completed_buffers(&self) -> TensorResult<()> {
        let mut in_flight = self.in_flight_buffers.lock().map_err(|e| {
            TensorError::InvalidOperation(format!("Mutex poison: {}", e))
        })?;

        if std::env::var("TL_DEBUG_BATCHING").is_ok() {
            eprintln!("[BATCH] Cleanup: {} in-flight buffers before", in_flight.len());
        }

        // Remove completed buffers (retain only non-completed ones)
        in_flight.retain(|buf| {
            let status = buf.status();
            let keep = !matches!(status, MTLCommandBufferStatus::Completed);

            if std::env::var("TL_DEBUG_BATCHING").is_ok() && !keep {
                eprintln!("[BATCH] Cleanup: removing completed buffer");
            }

            keep
        });

        if std::env::var("TL_DEBUG_BATCHING").is_ok() {
            eprintln!("[BATCH] Cleanup: {} in-flight buffers after", in_flight.len());
        }

        Ok(())
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
                let command_buffer = Self::create_command_buffer(&self.command_queue)?;
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

            // Save the old buffer before replacing it
            let old_buffer = command_buffer.clone();

            // Commit old buffer (send to GPU, but don't wait)
            old_buffer.commit();

            // Add to in-flight queue for tracking (Candle's pattern)
            // This allows GPU to work in parallel while we prepare next operations
            let mut in_flight = self.in_flight_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;
            in_flight.push(old_buffer);

            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!("[BATCH] Added buffer to in-flight queue (total: {})", in_flight.len());
            }

            drop(in_flight); // Release lock before creating new buffer

            // Replace with new buffer
            *command_buffer = Self::create_command_buffer(&self.command_queue)?;

            // Reset index for this thread
            *current_index = 0;
            flushed = true;

            // Periodically clean up completed buffers to prevent queue growth
            self.cleanup_completed_buffers()?;
        }

        // Increment index for this thread
        *current_index += 1;

        // Return cloned buffer (so we don't hold the mutex locks)
        Ok((flushed, command_buffer.clone()))
    }

    /// Get a command encoder (Candle-style: simple, no semaphore)
    ///
    /// This is the main entry point for GPU operations.
    /// Matches Candle's implementation exactly.
    ///
    /// Returns (flushed, encoder) where flushed indicates if a commit happened.
    pub fn command_encoder(
        &mut self,
    ) -> TensorResult<(bool, metal::ComputeCommandEncoder)> {
        // Get command buffer (may trigger flush)
        let (flushed, command_buffer) = self.get_or_flush_command_buffer()?;

        // Create encoder directly (no semaphore, matches Candle)
        let encoder = command_buffer.compute_command_encoder();

        Ok((flushed, encoder))
    }

    /// Wait for all pending command buffers to complete
    ///
    /// This is called when we need results from the GPU:
    /// - Before reading tensor data
    /// - At end of operation sequence
    ///
    /// CRITICAL: Now also waits for ALL in-flight buffers, not just current thread's buffer
    /// This prevents reading GPU data before previous operations complete
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

        // FIRST: Wait for ALL in-flight buffers from previous flushes
        // This is the KEY FIX: ensure all previously committed buffers complete
        {
            let mut in_flight = self.in_flight_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;

            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!("[BATCH] Waiting for {} in-flight buffers", in_flight.len());
            }

            // Wait for each in-flight buffer to complete
            for buf in in_flight.iter() {
                match buf.status() {
                    MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                        buf.commit();
                        buf.wait_until_completed();
                    }
                    MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                        buf.wait_until_completed();
                    }
                    MTLCommandBufferStatus::Completed => {
                        // Already done
                    }
                    MTLCommandBufferStatus::Error => {
                        eprintln!("[WARN] In-flight buffer in error state");
                    }
                }
            }

            // Clear all in-flight buffers after waiting
            in_flight.clear();

            if std::env::var("TL_DEBUG_BATCHING").is_ok() {
                eprintln!("[BATCH] All in-flight buffers completed and cleared");
            }
        }

        // SECOND: Handle current thread's command buffer
        // Extract current thread's command buffer, create new in its place
        let current = {
            let mut command_buffers = self.command_buffers.lock().map_err(|e| {
                TensorError::InvalidOperation(format!("Mutex poison: {}", e))
            })?;

            if let Some(command_buffer) = command_buffers.get_mut() {
                let current = command_buffer.clone();
                *command_buffer = Self::create_command_buffer(&self.command_queue)?;
                Some(current)
            } else {
                // No command buffer for this thread yet
                None
            }
        };

        // Only commit and wait if we have a buffer (matches Candle exactly)
        if let Some(current) = current {
            if std::env::var("TL_DEBUG_HANG").is_ok() {
                eprintln!("[HANG] wait_until_completed: status={:?}", current.status());
            }

            match current.status() {
                MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Committing buffer...");
                    }
                    current.commit();
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Waiting for completion...");
                    }
                    current.wait_until_completed();
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Completed!");
                    }
                }
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Buffer already committed, waiting...");
                    }
                    current.wait_until_completed();
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Completed!");
                    }
                }
                MTLCommandBufferStatus::Completed => {
                    if std::env::var("TL_DEBUG_HANG").is_ok() {
                        eprintln!("[HANG] Already completed, no wait needed");
                    }
                } // No action needed
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
            let command_buffer = Self::create_command_buffer(&self.command_queue)?;
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

impl Drop for Commands {
    /// Ensure all command buffers from all threads are properly completed
    ///
    /// In a multi-threaded environment, multiple threads may have in-flight
    /// command buffers. This ensures they all complete before the Commands
    /// manager is destroyed, preventing GPU resource leaks.
    fn drop(&mut self) {
        if std::env::var("TL_DEBUG_HANG").is_ok() {
            eprintln!("[DROP] Commands: Starting cleanup of all buffers");
        }

        // FIRST: Wait for all in-flight buffers from previous flushes
        if let Ok(mut in_flight) = self.in_flight_buffers.lock() {
            if std::env::var("TL_DEBUG_HANG").is_ok() {
                eprintln!("[DROP] Commands: Waiting for {} in-flight buffers", in_flight.len());
            }

            for buf in in_flight.iter() {
                match buf.status() {
                    MTLCommandBufferStatus::Committed
                    | MTLCommandBufferStatus::Scheduled
                    | MTLCommandBufferStatus::Enqueued => {
                        buf.commit();
                        buf.wait_until_completed();
                    }
                    MTLCommandBufferStatus::NotEnqueued => {
                        buf.commit();
                        buf.wait_until_completed();
                    }
                    MTLCommandBufferStatus::Completed => {
                        // Already done
                    }
                    MTLCommandBufferStatus::Error => {
                        eprintln!("[WARN] Commands: In-flight buffer in error state");
                    }
                }
            }

            in_flight.clear();

            if std::env::var("TL_DEBUG_HANG").is_ok() {
                eprintln!("[DROP] Commands: All in-flight buffers completed");
            }
        }

        // SECOND: Lock the command buffers map and wait for per-thread buffers
        if let Ok(command_buffers) = self.command_buffers.lock() {
            let thread_ids: Vec<thread::ThreadId> =
                command_buffers.inner.keys().copied().collect();

            if std::env::var("TL_DEBUG_HANG").is_ok() {
                eprintln!(
                    "[DROP] Commands: Found {} thread(s) with command buffers",
                    thread_ids.len()
                );
            }

            // Process each thread's command buffer
            for thread_id in thread_ids {
                if let Some(cmd_buf) = command_buffers.inner.get(&thread_id) {
                    let status = cmd_buf.status();

                    match status {
                        MTLCommandBufferStatus::Committed
                        | MTLCommandBufferStatus::Scheduled
                        | MTLCommandBufferStatus::Enqueued => {
                            if std::env::var("TL_DEBUG_HANG").is_ok() {
                                eprintln!(
                                    "[DROP] Commands: Thread {:?} has in-flight buffer (status={:?}), waiting...",
                                    thread_id, status
                                );
                            }
                            cmd_buf.commit();
                            cmd_buf.wait_until_completed();

                            if std::env::var("TL_DEBUG_HANG").is_ok() {
                                eprintln!(
                                    "[DROP] Commands: Thread {:?} buffer completed",
                                    thread_id
                                );
                            }
                        }
                        MTLCommandBufferStatus::NotEnqueued => {
                            if std::env::var("TL_DEBUG_HANG").is_ok() {
                                eprintln!(
                                    "[DROP] Commands: Thread {:?} buffer not enqueued (safe)",
                                    thread_id
                                );
                            }
                        }
                        MTLCommandBufferStatus::Completed => {
                            if std::env::var("TL_DEBUG_HANG").is_ok() {
                                eprintln!(
                                    "[DROP] Commands: Thread {:?} buffer already completed",
                                    thread_id
                                );
                            }
                        }
                        MTLCommandBufferStatus::Error => {
                            eprintln!(
                                "[WARN] Commands: Thread {:?} buffer in error state",
                                thread_id
                            );
                        }
                    }
                }
            }

            if std::env::var("TL_DEBUG_HANG").is_ok() {
                eprintln!("[DROP] Commands: All thread command buffers cleaned up");
            }
        } else {
            eprintln!("[WARN] Commands: Failed to lock command buffers during drop");
        }
    }
}
