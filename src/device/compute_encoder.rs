//! Compute command encoder with semaphore integration
//!
//! Based on Candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/encoder.rs
//!
//! This encoder wrapper ensures proper semaphore state management:
//! - Status set to Encoding when created
//! - Status reset to Available when dropped
//! - Guaranteed cleanup via Drop trait

use crate::device::{CommandSemaphore, CommandStatus};
use metal::{ComputeCommandEncoder as MTLComputeCommandEncoder, ComputeCommandEncoderRef};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

/// Wrapper around Metal ComputeCommandEncoder with semaphore integration
pub struct ComputeCommandEncoder {
    raw: MTLComputeCommandEncoder,
    semaphore: Arc<CommandSemaphore>,
    /// Track if encoding has ended to prevent double-call
    encoding_ended: AtomicBool,
}

impl ComputeCommandEncoder {
    /// Create a new encoder with semaphore integration
    pub(crate) fn new(raw: MTLComputeCommandEncoder, semaphore: Arc<CommandSemaphore>) -> Self {
        Self {
            raw,
            semaphore,
            encoding_ended: AtomicBool::new(false),
        }
    }

    /// Get the raw Metal encoder reference
    pub fn as_ref(&self) -> &ComputeCommandEncoderRef {
        &self.raw
    }

    /// Signal that encoding has ended by resetting semaphore to Available
    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    /// End encoding and reset semaphore state
    /// Safe to call multiple times - only the first call has effect
    pub fn end_encoding(&self) {
        // Only end encoding once
        if !self.encoding_ended.swap(true, Ordering::SeqCst) {
            self.raw.end_encoding();
            self.signal_encoding_ended();
        }
    }
}

/// Guaranteed cleanup via Drop trait
/// This ensures semaphore is always reset even if end_encoding() is not called
impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

/// Deref to allow direct access to Metal encoder methods
impl std::ops::Deref for ComputeCommandEncoder {
    type Target = ComputeCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
