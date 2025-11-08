//! Buffer handle CPU read - Candle-compliant implementation
//!
//! Candle pattern for reading GPU data:
//! 1. device.wait_until_completed() FIRST
//! 2. THEN read buffer contents
//!
//! This ensures ALL pending GPU operations complete before CPU reads.

use crate::device::{Device, MetalBuffer};
use crate::tensor::{BufferHandle, FloatType};

impl<T: FloatType> BufferHandle<T> {
    /// Read GPU buffer to CPU Vec - CANDLE PATTERN
    ///
    /// CRITICAL: This method does NOT wait for GPU!
    /// The caller MUST call device.wait_until_completed() BEFORE calling this.
    ///
    /// Candle's to_cpu() pattern:
    /// ```
    /// device.wait_until_completed()?;  // <-- FIRST
    /// let data = buffer.read_contents(); // <-- THEN
    /// ```
    pub fn to_cpu_vec_candle(&self) -> Vec<T> {
        match self {
            BufferHandle::CPU(data) => data.clone(),
            BufferHandle::Metal(metal_buf) => {
                // CANDLE PATTERN: Just read, don't wait
                // Caller MUST have called wait_until_completed() already!
                let ptr = metal_buf.buffer.contents() as *const T;
                let slice = unsafe {
                    std::slice::from_raw_parts(ptr, metal_buf.length)
                };
                slice.sync_and_read()
            }
        }
    }
}

/// Candle-compliant sync and read
///
/// This is the CORRECT way to read GPU data following Candle patterns.
pub fn sync_and_read_candle<T: FloatType>(
    buffer: &BufferHandle<T>,
    device: &Device,
) -> Vec<T> {
    // CANDLE PATTERN:
    // 1. Wait for ALL GPU operations to complete
    if let Device::Metal(ref metal_device) = device {
        // CRITICAL: flush + wait BEFORE reading
        metal_device.flush_if_needed().ok();
        metal_device.wait_until_completed().ok();
    }

    // 2. NOW it's safe to read buffer contents
    buffer.to_cpu_vec_candle()
}

/// Candle-compliant sync and read as f32
pub fn sync_and_read_f32_candle<T: FloatType>(
    buffer: &BufferHandle<T>,
    device: &Device,
) -> Vec<f32> {
    // Same pattern: wait THEN read
    let data = sync_and_read_candle(buffer, device);
    data.iter().map(|x| x.to_f32()).collect()
}
