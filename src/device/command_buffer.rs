//! Command buffer wrapper for Metal
//!
//! Based on candle's implementation:
//! https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/metal/command_buffer.rs

use metal::{CommandBuffer as MTLCommandBuffer, CommandBufferRef, MTLCommandBufferStatus};
use std::collections::HashMap;
use std::thread;

/// Wrapper around Metal CommandBuffer with additional functionality
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
}

/// Thread-local command buffer map
/// Each thread gets its own command buffer to avoid contention
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
