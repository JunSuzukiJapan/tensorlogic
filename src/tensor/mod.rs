//! Tensor type and operations

mod tensor;
mod buffer_handle;
mod shape;

pub use tensor::Tensor;
pub use buffer_handle::BufferHandle;
pub use shape::TensorShape;
