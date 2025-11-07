//! Tensor type and operations

mod tensor;
mod buffer_handle;
mod shape;
mod tensor_like;
mod float_type;
mod tensor_creation;
mod tensor_accessors;
mod tensor_transform;
mod tensor_io;
mod tensor_autograd;
mod tensor_convert;

pub use tensor::Tensor;
pub use buffer_handle::BufferHandle;
pub use shape::TensorShape;
pub use tensor_like::{TensorLike, TokenIdArray};
pub use float_type::FloatType;
pub use tensor_creation::TensorCreation;
pub use tensor_accessors::TensorAccessors;
pub use tensor_transform::TensorTransform;
pub use tensor_io::TensorIO;
pub use tensor_autograd::TensorAutograd;
pub use tensor_convert::TensorConvert;

// Type aliases for convenience
/// Tensor with f16 precision (default)
pub type TensorF16 = Tensor<half::f16>;

/// Tensor with f32 precision
pub type TensorF32 = Tensor<f32>;
