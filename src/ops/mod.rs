//! Tensor operations

mod helpers;
pub mod elementwise;
pub mod matmul;
pub mod activations;
pub mod broadcast;
pub mod reduce;
pub mod einsum;
pub mod fused;
pub mod advanced_fusion;
pub mod inplace;
pub mod tensor_ops;

// Re-export commonly used types
pub use fused::Activation;

