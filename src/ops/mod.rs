//! Tensor operations

pub mod elementwise;
pub mod matmul;
pub mod activations;
pub mod broadcast;
pub mod reduce;
pub mod einsum;
pub mod fused;
pub mod inplace;

pub use fused::Activation;
