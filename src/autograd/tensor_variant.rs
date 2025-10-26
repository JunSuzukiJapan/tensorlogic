//! Type-safe wrapper for Tensor<f16> and Tensor<f32>
//!
//! This allows AutogradContext to store both f16 and f32 tensors in a single HashMap,
//! enabling generic support for automatic differentiation with both types.

use crate::tensor::Tensor;
use half::f16;

/// Variant that can hold either Tensor<f16> or Tensor<f32>
#[derive(Debug, Clone)]
pub enum TensorVariant {
    /// f16 tensor
    F16(Tensor<f16>),
    /// f32 tensor
    F32(Tensor<f32>),
}

impl TensorVariant {
    /// Create from Tensor<f16>
    pub fn from_f16(tensor: Tensor<f16>) -> Self {
        TensorVariant::F16(tensor)
    }

    /// Create from Tensor<f32>
    pub fn from_f32(tensor: Tensor<f32>) -> Self {
        TensorVariant::F32(tensor)
    }

    /// Get reference to f16 tensor if this variant holds one
    pub fn as_f16(&self) -> Option<&Tensor<f16>> {
        match self {
            TensorVariant::F16(t) => Some(t),
            _ => None,
        }
    }

    /// Get reference to f32 tensor if this variant holds one
    pub fn as_f32(&self) -> Option<&Tensor<f32>> {
        match self {
            TensorVariant::F32(t) => Some(t),
            _ => None,
        }
    }

    /// Get mutable reference to f16 tensor if this variant holds one
    pub fn as_f16_mut(&mut self) -> Option<&mut Tensor<f16>> {
        match self {
            TensorVariant::F16(t) => Some(t),
            _ => None,
        }
    }

    /// Get mutable reference to f32 tensor if this variant holds one
    pub fn as_f32_mut(&mut self) -> Option<&mut Tensor<f32>> {
        match self {
            TensorVariant::F32(t) => Some(t),
            _ => None,
        }
    }

    /// Clone as f16 tensor if this variant holds one
    pub fn clone_f16(&self) -> Option<Tensor<f16>> {
        match self {
            TensorVariant::F16(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// Clone as f32 tensor if this variant holds one
    pub fn clone_f32(&self) -> Option<Tensor<f32>> {
        match self {
            TensorVariant::F32(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// Check if this variant holds f16 tensor
    pub fn is_f16(&self) -> bool {
        matches!(self, TensorVariant::F16(_))
    }

    /// Check if this variant holds f32 tensor
    pub fn is_f32(&self) -> bool {
        matches!(self, TensorVariant::F32(_))
    }

    /// Get type name as string
    pub fn type_name(&self) -> &'static str {
        match self {
            TensorVariant::F16(_) => "f16",
            TensorVariant::F32(_) => "f32",
        }
    }
}

impl From<Tensor<f16>> for TensorVariant {
    fn from(tensor: Tensor<f16>) -> Self {
        TensorVariant::F16(tensor)
    }
}

impl From<Tensor<f32>> for TensorVariant {
    fn from(tensor: Tensor<f32>) -> Self {
        TensorVariant::F32(tensor)
    }
}
