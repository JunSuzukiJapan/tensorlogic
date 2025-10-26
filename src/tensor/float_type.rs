/// Generic floating-point type trait for TensorLogic
///
/// This trait defines the required operations for floating-point types
/// used in tensor computations. Currently implemented for f16 and f32.

use half::f16;
use std::fmt::Debug;

/// Trait for floating-point types supported in tensor operations
pub trait FloatType: Copy + Clone + Debug + Send + Sync + 'static {
    /// The zero value for this type
    fn zero() -> Self;

    /// The one value for this type
    fn one() -> Self;

    /// Convert from f32 to this type
    fn from_f32(value: f32) -> Self;

    /// Convert from this type to f32
    fn to_f32(self) -> f32;

    /// Convert from f64 to this type
    fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }

    /// Convert from this type to f64
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Metal shader type name (e.g., "half" or "float")
    fn metal_type_name() -> &'static str;

    /// Size in bytes
    fn size_in_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Check if this is f16
    fn is_f16() -> bool {
        Self::size_in_bytes() == 2
    }

    /// Check if this is f32
    fn is_f32() -> bool {
        Self::size_in_bytes() == 4
    }
}

/// Implementation for f16 (half precision)
impl FloatType for f16 {
    #[inline]
    fn zero() -> Self {
        f16::from_f32(0.0)
    }

    #[inline]
    fn one() -> Self {
        f16::from_f32(1.0)
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    #[inline]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }

    #[inline]
    fn metal_type_name() -> &'static str {
        "half"
    }
}

/// Implementation for f32 (single precision)
impl FloatType for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn metal_type_name() -> &'static str {
        "float"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_float_type() {
        assert_eq!(f16::zero().to_f32(), 0.0);
        assert_eq!(f16::one().to_f32(), 1.0);
        assert_eq!(f16::from_f32(3.14).to_f32(), 3.14);
        assert_eq!(f16::metal_type_name(), "half");
        assert!(f16::is_f16());
        assert!(!f16::is_f32());
    }

    #[test]
    fn test_f32_float_type() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(f32::from_f32(3.14), 3.14);
        assert_eq!(f32::metal_type_name(), "float");
        assert!(!f32::is_f16());
        assert!(f32::is_f32());
    }

    #[test]
    fn test_conversions() {
        let f16_val = f16::from_f32(2.5);
        let f32_val = f32::from_f32(2.5);

        assert_eq!(f16_val.to_f32(), 2.5);
        assert_eq!(f32_val.to_f32(), 2.5);

        assert_eq!(f16_val.to_f64(), 2.5);
        assert_eq!(f32_val.to_f64(), 2.5);
    }
}
