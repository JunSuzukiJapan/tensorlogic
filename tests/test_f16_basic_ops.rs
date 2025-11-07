/// Comprehensive tests for f16 tensor basic operations
///
/// These tests mirror test_f32_basic_ops.rs but use f16 (half precision).
/// F16 tests are critical because:
/// - The project claims to focus on f16 for Metal GPU efficiency
/// - F16 has different precision characteristics (±65504, ~3-4 decimal digits)
/// - Numerical stability differs from f32
///
/// Tests cover:
/// - Basic arithmetic (add, sub, mul, div)
/// - Scalar operations
/// - Precision comparison with f32
/// - Overflow and underflow specific to f16
/// - Numerical stability

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use half::f16;

// Helper function to assert f16 tensors are close
fn assert_tensor_close_f16(result: &[f16], expected: &[f16], epsilon: f32) {
    assert_eq!(result.len(), expected.len(), "Length mismatch");
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r.to_f32() - e.to_f32()).abs();
        assert!(
            diff < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r.to_f32(), e.to_f32(), diff
        );
    }
}

// Basic Arithmetic Operations

#[test]
fn test_f16_addition() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create f16 tensors
    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0), f16::from_f32(8.0)],
        vec![2, 2]
    )?;

    // Add tensors
    let c = a.add(&b)?;
    let result = c.to_vec();

    // Verify results
    let expected = vec![f16::from_f32(6.0), f16::from_f32(8.0), f16::from_f32(10.0), f16::from_f32(12.0)];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    println!("✓ f16 addition test passed");
    Ok(())
}

#[test]
fn test_f16_subtraction() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0), f16::from_f32(40.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;

    let c = a.sub(&b)?;
    let result = c.to_vec();

    let expected = vec![f16::from_f32(9.0), f16::from_f32(18.0), f16::from_f32(27.0), f16::from_f32(36.0)];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    println!("✓ f16 subtraction test passed");
    Ok(())
}

#[test]
fn test_f16_multiplication() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0), f16::from_f32(5.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.5), f16::from_f32(2.5), f16::from_f32(3.5), f16::from_f32(4.5)],
        vec![2, 2]
    )?;

    let c = a.mul(&b)?;
    let result = c.to_vec();

    let expected = vec![f16::from_f32(3.0), f16::from_f32(7.5), f16::from_f32(14.0), f16::from_f32(22.5)];
    assert_tensor_close_f16(&result, &expected, 1e-2);

    println!("✓ f16 multiplication test passed");
    Ok(())
}

#[test]
fn test_f16_division() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0), f16::from_f32(40.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(8.0)],
        vec![2, 2]
    )?;

    let c = a.div(&b)?;
    let result = c.to_vec();

    let expected = vec![f16::from_f32(5.0), f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(5.0)];
    assert_tensor_close_f16(&result, &expected, 1e-2);

    println!("✓ f16 division test passed");
    Ok(())
}

// Scalar Operations

#[test]
fn test_f16_scalar_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;

    // Scalar multiplication
    let b = a.mul_scalar(f16::from_f32(2.5))?;
    let result = b.to_vec();
    let expected = vec![f16::from_f32(2.5), f16::from_f32(5.0), f16::from_f32(7.5), f16::from_f32(10.0)];
    assert_tensor_close_f16(&result, &expected, 1e-2);

    // Scalar addition
    let c = a.add_scalar(f16::from_f32(10.0))?;
    let result = c.to_vec();
    let expected = vec![f16::from_f32(11.0), f16::from_f32(12.0), f16::from_f32(13.0), f16::from_f32(14.0)];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    // Scalar division
    let d = a.div_scalar(f16::from_f32(2.0))?;
    let result = d.to_vec();
    let expected = vec![f16::from_f32(0.5), f16::from_f32(1.0), f16::from_f32(1.5), f16::from_f32(2.0)];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    println!("✓ f16 scalar operations test passed");
    Ok(())
}

// F16 Precision Tests

#[test]
fn test_f16_precision_vs_f32() -> TensorResult<()> {
    // Test that f16 has lower precision than f32
    let device = MetalDevice::new()?;

    // Use a value that shows precision difference
    let val_f32 = 1.234567f32;
    let val_f16 = f16::from_f32(val_f32);

    // F16 should have lower precision (approximately 3-4 decimal digits)
    let diff = (val_f32 - val_f16.to_f32()).abs();
    assert!(diff < 1e-3, "F16 precision should be within 1e-3 of f32");
    assert!(diff > 0.0, "F16 and f32 should have some precision difference");

    println!("✓ f16 precision vs f32 test passed");
    Ok(())
}

#[test]
fn test_f16_small_values() -> TensorResult<()> {
    // Test f16 with small values (near underflow)
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(0.001), f16::from_f32(0.0001), f16::from_f32(0.00001)],
        vec![3]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(0.002), f16::from_f32(0.0002), f16::from_f32(0.00002)],
        vec![3]
    )?;

    let c = a.add(&b)?;
    let result = c.to_vec();

    // Verify results are in the right ballpark (f16 may lose some precision)
    assert!(result[0].to_f32() > 0.002 && result[0].to_f32() < 0.004);
    assert!(result[1].to_f32() > 0.0002 && result[1].to_f32() < 0.0004);
    // Third value may underflow to 0 in f16

    println!("✓ f16 small values test passed");
    Ok(())
}

#[test]
fn test_f16_large_values() -> TensorResult<()> {
    // Test f16 with large values (near overflow)
    let device = MetalDevice::new()?;

    // F16 max is approximately 65504
    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1000.0), f16::from_f32(10000.0), f16::from_f32(60000.0)],
        vec![3]
    )?;

    let result = a.to_vec();

    // All should be finite (not overflow to Inf)
    for &val in &result {
        assert!(val.is_finite(), "Large f16 values should remain finite");
    }

    println!("✓ f16 large values test passed");
    Ok(())
}

#[test]
fn test_f16_overflow() -> TensorResult<()> {
    // Test f16 overflow behavior
    // F16 max is ~65504, values beyond this become Inf

    let large_val = f16::from_f32(70000.0);
    assert!(large_val.is_infinite(), "Values > 65504 should overflow to Inf in f16");

    let very_large_val = f16::from_f32(100000.0);
    assert!(very_large_val.is_infinite(), "Very large values should be Inf in f16");

    println!("✓ f16 overflow test passed");
    Ok(())
}

#[test]
fn test_f16_underflow() -> TensorResult<()> {
    // Test f16 underflow behavior
    // F16 min normal is ~6e-8, smaller values may underflow to 0

    let tiny_val = f16::from_f32(1e-10);
    // May underflow to 0 or become subnormal
    assert!(
        tiny_val == f16::ZERO || tiny_val.to_f32().abs() < 1e-7,
        "Tiny values may underflow in f16"
    );

    println!("✓ f16 underflow test passed");
    Ok(())
}

// Numerical Stability

#[test]
fn test_f16_addition_stability() -> TensorResult<()> {
    // Test that f16 addition maintains reasonable precision
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::ones(vec![100])?;
    let mut sum = a.clone();

    // Add 1.0 many times
    for _ in 0..100 {
        sum = sum.add_scalar(f16::ONE)?;
    }

    let result = sum.to_vec();

    // Each element should be approximately 101.0 (1.0 initial + 100 additions)
    for &val in &result {
        let expected = 101.0;
        let diff = (val.to_f32() - expected).abs();
        assert!(diff < 1.0, "F16 accumulation should maintain reasonable precision");
    }

    println!("✓ f16 addition stability test passed");
    Ok(())
}

#[test]
fn test_f16_multiplication_stability() -> TensorResult<()> {
    // Test f16 multiplication stability with values that might lose precision
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.1), f16::from_f32(1.2), f16::from_f32(1.3), f16::from_f32(1.4)],
        vec![4]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(0.9), f16::from_f32(0.8), f16::from_f32(0.7), f16::from_f32(0.6)],
        vec![4]
    )?;

    let c = a.mul(&b)?;
    let result = c.to_vec();

    // Verify results are close to expected (with f16 precision tolerance)
    let expected = [1.1 * 0.9, 1.2 * 0.8, 1.3 * 0.7, 1.4 * 0.6];
    for (i, &val) in result.iter().enumerate() {
        let diff = (val.to_f32() - expected[i]).abs();
        assert!(diff < 0.01, "F16 multiplication should be reasonably accurate");
    }

    println!("✓ f16 multiplication stability test passed");
    Ok(())
}

#[test]
fn test_f16_mixed_operations() -> TensorResult<()> {
    // Test combinations of operations
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(6.0), f16::from_f32(8.0)],
        vec![2, 2]
    )?;

    // (a * 2) + 3 - 1
    let b = a.mul_scalar(f16::from_f32(2.0))?
             .add_scalar(f16::from_f32(3.0))?
             .sub_scalar(f16::from_f32(1.0))?;

    let result = b.to_vec();

    // Expected: (2*2)+3-1=6, (4*2)+3-1=10, (6*2)+3-1=14, (8*2)+3-1=18
    let expected = vec![f16::from_f32(6.0), f16::from_f32(10.0), f16::from_f32(14.0), f16::from_f32(18.0)];
    assert_tensor_close_f16(&result, &expected, 1e-2);

    println!("✓ f16 mixed operations test passed");
    Ok(())
}

// Edge Cases

#[test]
fn test_f16_zero_operations() -> TensorResult<()> {
    // Test operations with zero
    let device = MetalDevice::new()?;

    let zeros = Tensor::<f16>::zeros(vec![2, 2])?;
    let ones = Tensor::<f16>::ones(vec![2, 2])?;

    // Add zeros
    let a = ones.add(&zeros)?;
    let result = a.to_vec();
    for &val in &result {
        assert_eq!(val, f16::ONE);
    }

    // Multiply by zero
    let b = ones.mul(&zeros)?;
    let result = b.to_vec();
    for &val in &result {
        assert_eq!(val, f16::ZERO);
    }

    println!("✓ f16 zero operations test passed");
    Ok(())
}

#[test]
fn test_f16_negative_values() -> TensorResult<()> {
    // Test operations with negative values
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(-1.0), f16::from_f32(-2.0), f16::from_f32(-3.0), f16::from_f32(-4.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;

    // Add
    let c = a.add(&b)?;
    let result = c.to_vec();
    for &val in &result {
        assert_eq!(val, f16::ZERO);
    }

    // Multiply
    let d = a.mul(&b)?;
    let result = d.to_vec();
    let expected = vec![f16::from_f32(-1.0), f16::from_f32(-4.0), f16::from_f32(-9.0), f16::from_f32(-16.0)];
    assert_tensor_close_f16(&result, &expected, 1e-2);

    println!("✓ f16 negative values test passed");
    Ok(())
}

#[test]
fn test_f16_special_values() -> TensorResult<()> {
    // Test special f16 values (Inf, NaN)
    let inf = f16::INFINITY;
    let neg_inf = f16::NEG_INFINITY;
    let nan = f16::NAN;

    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(nan.is_nan());

    // Inf + 1 = Inf
    let result = f16::from_f32(inf.to_f32() + 1.0);
    assert!(result.is_infinite());

    // NaN propagates
    let result = f16::from_f32(nan.to_f32() + 1.0);
    assert!(result.is_nan());

    println!("✓ f16 special values test passed");
    Ok(())
}

// Comparison with F32

#[test]
fn test_f16_f32_precision_difference() -> TensorResult<()> {
    // Demonstrate precision difference between f16 and f32
    let device = MetalDevice::new()?;

    // Create same data in f32 and f16
    let data_f32 = vec![1.234567f32, 9.876543f32, 0.123456f32, 7.654321f32];
    let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let a_f32 = Tensor::<f32>::from_vec(data_f32.clone(), vec![4])?;
    let a_f16 = Tensor::<f16>::from_vec(data_f16, vec![4])?;

    // Multiply by 2
    let b_f32 = a_f32.mul_scalar(2.0)?;
    let b_f16 = a_f16.mul_scalar(f16::from_f32(2.0))?;

    let result_f32 = b_f32.to_vec();
    let result_f16: Vec<f32> = b_f16.to_vec().iter().map(|&x| x.to_f32()).collect();

    // Compare precision
    for (i, (&r32, &r16)) in result_f32.iter().zip(result_f16.iter()).enumerate() {
        let diff = (r32 - r16).abs();
        // F16 should be less precise than f32
        assert!(diff < 0.01, "F16 result should be close to f32, but less precise");
        println!("  Element {}: f32={:.6}, f16={:.6}, diff={:.6}", i, r32, r16, diff);
    }

    println!("✓ f16 vs f32 precision difference test passed");
    Ok(())
}

#[test]
fn test_f16_tensor_size_optimization() -> TensorResult<()> {
    // Verify f16 uses half the memory of f32
    use std::mem;

    let size_f16 = mem::size_of::<f16>();
    let size_f32 = mem::size_of::<f32>();

    assert_eq!(size_f16, 2, "F16 should be 2 bytes");
    assert_eq!(size_f32, 4, "F32 should be 4 bytes");
    assert_eq!(size_f32, size_f16 * 2, "F32 should be exactly 2x f16 size");

    // For large tensors, this means significant memory savings
    let num_elements = 1_000_000;
    let memory_f16 = num_elements * size_f16;
    let memory_f32 = num_elements * size_f32;
    let savings = memory_f32 - memory_f16;

    println!("  Memory for 1M elements:");
    println!("    F16: {} bytes ({} MB)", memory_f16, memory_f16 / (1024 * 1024));
    println!("    F32: {} bytes ({} MB)", memory_f32, memory_f32 / (1024 * 1024));
    println!("    Savings: {} bytes ({} MB)", savings, savings / (1024 * 1024));

    println!("✓ f16 tensor size optimization test passed");
    Ok(())
}

// Accumulated Error Test

#[test]
fn test_f16_accumulated_error() -> TensorResult<()> {
    // Test that repeated operations accumulate error in f16
    let device = MetalDevice::new()?;

    let mut a = Tensor::<f16>::from_vec(vec![f16::from_f32(1.0); 4], vec![2, 2])?;

    // Repeatedly divide by 3 and multiply by 3 (should stay at 1.0, but will drift in f16)
    for _ in 0..10 {
        a = a.div_scalar(f16::from_f32(3.0))?;
        a = a.mul_scalar(f16::from_f32(3.0))?;
    }

    let result = a.to_vec();

    // Should still be close to 1.0, but may have some error
    for &val in &result {
        let diff = (val.to_f32() - 1.0).abs();
        assert!(diff < 0.1, "F16 should maintain reasonable precision even with accumulated operations");
    }

    println!("✓ f16 accumulated error test passed");
    Ok(())
}
