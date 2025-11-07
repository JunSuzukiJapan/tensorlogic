/// Comprehensive tests for Error Handling
///
/// Robust error handling is crucial for production systems.
/// Tests cover:
/// - Shape mismatches in operations
/// - Dimension errors
/// - Numerical errors (NaN, Inf, overflow, underflow)
/// - Out-of-bounds access
/// - Invalid parameters
/// - Empty tensors
/// - Device errors (when applicable)
///
/// These tests ensure the library fails gracefully with clear error messages
/// rather than producing silent errors or crashes.

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use half::f16;
use serial_test::serial;

// Shape Mismatch Errors

#[test]
#[serial]
#[should_panic(expected = "Shape mismatch")]
fn test_add_shape_mismatch_2x2_and_3x3() {
    let a = Tensor::<f32>::ones(vec![2, 2]).unwrap();
    let b = Tensor::<f32>::ones(vec![3, 3]).unwrap();
    let _ = a.add(&b).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "Shape mismatch")]
fn test_add_shape_mismatch_1d_and_2d() {
    let a = Tensor::<f32>::ones(vec![4]).unwrap();
    let b = Tensor::<f32>::ones(vec![4, 1]).unwrap();
    let _ = a.add(&b).unwrap();
}

#[test]
#[serial]
#[should_panic]
fn test_sub_shape_mismatch() {
    let a = Tensor::<f32>::ones(vec![2, 3]).unwrap();
    let b = Tensor::<f32>::ones(vec![3, 2]).unwrap();
    let _ = a.sub(&b).unwrap();
}

#[test]
#[serial]
#[should_panic]
fn test_mul_shape_mismatch() {
    let a = Tensor::<f32>::ones(vec![4, 5]).unwrap();
    let b = Tensor::<f32>::ones(vec![4, 6]).unwrap();
    let _ = a.mul(&b).unwrap();
}

#[test]
#[serial]
#[should_panic]
fn test_div_shape_mismatch() {
    let a = Tensor::<f32>::ones(vec![10]).unwrap();
    let b = Tensor::<f32>::ones(vec![20]).unwrap();
    let _ = a.div(&b).unwrap();
}

// Matrix Multiplication Errors

#[test]
#[serial]
#[should_panic]
fn test_matmul_incompatible_dimensions() {
    // [2, 3] @ [5, 7] - incompatible (inner dimensions don't match)
    let a = Tensor::<f32>::ones(vec![2, 3]).unwrap();
    let b = Tensor::<f32>::ones(vec![5, 7]).unwrap();
    let _ = a.matmul(&b).unwrap();
}

#[test]
#[serial]
#[should_panic]
fn test_matmul_1d_tensors() {
    // Matmul requires at least 2D tensors
    let a = Tensor::<f32>::ones(vec![5]).unwrap();
    let b = Tensor::<f32>::ones(vec![5]).unwrap();
    let _ = a.matmul(&b).unwrap();
}

#[test]
#[serial]
fn test_matmul_valid_dimensions() -> TensorResult<()> {
    // This should succeed: [2, 3] @ [3, 4] = [2, 4]
    let a = Tensor::<f32>::ones(vec![2, 3])?;
    let b = Tensor::<f32>::ones(vec![3, 4])?;
    let c = a.matmul(&b)?;

    assert_eq!(c.shape(), vec![2, 4]);

    println!("✓ Matmul valid dimensions test passed");
    Ok(())
}

// Reshape Errors

#[test]
#[serial]
#[should_panic]
fn test_reshape_incompatible_size() {
    // Cannot reshape [2, 3] (6 elements) to [2, 4] (8 elements)
    let a = Tensor::<f32>::ones(vec![2, 3]).unwrap();
    let _ = a.reshape(vec![2, 4]).unwrap();
}

#[test]
#[serial]
#[should_panic]
fn test_reshape_negative_dimension() {
    let a = Tensor::<f32>::ones(vec![4, 4]).unwrap();
    // Negative dimensions should be invalid
    // Note: This depends on implementation - may not panic if -1 is supported as "infer"
    let _ = a.reshape(vec![2, -8_i32 as usize]).unwrap();
}

#[test]
#[serial]
fn test_reshape_valid() -> TensorResult<()> {
    // Valid reshape: [2, 3] (6 elements) to [3, 2] (6 elements)
    let a = Tensor::<f32>::ones(vec![2, 3])?;
    let b = a.reshape(vec![3, 2])?;

    assert_eq!(b.shape(), vec![3, 2]);

    println!("✓ Reshape valid test passed");
    Ok(())
}

// Indexing / Slicing Errors
// Note: Tensor slice method is not implemented, only available on TokenIdArray

// #[test]
// #[serial]
// #[should_panic(expected = "Invalid dimension")]
// fn test_slice_invalid_dimension() {
//     let a = Tensor::<f32>::ones(vec![3, 4]).unwrap();
//     // Tensor is 2D, cannot slice dimension 2
//     let _ = a.slice(2, 0, 2).unwrap();
// }

// #[test]
// #[serial]
// #[should_panic]
// fn test_slice_out_of_bounds() {
//     let a = Tensor::<f32>::ones(vec![3, 4]).unwrap();
//     // Cannot slice [5:10] in dimension with size 3
//     let _ = a.slice(0, 5, 10).unwrap();
// }

// Reduction Errors

#[test]
#[serial]
#[should_panic(expected = "Invalid dimension")]
fn test_sum_invalid_dimension() {
    let a = Tensor::<f32>::ones(vec![3, 4]).unwrap();
    // Cannot sum over dimension 2 (tensor only has 2 dimensions)
    let _ = a.sum_dim(2).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "Invalid dimension")]
fn test_softmax_invalid_dimension() {
    let a = Tensor::<f32>::ones(vec![2, 3, 4]).unwrap();
    // Cannot apply softmax on dimension 3 (max is 2 for 3D tensor)
    let _ = a.softmax(3).unwrap();
}

// Numerical Stability Errors

#[test]
#[serial]
fn test_division_by_zero() -> TensorResult<()> {
    // Division by zero should produce Inf, not crash
    let a = Tensor::<f32>::ones(vec![2, 2])?;
    let b = Tensor::<f32>::zeros(vec![2, 2])?;

    let c = a.div(&b)?;
    let result = c.sync_and_read();

    // Should be Inf, not NaN or crash
    for &val in &result {
        assert!(val.is_infinite(), "Division by zero should produce Inf");
    }

    println!("✓ Division by zero test passed");
    Ok(())
}

#[test]
#[serial]
fn test_nan_propagation_add() -> TensorResult<()> {
    // NaN should propagate through operations
    let a = Tensor::<f32>::from_vec(vec![1.0, f32::NAN, 3.0], vec![3])?;
    let b = Tensor::<f32>::ones(vec![3])?;

    let c = a.add(&b)?;
    let result = c.sync_and_read();

    assert!(result[0].is_finite());
    assert!(result[1].is_nan(), "NaN should propagate in addition");
    assert!(result[2].is_finite());

    println!("✓ NaN propagation add test passed");
    Ok(())
}

#[test]
#[serial]
fn test_nan_propagation_mul() -> TensorResult<()> {
    // NaN * anything = NaN (even 0)
    let a = Tensor::<f32>::from_vec(vec![f32::NAN, f32::NAN], vec![2])?;
    let b = Tensor::<f32>::from_vec(vec![0.0, 5.0], vec![2])?;

    let c = a.mul(&b)?;
    let result = c.sync_and_read();

    assert!(result[0].is_nan(), "NaN * 0 should be NaN");
    assert!(result[1].is_nan(), "NaN * 5 should be NaN");

    println!("✓ NaN propagation mul test passed");
    Ok(())
}

#[test]
#[serial]
fn test_inf_propagation() -> TensorResult<()> {
    // Inf should propagate appropriately
    let a = Tensor::<f32>::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY], vec![2])?;
    let b = Tensor::<f32>::from_vec(vec![2.0, 2.0], vec![2])?;

    let c = a.mul(&b)?;
    let result = c.sync_and_read();

    assert!(result[0].is_infinite() && result[0] > 0.0, "Inf * 2 should be Inf");
    assert!(result[1].is_infinite() && result[1] < 0.0, "-Inf * 2 should be -Inf");

    println!("✓ Inf propagation test passed");
    Ok(())
}

#[test]
#[serial]
fn test_inf_minus_inf() -> TensorResult<()> {
    // Inf - Inf = NaN
    let a = Tensor::<f32>::from_vec(vec![f32::INFINITY], vec![1])?;
    let b = Tensor::<f32>::from_vec(vec![f32::INFINITY], vec![1])?;

    let c = a.sub(&b)?;
    let result = c.sync_and_read();

    assert!(result[0].is_nan(), "Inf - Inf should be NaN");

    println!("✓ Inf minus Inf test passed");
    Ok(())
}

#[test]
#[serial]
fn test_exp_overflow() -> TensorResult<()> {
    // exp of large number should produce Inf, not crash
    let a = Tensor::<f32>::from_vec(vec![1000.0], vec![1])?;

    let b = a.exp()?;
    let result = b.sync_and_read();

    assert!(result[0].is_infinite(), "exp(1000) should be Inf");

    println!("✓ Exp overflow test passed");
    Ok(())
}

#[test]
#[serial]
fn test_log_negative() -> TensorResult<()> {
    // log of negative number should produce NaN
    let a = Tensor::<f32>::from_vec(vec![-1.0, -10.0], vec![2])?;

    let b = a.log()?;
    let result = b.sync_and_read();

    assert!(result[0].is_nan(), "log(-1) should be NaN");
    assert!(result[1].is_nan(), "log(-10) should be NaN");

    println!("✓ Log negative test passed");
    Ok(())
}

#[test]
#[serial]
fn test_log_zero() -> TensorResult<()> {
    // log(0) = -Inf
    let a = Tensor::<f32>::from_vec(vec![0.0], vec![1])?;

    let b = a.log()?;
    let result = b.sync_and_read();

    assert!(
        result[0].is_infinite() && result[0] < 0.0,
        "log(0) should be -Inf"
    );

    println!("✓ Log zero test passed");
    Ok(())
}

#[test]
#[serial]
fn test_sqrt_negative() -> TensorResult<()> {
    // sqrt of negative should produce NaN
    let a = Tensor::<f32>::from_vec(vec![-4.0], vec![1])?;

    let b = a.sqrt()?;
    let result = b.sync_and_read();

    assert!(result[0].is_nan(), "sqrt(-4) should be NaN");

    println!("✓ Sqrt negative test passed");
    Ok(())
}

#[test]
#[serial]
fn test_pow_special_cases() -> TensorResult<()> {
    // Test special cases of pow
    let a = Tensor::<f32>::from_vec(
        vec![0.0, -1.0, 2.0, f32::INFINITY],
        vec![4]
    )?;
    let b = Tensor::<f32>::from_vec(
        vec![0.0, 0.5, 10.0, 2.0],
        vec![4]
    )?;

    let c = a.pow(&b)?;
    let result = c.sync_and_read();

    // 0^0 = 1 (by convention)
    assert!((result[0] - 1.0).abs() < 1e-5, "0^0 should be 1");

    // (-1)^0.5 = NaN (complex number)
    assert!(result[1].is_nan(), "(-1)^0.5 should be NaN");

    // 2^10 = 1024
    assert!((result[2] - 1024.0).abs() < 1e-3, "2^10 should be 1024");

    // Inf^2 = Inf
    assert!(result[3].is_infinite(), "Inf^2 should be Inf");

    println!("✓ Pow special cases test passed");
    Ok(())
}

// Empty Tensor Errors

#[test]
#[serial]
fn test_empty_tensor_operations() -> TensorResult<()> {
    // Operations on empty tensors should not crash
    let a = Tensor::<f32>::zeros(vec![0, 5])?;
    let b = Tensor::<f32>::zeros(vec![0, 5])?;

    let c = a.add(&b)?;
    assert_eq!(c.shape(), vec![0, 5]);

    println!("✓ Empty tensor operations test passed");
    Ok(())
}

#[test]
#[serial]
fn test_empty_tensor_sum() -> TensorResult<()> {
    // Sum of empty tensor should be 0
    let a = Tensor::<f32>::zeros(vec![0, 3])?;

    let sum = a.sum()?;
    let result = sum.sync_and_read();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 0.0);

    println!("✓ Empty tensor sum test passed");
    Ok(())
}

// Zero-sized Dimension Errors
// Note: Empty tensors with 0-sized dimensions may be allowed

// #[test]
// #[serial]
// #[should_panic]
// fn test_zeros_with_zero_dimension() {
//     // Creating tensor with a zero dimension should fail
//     let _ = Tensor::<f32>::zeros(vec![2, 0, 3]).unwrap();
// }

// #[test]
// #[serial]
// #[should_panic]
// fn test_ones_with_zero_dimension() {
//     let _ = Tensor::<f32>::ones(vec![0, 4]).unwrap();
// }

// Autograd Errors

#[test]
#[serial]
#[should_panic(expected = "requires_grad")]
fn test_backward_without_requires_grad() {
    use tensorlogic::tensor::TensorAutograd;

    // Calling backward on tensor without requires_grad should fail
    let a = Tensor::<f32>::ones(vec![2, 2]).unwrap();
    let _ = a.backward().unwrap();
}

// Type Conversion Errors

#[test]
#[serial]
fn test_f16_precision_loss() -> TensorResult<()> {
    // Very large numbers may lose precision in f16
    let large_f32 = 65536.0f32; // Larger than f16 max (~65504)

    let a = Tensor::<f32>::from_vec(vec![large_f32], vec![1])?;

    // Convert to f16
    let a_f16_vec: Vec<f16> = a.sync_and_read().iter().map(|&x| f16::from_f32(x)).collect();

    // f16 representation may be Inf
    assert!(
        a_f16_vec[0].is_infinite() || a_f16_vec[0].to_f32() > 60000.0,
        "Large f32 may overflow in f16"
    );

    println!("✓ f16 precision loss test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_underflow() -> TensorResult<()> {
    // Very small numbers may underflow to 0 in f16
    let tiny_f32 = 1e-10f32; // Smaller than f16 min (~6e-8)

    let tiny_f16 = f16::from_f32(tiny_f32);

    // May underflow to 0
    assert!(
        tiny_f16 == f16::ZERO || tiny_f16.to_f32().abs() < 1e-7,
        "Tiny f32 may underflow in f16"
    );

    println!("✓ f16 underflow test passed");
    Ok(())
}

// Concatenation Errors

#[test]
#[serial]
#[should_panic]
fn test_concat_dimension_mismatch() {
    // Cannot concat tensors with different shapes in non-concat dimensions
    let a = Tensor::<f32>::ones(vec![2, 3]).unwrap();
    let b = Tensor::<f32>::ones(vec![2, 5]).unwrap();

    // Concatenating along dim 0, but dim 1 doesn't match (3 vs 5)
    let _ = Tensor::concat(&[&a, &b], 0).unwrap();
}

#[test]
#[serial]
#[should_panic(expected = "Invalid dimension")]
fn test_concat_invalid_dimension() {
    let a = Tensor::<f32>::ones(vec![2, 3]).unwrap();
    let b = Tensor::<f32>::ones(vec![2, 3]).unwrap();

    // Dimension 2 doesn't exist
    let _ = Tensor::concat(&[&a, &b], 2).unwrap();
}

// Transpose Errors

#[test]
#[serial]
#[should_panic]
fn test_transpose_1d_tensor() {
    // Transpose requires at least 2D
    let a = Tensor::<f32>::ones(vec![5]).unwrap();
    let _ = a.transpose().unwrap();
}

// Activation Function Edge Cases

#[test]
#[serial]
fn test_relu_negative() -> TensorResult<()> {
    // ReLU of negative should be 0
    let a = Tensor::<f32>::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5])?;

    let b = a.relu()?;
    let result = b.sync_and_read();

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 5.0);

    println!("✓ ReLU negative test passed");
    Ok(())
}

#[test]
#[serial]
fn test_sigmoid_extreme_values() -> TensorResult<()> {
    // Sigmoid should saturate at extreme values
    let a = Tensor::<f32>::from_vec(vec![-1000.0, 0.0, 1000.0], vec![3])?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // sigmoid(-1000) ≈ 0
    assert!(result[0] < 0.001, "sigmoid(-1000) should be ~0");

    // sigmoid(0) = 0.5
    assert!((result[1] - 0.5).abs() < 0.01, "sigmoid(0) should be 0.5");

    // sigmoid(1000) ≈ 1
    assert!(result[2] > 0.999, "sigmoid(1000) should be ~1");

    println!("✓ Sigmoid extreme values test passed");
    Ok(())
}

#[test]
#[serial]
fn test_softmax_overflow_safety() -> TensorResult<()> {
    // Softmax should handle large values without overflow
    let a = Tensor::<f32>::from_vec(vec![1000.0, 1001.0, 1002.0], vec![1, 3])?;

    let b = a.softmax(1)?;
    let result = b.sync_and_read();

    // Should not be NaN or Inf
    for &val in &result {
        assert!(val.is_finite(), "Softmax output should be finite");
    }

    // Should sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "Softmax should sum to 1");

    println!("✓ Softmax overflow safety test passed");
    Ok(())
}

// Tensor Creation Errors

#[test]
#[serial]
#[should_panic]
fn test_from_vec_size_mismatch() {
    // Vector size doesn't match shape
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let _ = Tensor::<f32>::from_vec(data, vec![2, 3]).unwrap(); // Expects 6 elements
}

#[test]
#[serial]
#[should_panic]
fn test_from_vec_empty_shape() {
    let data = vec![1.0, 2.0, 3.0];
    let _ = Tensor::<f32>::from_vec(data, vec![]).unwrap(); // Empty shape
}

// Layer Normalization Errors

#[test]
#[serial]
#[should_panic]
fn test_layer_norm_invalid_dimension() {
    let a = Tensor::<f32>::ones(vec![2, 3, 4]).unwrap();
    let weight = Tensor::<f32>::ones(vec![5]).unwrap(); // Wrong size

    let _ = a.layer_norm(&weight, None).unwrap();
}

// Device Errors (if Metal is available)

#[test]
#[serial]
fn test_device_availability() -> TensorResult<()> {
    // Check if Metal device is available
    match MetalDevice::new() {
        Ok(_device) => {
            println!("✓ Metal device available");
        }
        Err(_e) => {
            println!("✓ Metal device not available (expected on non-Apple platforms)");
        }
    }

    Ok(())
}

// Stress Test: Multiple Errors in Sequence

#[test]
#[serial]
fn test_error_recovery() -> TensorResult<()> {
    // Test that errors don't leave system in bad state
    let a = Tensor::<f32>::ones(vec![2, 2])?;
    let b = Tensor::<f32>::ones(vec![3, 3])?;

    // This should fail
    let result1 = a.add(&b);
    assert!(result1.is_err(), "Expected error for shape mismatch");

    // But we should be able to do valid operations afterward
    let c = Tensor::<f32>::ones(vec![2, 2])?;
    let d = a.add(&c)?;

    assert_eq!(d.shape(), vec![2, 2]);

    println!("✓ Error recovery test passed");
    Ok(())
}

#[test]
#[serial]
fn test_chain_operations_with_error() -> TensorResult<()> {
    // Test error in chain of operations
    let a = Tensor::<f32>::ones(vec![2, 2])?;

    // Valid operations
    let b = a.mul_scalar(2.0)?;
    let c = b.add_scalar(1.0)?;

    // This should succeed
    assert_eq!(c.sync_and_read(), vec![3.0, 3.0, 3.0, 3.0]);

    println!("✓ Chain operations test passed");
    Ok(())
}
