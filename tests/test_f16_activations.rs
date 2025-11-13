#![allow(unused_variables)]
/// Comprehensive tests for f16 activation functions
///
/// Activation functions are critical for neural networks.
/// F16 tests ensure that activations work correctly with half precision.
///
/// Tests cover:
/// - ReLU and variants (LeakyReLU)
/// - GELU
/// - Sigmoid and Tanh
/// - Softmax
/// - Numerical stability with f16 precision
/// - Edge cases (large/small values, negatives, zeros)

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};
use half::f16;
use serial_test::serial;

// Helper function
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

// ReLU Tests

#[test]
#[serial]
fn test_f16_relu_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-2.0), f16::from_f32(-1.0), f16::from_f32(0.0),
            f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
        ],
        vec![6]
    )?;

    let b = a.relu()?;
    let result = b.sync_and_read();

    let expected = vec![
        f16::ZERO, f16::ZERO, f16::ZERO,
        f16::ONE, f16::from_f32(2.0), f16::from_f32(3.0),
    ];
    assert_tensor_close_f16(&result, &expected, 1e-3);

    println!("✓ f16 relu basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_relu_all_negative() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(-1.0), f16::from_f32(-5.0), f16::from_f32(-10.0), f16::from_f32(-100.0)],
        vec![4]
    )?;

    let b = a.relu()?;
    let result = b.sync_and_read();

    // All should be zero
    for &val in &result {
        assert_eq!(val, f16::ZERO);
    }

    println!("✓ f16 relu all negative test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_relu_all_positive() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(5.0), f16::from_f32(10.0), f16::from_f32(100.0)],
        vec![4]
    )?;

    let b = a.relu()?;
    let result = b.sync_and_read();

    // Should be unchanged
    assert_tensor_close_f16(
        &result,
        &vec![f16::from_f32(1.0), f16::from_f32(5.0), f16::from_f32(10.0), f16::from_f32(100.0)],
        1e-2
    );

    println!("✓ f16 relu all positive test passed");
    Ok(())
}

// GELU Tests

#[test]
#[serial]
fn test_f16_gelu_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-3.0), f16::from_f32(-1.0), f16::from_f32(0.0),
            f16::from_f32(1.0), f16::from_f32(3.0),
        ],
        vec![5]
    )?;

    let b = a.gelu()?;
    let result = b.sync_and_read();

    // GELU properties:
    // GELU(0) ≈ 0
    // GELU(x) ≈ x for large positive x
    // GELU(x) ≈ 0 for large negative x

    assert!(result[2].to_f32().abs() < 0.1, "GELU(0) should be ~0");
    assert!(result[4].to_f32() > 2.5, "GELU(3) should be close to 3");
    assert!(result[0].to_f32().abs() < 0.1, "GELU(-3) should be ~0");

    println!("✓ f16 gelu basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_gelu_smooth() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // GELU should be smooth (continuous and differentiable)
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(-0.1), f16::from_f32(0.0), f16::from_f32(0.1)],
        vec![3]
    )?;

    let b = a.gelu()?;
    let result = b.sync_and_read();

    // Should transition smoothly through zero
    assert!(result[0].to_f32() < 0.0);
    assert!(result[1].to_f32().abs() < 0.1);
    assert!(result[2].to_f32() > 0.0);

    println!("✓ f16 gelu smooth test passed");
    Ok(())
}

// Sigmoid Tests

#[test]
#[serial]
fn test_f16_sigmoid_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-10.0), f16::from_f32(-1.0), f16::from_f32(0.0),
            f16::from_f32(1.0), f16::from_f32(10.0),
        ],
        vec![5]
    )?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // Sigmoid properties:
    // sigmoid(0) = 0.5
    // sigmoid(x) → 0 as x → -∞
    // sigmoid(x) → 1 as x → +∞

    assert!(result[2].to_f32() > 0.49 && result[2].to_f32() < 0.51, "Sigmoid(0) should be 0.5");
    assert!(result[0].to_f32() < 0.01, "Sigmoid(-10) should be ~0");
    assert!(result[4].to_f32() > 0.99, "Sigmoid(10) should be ~1");

    println!("✓ f16 sigmoid basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_range() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Sigmoid output should always be in (0, 1)
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-1000.0), f16::from_f32(-100.0), f16::from_f32(-10.0),
            f16::from_f32(0.0), f16::from_f32(10.0), f16::from_f32(100.0), f16::from_f32(1000.0),
        ],
        vec![7]
    )?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // All values should be in (0, 1)
    for &val in &result {
        assert!(val.to_f32() >= 0.0 && val.to_f32() <= 1.0, "Sigmoid should be in [0, 1]");
    }

    println!("✓ f16 sigmoid range test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_extreme_negative() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test very large negative values like those seen in FFN gates
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-10000.0), f16::from_f32(-9840.0), f16::from_f32(-6208.0),
            f16::from_f32(-5072.0), f16::from_f32(-3000.0),
        ],
        vec![5]
    )?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // All values should be very close to 0 but finite
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Sigmoid of large negative value at {} should be finite, got {:?}", i, val);
        assert!(val.to_f32() >= 0.0 && val.to_f32() <= 1.0, "Sigmoid at {} should be in [0, 1]", i);
        // For very large negative values, sigmoid should be essentially 0
        assert!(val.to_f32() < 0.001, "Sigmoid of large negative value at {} should be ~0, got {}", i, val.to_f32());
    }

    // Test sum doesn't overflow to inf
    let sum: f32 = result.iter().map(|&x| x.to_f32()).sum();
    assert!(sum.is_finite(), "Sum of sigmoid outputs should be finite");

    println!("✓ f16 sigmoid extreme negative test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_sum_stability() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test that sum() doesn't produce inf on sigmoid outputs
    let device = MetalDevice::new()?;

    // Create a large tensor with FFN-like negative values
    let size = 100;
    let values: Vec<f16> = (0..size)
        .map(|i| f16::from_f32(-5000.0 - (i as f32) * 10.0))
        .collect();

    let a = Tensor::<f16>::from_vec(values, vec![size])?;
    let b = a.sigmoid()?;

    // Use Tensor::sum() method
    let sum_result = b.sum()?;

    assert!(sum_result.is_finite(), "Tensor::sum() of sigmoid should be finite, got {:?}", sum_result);
    assert!(sum_result.to_f32() >= 0.0, "Sum should be non-negative");

    println!("✓ f16 sigmoid sum stability test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_ffn_exact_values() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test exact values observed in FFN: -6208, -5072, -9840
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-6208.0),
            f16::from_f32(-5072.0),
            f16::from_f32(-9840.0),
        ],
        vec![3]
    )?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // Each individual value must be finite
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "sigmoid at index {} should be finite, got {:?}", i, val);
        assert!(val.to_f32() >= 0.0 && val.to_f32() <= 1.0, "sigmoid at {} should be in [0,1]", i);
        assert!(val.to_f32() < 1e-6, "sigmoid of large negative should be ~0, got {}", val.to_f32());
    }

    // Sum via CPU should be finite
    let cpu_sum: f32 = result.iter().map(|&x| x.to_f32()).sum();
    assert!(cpu_sum.is_finite(), "CPU sum should be finite, got {}", cpu_sum);

    // Tensor::sum() via GPU should also be finite
    let gpu_sum = b.sum()?;
    assert!(gpu_sum.is_finite(), "GPU sum should be finite, got {:?}", gpu_sum);

    println!("✓ f16 sigmoid FFN exact values test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_large_tensor_ffn_size() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test with FFN-like tensor dimensions: [46, 5632]
    let device = MetalDevice::new()?;

    let rows = 46;
    let cols = 100; // Use smaller cols for faster test, but still large
    let size = rows * cols;

    // Fill with large negative values similar to FFN gates
    let values: Vec<f16> = (0..size)
        .map(|i| {
            let base = -6000.0;
            let variation = (i % 100) as f32 * 10.0;
            f16::from_f32(base - variation)
        })
        .collect();

    let a = Tensor::<f16>::from_vec(values, vec![rows, cols])?;
    let b = a.sigmoid()?;

    // Test Tensor::sum() on large tensor
    let sum_result = b.sum()?;
    assert!(sum_result.is_finite(), "Sum of large sigmoid tensor should be finite, got {:?}", sum_result);
    assert!(sum_result.to_f32() >= 0.0, "Sum should be non-negative");
    assert!(sum_result.to_f32() < 1.0, "Sum of near-zero values should be close to 0");

    // Verify a few individual values
    let result = b.sync_and_read();
    for i in (0..size).step_by(size / 10) {
        assert!(result[i].is_finite(), "Value at {} should be finite", i);
        assert!(result[i].to_f32() >= 0.0 && result[i].to_f32() <= 1.0);
    }

    println!("✓ f16 sigmoid large tensor FFN size test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_sigmoid_mixed_large_values() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test mix of large positive and negative values
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-10000.0), f16::from_f32(-6208.0), f16::from_f32(-1000.0),
            f16::from_f32(0.0),
            f16::from_f32(1000.0), f16::from_f32(6208.0), f16::from_f32(10000.0),
        ],
        vec![7]
    )?;

    let b = a.sigmoid()?;
    let result = b.sync_and_read();

    // Negative large values → ~0
    assert!(result[0].to_f32() < 1e-6, "sigmoid(-10000) should be ~0");
    assert!(result[1].to_f32() < 1e-6, "sigmoid(-6208) should be ~0");
    assert!(result[2].to_f32() < 0.001, "sigmoid(-1000) should be ~0");

    // Zero → 0.5
    assert!((result[3].to_f32() - 0.5).abs() < 0.01, "sigmoid(0) should be ~0.5");

    // Positive large values → ~1
    assert!(result[4].to_f32() > 0.999, "sigmoid(1000) should be ~1");
    assert!(result[5].to_f32() > 0.999, "sigmoid(6208) should be ~1");
    assert!(result[6].to_f32() > 0.999, "sigmoid(10000) should be ~1");

    // All should be finite
    for &val in &result {
        assert!(val.is_finite(), "All values should be finite");
    }

    // Sum should be finite
    let sum_result = b.sum()?;
    assert!(sum_result.is_finite(), "Sum should be finite");

    println!("✓ f16 sigmoid mixed large values test passed");
    Ok(())
}

// Tanh Tests

#[test]
#[serial]
fn test_f16_tanh_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-10.0), f16::from_f32(-1.0), f16::from_f32(0.0),
            f16::from_f32(1.0), f16::from_f32(10.0),
        ],
        vec![5]
    )?;

    let b = a.tanh()?;
    let result = b.sync_and_read();

    // Tanh properties:
    // tanh(0) = 0
    // tanh(x) → -1 as x → -∞
    // tanh(x) → 1 as x → +∞

    assert!(result[2].to_f32().abs() < 0.01, "Tanh(0) should be 0");
    assert!(result[0].to_f32() < -0.99, "Tanh(-10) should be ~-1");
    assert!(result[4].to_f32() > 0.99, "Tanh(10) should be ~1");

    println!("✓ f16 tanh basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_tanh_range() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Tanh output should always be in (-1, 1)
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-100.0), f16::from_f32(-10.0), f16::from_f32(-1.0),
            f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(10.0), f16::from_f32(100.0),
        ],
        vec![7]
    )?;

    let b = a.tanh()?;
    let result = b.sync_and_read();

    // All values should be in (-1, 1)
    for &val in &result {
        assert!(val.to_f32() >= -1.0 && val.to_f32() <= 1.0, "Tanh should be in [-1, 1]");
    }

    println!("✓ f16 tanh range test passed");
    Ok(())
}

// Softmax Tests

#[test]
#[serial]
fn test_f16_softmax_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![1, 3]
    )?;

    let b = a.softmax()?; // dim=1
    let result = b.sync_and_read();

    // Softmax properties:
    // 1. All values should be positive
    // 2. Sum should be 1.0
    // 3. Larger inputs should have larger outputs

    for &val in &result {
        assert!(val.to_f32() > 0.0, "Softmax should be positive");
    }

    let sum: f32 = result.iter().map(|&x| x.to_f32()).sum();
    assert!((sum - 1.0).abs() < 0.01, "Softmax should sum to 1");

    assert!(result[2].to_f32() > result[1].to_f32());
    assert!(result[1].to_f32() > result[0].to_f32());

    println!("✓ f16 softmax basic test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_softmax_uniform() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Softmax of uniform values should give uniform distribution
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::ONE, f16::ONE, f16::ONE, f16::ONE],
        vec![1, 4]
    )?;

    let b = a.softmax()?;
    let result = b.sync_and_read();

    // Each should be approximately 0.25
    for &val in &result {
        assert!((val.to_f32() - 0.25).abs() < 0.01, "Uniform softmax should be 0.25 each");
    }

    println!("✓ f16 softmax uniform test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_softmax_overflow_safety() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Softmax should handle large values without overflow
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1000.0), f16::from_f32(1001.0), f16::from_f32(1002.0)],
        vec![1, 3]
    )?;

    let b = a.softmax()?;
    let result = b.sync_and_read();

    // Should not be NaN or Inf
    for &val in &result {
        assert!(val.is_finite(), "Softmax should be finite even with large inputs");
    }

    // Should sum to 1
    let sum: f32 = result.iter().map(|&x| x.to_f32()).sum();
    assert!((sum - 1.0).abs() < 0.01, "Softmax should sum to 1");

    println!("✓ f16 softmax overflow safety test passed");
    Ok(())
}

// LeakyReLU Tests

#[test]
#[serial]
#[ignore] // TODO: leaky_relu() not yet implemented
fn test_f16_leaky_relu() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-2.0), f16::from_f32(-1.0), f16::from_f32(0.0),
            f16::from_f32(1.0), f16::from_f32(2.0),
        ],
        vec![5]
    )?;

    let _negative_slope = 0.01;
    // let b = a.leaky_relu(negative_slope)?;
    // let result = b.sync_and_read();

    // // LeakyReLU(x) = x if x > 0, else negative_slope * x
    // assert!((result[0].to_f32() - (-2.0 * negative_slope)).abs() < 0.01);
    // assert!((result[1].to_f32() - (-1.0 * negative_slope)).abs() < 0.01);
    // assert!(result[2].to_f32().abs() < 0.01); // 0
    // assert!((result[3].to_f32() - 1.0).abs() < 0.01);
    // assert!((result[4].to_f32() - 2.0).abs() < 0.01);

    // println!("✓ f16 leaky relu test passed");
    Ok(())
}

// Comparison with F32

#[test]
#[serial]
fn test_f16_f32_activation_comparison() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Compare f16 and f32 activation outputs
    let device = MetalDevice::new()?;

    let data_f32 = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let a_f32 = Tensor::<f32>::from_vec(data_f32, vec![5])?;
    let a_f16 = Tensor::<f16>::from_vec(data_f16, vec![5])?;

    // Test ReLU
    let relu_f32 = a_f32.relu()?;
    let relu_f16 = a_f16.relu()?;

    let result_f32 = relu_f32.sync_and_read();
    let result_f16: Vec<f32> = relu_f16.sync_and_read().iter().map(|&x| x.to_f32()).collect();

    for (i, (&r32, &r16)) in result_f32.iter().zip(result_f16.iter()).enumerate() {
        let diff = (r32 - r16).abs();
        assert!(diff < 0.01, "F16 and F32 ReLU should be very close");
    }

    println!("✓ f16 vs f32 activation comparison test passed");
    Ok(())
}

// Edge Cases

#[test]
#[serial]
fn test_f16_activation_with_zeros() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let zeros = Tensor::<f16>::zeros(&device, vec![4])?;

    // ReLU of zeros should be zeros
    let relu_result = zeros.relu()?;
    for &val in &relu_result.sync_and_read() {
        assert_eq!(val, f16::ZERO);
    }

    // Sigmoid of zeros should be 0.5
    let sigmoid_result = zeros.sigmoid()?;
    for &val in &sigmoid_result.sync_and_read() {
        assert!((val.to_f32() - 0.5).abs() < 0.01);
    }

    // Tanh of zeros should be zeros
    let tanh_result = zeros.tanh()?;
    for &val in &tanh_result.sync_and_read() {
        assert!(val.to_f32().abs() < 0.01);
    }

    println!("✓ f16 activation with zeros test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_activation_numerical_stability() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test activations with values that might cause numerical issues
    let device = MetalDevice::new()?;

    // Very large positive and negative values
    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(-1000.0), f16::from_f32(-100.0), f16::from_f32(100.0), f16::from_f32(1000.0)],
        vec![4]
    )?;

    // ReLU should handle large values
    let relu_result = a.relu()?;
    for &val in &relu_result.sync_and_read() {
        assert!(val.is_finite(), "ReLU should produce finite values");
    }

    // Sigmoid should saturate gracefully
    let sigmoid_result = a.sigmoid()?;
    for &val in &sigmoid_result.sync_and_read() {
        assert!(val.is_finite(), "Sigmoid should produce finite values");
        assert!(val.to_f32() >= 0.0 && val.to_f32() <= 1.0);
    }

    println!("✓ f16 activation numerical stability test passed");
    Ok(())
}

// Multi-dimensional Tests

#[test]
#[serial]
fn test_f16_activation_2d() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;

    let a = Tensor::<f16>::from_vec(
        vec![
            f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(1.0),
            f16::from_f32(-2.0), f16::from_f32(0.0), f16::from_f32(2.0),
        ],
        vec![2, 3]
    )?;

    let b = a.relu()?;
    let result = b.sync_and_read();

    assert_eq!(result.len(), 6);
    assert_eq!(b.shape().dims(), &[2, 3]);

    println!("✓ f16 activation 2D test passed");
    Ok(())
}

#[test]
#[serial]
fn test_f16_activation_batched() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test activation on batched data
    let device = MetalDevice::new()?;

    let batch_size = 4;
    let features = 8;

    let a = Tensor::<f16>::ones(&device, vec![batch_size, features])?;

    let b = a.relu()?;
    assert_eq!(b.shape().dims(), &[batch_size, features]);

    let c = a.sigmoid()?;
    assert_eq!(c.shape().dims(), &[batch_size, features]);

    println!("✓ f16 activation batched test passed");
    Ok(())
}

// Sum Reduction Tests for Large Tensors

#[test]
#[serial]
fn test_f16_sum_simple() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test 1: 1D tensor
    let a1d = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![4]
    )?;
    let sum1d = a1d.sum()?;
    assert!((sum1d.to_f32() - 10.0).abs() < 0.1);

    // Test 2: 2D tensor [2, 2]
    let a2d = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;
    let sum2d = a2d.sum()?;
    assert!((sum2d.to_f32() - 10.0).abs() < 0.1);

    // Test 3: 2D tensor [1, 4]
    let a1x4 = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![1, 4]
    )?;
    let sum1x4 = a1x4.sum()?;
    assert!((sum1x4.to_f32() - 10.0).abs() < 0.1);

    // Test 4: 2D tensor [4, 1]
    let a4x1 = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![4, 1]
    )?;
    let sum4x1 = a4x1.sum()?;
    assert!((sum4x1.to_f32() - 10.0).abs() < 0.1);
    Ok(())
}

#[test]
#[serial]
fn test_f16_sum_large_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test with FFN-sized tensor: [46, 5632] = 259,072 elements
    // This is the exact size that caused overflow in chat demo
    let rows = 46;
    let cols = 5632;
    let size = rows * cols;

    // Create tensor with small positive values (similar to sigmoid outputs)
    // Note: Use small values (0.15 avg) so sum doesn't exceed f16 max (65,504)
    // 259,072 × 0.15 ≈ 38,861 which is within f16 range
    // This simulates sigmoid(large negative) outputs which are close to 0
    let values: Vec<f16> = (0..size)
        .map(|i| f16::from_f32(0.1 + (i % 100) as f32 * 0.001))
        .collect();

    let a = Tensor::<f16>::from_vec(values.clone(), vec![rows, cols])?;
    let gpu_sum = a.sum()?;

    // Calculate expected sum on CPU with f32 precision
    let expected_sum: f32 = values.iter().map(|&x| x.to_f32()).sum();

    // The sum should be finite
    assert!(gpu_sum.is_finite(), "GPU sum should be finite, got {:?}", gpu_sum);

    // The sum should be positive
    assert!(gpu_sum > f16::ZERO, "GPU sum should be positive, got {:?}", gpu_sum);

    // The sum should be reasonable (within 1% of expected)
    // f32 accumulation should be very accurate
    let gpu_f32 = gpu_sum.to_f32();
    let error_pct = ((gpu_f32 - expected_sum).abs() / expected_sum) * 100.0;
    assert!(error_pct < 1.0, "Sum error too large: {:.2}%", error_pct);
    Ok(())
}

#[test]
#[serial]
fn test_f16_sum_after_sigmoid_large() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test the exact scenario that fails: sigmoid then sum on large tensor
    let rows = 46;
    let cols = 5632;
    let size = rows * cols;

    println!("Testing sigmoid→sum on large tensor [{}, {}]", rows, cols);

    // Create large negative values (simulating FFN gate output)
    let values: Vec<f16> = (0..size)
        .map(|i| {
            let base = -6000.0;
            let variation = (i % 100) as f32 * 10.0;
            f16::from_f32(base - variation)
        })
        .collect();

    let a = Tensor::<f16>::from_vec(values, vec![rows, cols])?;

    // Apply sigmoid
    let sig = a.sigmoid()?;

    // Sum the result
    let gpu_sum = sig.sum()?;

    println!("Sigmoid sum: {:?}", gpu_sum);

    // The sum should be finite (not inf)
    assert!(gpu_sum.is_finite(), "Sigmoid sum should be finite, got {:?}", gpu_sum);

    // For large negative values, sigmoid ≈ 0, so sum should be small but positive
    assert!(gpu_sum >= f16::ZERO, "Sigmoid sum should be non-negative");

    // With 258,872 elements each ≈ 0, sum should be < 1000
    assert!(gpu_sum.to_f32() < 1000.0, "Sigmoid sum should be small for large negative inputs, got {}", gpu_sum.to_f32());

    println!("✓ f16 sigmoid→sum large tensor test passed");
    Ok(())
}
