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
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
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

// Tanh Tests

#[test]
#[serial]
fn test_f16_tanh_basic() -> TensorResult<()> {
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
