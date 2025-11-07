//! Comprehensive GPU Operations Test Suite
//!
//! Tests all GPU operations to ensure correctness, memory safety, and performance.
//! Each test validates a specific GPU operation under various conditions.
//!
//! NOTE: All tests in this file use #[serial] to avoid GPU resource contention
//! and prevent deadlocks when running tests in parallel.

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorAccessors, TensorIO};
use half::f16;
use serial_test::serial;

/// Tolerance for floating point comparisons
const EPSILON_F32: f32 = 1e-4;
const EPSILON_F16: f32 = 1e-2; // f16 has lower precision

/// Helper function to create f32 tensor from Vec
fn tensor_f32(device: &MetalDevice, data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::<f32>::from_vec_gpu(device, data, shape).expect("Failed to create tensor")
}

/// Helper function to create f16 tensor from Vec<f32>
fn tensor_f16(device: &MetalDevice, data: Vec<f32>, shape: Vec<usize>) -> Tensor<f16> {
    let f16_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    Tensor::<f16>::from_vec_gpu(device, f16_data, shape).expect("Failed to create tensor")
}

/// Helper function to compare two f32 slices with tolerance
fn assert_close_f32(a: &[f32], b: &[f32], epsilon: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: Length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < epsilon,
            "{}: Mismatch at index {}: {} vs {} (diff: {})",
            msg, i, x, y, diff
        );
    }
}

/// Helper function to compare f16 tensors
fn assert_close_f16(a: &[f16], b: &[f16], epsilon: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: Length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x.to_f32() - y.to_f32()).abs();
        assert!(
            diff < epsilon,
            "{}: Mismatch at index {}: {} vs {} (diff: {})",
            msg, i, x, y, diff
        );
    }
}

// ============================================================================
// MATMUL OPERATIONS
// ============================================================================

#[test]
#[serial]
fn test_matmul_basic_2x2_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // A = [[1, 2],    B = [[5, 6],
    //      [3, 4]]         [7, 8]]
    // Expected: A @ B = [[19, 22], [43, 50]]

    let a = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = tensor_f32(&device, vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = a.matmul(&b).expect("matmul failed");

    let expected = vec![19.0, 22.0, 43.0, 50.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "matmul_basic_2x2");
}

#[test]
#[serial]
fn test_matmul_basic_2x2_f16() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let a = tensor_f16(&device, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = tensor_f16(&device, vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = a.matmul(&b).expect("matmul failed");

    let expected: Vec<f16> = vec![19.0, 22.0, 43.0, 50.0].iter().map(|&x| f16::from_f32(x)).collect();
    let actual = result.sync_and_read();
    assert_close_f16(&actual, &expected, EPSILON_F16, "matmul_basic_2x2_f16");
}

#[test]
#[serial]
fn test_matmul_rectangular_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // A: 3x2, B: 2x4 -> Result: 3x4
    let a = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let b = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);

    let result = a.matmul(&b).expect("matmul failed");

    assert_eq!(result.dims(), &[3, 4], "matmul shape mismatch");

    // Verify first row: [1*1 + 2*5, 1*2 + 2*6, 1*3 + 2*7, 1*4 + 2*8] = [11, 14, 17, 20]
    let actual = result.sync_and_read();
    assert_eq!(actual.len(), 12, "Result should have 12 elements");
    assert!((actual[0] - 11.0).abs() < EPSILON_F32);
    assert!((actual[1] - 14.0).abs() < EPSILON_F32);
}

#[test]
#[serial]
fn test_matmul_identity_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // A @ I = A
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let a = tensor_f32(&device, a_data.clone(), vec![2, 2]);
    let i = tensor_f32(&device, vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // Identity

    let result = a.matmul(&i).expect("matmul with identity failed");

    let actual = result.sync_and_read();
    assert_close_f32(&actual, &a_data, EPSILON_F32, "matmul_identity");
}

// ============================================================================
// SOFTMAX OPERATIONS
// ============================================================================

#[test]
#[serial]
fn test_softmax_1d_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![1.0, 2.0, 3.0], vec![3]);
    let result = input.softmax().expect("softmax failed");

    let actual = result.sync_and_read();

    // Verify sum equals 1.0
    let sum: f32 = actual.iter().sum();
    assert!((sum - 1.0).abs() < EPSILON_F32, "Softmax sum should be 1.0, got {}", sum);

    // Verify values are in [0, 1]
    for &val in actual.iter() {
        assert!(val >= 0.0 && val <= 1.0, "Softmax value out of range: {}", val);
    }

    // Verify monotonicity (larger input -> larger output)
    assert!(actual[0] < actual[1] && actual[1] < actual[2], "Softmax should be monotonic");
}

#[test]
#[serial]
fn test_softmax_numerical_stability_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Large values that could cause overflow without proper implementation
    let input = tensor_f32(&device, vec![100.0, 200.0, 300.0], vec![3]);
    let result = input.softmax().expect("softmax failed");

    let actual = result.sync_and_read();

    // Should not have NaN or Inf
    for &val in actual.iter() {
        assert!(val.is_finite(), "Softmax produced non-finite value: {}", val);
    }

    // Sum should still be 1.0
    let sum: f32 = actual.iter().sum();
    assert!((sum - 1.0).abs() < EPSILON_F32, "Softmax sum should be 1.0 even with large values");
}

// ============================================================================
// RMS NORMALIZATION
// ============================================================================

#[test]
#[serial]
fn test_rms_norm_basic_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Input: [1, 2, 3, 4]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = tensor_f32(&device, input.clone(), vec![4]);
    let weight = Tensor::<f32>::ones(&device, vec![4]).expect("Failed to create weight");

    let result = tensor.rms_norm(vec![4], &weight, 1e-5).expect("rms_norm failed");

    let actual = result.sync_and_read();

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();

    // Normalized values should be input / rms
    for (i, &val) in actual.iter().enumerate() {
        let expected = input[i] / rms;
        assert!((val - expected).abs() < 1e-3, "RMS norm mismatch at {}: {} vs {}", i, val, expected);
    }
}

// ============================================================================
// SCALAR OPERATIONS
// ============================================================================

#[test]
#[serial]
fn test_div_scalar_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![2.0, 4.0, 6.0, 8.0], vec![4]);
    let result = input.div_scalar(2.0).expect("div_scalar failed");

    let expected = vec![1.0, 2.0, 3.0, 4.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "div_scalar");
}

#[test]
#[serial]
fn test_mul_scalar_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let result = input.mul_scalar(2.5).expect("mul_scalar failed");

    let expected = vec![2.5, 5.0, 7.5, 10.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "mul_scalar");
}

#[test]
#[serial]
fn test_add_scalar_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let result = input.add_scalar(10.0).expect("add_scalar failed");

    let expected = vec![11.0, 12.0, 13.0, 14.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "add_scalar");
}

// ============================================================================
// ELEMENTWISE OPERATIONS
// ============================================================================

#[test]
#[serial]
fn test_elementwise_add_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let a = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = tensor_f32(&device, vec![10.0, 20.0, 30.0, 40.0], vec![4]);

    let result = a.add(&b).expect("add failed");

    let expected = vec![11.0, 22.0, 33.0, 44.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "elementwise_add");
}

#[test]
#[serial]
fn test_elementwise_mul_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let a = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = tensor_f32(&device, vec![2.0, 3.0, 4.0, 5.0], vec![4]);

    let result = a.mul(&b).expect("mul failed");

    let expected = vec![2.0, 6.0, 12.0, 20.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "elementwise_mul");
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

#[test]
#[serial]
fn test_relu_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let result = input.relu().expect("relu failed");

    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "relu");
}

#[test]
#[serial]
fn test_gelu_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let input = tensor_f32(&device, vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let result = input.gelu().expect("gelu failed");

    let actual = result.sync_and_read();

    // GELU properties:
    // - gelu(0) ≈ 0
    // - gelu(x) ≈ x for large positive x
    // - gelu(x) ≈ 0 for large negative x
    assert!(actual[2].abs() < 0.1, "GELU(0) should be near 0");
    assert!(actual[4] > 1.5, "GELU(2) should be close to 2");
    assert!(actual[0].abs() < 0.1, "GELU(-2) should be near 0");
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[test]
#[serial]
fn test_large_matmul_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // 100x100 @ 100x100
    let size = 100;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();

    let a = tensor_f32(&device, a_data, vec![size, size]);
    let b = tensor_f32(&device, b_data, vec![size, size]);

    let result = a.matmul(&b).expect("large matmul failed");

    assert_eq!(result.dims(), &[size, size], "Large matmul shape mismatch");
}

#[test]
#[serial]
fn test_buffer_reuse() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Create and destroy multiple tensors to test buffer pool
    for _ in 0..10 {
    let device = MetalDevice::new()?;
        let tensor = tensor_f32(&device, vec![1.0; 1000], vec![1000]);
        let result = tensor.mul_scalar(2.0).expect("mul_scalar failed");
        drop(result);
    }

    // If we get here without hanging, buffer pool is working
}

#[test]
#[serial]
fn test_concurrent_operations_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Execute multiple operations in sequence
    let tensor = tensor_f32(&device, vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let r1 = tensor.mul_scalar(2.0).expect("op1 failed");
    let r2 = r1.add(&tensor).expect("op2 failed");
    let r3 = r2.div_scalar(3.0).expect("op3 failed");

    // Verify final result is valid
    let actual = r3.sync_and_read();
    for &val in actual.iter() {
        assert!(val.is_finite(), "Concurrent ops produced non-finite value");
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

#[test]
#[serial]
fn test_zero_tensor_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let zeros = Tensor::<f32>::zeros(&device, vec![4]).expect("Failed to create zeros");
    let result = zeros.mul_scalar(5.0).expect("zero tensor mul failed");

    let expected = vec![0.0; 4];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "zero_tensor");
}

#[test]
#[serial]
fn test_single_element_f32() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    let single = tensor_f32(&device, vec![42.0], vec![1]);
    let result = single.div_scalar(2.0).expect("single element failed");

    let expected = vec![21.0];
    let actual = result.sync_and_read();
    assert_close_f32(&actual, &expected, EPSILON_F32, "single_element");
}

// ============================================================================
// MEMORY SAFETY
// ============================================================================

#[test]
#[serial]
fn test_f16_buffer_recycling() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Create and drop many f16 tensors to test recycling
    for _ in 0..20 {
    let device = MetalDevice::new()?;
        let t = tensor_f16(&device, vec![1.0; 500], vec![500]);
        let r = t.mul_scalar(f16::from_f32(2.0)).expect("mul failed");
        drop(r);
    }
}

#[test]
#[serial]
fn test_f32_buffer_recycling() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Create and drop many f32 tensors to test recycling
    for _ in 0..20 {
    let device = MetalDevice::new()?;
        let t = tensor_f32(&device, vec![1.0; 500], vec![500]);
        let r = t.mul_scalar(2.0).expect("mul failed");
        drop(r);
    }
}
