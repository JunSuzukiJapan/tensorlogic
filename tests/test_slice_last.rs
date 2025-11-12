/// Comprehensive tests for slice_last() operation
///
/// Tests verify mathematical correctness of slice_last along axis 0.
/// Critical for validating GPU implementation vs CPU fallback.
///
/// Tests cover:
/// - 2D tensors: [I, D] -> [D]
/// - 3D tensors: [I, H, D] -> [H, D]
/// - 4D tensors: [B, I, H, D] -> [I, H, D]
/// - F16 and F32 precision
/// - Edge cases: small/large sizes, single element
/// - Known value verification

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};
use half::f16;

// Helper to assert tensors are close
fn assert_close_f32(result: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(result.len(), expected.len(), "Length mismatch");
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        assert!(
            diff < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r, e, diff
        );
    }
}

fn assert_close_f16(result: &[f16], expected: &[f16], epsilon: f32) {
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

#[test]
fn test_slice_last_2d_f32() -> TensorResult<()> {
    // Test 2D tensor: [3, 4] -> [4]
    let device = MetalDevice::new()?;

    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,      // i=0
        5.0, 6.0, 7.0, 8.0,      // i=1
        9.0, 10.0, 11.0, 12.0,   // i=2 (last)
    ];

    let x = Tensor::<f32>::from_vec_gpu(&device, data, vec![3, 4])?;
    let result = x.slice_last(0)?;

    // Verify shape
    assert_eq!(result.shape().dims(), &[4]);

    // Verify data: should be last row [9, 10, 11, 12]
    let result_data = result.sync_and_read();
    let expected = vec![9.0, 10.0, 11.0, 12.0];
    assert_close_f32(&result_data, &expected, 1e-6);

    println!("✓ 2D F32 slice_last test passed");
    Ok(())
}

#[test]
fn test_slice_last_2d_f16() -> TensorResult<()> {
    // Test 2D tensor: [3, 4] -> [4]
    let device = MetalDevice::new()?;

    let data: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0),
        f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0), f16::from_f32(8.0),
        f16::from_f32(9.0), f16::from_f32(10.0), f16::from_f32(11.0), f16::from_f32(12.0),
    ];

    let x = Tensor::<f16>::from_vec_gpu(&device, data, vec![3, 4])?;
    let result = x.slice_last(0)?;

    // Verify shape
    assert_eq!(result.shape().dims(), &[4]);

    // Verify data: should be last row [9, 10, 11, 12]
    let result_data = result.sync_and_read();
    let expected = vec![
        f16::from_f32(9.0), f16::from_f32(10.0),
        f16::from_f32(11.0), f16::from_f32(12.0)
    ];
    assert_close_f16(&result_data, &expected, 1e-3);

    println!("✓ 2D F16 slice_last test passed");
    Ok(())
}

#[test]
fn test_slice_last_3d_f32() -> TensorResult<()> {
    // Test 3D tensor: [2, 3, 4] -> [3, 4]
    let device = MetalDevice::new()?;

    let data: Vec<f32> = vec![
        // i=0
        1.0, 2.0, 3.0, 4.0,       // h=0
        5.0, 6.0, 7.0, 8.0,       // h=1
        9.0, 10.0, 11.0, 12.0,    // h=2
        // i=1 (last)
        13.0, 14.0, 15.0, 16.0,   // h=0
        17.0, 18.0, 19.0, 20.0,   // h=1
        21.0, 22.0, 23.0, 24.0,   // h=2
    ];

    let x = Tensor::<f32>::from_vec_gpu(&device, data, vec![2, 3, 4])?;
    let result = x.slice_last(0)?;

    // Verify shape: [3, 4]
    assert_eq!(result.shape().dims(), &[3, 4]);

    // Verify data: should be last i slice [13..=24]
    let result_data = result.sync_and_read();
    let expected = vec![
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
    ];
    assert_close_f32(&result_data, &expected, 1e-6);

    println!("✓ 3D F32 slice_last test passed");
    Ok(())
}

#[test]
fn test_slice_last_3d_f16() -> TensorResult<()> {
    // Test 3D tensor: [2, 3, 4] -> [3, 4]
    let device = MetalDevice::new()?;

    let data: Vec<f16> = (1..=24)
        .map(|i| f16::from_f32(i as f32))
        .collect();

    let x = Tensor::<f16>::from_vec_gpu(&device, data, vec![2, 3, 4])?;
    let result = x.slice_last(0)?;

    // Verify shape: [3, 4]
    assert_eq!(result.shape().dims(), &[3, 4]);

    // Verify data: should be last i slice [13..=24]
    let result_data = result.sync_and_read();
    let expected: Vec<f16> = (13..=24)
        .map(|i| f16::from_f32(i as f32))
        .collect();
    assert_close_f16(&result_data, &expected, 1e-3);

    println!("✓ 3D F16 slice_last test passed");
    Ok(())
}

#[test]
fn test_slice_last_realistic_sizes() -> TensorResult<()> {
    // Test with realistic transformer sizes
    // Simulates: [seq_len=35, heads=22, head_dim=64] -> [heads=22, head_dim=64]
    let device = MetalDevice::new()?;

    let seq_len = 35;
    let heads = 22;
    let head_dim = 64;

    // Create data where each position has a unique pattern
    let data: Vec<f16> = (0..(seq_len * heads * head_dim))
        .map(|i| f16::from_f32((i % 100) as f32 * 0.01))
        .collect();

    let x = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![seq_len, heads, head_dim])?;
    let result = x.slice_last(0)?;

    // Verify shape: [heads, head_dim]
    assert_eq!(result.shape().dims(), &[heads, head_dim]);

    // Verify we got the last seq_len slice
    let result_data = result.sync_and_read();
    let expected_start = (seq_len - 1) * heads * head_dim;
    let expected: Vec<f16> = data[expected_start..].to_vec();

    assert_eq!(result_data.len(), expected.len());
    assert_close_f16(&result_data, &expected, 1e-3);

    println!("✓ Realistic transformer size test passed");
    Ok(())
}

#[test]
fn test_slice_last_small_tensor() -> TensorResult<()> {
    // Test edge case: [2, 1] -> [1]
    let device = MetalDevice::new()?;

    let data = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let x = Tensor::<f16>::from_vec_gpu(&device, data, vec![2, 1])?;
    let result = x.slice_last(0)?;

    assert_eq!(result.shape().dims(), &[1]);
    let result_data = result.sync_and_read();
    assert_close_f16(&result_data, &[f16::from_f32(2.0)], 1e-3);

    println!("✓ Small tensor test passed");
    Ok(())
}

#[test]
fn test_slice_last_large_last_dim() -> TensorResult<()> {
    // Test with large last dimension: [5, 2048] -> [2048]
    let device = MetalDevice::new()?;

    let i_dim = 5;
    let d_dim = 2048;

    let data: Vec<f16> = (0..(i_dim * d_dim))
        .map(|idx| f16::from_f32((idx % 1000) as f32 * 0.001))
        .collect();

    let x = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![i_dim, d_dim])?;
    let result = x.slice_last(0)?;

    // Verify shape
    assert_eq!(result.shape().dims(), &[d_dim]);

    // Verify we got the last row
    let result_data = result.sync_and_read();
    let expected_start = (i_dim - 1) * d_dim;
    let expected: Vec<f16> = data[expected_start..].to_vec();

    assert_eq!(result_data.len(), expected.len());
    // Check first and last few elements
    for i in 0..10 {
        let diff = (result_data[i].to_f32() - expected[i].to_f32()).abs();
        assert!(diff < 1e-3, "Mismatch at start index {}", i);
    }
    for i in (d_dim - 10)..d_dim {
        let diff = (result_data[i].to_f32() - expected[i].to_f32()).abs();
        assert!(diff < 1e-3, "Mismatch at end index {}", i);
    }

    println!("✓ Large last dimension test passed");
    Ok(())
}

#[test]
fn test_slice_last_sequential_values() -> TensorResult<()> {
    // Test with sequential values to verify no data corruption
    // [10, 8] -> [8]
    let device = MetalDevice::new()?;

    let data: Vec<f32> = (0..80).map(|i| i as f32).collect();
    let x = Tensor::<f32>::from_vec_gpu(&device, data, vec![10, 8])?;
    let result = x.slice_last(0)?;

    assert_eq!(result.shape().dims(), &[8]);

    // Last row should be [72, 73, 74, 75, 76, 77, 78, 79]
    let result_data = result.sync_and_read();
    let expected: Vec<f32> = (72..80).map(|i| i as f32).collect();
    assert_close_f32(&result_data, &expected, 1e-6);

    println!("✓ Sequential values test passed");
    Ok(())
}

#[test]
fn test_slice_last_4d_tensor() -> TensorResult<()> {
    // Test 4D tensor: [2, 3, 4, 5] -> [3, 4, 5]
    let device = MetalDevice::new()?;

    let b = 2;
    let i = 3;
    let h = 4;
    let d = 5;
    let total = b * i * h * d;

    let data: Vec<f16> = (0..total)
        .map(|idx| f16::from_f32((idx % 100) as f32 * 0.1))
        .collect();

    let x = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![b, i, h, d])?;
    let result = x.slice_last(0)?;

    // Verify shape: [3, 4, 5]
    assert_eq!(result.shape().dims(), &[i, h, d]);

    // Verify we got the last b slice
    let result_data = result.sync_and_read();
    let expected_start = (b - 1) * i * h * d;
    let expected: Vec<f16> = data[expected_start..].to_vec();

    assert_eq!(result_data.len(), expected.len());
    assert_close_f16(&result_data, &expected, 1e-2);

    println!("✓ 4D tensor test passed");
    Ok(())
}

#[test]
fn test_slice_last_alternating_pattern() -> TensorResult<()> {
    // Test with alternating values to detect any pattern-based bugs
    let device = MetalDevice::new()?;

    let data: Vec<f16> = vec![
        // i=0
        f16::from_f32(1.0), f16::from_f32(-1.0), f16::from_f32(1.0), f16::from_f32(-1.0),
        // i=1
        f16::from_f32(2.0), f16::from_f32(-2.0), f16::from_f32(2.0), f16::from_f32(-2.0),
        // i=2 (last)
        f16::from_f32(3.0), f16::from_f32(-3.0), f16::from_f32(3.0), f16::from_f32(-3.0),
    ];

    let x = Tensor::<f16>::from_vec_gpu(&device, data, vec![3, 4])?;
    let result = x.slice_last(0)?;

    let result_data = result.sync_and_read();
    let expected = vec![
        f16::from_f32(3.0), f16::from_f32(-3.0),
        f16::from_f32(3.0), f16::from_f32(-3.0)
    ];
    assert_close_f16(&result_data, &expected, 1e-3);

    println!("✓ Alternating pattern test passed");
    Ok(())
}

#[test]
fn test_slice_last_zeros() -> TensorResult<()> {
    // Test with all zeros (edge case)
    let device = MetalDevice::new()?;

    let x = Tensor::<f16>::zeros(&device, vec![5, 10])?;
    let result = x.slice_last(0)?;

    assert_eq!(result.shape().dims(), &[10]);

    let result_data = result.sync_and_read();
    for &val in &result_data {
        assert_eq!(val.to_f32(), 0.0, "Expected all zeros");
    }

    println!("✓ All zeros test passed");
    Ok(())
}

#[test]
fn test_slice_last_ones() -> TensorResult<()> {
    // Test with all ones
    let device = MetalDevice::new()?;

    let x = Tensor::<f32>::ones(&device, vec![7, 12])?;
    let result = x.slice_last(0)?;

    assert_eq!(result.shape().dims(), &[12]);

    let result_data = result.sync_and_read();
    for &val in &result_data {
        assert!((val - 1.0).abs() < 1e-6, "Expected all ones, got {}", val);
    }

    println!("✓ All ones test passed");
    Ok(())
}
