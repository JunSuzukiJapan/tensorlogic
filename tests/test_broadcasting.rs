//! Comprehensive tests for broadcasting operations
//! Tests broadcast_to with various shape combinations and edge cases

use half::f16;
use tensorlogic::prelude::*;

// ============================================================================
// Basic Broadcasting Tests
// ============================================================================

#[test]
fn test_broadcast_scalar_to_vector() -> TensorResult<()> {
    // [1] -> [5]
    let a = Tensor::from_vec(vec![f16::from_f32(42.0)], vec![1])?;
    let target_shape = TensorShape::new(vec![5]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[5]);
    let values = b.sync_and_read();
    for val in values {
        assert!((val.to_f32() - 42.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_1d_to_2d() -> TensorResult<()> {
    // [3] -> [4, 3]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ],
        vec![3],
    )?;

    let target_shape = TensorShape::new(vec![4, 3]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[4, 3]);
    let values = b.sync_and_read();

    // Each row should be [1, 2, 3]
    for row in 0..4 {
        assert!((values[row * 3].to_f32() - 1.0).abs() < 1e-3);
        assert!((values[row * 3 + 1].to_f32() - 2.0).abs() < 1e-3);
        assert!((values[row * 3 + 2].to_f32() - 3.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_1d_to_3d() -> TensorResult<()> {
    // [2] -> [3, 4, 2]
    let a = Tensor::from_vec(
        vec![f16::from_f32(10.0), f16::from_f32(20.0)],
        vec![2],
    )?;

    let target_shape = TensorShape::new(vec![3, 4, 2]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[3, 4, 2]);
    let values = b.sync_and_read();

    // Every pair should be [10, 20]
    for i in (0..values.len()).step_by(2) {
        assert!((values[i].to_f32() - 10.0).abs() < 1e-3);
        assert!((values[i + 1].to_f32() - 20.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_column_vector() -> TensorResult<()> {
    // [3, 1] -> [3, 4]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![3, 1],
    )?;

    let target_shape = TensorShape::new(vec![3, 4]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[3, 4]);
    let values = b.sync_and_read();

    // Row 0: all 10.0
    for i in 0..4 {
        assert!((values[i].to_f32() - 10.0).abs() < 1e-3);
    }
    // Row 1: all 20.0
    for i in 4..8 {
        assert!((values[i].to_f32() - 20.0).abs() < 1e-3);
    }
    // Row 2: all 30.0
    for i in 8..12 {
        assert!((values[i].to_f32() - 30.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_row_vector() -> TensorResult<()> {
    // [1, 4] -> [3, 4]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![1, 4],
    )?;

    let target_shape = TensorShape::new(vec![3, 4]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[3, 4]);
    let values = b.sync_and_read();

    // Each row should be [1, 2, 3, 4]
    for row in 0..3 {
        for col in 0..4 {
            let expected = (col + 1) as f32;
            assert!((values[row * 4 + col].to_f32() - expected).abs() < 1e-3);
        }
    }
    Ok(())
}

// ============================================================================
// Complex Broadcasting Patterns
// ============================================================================

#[test]
fn test_broadcast_2d_to_3d() -> TensorResult<()> {
    // [2, 3] -> [4, 2, 3]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    let target_shape = TensorShape::new(vec![4, 2, 3]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[4, 2, 3]);
    let values = b.sync_and_read();

    // Each of the 4 batches should be identical
    let original = a.sync_and_read();
    for batch in 0..4 {
        for i in 0..6 {
            let idx = batch * 6 + i;
            assert!((values[idx].to_f32() - original[i].to_f32()).abs() < 1e-3);
        }
    }
    Ok(())
}

#[test]
fn test_broadcast_multiple_dims() -> TensorResult<()> {
    // [1, 3, 1] -> [2, 3, 4]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![1, 3, 1],
    )?;

    let target_shape = TensorShape::new(vec![2, 3, 4]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[2, 3, 4]);
    let values = b.sync_and_read();

    // Each [3, 4] slice should have rows of all 10s, all 20s, all 30s
    for batch in 0..2 {
        for row in 0..3 {
            let expected = (row + 1) as f32 * 10.0;
            for col in 0..4 {
                let idx = batch * 12 + row * 4 + col;
                assert!((values[idx].to_f32() - expected).abs() < 1e-2);
            }
        }
    }
    Ok(())
}

#[test]
fn test_broadcast_4d() -> TensorResult<()> {
    // [1, 1, 2, 1] -> [2, 3, 2, 4]
    let a = Tensor::from_vec(
        vec![f16::from_f32(5.0), f16::from_f32(10.0)],
        vec![1, 1, 2, 1],
    )?;

    let target_shape = TensorShape::new(vec![2, 3, 2, 4]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[2, 3, 2, 4]);
    Ok(())
}

// ============================================================================
// Identity Broadcasting (No-op)
// ============================================================================

#[test]
fn test_broadcast_identity() -> TensorResult<()> {
    // [3, 4] -> [3, 4] (no change)
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let target_shape = TensorShape::new(vec![3, 4]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[3, 4]);

    let a_vals = a.sync_and_read();
    let b_vals = b.sync_and_read();

    for i in 0..12 {
        assert!((a_vals[i].to_f32() - b_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_single_element() -> TensorResult<()> {
    // [1, 1, 1] -> [1, 1, 1] (still identity)
    let a = Tensor::from_vec(vec![f16::from_f32(123.0)], vec![1, 1, 1])?;

    let target_shape = TensorShape::new(vec![1, 1, 1]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[1, 1, 1]);
    assert!((b.sync_and_read()[0].to_f32() - 123.0).abs() < 1e-2);
    Ok(())
}

// ============================================================================
// Broadcasting with Element-wise Operations
// ============================================================================

#[test]
fn test_broadcast_add() -> TensorResult<()> {
    // Simulate: [2, 3] + [3] = [2, 3]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    let b = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![3],
    )?;

    // Broadcast b to [2, 3]
    let target_shape = TensorShape::new(vec![2, 3]);
    let b_broadcast = b.broadcast_to(&target_shape)?;

    let result = a.add(&b_broadcast)?;
    let values = result.sync_and_read();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    assert!((values[0].to_f32() - 11.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 22.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 33.0).abs() < 1e-3);
    assert!((values[3].to_f32() - 14.0).abs() < 1e-3);
    assert!((values[4].to_f32() - 25.0).abs() < 1e-3);
    assert!((values[5].to_f32() - 36.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_broadcast_multiply() -> TensorResult<()> {
    // [2, 1] * [1, 3] = [2, 3]
    let a = Tensor::from_vec(
        vec![f16::from_f32(2.0), f16::from_f32(3.0)],
        vec![2, 1],
    )?;

    let b = Tensor::from_vec(
        vec![
            f16::from_f32(10.0),
            f16::from_f32(20.0),
            f16::from_f32(30.0),
        ],
        vec![1, 3],
    )?;

    let target_shape = TensorShape::new(vec![2, 3]);
    let a_broadcast = a.broadcast_to(&target_shape)?;
    let b_broadcast = b.broadcast_to(&target_shape)?;

    let result = a_broadcast.mul(&b_broadcast)?;
    let values = result.sync_and_read();

    // Expected: [[20, 40, 60], [30, 60, 90]]
    assert!((values[0].to_f32() - 20.0).abs() < 1e-2);
    assert!((values[1].to_f32() - 40.0).abs() < 1e-2);
    assert!((values[2].to_f32() - 60.0).abs() < 1e-2);
    assert!((values[3].to_f32() - 30.0).abs() < 1e-2);
    assert!((values[4].to_f32() - 60.0).abs() < 1e-2);
    assert!((values[5].to_f32() - 90.0).abs() < 1e-2);
    Ok(())
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_broadcast_empty_leading_dims() -> TensorResult<()> {
    // [3] -> [1, 1, 3]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ],
        vec![3],
    )?;

    let target_shape = TensorShape::new(vec![1, 1, 3]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[1, 1, 3]);
    let values = b.sync_and_read();
    assert!((values[0].to_f32() - 1.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 2.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 3.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_broadcast_all_ones() -> TensorResult<()> {
    // [1, 1, 1] -> [5, 5, 5]
    let a = Tensor::from_vec(vec![f16::from_f32(7.0)], vec![1, 1, 1])?;

    let target_shape = TensorShape::new(vec![5, 5, 5]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[5, 5, 5]);
    let values = b.sync_and_read();
    assert_eq!(values.len(), 125);

    for val in values {
        assert!((val.to_f32() - 7.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_broadcast_partial_ones() -> TensorResult<()> {
    // [2, 1, 3, 1] -> [2, 4, 3, 5]
    let data: Vec<f16> = (0..6).map(|i| f16::from_f32(i as f32)).collect();
    let a = Tensor::from_vec(data, vec![2, 1, 3, 1])?;

    let target_shape = TensorShape::new(vec![2, 4, 3, 5]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[2, 4, 3, 5]);
    assert_eq!(b.numel(), 2 * 4 * 3 * 5);
    Ok(())
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_broadcast_incompatible_dims() {
    // [3] cannot broadcast to [2]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ],
        vec![3],
    )
    .unwrap();

    let target_shape = TensorShape::new(vec![2]);
    assert!(a.broadcast_to(&target_shape).is_err());
}

#[test]
fn test_broadcast_conflicting_dims() {
    // [2, 3] cannot broadcast to [2, 4]
    let a = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3]).unwrap();

    let target_shape = TensorShape::new(vec![2, 4]);
    assert!(a.broadcast_to(&target_shape).is_err());
}

#[test]
fn test_broadcast_smaller_target() {
    // [3, 4] cannot broadcast to [3] (reduction, not broadcast)
    let a = Tensor::from_vec(vec![f16::ONE; 12], vec![3, 4]).unwrap();

    let target_shape = TensorShape::new(vec![3]);
    assert!(a.broadcast_to(&target_shape).is_err());
}

#[test]
fn test_broadcast_middle_dim_mismatch() {
    // [2, 3, 4] cannot broadcast to [2, 5, 4]
    let a = Tensor::from_vec(vec![f16::ONE; 24], vec![2, 3, 4]).unwrap();

    let target_shape = TensorShape::new(vec![2, 5, 4]);
    assert!(a.broadcast_to(&target_shape).is_err());
}

// ============================================================================
// Large Scale Broadcasting
// ============================================================================

#[test]
fn test_broadcast_large_expansion() -> TensorResult<()> {
    // [1] -> [100, 100]
    let a = Tensor::from_vec(vec![f16::from_f32(3.14)], vec![1])?;

    let target_shape = TensorShape::new(vec![100, 100]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[100, 100]);
    assert_eq!(b.numel(), 10000);

    let values = b.sync_and_read();
    for val in values.iter().take(100) {
        assert!((val.to_f32() - 3.14).abs() < 0.01);
    }
    Ok(())
}

#[test]
fn test_broadcast_high_dimensional() -> TensorResult<()> {
    // [1, 1, 1, 1, 2] -> [2, 3, 4, 5, 2]
    let a = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![1, 1, 1, 1, 2],
    )?;

    let target_shape = TensorShape::new(vec![2, 3, 4, 5, 2]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[2, 3, 4, 5, 2]);
    assert_eq!(b.numel(), 2 * 3 * 4 * 5 * 2);
    Ok(())
}

#[test]
fn test_broadcast_alternating_pattern() -> TensorResult<()> {
    // Test pattern where some dims are 1, some are not
    // [1, 5, 1, 3] -> [7, 5, 9, 3]
    let data: Vec<f16> = (0..15).map(|i| f16::from_f32(i as f32)).collect();
    let a = Tensor::from_vec(data, vec![1, 5, 1, 3])?;

    let target_shape = TensorShape::new(vec![7, 5, 9, 3]);
    let b = a.broadcast_to(&target_shape)?;

    assert_eq!(b.shape().dims(), &[7, 5, 9, 3]);
    assert_eq!(b.numel(), 7 * 5 * 9 * 3);
    Ok(())
}

// ============================================================================
// Broadcasting with Different Data Patterns
// ============================================================================

#[test]
fn test_broadcast_negative_values() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(-1.0),
            f16::from_f32(-2.0),
            f16::from_f32(-3.0),
        ],
        vec![3],
    )?;

    let target_shape = TensorShape::new(vec![2, 3]);
    let b = a.broadcast_to(&target_shape)?;

    let values = b.sync_and_read();
    assert!((values[0].to_f32() - (-1.0)).abs() < 1e-3);
    assert!((values[1].to_f32() - (-2.0)).abs() < 1e-3);
    assert!((values[2].to_f32() - (-3.0)).abs() < 1e-3);
    assert!((values[3].to_f32() - (-1.0)).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_broadcast_zeros() -> TensorResult<()> {
    let a = Tensor::from_vec(vec![f16::ZERO], vec![1])?;

    let target_shape = TensorShape::new(vec![10, 10]);
    let b = a.broadcast_to(&target_shape)?;

    let values = b.sync_and_read();
    for val in values {
        assert!((val.to_f32()).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_broadcast_large_values() -> TensorResult<()> {
    // Test with values near F16 max
    let a = Tensor::from_vec(vec![f16::from_f32(30000.0)], vec![1])?;

    let target_shape = TensorShape::new(vec![3, 3]);
    let b = a.broadcast_to(&target_shape)?;

    let values = b.sync_and_read();
    for val in values {
        assert!((val.to_f32() - 30000.0).abs() < 100.0);
    }
    Ok(())
}
