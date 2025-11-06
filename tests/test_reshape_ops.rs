//! Comprehensive tests for reshape and transformation operations
//! Tests reshape, flatten, transpose, permute with various shapes and edge cases

use half::f16;
use tensorlogic::prelude::*;

// ============================================================================
// Reshape Operations Tests
// ============================================================================

#[test]
fn test_reshape_1d_to_2d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..6).map(|i| f16::from_f32(i as f32)).collect(),
        vec![6],
    )?;

    let b = a.reshape(vec![2, 3])?;

    assert_eq!(b.shape().dims(), &[2, 3]);
    let values = b.to_vec();
    for i in 0..6 {
        assert!((values[i].to_f32() - i as f32).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_reshape_2d_to_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let b = a.reshape(vec![12])?;

    assert_eq!(b.shape().dims(), &[12]);
    let values = b.to_vec();
    for i in 0..12 {
        assert!((values[i].to_f32() - i as f32).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_reshape_2d_to_2d() -> TensorResult<()> {
    // [3, 4] -> [4, 3]
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let b = a.reshape(vec![4, 3])?;

    assert_eq!(b.shape().dims(), &[4, 3]);
    assert_eq!(b.numel(), 12);
    Ok(())
}

#[test]
fn test_reshape_3d_to_2d() -> TensorResult<()> {
    // [2, 3, 4] -> [6, 4]
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.reshape(vec![6, 4])?;

    assert_eq!(b.shape().dims(), &[6, 4]);
    assert_eq!(b.numel(), 24);
    Ok(())
}

#[test]
fn test_reshape_2d_to_3d() -> TensorResult<()> {
    // [6, 4] -> [2, 3, 4]
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![6, 4],
    )?;

    let b = a.reshape(vec![2, 3, 4])?;

    assert_eq!(b.shape().dims(), &[2, 3, 4]);
    assert_eq!(b.numel(), 24);
    Ok(())
}

#[test]
fn test_reshape_to_scalar_like() -> TensorResult<()> {
    // [1] -> [1] (identity)
    let a = Tensor::from_vec(vec![f16::from_f32(42.0)], vec![1])?;

    let b = a.reshape(vec![1])?;

    assert_eq!(b.shape().dims(), &[1]);
    assert!((b.to_vec()[0].to_f32() - 42.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_reshape_multiple_times() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![24],
    )?;

    let b = a.reshape(vec![2, 12])?;
    let c = b.reshape(vec![4, 6])?;
    let d = c.reshape(vec![2, 3, 4])?;

    assert_eq!(d.shape().dims(), &[2, 3, 4]);
    Ok(())
}

#[test]
fn test_reshape_high_dimensional() -> TensorResult<()> {
    // [2, 3, 4, 5] -> [6, 20]
    let numel = 2 * 3 * 4 * 5;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect(),
        vec![2, 3, 4, 5],
    )?;

    let b = a.reshape(vec![6, 20])?;

    assert_eq!(b.shape().dims(), &[6, 20]);
    assert_eq!(b.numel(), numel);
    Ok(())
}

// ============================================================================
// Flatten Operations Tests
// ============================================================================

#[test]
fn test_flatten_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..10).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10],
    )?;

    let b = a.flatten()?;

    assert_eq!(b.shape().dims(), &[10]);
    Ok(())
}

#[test]
fn test_flatten_2d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let b = a.flatten()?;

    assert_eq!(b.shape().dims(), &[12]);
    let values = b.to_vec();
    for i in 0..12 {
        assert!((values[i].to_f32() - i as f32).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_flatten_3d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.flatten()?;

    assert_eq!(b.shape().dims(), &[24]);
    assert_eq!(b.numel(), 24);
    Ok(())
}

#[test]
fn test_flatten_4d() -> TensorResult<()> {
    let numel = 2 * 3 * 4 * 5;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4, 5],
    )?;

    let b = a.flatten()?;

    assert_eq!(b.shape().dims(), &[numel]);
    Ok(())
}

#[test]
fn test_flatten_then_reshape() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let flat = a.flatten()?;
    let reshaped = flat.reshape(vec![4, 6])?;

    assert_eq!(reshaped.shape().dims(), &[4, 6]);
    assert_eq!(reshaped.numel(), 24);
    Ok(())
}

// ============================================================================
// Transpose Operations Tests (2D only)
// ============================================================================

#[test]
fn test_transpose_square_matrix() -> TensorResult<()> {
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    let a = Tensor::from_vec(
        (1..=9).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 3],
    )?;

    let b = a.transpose()?;

    assert_eq!(b.shape().dims(), &[3, 3]);
    let values = b.to_vec();

    // Transposed: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    assert!((values[0].to_f32() - 1.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 4.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 7.0).abs() < 1e-3);
    assert!((values[3].to_f32() - 2.0).abs() < 1e-3);
    assert!((values[4].to_f32() - 5.0).abs() < 1e-3);
    assert!((values[5].to_f32() - 8.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_transpose_rectangular() -> TensorResult<()> {
    // [2, 3] -> [3, 2]
    let a = Tensor::from_vec(
        (1..=6).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3],
    )?;

    let b = a.transpose()?;

    assert_eq!(b.shape().dims(), &[3, 2]);
    let values = b.to_vec();

    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    assert!((values[0].to_f32() - 1.0).abs() < 1e-3);
    assert!((values[1].to_f32() - 4.0).abs() < 1e-3);
    assert!((values[2].to_f32() - 2.0).abs() < 1e-3);
    assert!((values[3].to_f32() - 5.0).abs() < 1e-3);
    assert!((values[4].to_f32() - 3.0).abs() < 1e-3);
    assert!((values[5].to_f32() - 6.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_transpose_row_vector() -> TensorResult<()> {
    // [1, 4] -> [4, 1]
    let a = Tensor::from_vec(
        (1..=4).map(|i| f16::from_f32(i as f32)).collect(),
        vec![1, 4],
    )?;

    let b = a.transpose()?;

    assert_eq!(b.shape().dims(), &[4, 1]);
    Ok(())
}

#[test]
fn test_transpose_column_vector() -> TensorResult<()> {
    // [4, 1] -> [1, 4]
    let a = Tensor::from_vec(
        (1..=4).map(|i| f16::from_f32(i as f32)).collect(),
        vec![4, 1],
    )?;

    let b = a.transpose()?;

    assert_eq!(b.shape().dims(), &[1, 4]);
    Ok(())
}

#[test]
fn test_transpose_twice() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (1..=6).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3],
    )?;

    let b = a.transpose()?;
    let c = b.transpose()?;

    assert_eq!(c.shape().dims(), &[2, 3]);

    let a_vals = a.to_vec();
    let c_vals = c.to_vec();

    for i in 0..6 {
        assert!((a_vals[i].to_f32() - c_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

// ============================================================================
// Permute Operations Tests (N-D)
// ============================================================================

#[test]
fn test_permute_3d_simple() -> TensorResult<()> {
    // [2, 3, 4] with dims [0, 2, 1] -> [2, 4, 3]
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.permute(vec![0, 2, 1])?;

    assert_eq!(b.shape().dims(), &[2, 4, 3]);
    Ok(())
}

#[test]
fn test_permute_3d_reverse() -> TensorResult<()> {
    // [2, 3, 4] with dims [2, 1, 0] -> [4, 3, 2]
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.permute(vec![2, 1, 0])?;

    assert_eq!(b.shape().dims(), &[4, 3, 2]);
    Ok(())
}

#[test]
fn test_permute_4d() -> TensorResult<()> {
    // [2, 3, 4, 5] with dims [0, 2, 1, 3] -> [2, 4, 3, 5]
    let numel = 2 * 3 * 4 * 5;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect(),
        vec![2, 3, 4, 5],
    )?;

    let b = a.permute(vec![0, 2, 1, 3])?;

    assert_eq!(b.shape().dims(), &[2, 4, 3, 5]);
    Ok(())
}

#[test]
fn test_permute_identity() -> TensorResult<()> {
    // Permute with identity permutation [0, 1, 2]
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.permute(vec![0, 1, 2])?;

    assert_eq!(b.shape().dims(), &[2, 3, 4]);

    let a_vals = a.to_vec();
    let b_vals = b.to_vec();

    for i in 0..24 {
        assert!((a_vals[i].to_f32() - b_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_permute_2d_as_transpose() -> TensorResult<()> {
    // Permute [1, 0] should be same as transpose
    let a = Tensor::from_vec(
        (1..=6).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3],
    )?;

    let transposed = a.transpose()?;
    let permuted = a.permute(vec![1, 0])?;

    assert_eq!(transposed.shape().dims(), permuted.shape().dims());

    let t_vals = transposed.to_vec();
    let p_vals = permuted.to_vec();

    for i in 0..6 {
        assert!((t_vals[i].to_f32() - p_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_permute_batch_first_to_seq_first() -> TensorResult<()> {
    // Common in NLP: [batch, seq, features] -> [seq, batch, features]
    // [2, 5, 3] with dims [1, 0, 2] -> [5, 2, 3]
    let a = Tensor::from_vec(
        (0..30).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 5, 3],
    )?;

    let b = a.permute(vec![1, 0, 2])?;

    assert_eq!(b.shape().dims(), &[5, 2, 3]);
    Ok(())
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_reshape_wrong_numel() {
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )
    .unwrap();

    // Cannot reshape 12 elements to [2, 7] = 14 elements
    assert!(a.reshape(vec![2, 7]).is_err());
}

#[test]
fn test_reshape_empty_shape() {
    let a = Tensor::from_vec(
        vec![f16::from_f32(1.0)],
        vec![1],
    )
    .unwrap();

    // Empty shape should fail
    assert!(a.reshape(vec![]).is_err());
}

#[test]
fn test_transpose_non_2d() {
    // 1D tensor
    let a = Tensor::from_vec(
        (0..10).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10],
    )
    .unwrap();

    assert!(a.transpose().is_err());

    // 3D tensor
    let b = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )
    .unwrap();

    assert!(b.transpose().is_err());
}

#[test]
fn test_permute_invalid_dims() {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )
    .unwrap();

    // Wrong number of dims
    assert!(a.permute(vec![0, 1]).is_err());

    // Duplicate dims
    assert!(a.permute(vec![0, 1, 1]).is_err());

    // Out of range dims
    assert!(a.permute(vec![0, 1, 5]).is_err());
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_reshape_large_tensor() -> TensorResult<()> {
    let numel = 1000;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect(),
        vec![1000],
    )?;

    let b = a.reshape(vec![10, 100])?;
    assert_eq!(b.shape().dims(), &[10, 100]);

    let c = b.reshape(vec![25, 40])?;
    assert_eq!(c.shape().dims(), &[25, 40]);

    let d = c.reshape(vec![20, 50])?;
    assert_eq!(d.shape().dims(), &[20, 50]);

    Ok(())
}

#[test]
fn test_flatten_large_tensor() -> TensorResult<()> {
    let numel = 1000;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect(),
        vec![10, 10, 10],
    )?;

    let flat = a.flatten()?;

    assert_eq!(flat.shape().dims(), &[1000]);
    assert_eq!(flat.numel(), 1000);
    Ok(())
}

#[test]
fn test_transpose_large_matrix() -> TensorResult<()> {
    let numel = 100 * 50;
    let a = Tensor::from_vec(
        (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect(),
        vec![100, 50],
    )?;

    let b = a.transpose()?;

    assert_eq!(b.shape().dims(), &[50, 100]);
    assert_eq!(b.numel(), 5000);
    Ok(())
}

// ============================================================================
// Chained Operations Tests
// ============================================================================

#[test]
fn test_reshape_flatten_chain() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![24],
    )?;

    let b = a.reshape(vec![2, 3, 4])?;
    let c = b.flatten()?;
    let d = c.reshape(vec![6, 4])?;

    assert_eq!(d.shape().dims(), &[6, 4]);
    Ok(())
}

#[test]
fn test_transpose_reshape_chain() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let b = a.transpose()?;
    let c = b.reshape(vec![2, 6])?;

    assert_eq!(c.shape().dims(), &[2, 6]);
    Ok(())
}

#[test]
fn test_permute_reshape_chain() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let b = a.permute(vec![2, 0, 1])?;
    let c = b.reshape(vec![8, 3])?;

    assert_eq!(c.shape().dims(), &[8, 3]);
    assert_eq!(c.numel(), 24);
    Ok(())
}

// ============================================================================
// Shape Preservation Tests
// ============================================================================

#[test]
fn test_operations_preserve_data() -> TensorResult<()> {
    let data: Vec<f16> = (0..12).map(|i| f16::from_f32(i as f32)).collect();
    let a = Tensor::from_vec(data.clone(), vec![12])?;

    let b = a.reshape(vec![3, 4])?;
    let c = b.flatten()?;

    let c_vals = c.to_vec();

    for i in 0..12 {
        assert!((data[i].to_f32() - c_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_reshape_identity_preserves_order() -> TensorResult<()> {
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![2, 3, 4],
    )?;

    let flat = a.flatten()?;
    let reshaped = flat.reshape(vec![2, 3, 4])?;

    let a_vals = a.to_vec();
    let r_vals = reshaped.to_vec();

    for i in 0..24 {
        assert!((a_vals[i].to_f32() - r_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}
