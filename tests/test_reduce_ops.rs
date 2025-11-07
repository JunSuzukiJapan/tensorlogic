//! Comprehensive tests for reduction operations
//! Tests sum, mean, max, min, argmax, argmin with various axes and edge cases

use half::f16;
use tensorlogic::prelude::*;

// ============================================================================
// Sum Operations Tests
// ============================================================================

#[test]
fn test_sum_1d_basic() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![4],
    )?;

    let sum = a.sum()?;
    assert!((sum.to_f32() - 10.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_sum_2d_all() -> TensorResult<()> {
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

    let sum = a.sum()?;
    assert!((sum.to_f32() - 21.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_sum_dim_rows() -> TensorResult<()> {
    // [[1, 2, 3],
    //  [4, 5, 6]]
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

    // Sum along rows (dim 0) -> [5, 7, 9]
    let sum0 = a.sum_dim(0, false)?;
    assert_eq!(sum0.shape().dims(), &[3]);
    let result = sum0.sync_and_read();
    assert!((result[0].to_f32() - 5.0).abs() < 1e-3);
    assert!((result[1].to_f32() - 7.0).abs() < 1e-3);
    assert!((result[2].to_f32() - 9.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_sum_dim_cols() -> TensorResult<()> {
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

    // Sum along cols (dim 1) -> [6, 15]
    let sum1 = a.sum_dim(1, false)?;
    assert_eq!(sum1.shape().dims(), &[2]);
    let result = sum1.sync_and_read();
    assert!((result[0].to_f32() - 6.0).abs() < 1e-3);
    assert!((result[1].to_f32() - 15.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_sum_dim_keepdim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![2, 2],
    )?;

    // Sum with keepdim=true
    let sum0 = a.sum_dim(0, true)?;
    assert_eq!(sum0.shape().dims(), &[1, 2]);

    let sum1 = a.sum_dim(1, true)?;
    assert_eq!(sum1.shape().dims(), &[2, 1]);
    Ok(())
}

#[test]
fn test_sum_3d_tensor() -> TensorResult<()> {
    // [2, 2, 3] tensor
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32 + 1.0)).collect(),
        vec![2, 2, 3],
    )?;

    let sum = a.sum()?;
    assert!((sum.to_f32() - 78.0).abs() < 1e-2); // 1+2+...+12 = 78
    Ok(())
}

#[test]
fn test_sum_negative_values() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(-1.0),
            f16::from_f32(2.0),
            f16::from_f32(-3.0),
            f16::from_f32(4.0),
        ],
        vec![4],
    )?;

    let sum = a.sum()?;
    assert!((sum.to_f32() - 2.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_sum_zeros() -> TensorResult<()> {
    let a = Tensor::from_vec(vec![f16::ZERO; 10], vec![10])?;
    let sum = a.sum()?;
    assert!((sum.to_f32()).abs() < 1e-3);
    Ok(())
}

// ============================================================================
// Mean Operations Tests
// ============================================================================

#[test]
fn test_mean_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(8.0),
        ],
        vec![4],
    )?;

    let mean = a.mean()?;
    assert!((mean.to_f32() - 5.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_mean_2d_all() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![2, 2],
    )?;

    let mean = a.mean()?;
    assert!((mean.to_f32() - 2.5).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_mean_dim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(8.0),
        ],
        vec![2, 2],
    )?;

    // Mean along dim 1 -> [(2+4)/2, (6+8)/2] = [3.0, 7.0]
    let mean1 = a.mean_dim(1, false)?;
    let result = mean1.sync_and_read();
    assert!((result[0].to_f32() - 3.0).abs() < 1e-3);
    assert!((result[1].to_f32() - 7.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_mean_dim_keepdim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![2, 2],
    )?;

    let mean0 = a.mean_dim(0, true)?;
    assert_eq!(mean0.shape().dims(), &[1, 2]);

    let mean1 = a.mean_dim(1, true)?;
    assert_eq!(mean1.shape().dims(), &[2, 1]);
    Ok(())
}

#[test]
fn test_mean_negative() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(-2.0),
            f16::from_f32(2.0),
            f16::from_f32(-4.0),
            f16::from_f32(4.0),
        ],
        vec![4],
    )?;

    let mean = a.mean()?;
    assert!((mean.to_f32()).abs() < 1e-3);
    Ok(())
}

// ============================================================================
// Max/Min Operations Tests
// ============================================================================

#[test]
fn test_max_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(3.0),
            f16::from_f32(7.0),
            f16::from_f32(2.0),
            f16::from_f32(9.0),
            f16::from_f32(1.0),
        ],
        vec![5],
    )?;

    let max = a.max()?;
    assert!((max.to_f32() - 9.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_min_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(3.0),
            f16::from_f32(7.0),
            f16::from_f32(2.0),
            f16::from_f32(9.0),
            f16::from_f32(1.0),
        ],
        vec![5],
    )?;

    let min = a.min()?;
    assert!((min.to_f32() - 1.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_max_negative() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(-5.0),
            f16::from_f32(-1.0),
            f16::from_f32(-10.0),
            f16::from_f32(-3.0),
        ],
        vec![4],
    )?;

    let max = a.max()?;
    assert!((max.to_f32() - (-1.0)).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_min_negative() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(-5.0),
            f16::from_f32(-1.0),
            f16::from_f32(-10.0),
            f16::from_f32(-3.0),
        ],
        vec![4],
    )?;

    let min = a.min()?;
    assert!((min.to_f32() - (-10.0)).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_max_all_same() -> TensorResult<()> {
    let a = Tensor::from_vec(vec![f16::from_f32(5.0); 10], vec![10])?;
    let max = a.max()?;
    assert!((max.to_f32() - 5.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_min_all_same() -> TensorResult<()> {
    let a = Tensor::from_vec(vec![f16::from_f32(5.0); 10], vec![10])?;
    let min = a.min()?;
    assert!((min.to_f32() - 5.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_max_2d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),
            f16::from_f32(3.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
            f16::from_f32(7.0),
        ],
        vec![2, 3],
    )?;

    let max = a.max()?;
    assert!((max.to_f32() - 9.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_min_2d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(5.0),
            f16::from_f32(3.0),
            f16::from_f32(8.0),
            f16::from_f32(1.0),
            f16::from_f32(6.0),
            f16::from_f32(2.0),
        ],
        vec![2, 3],
    )?;

    let min = a.min()?;
    assert!((min.to_f32() - 1.0).abs() < 1e-3);
    Ok(())
}

// ============================================================================
// Argmax/Argmin Tests
// ============================================================================

#[test]
fn test_argmax_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(3.0),
            f16::from_f32(7.0),
            f16::from_f32(2.0),
            f16::from_f32(9.0),
            f16::from_f32(1.0),
        ],
        vec![5],
    )?;

    let idx = a.argmax(None, false)?;
    let values = idx.sync_and_read();
    assert!((values[0].to_f32() - 3.0).abs() < 1e-3); // Index 3
    Ok(())
}

#[test]
fn test_argmin_1d() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(3.0),
            f16::from_f32(7.0),
            f16::from_f32(2.0),
            f16::from_f32(9.0),
            f16::from_f32(1.0),
        ],
        vec![5],
    )?;

    let idx = a.argmin(None, false)?;
    let values = idx.sync_and_read();
    assert!((values[0].to_f32() - 4.0).abs() < 1e-3); // Index 4
    Ok(())
}

#[test]
fn test_argmax_dim() -> TensorResult<()> {
    // [[1, 5, 3],
    //  [9, 2, 7]]
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),
            f16::from_f32(3.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
            f16::from_f32(7.0),
        ],
        vec![2, 3],
    )?;

    // Argmax along dim 1 (columns) -> [1, 0] (indices of max in each row)
    let idx = a.argmax(Some(1), false)?;
    let values = idx.sync_and_read();
    assert!((values[0].to_f32() - 1.0).abs() < 1e-3); // Max of row 0 is at index 1
    assert!((values[1].to_f32() - 0.0).abs() < 1e-3); // Max of row 1 is at index 0
    Ok(())
}

#[test]
fn test_argmin_dim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),
            f16::from_f32(3.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
            f16::from_f32(7.0),
        ],
        vec![2, 3],
    )?;

    // Argmin along dim 1 (columns) -> [0, 1] (indices of min in each row)
    let idx = a.argmin(Some(1), false)?;
    let values = idx.sync_and_read();
    assert!((values[0].to_f32() - 0.0).abs() < 1e-3); // Min of row 0 is at index 0
    assert!((values[1].to_f32() - 1.0).abs() < 1e-3); // Min of row 1 is at index 1
    Ok(())
}

#[test]
fn test_argmax_keepdim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
        ],
        vec![2, 2],
    )?;

    let idx = a.argmax(Some(1), true)?;
    assert_eq!(idx.shape().dims(), &[2, 1]);
    Ok(())
}

#[test]
fn test_argmin_keepdim() -> TensorResult<()> {
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(5.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
        ],
        vec![2, 2],
    )?;

    let idx = a.argmin(Some(1), true)?;
    assert_eq!(idx.shape().dims(), &[2, 1]);
    Ok(())
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_sum_dim_invalid() {
    let a = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3]).unwrap();

    // Dimension 2 doesn't exist (only 0 and 1)
    assert!(a.sum_dim(2, false).is_err());
}

#[test]
fn test_mean_dim_invalid() {
    let a = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3]).unwrap();

    assert!(a.mean_dim(5, false).is_err());
}

#[test]
fn test_argmax_invalid_dim() {
    let a = Tensor::from_vec(vec![f16::ONE; 6], vec![2, 3]).unwrap();

    assert!(a.argmax(Some(10), false).is_err());
}

#[test]
fn test_reduce_large_tensor() -> TensorResult<()> {
    // Test with larger tensor (1000 elements)
    let data: Vec<f16> = (0..1000)
        .map(|i| f16::from_f32((i % 100) as f32))
        .collect();
    let a = Tensor::from_vec(data, vec![10, 100])?;

    let sum = a.sum()?;
    assert!(sum.to_f32() > 0.0);

    let mean = a.mean()?;
    assert!(mean.to_f32() > 0.0);

    let max = a.max()?;
    assert!((max.to_f32() - 99.0).abs() < 1e-2);

    let min = a.min()?;
    assert!((min.to_f32()).abs() < 1e-2);
    Ok(())
}

#[test]
fn test_sum_very_large_values() -> TensorResult<()> {
    // Test near F16 max (65504)
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(30000.0),
            f16::from_f32(30000.0),
        ],
        vec![2],
    )?;

    let sum = a.sum()?;
    // Should handle large sums (may overflow to infinity in F16)
    assert!(sum.to_f32() > 50000.0 || sum.is_infinite());
    Ok(())
}

#[test]
fn test_mean_precision() -> TensorResult<()> {
    // Test mean with values that might lose precision in F16
    let a = Tensor::from_vec(
        vec![
            f16::from_f32(0.001),
            f16::from_f32(0.002),
            f16::from_f32(0.003),
            f16::from_f32(0.004),
        ],
        vec![4],
    )?;

    let mean = a.mean()?;
    // F16 precision: ~0.001, so expect 0.0025 ± 0.001
    assert!((mean.to_f32() - 0.0025).abs() < 0.001);
    Ok(())
}
