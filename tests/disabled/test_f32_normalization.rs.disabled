/// Test f32 normalization operations
/// Tests layer_norm, rms_norm, batch_norm, and related operations

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use tensorlogic::ops::normalization::TensorNormalization;

#[test]
fn test_f32_rms_norm() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input tensor
    let input = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4]
    )?;

    // Create weight (gamma)
    let weight = Tensor::<f32>::ones(&device, vec![4])?;

    // Apply RMS normalization
    let output = input.rms_norm(&weight, 1e-5)?;

    // Verify output shape
    assert_eq!(output.shape(), &[4]);

    let data = output.to_vec();

    // Verify all values are finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    // RMS norm should reduce variance
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / data.len() as f32;

    // Variance should be relatively small after normalization
    assert!(variance < 2.0);

    println!("✓ f32 rms_norm test passed");
    Ok(())
}

#[test]
fn test_f32_layer_norm() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input tensor [2, 4] - batch of 2, feature dim 4
    let input = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0],
        vec![2, 4]
    )?;

    // Create weight and bias
    let weight = Tensor::<f32>::ones(&device, vec![4])?;
    let bias = Tensor::<f32>::zeros(&device, vec![4])?;

    // Apply layer normalization
    let output = input.layer_norm(&weight, &bias, 1e-5)?;

    // Verify output shape
    assert_eq!(output.shape(), &[2, 4]);

    let data = output.to_vec();

    // Verify all values are finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    // For each batch, mean should be close to 0 (due to zero bias)
    for batch in 0..2 {
        let start = batch * 4;
        let end = start + 4;
        let batch_data = &data[start..end];
        let mean: f32 = batch_data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.1);
    }

    println!("✓ f32 layer_norm test passed");
    Ok(())
}

#[test]
fn test_f32_layer_norm_with_bias() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input
    let input = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4]
    )?;

    // Weight = 1.0, bias = 2.0
    let weight = Tensor::<f32>::ones(&device, vec![4])?;
    let bias = Tensor::<f32>::full(&device, vec![4], 2.0)?;

    // Apply layer norm
    let output = input.layer_norm(&weight, &bias, 1e-5)?;

    let data = output.to_vec();

    // All values should be finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    // Mean should be shifted by bias (approximately 2.0)
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!((mean - 2.0).abs() < 0.5);

    println!("✓ f32 layer_norm with bias test passed");
    Ok(())
}

#[test]
fn test_f32_rms_norm_multidim() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create multi-dimensional input [2, 3, 4]
    let input = Tensor::<f32>::from_vec(
        (0..24).map(|i| i as f32 + 1.0).collect(),
        vec![2, 3, 4]
    )?;

    // Weight for last dimension
    let weight = Tensor::<f32>::ones(&device, vec![4])?;

    // Apply RMS norm
    let output = input.rms_norm(&weight, 1e-5)?;

    // Verify shape preserved
    assert_eq!(output.shape(), &[2, 3, 4]);

    let data = output.to_vec();

    // All finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    println!("✓ f32 rms_norm multidim test passed");
    Ok(())
}

#[test]
fn test_f32_normalization_stability() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input with large values
    let input = Tensor::<f32>::from_vec(
        vec![100.0, 200.0, 300.0, 400.0],
        vec![4]
    )?;

    let weight = Tensor::<f32>::ones(&device, vec![4])?;

    // RMS norm should handle large values
    let output = input.rms_norm(&weight, 1e-5)?;

    let data = output.to_vec();

    // All should be finite (no overflow/underflow)
    for val in data.iter() {
        assert!(val.is_finite());
        assert!(!val.is_nan());
    }

    println!("✓ f32 normalization stability test passed");
    Ok(())
}

#[test]
fn test_f32_normalization_small_values() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input with small values
    let input = Tensor::<f32>::from_vec(
        vec![0.001, 0.002, 0.003, 0.004],
        vec![4]
    )?;

    let weight = Tensor::<f32>::ones(&device, vec![4])?;
    let bias = Tensor::<f32>::zeros(&device, vec![4])?;

    // Layer norm should handle small values
    let output = input.layer_norm(&weight, &bias, 1e-5)?;

    let data = output.to_vec();

    // All should be finite
    for val in data.iter() {
        assert!(val.is_finite());
        assert!(!val.is_nan());
    }

    println!("✓ f32 normalization small values test passed");
    Ok(())
}

#[test]
fn test_f32_layer_norm_zero_mean() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input with mean already close to zero
    let input = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 1.0, 2.0],
        vec![4]
    )?;

    let weight = Tensor::<f32>::ones(&device, vec![4])?;
    let bias = Tensor::<f32>::zeros(&device, vec![4])?;

    let output = input.layer_norm(&weight, &bias, 1e-5)?;

    let data = output.to_vec();

    // Mean should be close to zero
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.1);

    // Variance should be close to 1
    let variance: f32 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / data.len() as f32;
    assert!((variance - 1.0).abs() < 0.2);

    println!("✓ f32 layer_norm zero mean test passed");
    Ok(())
}

#[test]
fn test_f32_rms_norm_weight_scaling() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let input = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4]
    )?;

    // Weight = 2.0 (should scale output by 2)
    let weight = Tensor::<f32>::full(&device, vec![4], 2.0)?;

    let output = input.rms_norm(&weight, 1e-5)?;

    // Compare with weight = 1.0
    let weight_ones = Tensor::<f32>::ones(&device, vec![4])?;
    let output_ones = input.rms_norm(&weight_ones, 1e-5)?;

    let data = output.to_vec();
    let data_ones = output_ones.to_vec();

    // Output with weight=2.0 should be approximately 2x output with weight=1.0
    for i in 0..4 {
        assert!((data[i] - 2.0 * data_ones[i]).abs() < 0.1);
    }

    println!("✓ f32 rms_norm weight scaling test passed");
    Ok(())
}
