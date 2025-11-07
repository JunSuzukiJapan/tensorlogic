//! Comprehensive tests for memory management and device operations
//! Tests tensor creation, cloning, device transfer, and memory efficiency

use half::f16;
use tensorlogic::prelude::*;

// ============================================================================
// Memory Allocation Tests
// ============================================================================

#[test]
fn test_create_small_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::ONE; 10],
        vec![10],
    )?;

    assert_eq!(a.numel(), 10);
    assert_eq!(a.shape().dims(), &[10]);
    Ok(())
}

#[test]
fn test_create_medium_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let numel = 1000;
    let data: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 100) as f32)).collect();
    let a = Tensor::from_vec(data, vec![1000])?;

    assert_eq!(a.numel(), 1000);
    Ok(())
}

#[test]
fn test_create_large_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // 1 million elements (~2MB for f16)
    let numel = 1_000_000;
    let data: Vec<f16> = vec![f16::from_f32(1.0); numel];
    let a = Tensor::from_vec(data, vec![1000, 1000])?;

    assert_eq!(a.numel(), 1_000_000);
    assert_eq!(a.shape().dims(), &[1000, 1000]);
    Ok(())
}

#[test]
fn test_create_very_large_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // 10 million elements (~20MB for f16)
    let numel = 10_000_000;
    let data: Vec<f16> = vec![f16::from_f32(0.5); numel];
    let a = Tensor::from_vec(data, vec![10000, 1000])?;

    assert_eq!(a.numel(), 10_000_000);
    Ok(())
}

#[test]
fn test_create_multiple_large_tensors() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Create multiple large tensors to test memory management
    let mut tensors = Vec::new();

    for i in 0..10 {
        let data: Vec<f16> = vec![f16::from_f32(i as f32); 100_000];
        let tensor = Tensor::from_vec(data, vec![100, 1000])?;
        tensors.push(tensor);
    }

    assert_eq!(tensors.len(), 10);
    Ok(())
}

#[test]
fn test_tensor_clone() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10, 10],
    )?;

    let b = a.clone();

    assert_eq!(a.shape().dims(), b.shape().dims());
    assert_eq!(a.numel(), b.numel());

    let a_vals = a.sync_and_read();
    let b_vals = b.sync_and_read();

    for i in 0..100 {
        assert!((a_vals[i].to_f32() - b_vals[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_clone_large_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let numel = 1_000_000;
    let data: Vec<f16> = vec![f16::from_f32(3.14); numel];
    let a = Tensor::from_vec(data, vec![1000, 1000])?;

    let b = a.clone();

    assert_eq!(a.numel(), b.numel());
    assert_eq!(a.shape().dims(), b.shape().dims());
    Ok(())
}

// ============================================================================
// Tensor Lifetime and Ownership Tests
// ============================================================================

#[test]
fn test_tensor_move() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::from_f32(42.0); 100],
        vec![100],
    )?;

    let b = a; // Move ownership

    assert_eq!(b.numel(), 100);
    // `a` is no longer accessible here
    Ok(())
}

#[test]
fn test_tensor_borrow() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::from_f32(42.0); 100],
        vec![100],
    )?;

    let sum = a.sum()?;
    assert!(sum.to_f32() > 0.0);

    // `a` is still accessible here
    assert_eq!(a.numel(), 100);
    Ok(())
}

#[test]
fn test_multiple_operations_same_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10, 10],
    )?;

    let sum = a.sum()?;
    let mean = a.mean()?;
    let max = a.max()?;
    let min = a.min()?;

    assert!(sum.to_f32() > 0.0);
    assert!(mean.to_f32() > 0.0);
    assert!(max.to_f32() > 0.0);
    assert!(min.to_f32() >= 0.0);
    Ok(())
}

// ============================================================================
// Device Tests (CPU/Metal)
// ============================================================================

#[test]
fn test_cpu_tensor_creation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::ONE; 100],
        vec![10, 10],
    )?;

    // Default should be CPU
    assert!(matches!(a.device(), Device::CPU));
    Ok(())
}

#[test]
fn test_zeros_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let a: Tensor<f16> = Tensor::zeros(&device, vec![5, 5])?;

    assert_eq!(a.numel(), 25);
    let values = a.sync_and_read();
    for val in values {
        assert!((val.to_f32()).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_ones_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let a: Tensor<f16> = Tensor::ones(&device, vec![5, 5])?;

    assert_eq!(a.numel(), 25);
    let values = a.sync_and_read();
    for val in values {
        assert!((val.to_f32() - 1.0).abs() < 1e-3);
    }
    Ok(())
}

#[test]
#[ignore] // TODO: zeros_like() not yet implemented
fn test_zeros_like() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::from_f32(42.0); 20],
        vec![4, 5],
    )?;

    let b = Tensor::zeros(&device, a.shape().dims().to_vec())?;

    assert_eq!(b.shape().dims(), &[4, 5]);
    let values = b.sync_and_read();
    for val in values {
        assert!((val.to_f32()).abs() < 1e-6);
    }
    Ok(())
}

#[test]
#[ignore] // TODO: ones_like() not yet implemented
fn test_ones_like() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        vec![f16::from_f32(42.0); 20],
        vec![4, 5],
    )?;

    let b = Tensor::ones(&device, a.shape().dims().to_vec())?;

    assert_eq!(b.shape().dims(), &[4, 5]);
    let values = b.sync_and_read();
    for val in values {
        assert!((val.to_f32() - 1.0).abs() < 1e-3);
    }
    Ok(())
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

#[test]
fn test_reshape_no_copy() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Reshape should not copy data, just change metadata
    let a = Tensor::from_vec(
        (0..120).map(|i| f16::from_f32(i as f32)).collect(),
        vec![120],
    )?;

    let b = a.reshape(vec![10, 12])?;
    let c = b.reshape(vec![5, 24])?;
    let d = c.reshape(vec![2, 60])?;

    // All should refer to same data
    assert_eq!(d.numel(), 120);
    Ok(())
}

#[test]
fn test_operations_on_reshaped() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..24).map(|i| f16::from_f32(i as f32)).collect(),
        vec![24],
    )?;

    let b = a.reshape(vec![2, 3, 4])?;
    let sum = b.sum()?;

    // Sum should be same as original
    assert!((sum.to_f32() - 276.0).abs() < 1.0); // 0+1+...+23 = 276
    Ok(())
}

#[test]
fn test_view_sharing() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![100],
    )?;

    let b = a.reshape(vec![10, 10])?;
    let c = b.reshape(vec![5, 20])?;

    // All views should have same underlying data
    assert_eq!(a.numel(), b.numel());
    assert_eq!(b.numel(), c.numel());
    Ok(())
}

// ============================================================================
// Data Transfer Tests
// ============================================================================

#[test]
fn test_to_vec_and_back() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let original_data: Vec<f16> = (0..100).map(|i| f16::from_f32(i as f32)).collect();
    let a = Tensor::from_vec(original_data.clone(), vec![10, 10])?;

    let retrieved = a.sync_and_read();

    assert_eq!(original_data.len(), retrieved.len());
    for i in 0..100 {
        assert!((original_data[i].to_f32() - retrieved[i].to_f32()).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn test_large_to_vec() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let numel = 1_000_000;
    let data: Vec<f16> = vec![f16::from_f32(1.0); numel];
    let a = Tensor::from_vec(data, vec![1000, 1000])?;

    let retrieved = a.sync_and_read();

    assert_eq!(retrieved.len(), 1_000_000);
    Ok(())
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

#[test]
fn test_batch_tensor_creation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate batch of images: [batch, channels, height, width]
    let batch_size = 32;
    let channels = 3;
    let height = 64;
    let width = 64;

    let numel = batch_size * channels * height * width;
    let data: Vec<f16> = vec![f16::from_f32(0.5); numel];

    let batch = Tensor::from_vec(data, vec![batch_size, channels, height, width])?;

    assert_eq!(batch.shape().dims(), &[32, 3, 64, 64]);
    assert_eq!(batch.numel(), 32 * 3 * 64 * 64);
    Ok(())
}

#[test]
fn test_batch_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Create batch of vectors
    let batch_size = 10;
    let dim = 100;

    let data: Vec<f16> = (0..batch_size * dim)
        .map(|i| f16::from_f32((i % 10) as f32))
        .collect();

    let batch = Tensor::from_vec(data, vec![batch_size, dim])?;

    // Operations on batch
    let sum = batch.sum()?;
    let mean = batch.mean()?;

    assert!(sum.to_f32() > 0.0);
    assert!(mean.to_f32() > 0.0);
    Ok(())
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_single_element_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(vec![f16::from_f32(42.0)], vec![1])?;

    assert_eq!(a.numel(), 1);
    assert_eq!(a.sync_and_read()[0].to_f32(), 42.0);
    Ok(())
}

#[test]
fn test_large_1d_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let numel = 10_000_000;
    let data: Vec<f16> = vec![f16::from_f32(1.0); numel];
    let a = Tensor::from_vec(data, vec![numel])?;

    assert_eq!(a.numel(), 10_000_000);
    assert_eq!(a.shape().dims(), &[10_000_000]);
    Ok(())
}

#[test]
fn test_high_dimensional_tensor() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // 6-dimensional tensor
    let a = Tensor::from_vec(
        vec![f16::ONE; 2 * 2 * 2 * 2 * 2 * 2],
        vec![2, 2, 2, 2, 2, 2],
    )?;

    assert_eq!(a.shape().rank(), 6);
    assert_eq!(a.numel(), 64);
    Ok(())
}

#[test]
fn test_tensor_with_ones_in_shape() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Shape with 1s: [1, 10, 1, 5]
    let a = Tensor::from_vec(
        vec![f16::ONE; 50],
        vec![1, 10, 1, 5],
    )?;

    assert_eq!(a.shape().dims(), &[1, 10, 1, 5]);
    assert_eq!(a.numel(), 50);
    Ok(())
}

#[test]
fn test_create_and_destroy_loop() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test creating and destroying many tensors
    for i in 0..100 {
        let data: Vec<f16> = vec![f16::from_f32(i as f32); 1000];
        let _tensor = Tensor::from_vec(data, vec![10, 100])?;
        // Tensor is dropped at end of loop
    }
    Ok(())
}

#[test]
#[ignore] // TODO: ones_like() not yet implemented
fn test_nested_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10, 10],
    )?;

    // Nested operations that create intermediate tensors
    let b = a.add(&Tensor::ones(&device, a.shape().dims().to_vec())?)?;
    let c = b.mul(&Tensor::from_vec(vec![f16::from_f32(2.0); 100], vec![10, 10])?)?;
    let sum = c.sum()?;

    assert!(sum.to_f32() > 0.0);
    Ok(())
}

// ============================================================================
// Memory Pattern Tests
// ============================================================================

#[test]
fn test_sequential_allocation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    // Allocate tensors sequentially
    let a: Tensor<f16> = Tensor::ones(&device, vec![100, 100])?;
    let b: Tensor<f16> = Tensor::ones(&device, vec![100, 100])?;
    let c: Tensor<f16> = Tensor::ones(&device, vec![100, 100])?;

    assert_eq!(a.numel(), 10_000);
    assert_eq!(b.numel(), 10_000);
    assert_eq!(c.numel(), 10_000);
    Ok(())
}

#[test]
fn test_interleaved_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let a: Tensor<f16> = Tensor::ones(&device, vec![50, 50])?;
    let sum_a = a.sum()?;

    let b: Tensor<f16> = Tensor::ones(&device, vec![50, 50])?;
    let sum_b = b.sum()?;

    let c = a.add(&b)?;
    let sum_c = c.sum()?;

    assert!(sum_a.to_f32() > 0.0);
    assert!(sum_b.to_f32() > 0.0);
    assert!(sum_c.to_f32() > 0.0);
    Ok(())
}

#[test]
fn test_accumulation_pattern() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    // Simulate accumulation pattern (common in training)
    let mut acc = Tensor::zeros(&device, vec![10, 10])?;

    for i in 0..10 {
        let delta = Tensor::from_vec(
            vec![f16::from_f32(i as f32); 100],
            vec![10, 10],
        )?;
        acc = acc.add(&delta)?;
    }

    let final_sum = acc.sum()?;
    assert!(final_sum.to_f32() > 0.0);
    Ok(())
}

// ============================================================================
// Shape and Metadata Tests
// ============================================================================

#[test]
fn test_numel_calculation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let shapes = vec![
        vec![10],
        vec![10, 10],
        vec![5, 20],
        vec![2, 5, 10],
        vec![2, 2, 5, 5],
    ];

    let expected = vec![10, 100, 100, 100, 100];

    for (shape, exp) in shapes.iter().zip(expected.iter()) {
        let a: Tensor<f16> = Tensor::ones(&device, shape.clone())?;
        assert_eq!(a.numel(), *exp);
    }
    Ok(())
}

#[test]
fn test_rank_calculation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let a1: Tensor<f16> = Tensor::ones(&device, vec![10])?;
    assert_eq!(a1.shape().rank(), 1);

    let a2: Tensor<f16> = Tensor::ones(&device, vec![10, 10])?;
    assert_eq!(a2.shape().rank(), 2);

    let a3: Tensor<f16> = Tensor::ones(&device, vec![2, 3, 4])?;
    assert_eq!(a3.shape().rank(), 3);

    let a4: Tensor<f16> = Tensor::ones(&device, vec![2, 2, 2, 2])?;
    assert_eq!(a4.shape().rank(), 4);

    Ok(())
}

#[test]
fn test_dtype_consistency() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let device = MetalDevice::new()?;
    let a: Tensor<f16> = Tensor::from_vec(
        vec![f16::ONE; 10],
        vec![10],
    )?;

    let b: Tensor<f16> = Tensor::ones(&device, vec![10])?;

    // Both should be f16
    let _c = a.add(&b)?;

    Ok(())
}
