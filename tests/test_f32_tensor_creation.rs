/// Test f32 tensor creation methods
/// Tests zeros, ones, from_vec, reshape, and other tensor creation operations

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors, TensorTransform};

#[test]
fn test_f32_zeros() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create zeros tensor
    let t = Tensor::<f32>::zeros(&device, vec![3, 4])?;
    let data = t.to_vec();

    // Verify all elements are zero
    assert_eq!(data.len(), 12);
    for val in data.iter() {
        assert!(val.abs() < 1e-6);
    }

    // Verify shape
    assert_eq!(t.shape(), &[3, 4]);

    println!("✓ f32 zeros test passed");
    Ok(())
}

#[test]
fn test_f32_ones() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create ones tensor
    let t = Tensor::<f32>::ones(&device, vec![2, 3])?;
    let data = t.to_vec();

    // Verify all elements are one
    assert_eq!(data.len(), 6);
    for val in data.iter() {
        assert!((val - 1.0).abs() < 1e-6);
    }

    // Verify shape
    assert_eq!(t.shape(), &[2, 3]);

    println!("✓ f32 ones test passed");
    Ok(())
}

#[test]
fn test_f32_from_vec() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor from vec
    let data_in = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    let t = Tensor::<f32>::from_vec(data_in.clone(), vec![2, 3])?;
    let data_out = t.to_vec();

    // Verify data matches
    assert_eq!(data_out.len(), 6);
    for i in 0..6 {
        assert!((data_out[i] - data_in[i]).abs() < 1e-6);
    }

    // Verify shape
    assert_eq!(t.shape(), &[2, 3]);

    println!("✓ f32 from_vec test passed");
    Ok(())
}

#[test]
fn test_f32_full() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor filled with specific value
    let t = Tensor::<f32>::full(&device, vec![3, 2], 7.5)?;
    let data = t.to_vec();

    // Verify all elements are 7.5
    assert_eq!(data.len(), 6);
    for val in data.iter() {
        assert!((val - 7.5).abs() < 1e-6);
    }

    println!("✓ f32 full test passed");
    Ok(())
}

#[test]
fn test_f32_reshape() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor and reshape
    let t1 = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    let t2 = t1.reshape(&[3, 2])?;

    // Verify data is preserved
    let data = t2.to_vec();
    assert_eq!(data.len(), 6);
    for i in 0..6 {
        assert!((data[i] - (i as f32 + 1.0)).abs() < 1e-6);
    }

    // Verify new shape
    assert_eq!(t2.shape(), &[3, 2]);

    println!("✓ f32 reshape test passed");
    Ok(())
}

#[test]
fn test_f32_transpose() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create 2x3 tensor
    let t1 = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    // Transpose to 3x2
    let t2 = t1.transpose()?;

    // Verify shape
    assert_eq!(t2.shape(), &[3, 2]);

    // Verify data is correctly transposed
    let data = t2.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-6);  // [0,0]
    assert!((data[1] - 4.0).abs() < 1e-6);  // [0,1]
    assert!((data[2] - 2.0).abs() < 1e-6);  // [1,0]
    assert!((data[3] - 5.0).abs() < 1e-6);  // [1,1]
    assert!((data[4] - 3.0).abs() < 1e-6);  // [2,0]
    assert!((data[5] - 6.0).abs() < 1e-6);  // [2,1]

    println!("✓ f32 transpose test passed");
    Ok(())
}

#[test]
fn test_f32_clone() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create original tensor
    let t1 = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;

    // Clone tensor
    let t2 = t1.clone();

    // Verify data matches
    let data1 = t1.to_vec();
    let data2 = t2.to_vec();

    assert_eq!(data1.len(), data2.len());
    for i in 0..data1.len() {
        assert!((data1[i] - data2[i]).abs() < 1e-6);
    }

    // Verify shape matches
    assert_eq!(t1.shape(), t2.shape());

    println!("✓ f32 clone test passed");
    Ok(())
}

#[test]
fn test_f32_slice_and_concat() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create two tensors
    let t1 = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
    let t2 = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;

    // Concatenate
    let t3 = Tensor::<f32>::concat(&[&t1, &t2], 0)?;

    // Verify concatenated tensor
    assert_eq!(t3.shape(), &[6]);
    let data = t3.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 2.0).abs() < 1e-6);
    assert!((data[2] - 3.0).abs() < 1e-6);
    assert!((data[3] - 4.0).abs() < 1e-6);
    assert!((data[4] - 5.0).abs() < 1e-6);
    assert!((data[5] - 6.0).abs() < 1e-6);

    println!("✓ f32 concat test passed");
    Ok(())
}
