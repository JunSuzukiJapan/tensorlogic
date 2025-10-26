/// Test f32 tensor basic operations
/// Tests addition, subtraction, multiplication, division with f32 tensors

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};

#[test]
fn test_f32_addition() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create f32 tensors
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    // Add tensors using method
    let c = a.add(&b)?;
    let result = c.to_vec();

    // Verify results
    assert!((result[0] - 6.0).abs() < 1e-5);
    assert!((result[1] - 8.0).abs() < 1e-5);
    assert!((result[2] - 10.0).abs() < 1e-5);
    assert!((result[3] - 12.0).abs() < 1e-5);

    println!("✓ f32 addition test passed");
    Ok(())
}

#[test]
fn test_f32_subtraction() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    let c = a.sub(&b)?;
    let result = c.to_vec();

    assert!((result[0] - 9.0).abs() < 1e-5);
    assert!((result[1] - 18.0).abs() < 1e-5);
    assert!((result[2] - 27.0).abs() < 1e-5);
    assert!((result[3] - 36.0).abs() < 1e-5);

    println!("✓ f32 subtraction test passed");
    Ok(())
}

#[test]
fn test_f32_multiplication() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2])?;

    let c = a.mul(&b)?;
    let result = c.to_vec();

    assert!((result[0] - 3.0).abs() < 1e-5);
    assert!((result[1] - 7.5).abs() < 1e-5);
    assert!((result[2] - 14.0).abs() < 1e-5);
    assert!((result[3] - 22.5).abs() < 1e-5);

    println!("✓ f32 multiplication test passed");
    Ok(())
}

#[test]
fn test_f32_division() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![2.0, 4.0, 5.0, 8.0], vec![2, 2])?;

    let c = a.div(&b)?;
    let result = c.to_vec();

    assert!((result[0] - 5.0).abs() < 1e-5);
    assert!((result[1] - 5.0).abs() < 1e-5);
    assert!((result[2] - 6.0).abs() < 1e-5);
    assert!((result[3] - 5.0).abs() < 1e-5);

    println!("✓ f32 division test passed");
    Ok(())
}

#[test]
fn test_f32_scalar_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    // Scalar multiplication
    let b = a.mul_scalar(2.5)?;
    let result = b.to_vec();
    assert!((result[0] - 2.5).abs() < 1e-5);
    assert!((result[1] - 5.0).abs() < 1e-5);
    assert!((result[2] - 7.5).abs() < 1e-5);
    assert!((result[3] - 10.0).abs() < 1e-5);

    // Scalar addition
    let c = a.add_scalar(10.0)?;
    let result = c.to_vec();
    assert!((result[0] - 11.0).abs() < 1e-5);
    assert!((result[1] - 12.0).abs() < 1e-5);
    assert!((result[2] - 13.0).abs() < 1e-5);
    assert!((result[3] - 14.0).abs() < 1e-5);

    println!("✓ f32 scalar operations test passed");
    Ok(())
}

#[test]
fn test_f32_combined_operations() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test (a + b) * c - d
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2])?;
    let b = Tensor::<f32>::from_vec(vec![3.0, 4.0], vec![2])?;
    let c = Tensor::<f32>::from_vec(vec![2.0, 2.0], vec![2])?;
    let d = Tensor::<f32>::from_vec(vec![1.0, 1.0], vec![2])?;

    let temp = a.add(&b)?;  // [4.0, 6.0]
    let temp2 = temp.mul(&c)?;  // [8.0, 12.0]
    let result_tensor = temp2.sub(&d)?;  // [7.0, 11.0]
    let result = result_tensor.to_vec();

    assert!((result[0] - 7.0).abs() < 1e-5);
    assert!((result[1] - 11.0).abs() < 1e-5);

    println!("✓ f32 combined operations test passed");
    Ok(())
}
