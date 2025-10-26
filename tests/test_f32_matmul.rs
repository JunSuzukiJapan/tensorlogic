/// Test f32 matrix multiplication and linear operations
/// Tests matmul, linear, einsum, and related operations

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use tensorlogic::ops::matmul::TensorMatmul;

#[test]
fn test_f32_matmul_2x2() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create 2x2 matrices
    // A = [[1, 2],
    //      [3, 4]]
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;

    // B = [[5, 6],
    //      [7, 8]]
    let b = Tensor::<f32>::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2]
    )?;

    // C = A @ B
    let c = a.matmul(&b)?;
    let data = c.to_vec();

    // Expected result:
    // [[1*5 + 2*7, 1*6 + 2*8],   [[19, 22],
    //  [3*5 + 4*7, 3*6 + 4*8]] =  [43, 50]]

    assert!((data[0] - 19.0).abs() < 1e-5);
    assert!((data[1] - 22.0).abs() < 1e-5);
    assert!((data[2] - 43.0).abs() < 1e-5);
    assert!((data[3] - 50.0).abs() < 1e-5);

    println!("✓ f32 matmul 2x2 test passed");
    Ok(())
}

#[test]
fn test_f32_matmul_rectangular() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // A: 2x3 matrix
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    // B: 3x2 matrix
    let b = Tensor::<f32>::from_vec(
        vec![7.0, 8.0,
             9.0, 10.0,
             11.0, 12.0],
        vec![3, 2]
    )?;

    // C: 2x2 matrix
    let c = a.matmul(&b)?;

    // Verify shape
    assert_eq!(c.shape(), &[2, 2]);

    let data = c.to_vec();

    // Expected:
    // [[1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12],   [[58,  64],
    //  [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]] =  [139, 154]]

    assert!((data[0] - 58.0).abs() < 1e-4);
    assert!((data[1] - 64.0).abs() < 1e-4);
    assert!((data[2] - 139.0).abs() < 1e-4);
    assert!((data[3] - 154.0).abs() < 1e-4);

    println!("✓ f32 matmul rectangular test passed");
    Ok(())
}

#[test]
fn test_f32_matmul_vector() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Matrix: 3x3
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0,
             7.0, 8.0, 9.0],
        vec![3, 3]
    )?;

    // Vector: 3x1 (column vector)
    let v = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0],
        vec![3, 1]
    )?;

    // Result: 3x1
    let result = a.matmul(&v)?;

    assert_eq!(result.shape(), &[3, 1]);

    let data = result.to_vec();

    // Expected:
    // [1*1 + 2*2 + 3*3]   [14]
    // [4*1 + 5*2 + 6*3] = [32]
    // [7*1 + 8*2 + 9*3]   [50]

    assert!((data[0] - 14.0).abs() < 1e-4);
    assert!((data[1] - 32.0).abs() < 1e-4);
    assert!((data[2] - 50.0).abs() < 1e-4);

    println!("✓ f32 matmul vector test passed");
    Ok(())
}

#[test]
fn test_f32_linear_layer() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Input: batch_size=2, input_dim=3
    let input = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    // Weight: input_dim=3, output_dim=2 (transposed for linear layer)
    let weight = Tensor::<f32>::from_vec(
        vec![0.1, 0.2, 0.3,
             0.4, 0.5, 0.6],
        vec![2, 3]
    )?;

    // Bias: output_dim=2
    let bias = Tensor::<f32>::from_vec(
        vec![1.0, 2.0],
        vec![2]
    )?;

    // Linear: output = input @ weight^T + bias
    let weight_t = weight.transpose()?;
    let matmul_result = input.matmul(&weight_t)?;
    let output = matmul_result.add(&bias.reshape(&[1, 2])?)?;

    // Verify shape: [2, 2]
    assert_eq!(output.shape(), &[2, 2]);

    let data = output.to_vec();

    // All values should be finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    println!("✓ f32 linear layer test passed");
    Ok(())
}

#[test]
fn test_f32_batch_matmul() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Batch of 2 matrices, each 2x3
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0,
             // second batch
             7.0, 8.0, 9.0,
             10.0, 11.0, 12.0],
        vec![2, 2, 3]
    )?;

    // Batch of 2 matrices, each 3x2
    let b = Tensor::<f32>::from_vec(
        vec![1.0, 2.0,
             3.0, 4.0,
             5.0, 6.0,
             // second batch
             7.0, 8.0,
             9.0, 10.0,
             11.0, 12.0],
        vec![2, 3, 2]
    )?;

    // Batch matmul
    let c = a.matmul(&b)?;

    // Verify shape: [2, 2, 2]
    assert_eq!(c.shape(), &[2, 2, 2]);

    let data = c.to_vec();

    // Verify all values are finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    println!("✓ f32 batch matmul test passed");
    Ok(())
}

#[test]
fn test_f32_identity_matmul() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create identity matrix
    let identity = Tensor::<f32>::from_vec(
        vec![1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0],
        vec![3, 3]
    )?;

    // Create test matrix
    let a = Tensor::<f32>::from_vec(
        vec![2.0, 3.0, 4.0,
             5.0, 6.0, 7.0,
             8.0, 9.0, 10.0],
        vec![3, 3]
    )?;

    // A @ I should equal A
    let result = a.matmul(&identity)?;
    let data_a = a.to_vec();
    let data_result = result.to_vec();

    for i in 0..9 {
        assert!((data_a[i] - data_result[i]).abs() < 1e-5);
    }

    println!("✓ f32 identity matmul test passed");
    Ok(())
}

#[test]
fn test_f32_matmul_chain() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test (A @ B) @ C
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2])?;
    let c = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2])?;

    // Chain multiplication
    let ab = a.matmul(&b)?;
    let abc = ab.matmul(&c)?;

    // Verify shape
    assert_eq!(abc.shape(), &[2, 2]);

    // Verify all finite
    let data = abc.to_vec();
    for val in data.iter() {
        assert!(val.is_finite());
    }

    println!("✓ f32 matmul chain test passed");
    Ok(())
}

#[test]
fn test_f32_outer_product() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Column vector: 3x1
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])?;

    // Row vector: 1x3
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], vec![1, 3])?;

    // Outer product: 3x3
    let result = a.matmul(&b)?;

    assert_eq!(result.shape(), &[3, 3]);

    let data = result.to_vec();

    // Expected:
    // [1*4, 1*5, 1*6]   [4,  5,  6]
    // [2*4, 2*5, 2*6] = [8,  10, 12]
    // [3*4, 3*5, 3*6]   [12, 15, 18]

    let expected = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
    for i in 0..9 {
        assert!((data[i] - expected[i]).abs() < 1e-5);
    }

    println!("✓ f32 outer product test passed");
    Ok(())
}
