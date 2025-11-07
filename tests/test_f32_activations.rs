/// Test f32 activation functions
/// Tests relu, gelu, softmax, sigmoid, tanh, and other activation functions

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors};
use tensorlogic::ops::activations::{TensorActivations, Activation};

#[test]
fn test_f32_relu() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor with positive and negative values
    let t = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        vec![6]
    )?;

    // Apply ReLU
    let result = t.relu()?;
    let data = result.sync_and_read();

    // Verify ReLU output (max(0, x))
    assert!((data[0] - 0.0).abs() < 1e-6);  // -2.0 -> 0.0
    assert!((data[1] - 0.0).abs() < 1e-6);  // -1.0 -> 0.0
    assert!((data[2] - 0.0).abs() < 1e-6);  // 0.0 -> 0.0
    assert!((data[3] - 1.0).abs() < 1e-6);  // 1.0 -> 1.0
    assert!((data[4] - 2.0).abs() < 1e-6);  // 2.0 -> 2.0
    assert!((data[5] - 3.0).abs() < 1e-6);  // 3.0 -> 3.0

    println!("✓ f32 relu test passed");
    Ok(())
}

#[test]
fn test_f32_gelu() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor
    let t = Tensor::<f32>::from_vec(
        vec![-1.0, 0.0, 1.0, 2.0],
        vec![4]
    )?;

    // Apply GELU
    let result = t.gelu()?;
    let data = result.sync_and_read();

    // GELU(0) should be approximately 0
    assert!(data[1].abs() < 0.1);

    // GELU is approximately linear for large positive values
    // GELU(2.0) should be close to 2.0
    assert!((data[3] - 2.0).abs() < 0.1);

    // Verify all values are finite
    for val in data.iter() {
        assert!(val.is_finite());
    }

    println!("✓ f32 gelu test passed");
    Ok(())
}

#[test]
fn test_f32_softmax() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor
    let t = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4]
    )?;

    // Apply softmax
    let result = t.softmax(0)?;
    let data = result.sync_and_read();

    // Verify sum is approximately 1.0
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Verify all values are positive
    for val in data.iter() {
        assert!(*val > 0.0);
        assert!(*val < 1.0);
    }

    // Verify values are in increasing order (since input is increasing)
    for i in 0..3 {
        assert!(data[i] < data[i+1]);
    }

    println!("✓ f32 softmax test passed");
    Ok(())
}

#[test]
fn test_f32_sigmoid() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor
    let t = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![5]
    )?;

    // Apply sigmoid
    let result = t.sigmoid()?;
    let data = result.sync_and_read();

    // Verify sigmoid(0) ≈ 0.5
    assert!((data[2] - 0.5).abs() < 0.01);

    // Verify all values are in (0, 1)
    for val in data.iter() {
        assert!(*val > 0.0 && *val < 1.0);
    }

    // Verify sigmoid is monotonically increasing
    for i in 0..4 {
        assert!(data[i] < data[i+1]);
    }

    println!("✓ f32 sigmoid test passed");
    Ok(())
}

#[test]
fn test_f32_tanh() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor
    let t = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![5]
    )?;

    // Apply tanh
    let result = t.tanh()?;
    let data = result.sync_and_read();

    // Verify tanh(0) ≈ 0
    assert!(data[2].abs() < 0.01);

    // Verify all values are in (-1, 1)
    for val in data.iter() {
        assert!(*val > -1.0 && *val < 1.0);
    }

    // Verify tanh is monotonically increasing
    for i in 0..4 {
        assert!(data[i] < data[i+1]);
    }

    // Verify tanh is odd function: tanh(-x) = -tanh(x)
    assert!((data[0] + data[4]).abs() < 0.01);  // tanh(-2) + tanh(2) ≈ 0
    assert!((data[1] + data[3]).abs() < 0.01);  // tanh(-1) + tanh(1) ≈ 0

    println!("✓ f32 tanh test passed");
    Ok(())
}

#[test]
fn test_f32_swiglu() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create input tensors for SwiGLU
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
    let gate = Tensor::<f32>::from_vec(vec![0.5, 1.0, 1.5, 2.0], vec![4])?;

    // Apply SwiGLU: x * sigmoid(gate)
    let gate_sigmoid = gate.sigmoid()?;
    let result = x.mul(&gate_sigmoid)?;
    let data = result.sync_and_read();

    // Verify all values are positive (since input is positive)
    for val in data.iter() {
        assert!(*val > 0.0);
        assert!(val.is_finite());
    }

    println!("✓ f32 swiglu test passed");
    Ok(())
}

#[test]
fn test_f32_activation_combinations() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Test combining activations
    let t = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![5]
    )?;

    // Apply ReLU then softmax
    let t1 = t.relu()?;
    let t2 = t1.softmax(0)?;

    let data = t2.sync_and_read();

    // Verify sum is 1.0
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Verify all non-negative
    for val in data.iter() {
        assert!(*val >= 0.0);
    }

    println!("✓ f32 activation combinations test passed");
    Ok(())
}

#[test]
fn test_f32_leaky_relu() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Create tensor with negative values
    let t = Tensor::<f32>::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![5]
    )?;

    // Apply LeakyReLU with negative_slope = 0.01
    let result = t.leaky_relu(0.01)?;
    let data = result.sync_and_read();

    // Verify negative values are scaled by 0.01
    assert!((data[0] - (-0.02)).abs() < 1e-6);  // -2.0 * 0.01
    assert!((data[1] - (-0.01)).abs() < 1e-6);  // -1.0 * 0.01

    // Verify zero stays zero
    assert!(data[2].abs() < 1e-6);

    // Verify positive values unchanged
    assert!((data[3] - 1.0).abs() < 1e-6);
    assert!((data[4] - 2.0).abs() < 1e-6);

    println!("✓ f32 leaky_relu test passed");
    Ok(())
}
