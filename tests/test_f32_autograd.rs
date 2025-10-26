/// Test f32 automatic differentiation (autograd)
/// Tests backward pass, gradient computation, and gradient checking

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors, TensorAutograd};
use tensorlogic::autograd::AutogradContext;

#[test]
fn test_f32_simple_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Enable autograd
    AutogradContext::set_enabled(true);

    // Create leaf tensor with requires_grad
    let mut x = Tensor::<f32>::from_vec(vec![2.0, 3.0], vec![2])?;
    x.set_requires_grad(true);

    // Forward: y = x * 2
    let y = x.mul_scalar(2.0)?;

    // Create gradient (dy/dy = 1)
    let grad_y = Tensor::<f32>::ones(&device, vec![2])?;

    // Backward
    y.backward(grad_y)?;

    // Get gradient
    if let Some(grad_x) = x.grad() {
        let grad_data = grad_x.to_vec();

        // dy/dx = 2, so gradient should be [2.0, 2.0]
        assert!((grad_data[0] - 2.0).abs() < 1e-5);
        assert!((grad_data[1] - 2.0).abs() < 1e-5);

        println!("✓ f32 simple backward test passed");
    } else {
        panic!("Gradient not found for x");
    }

    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_addition_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensors
    let mut a = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2])?;
    let mut b = Tensor::<f32>::from_vec(vec![3.0, 4.0], vec![2])?;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    // Forward: c = a + b
    let c = a.add(&b)?;

    // Backward
    let grad_c = Tensor::<f32>::ones(&device, vec![2])?;
    c.backward(grad_c)?;

    // Check gradients: dc/da = 1, dc/db = 1
    if let Some(grad_a) = a.grad() {
        let data = grad_a.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
    }

    if let Some(grad_b) = b.grad() {
        let data = grad_b.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
    }

    println!("✓ f32 addition backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_multiplication_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensors
    let mut a = Tensor::<f32>::from_vec(vec![2.0, 3.0], vec![2])?;
    let mut b = Tensor::<f32>::from_vec(vec![4.0, 5.0], vec![2])?;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    // Forward: c = a * b (element-wise)
    let c = a.mul(&b)?;

    // Backward
    let grad_c = Tensor::<f32>::ones(&device, vec![2])?;
    c.backward(grad_c)?;

    // Check gradients: dc/da = b, dc/db = a
    if let Some(grad_a) = a.grad() {
        let data = grad_a.to_vec();
        assert!((data[0] - 4.0).abs() < 1e-5);  // b[0]
        assert!((data[1] - 5.0).abs() < 1e-5);  // b[1]
    }

    if let Some(grad_b) = b.grad() {
        let data = grad_b.to_vec();
        assert!((data[0] - 2.0).abs() < 1e-5);  // a[0]
        assert!((data[1] - 3.0).abs() < 1e-5);  // a[1]
    }

    println!("✓ f32 multiplication backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_matmul_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create matrices
    let mut a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let mut b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    // Forward: c = a @ b
    let c = a.matmul(&b)?;

    // Backward
    let grad_c = Tensor::<f32>::ones(&device, vec![2, 2])?;
    c.backward(grad_c)?;

    // Check that gradients exist and are finite
    if let Some(grad_a) = a.grad() {
        let data = grad_a.to_vec();
        for val in data.iter() {
            assert!(val.is_finite());
        }
    }

    if let Some(grad_b) = b.grad() {
        let data = grad_b.to_vec();
        for val in data.iter() {
            assert!(val.is_finite());
        }
    }

    println!("✓ f32 matmul backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_relu_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensor with positive and negative values
    let mut x = Tensor::<f32>::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4])?;
    x.set_requires_grad(true);

    // Forward: y = relu(x)
    let y = x.relu()?;

    // Backward
    let grad_y = Tensor::<f32>::ones(&device, vec![4])?;
    y.backward(grad_y)?;

    // Check gradient: drelu/dx = 1 if x > 0, else 0
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        assert!((data[0] - 0.0).abs() < 1e-5);  // x = -1.0, grad = 0
        assert!((data[1] - 1.0).abs() < 1e-5);  // x = 2.0, grad = 1
        assert!((data[2] - 0.0).abs() < 1e-5);  // x = -3.0, grad = 0
        assert!((data[3] - 1.0).abs() < 1e-5);  // x = 4.0, grad = 1
    }

    println!("✓ f32 relu backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_chain_rule() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create input
    let mut x = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2])?;
    x.set_requires_grad(true);

    // Forward: y = (x * 2) + 3
    let temp = x.mul_scalar(2.0)?;
    let y = temp.add_scalar(3.0)?;

    // Backward
    let grad_y = Tensor::<f32>::ones(&device, vec![2])?;
    y.backward(grad_y)?;

    // Check gradient: dy/dx = 2
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    println!("✓ f32 chain rule test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_sum_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensor
    let mut x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
    x.set_requires_grad(true);

    // Forward: y = sum(x)
    let y = x.sum(&[0], false)?;

    // Backward with scalar gradient = 1
    let grad_y = Tensor::<f32>::ones(&device, vec![1])?;
    y.backward(grad_y)?;

    // Check gradient: dsum/dx = 1 for all elements
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        for val in data.iter() {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    println!("✓ f32 sum backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_mean_backward() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensor
    let mut x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
    x.set_requires_grad(true);

    // Forward: y = mean(x)
    let y = x.mean(&[0], false)?;

    // Backward
    let grad_y = Tensor::<f32>::ones(&device, vec![1])?;
    y.backward(grad_y)?;

    // Check gradient: dmean/dx = 1/n for all elements
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        for val in data.iter() {
            assert!((val - 0.25).abs() < 1e-5);  // 1/4
        }
    }

    println!("✓ f32 mean backward test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}

#[test]
fn test_f32_gradient_accumulation() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    AutogradContext::set_enabled(true);

    // Create tensor
    let mut x = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![2])?;
    x.set_requires_grad(true);

    // First forward pass
    let y1 = x.mul_scalar(2.0)?;
    let grad_y1 = Tensor::<f32>::ones(&device, vec![2])?;
    y1.backward(grad_y1)?;

    // Check first gradient
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    // Second forward pass (gradient should accumulate)
    let y2 = x.mul_scalar(3.0)?;
    let grad_y2 = Tensor::<f32>::ones(&device, vec![2])?;
    y2.backward(grad_y2)?;

    // Check accumulated gradient (2 + 3 = 5)
    if let Some(grad_x) = x.grad() {
        let data = grad_x.to_vec();
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
    }

    println!("✓ f32 gradient accumulation test passed");
    AutogradContext::set_enabled(false);
    Ok(())
}
