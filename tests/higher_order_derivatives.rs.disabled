//! Integration tests for higher-order derivatives

use tensorlogic::tensor::Tensor;
use tensorlogic::autograd::AutogradContext;
use half::f16;

#[test]
#[ignore] // TODO: Full second derivative requires gradient computation to create graph
fn test_second_derivative_simple() {
    // f(x) = x², f'(x) = 2x, f''(x) = 2
    // NOTE: This test is ignored because full second derivative support
    // requires the backward pass itself to create a computation graph.
    // Current implementation sets gradients as plain tensors without operations.
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    // Forward: y = x²
    let mut y = x.mul(&x).unwrap();

    // First backward: compute dy/dx = 2x = 6
    y.backward_create_graph().unwrap();

    // Retrieve updated tensor from registry
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad1 = x.grad().unwrap();
    let grad1_val = grad1.to_vec()[0].to_f32();

    // Expected: 2 * 3 = 6
    assert!((grad1_val - 6.0).abs() < 0.1, "First derivative: expected 6.0, got {}", grad1_val);

    // Get gradient as a new tensor with computation graph
    let mut grad1_tensor = x.grad().unwrap().clone();

    // Zero out x's gradient for second backward
    x.zero_grad();
    // Re-register x after modifying it
    AutogradContext::register_tensor(x.grad_node().unwrap(), x.clone());

    // Second backward: compute d(dy/dx)/dx = d²y/dx² = 2
    grad1_tensor.backward().unwrap();

    // Retrieve updated tensor from registry again
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad2 = x.grad().unwrap();
    let grad2_val = grad2.to_vec()[0].to_f32();

    // Expected: 2
    assert!((grad2_val - 2.0).abs() < 0.5, "Second derivative: expected 2.0, got {}", grad2_val);
}

#[test]
#[ignore] // TODO: Full second derivative requires gradient computation to create graph
fn test_second_derivative_cubic() {
    // f(x) = x³, f'(x) = 3x², f''(x) = 6x
    // NOTE: Ignored for same reason as test_second_derivative_simple
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    // Forward: y = x³ = x * x * x
    let x2 = x.mul(&x).unwrap();
    let mut y = x2.mul(&x).unwrap();

    // First backward: dy/dx = 3x² = 3 * 4 = 12
    y.backward_create_graph().unwrap();

    // Retrieve updated tensor from registry
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad1 = x.grad().unwrap();
    let grad1_val = grad1.to_vec()[0].to_f32();

    // Expected: 3 * 2² = 12
    assert!((grad1_val - 12.0).abs() < 0.5, "First derivative: expected 12.0, got {}", grad1_val);

    // Get gradient as a new tensor
    let mut grad1_tensor = x.grad().unwrap().clone();

    // Zero out x's gradient
    x.zero_grad();
    // Re-register x after modifying it
    AutogradContext::register_tensor(x.grad_node().unwrap(), x.clone());

    // Second backward: d²y/dx² = 6x = 6 * 2 = 12
    grad1_tensor.backward().unwrap();

    // Retrieve updated tensor from registry again
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad2 = x.grad().unwrap();
    let grad2_val = grad2.to_vec()[0].to_f32();

    // Expected: 6 * 2 = 12
    assert!((grad2_val - 12.0).abs() < 1.0, "Second derivative: expected 12.0, got {}", grad2_val);
}

#[test]
fn test_create_graph_flag() {
    // Test that create_graph=false doesn't allow second derivatives
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.mul(&x).unwrap();

    // Standard backward (create_graph=false)
    y.backward().unwrap();

    // Retrieve updated tensor from registry
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad1 = x.grad().unwrap();

    // Gradient should not have requires_grad=true
    assert!(!grad1.requires_grad(),
        "Gradient should not require grad with standard backward");
}

#[test]
fn test_gradient_requires_grad_with_create_graph() {
    // Test that gradients have requires_grad=true with create_graph
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.mul(&x).unwrap();

    // Backward with create_graph
    y.backward_create_graph().unwrap();

    // Retrieve updated tensor from registry
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();

    let grad = x.grad().unwrap();

    // Gradient should have requires_grad=true
    assert!(grad.requires_grad(), "Gradient should require grad with create_graph");

    // Gradient value should be 2*x = 4
    let grad_val = grad.to_vec()[0].to_f32();
    assert!((grad_val - 4.0).abs() < 0.5, "First derivative: expected 4.0, got {}", grad_val);
}
