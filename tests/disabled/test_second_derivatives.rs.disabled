//! Tests for second-order derivatives (Phase 8.6)

use tensorlogic::autograd::AutogradContext;
use tensorlogic::tensor::Tensor;

#[test]
fn test_second_derivative_simple() {
    // f(x) = x²
    // f'(x) = 2x
    // f''(x) = 2

    let mut x = Tensor::from_vec(vec![half::f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    // Forward: f(x) = x²
    let mut y = x.mul(&x).unwrap();

    // First derivative with create_graph
    y.backward_create_graph().unwrap();

    // Retrieve x from registry to get gradient
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let grad = x.grad().unwrap();

    // f'(3) = 2*3 = 6
    assert!((grad.sync_and_read()[0].to_f32() - 6.0).abs() < 0.1);

    // Gradient should have requires_grad=true
    assert!(grad.requires_grad());

    // TODO: Second derivative requires gradient computation to be recorded as operation
    // This is deferred to full Phase 8.6 implementation
    // For now, we verify that gradient has requires_grad set correctly
}

#[test]
fn test_gradient_requires_grad_with_create_graph() {
    // Test that gradients have requires_grad=true when using create_graph

    let mut x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.mul(&x).unwrap(); // y = x²

    // Backward with create_graph
    y.backward_create_graph().unwrap();

    // Retrieve updated x from registry
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let grad = x.grad().unwrap();

    // Gradient should be 2*x = 4.0
    assert!((grad.sync_and_read()[0].to_f32() - 4.0).abs() < 0.1);

    // With create_graph, gradient should have requires_grad=true
    assert!(grad.requires_grad());
    assert!(grad.grad_node().is_some());
}

#[test]
fn test_gradient_without_create_graph() {
    // Test that gradients do NOT have requires_grad without create_graph

    let mut x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.mul(&x).unwrap();

    // Backward WITHOUT create_graph
    y.backward().unwrap();

    // Retrieve updated x
    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let grad = x.grad().unwrap();

    // Gradient value should still be correct
    assert!((grad.sync_and_read()[0].to_f32() - 4.0).abs() < 0.1);

    // Without create_graph, gradient should NOT have requires_grad
    assert!(!grad.requires_grad());
}

#[test]
fn test_chain_rule_first_derivative() {
    // f(x) = (x²)²
    // f'(x) = 2*(x²) * 2x = 4x³

    let mut x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let x_sq = x.mul(&x).unwrap(); // x²
    let mut y = x_sq.mul(&x_sq).unwrap(); // (x²)²

    y.backward().unwrap();

    x = AutogradContext::get_tensor(x.grad_node().unwrap()).unwrap();
    let grad = x.grad().unwrap();

    // f'(2) = 4 * 2³ = 4 * 8 = 32
    assert!((grad.sync_and_read()[0].to_f32() - 32.0).abs() < 0.5);
}
