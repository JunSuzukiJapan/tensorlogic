//! Minimal test for backward_create_graph

use tensorlogic::tensor::Tensor;
use tensorlogic::autograd::AutogradContext;
use half::f16;

#[test]
fn test_basic_backward() {
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    println!("Before mul:");
    println!("  x.grad_node() = {:?}", x.grad_node());
    println!("  x.requires_grad() = {}", x.requires_grad());

    let mut y = x.mul(&x).unwrap();

    println!("After mul:");
    println!("  y.grad_node() = {:?}", y.grad_node());
    println!("  y.requires_grad() = {}", y.requires_grad());

    y.backward().unwrap();

    println!("After backward():");

    // IMPORTANT: Must retrieve tensor from registry to get updated gradients
    let x_node_id = x.grad_node().unwrap();
    x = AutogradContext::get_tensor(x_node_id).unwrap();

    if let Some(grad) = x.grad() {
        println!("  x.grad() = {:?}", grad.to_vec());
        let grad_val = grad.to_vec()[0].to_f32();
        assert!((grad_val - 6.0).abs() < 0.5, "Expected gradient ~6.0, got {}", grad_val);
    } else {
        panic!("x.grad() should not be None after backward()");
    }
}

#[test]
fn test_backward_create_graph() {
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.mul(&x).unwrap();

    y.backward_create_graph().unwrap();

    println!("After backward_create_graph():");

    // IMPORTANT: Must retrieve tensor from registry to get updated gradients
    let x_node_id = x.grad_node().unwrap();
    x = AutogradContext::get_tensor(x_node_id).unwrap();

    if let Some(grad) = x.grad() {
        println!("  x.grad() = {:?}", grad.to_vec());
        println!("  x.grad().requires_grad() = {}", grad.requires_grad());
        assert!(grad.requires_grad(), "Gradient should require grad with create_graph");
        let grad_val = grad.to_vec()[0].to_f32();
        assert!((grad_val - 6.0).abs() < 0.5, "Expected gradient ~6.0, got {}", grad_val);
    } else {
        panic!("x.grad() should not be None after backward_create_graph()");
    }
}
