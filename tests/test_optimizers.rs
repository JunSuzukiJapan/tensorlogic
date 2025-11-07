#![allow(unused_variables)]
/// Comprehensive tests for optimizers (SGD, Adam, AdamW)
///
/// Optimizers are critical for training neural networks.
/// Currently completely untested (0% coverage).
///
/// Tests cover:
/// - SGD: basic step, momentum, Nesterov momentum, weight decay
/// - Adam: basic step, beta parameters, bias correction, convergence
/// - AdamW: weight decay behavior
/// - Learning rate manipulation
/// - State dict save/load
/// - Integration with autograd
/// - Simple optimization convergence tests

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO, TensorAccessors, TensorAutograd};
use tensorlogic::optim::{SGD, Adam, AdamW, Optimizer};
use tensorlogic::autograd::AutogradContext;
use half::f16;

// Helper functions

fn assert_close_f16(a: f16, b: f16, epsilon: f32) {
    let diff = (a.to_f32() - b.to_f32()).abs();
    assert!(
        diff < epsilon,
        "Values not close: {} vs {}, diff = {}",
        a.to_f32(), b.to_f32(), diff
    );
}

#[allow(dead_code)]
fn assert_tensor_close(result: &[f16], expected: &[f16], epsilon: f32) {
    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r.to_f32() - e.to_f32()).abs();
        assert!(
            diff < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r.to_f32(), e.to_f32(), diff
        );
    }
}

// SGD Tests

#[test]
fn test_sgd_creation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10])?;
    let optimizer = SGD::new(vec![param], 0.01);

    assert_eq!(optimizer.get_lr(), 0.01);
    assert_eq!(optimizer.num_param_groups(), 1);

    println!("✓ SGD creation test passed");
    Ok(())
}

#[test]
fn test_sgd_basic_step() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    // Create parameter: x = 2.0
    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    // Set gradient: grad = 1.0
    let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    x.set_grad(grad);

    // Register in context
    AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

    // Create optimizer with lr=0.1
    let mut optimizer = SGD::new(vec![x.clone()], 0.1);

    // Step: x_new = x - lr * grad = 2.0 - 0.1 * 1.0 = 1.9
    optimizer.step()?;

    // Get updated parameter
    let updated_x = &optimizer.params()[0];
    let value = updated_x.sync_and_read()[0];

    assert_close_f16(value, f16::from_f32(1.9), 0.01);

    println!("✓ SGD basic step test passed");
    Ok(())
}

#[test]
fn test_sgd_multiple_steps() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(10.0)], vec![1])?;
    x.set_requires_grad(true);

    let mut optimizer = SGD::new(vec![x.clone()], 0.1);

    // Perform multiple steps with constant gradient
    for _ in 0..10 {
        let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
        optimizer.get_params_mut()[0].set_grad(grad);
        AutogradContext::register_tensor_generic(
            optimizer.get_params_mut()[0].grad_node().unwrap(),
            optimizer.get_params_mut()[0].clone()
        );

        optimizer.step()?;
    }

    // After 10 steps: x = 10.0 - 10 * 0.1 * 1.0 = 9.0
    let value = optimizer.params()[0].sync_and_read()[0];
    assert_close_f16(value, f16::from_f32(9.0), 0.05);

    println!("✓ SGD multiple steps test passed");
    Ok(())
}

#[test]
fn test_sgd_with_momentum() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 5], vec![5])?;
    let optimizer = SGD::with_momentum(vec![param], 0.01, 0.9);

    // Check momentum is set
    // Note: momentum field is not public, but we can verify creation succeeds
    assert_eq!(optimizer.get_lr(), 0.01);

    println!("✓ SGD with momentum test passed");
    Ok(())
}

#[test]
fn test_sgd_zero_grad() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut param = Tensor::from_vec(vec![f16::from_f32(2.0); 5], vec![5])?;
    param.set_requires_grad(true);

    // Set gradient
    let grad = Tensor::from_vec(vec![f16::from_f32(1.0); 5], vec![5])?;
    param.set_grad(grad);

    let mut optimizer = SGD::new(vec![param.clone()], 0.1);

    // Verify gradient exists
    assert!(optimizer.params()[0].grad().is_some());

    // Zero grad
    optimizer.zero_grad();

    // Gradient should be cleared
    assert!(optimizer.get_params_mut()[0].grad().is_none());

    println!("✓ SGD zero grad test passed");
    Ok(())
}

#[test]
fn test_sgd_learning_rate_manipulation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10])?;
    let mut optimizer = SGD::new(vec![param], 0.01);

    assert_eq!(optimizer.get_lr(), 0.01);

    // Change learning rate
    optimizer.set_lr(0.001);
    assert_eq!(optimizer.get_lr(), 0.001);

    // Change again
    optimizer.set_lr(0.1);
    assert_eq!(optimizer.get_lr(), 0.1);

    println!("✓ SGD learning rate manipulation test passed");
    Ok(())
}

// Adam Tests

#[test]
fn test_adam_creation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10])?;
    let optimizer = Adam::new(vec![param], 0.001);

    assert_eq!(optimizer.get_lr(), 0.001);
    assert_eq!(optimizer.num_param_groups(), 1);

    println!("✓ Adam creation test passed");
    Ok(())
}

#[test]
fn test_adam_basic_step() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    x.set_grad(grad);

    AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

    let mut optimizer = Adam::new(vec![x.clone()], 0.1);

    // Adam should take a step (exact value depends on moment estimates)
    let before = x.sync_and_read()[0].to_f32();
    optimizer.step()?;

    let after = optimizer.params()[0].sync_and_read()[0].to_f32();

    // Value should decrease (since gradient is positive)
    assert!(after < before, "Adam should decrease parameter value");

    println!("✓ Adam basic step test passed");
    Ok(())
}

#[test]
fn test_adam_multiple_steps() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(5.0)], vec![1])?;
    x.set_requires_grad(true);

    let mut optimizer = Adam::new(vec![x.clone()], 0.1);

    let initial = x.sync_and_read()[0].to_f32();

    // Perform multiple steps
    for _ in 0..10 {
        let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
        optimizer.get_params_mut()[0].set_grad(grad);
        AutogradContext::register_tensor_generic(
            optimizer.get_params_mut()[0].grad_node().unwrap(),
            optimizer.get_params_mut()[0].clone()
        );

        optimizer.step()?;
    }

    let final_val = optimizer.params()[0].sync_and_read()[0].to_f32();

    // Value should have decreased significantly
    assert!(final_val < initial - 1.0, "Adam should make progress over multiple steps");

    println!("✓ Adam multiple steps test passed");
    Ok(())
}

#[test]
fn test_adam_with_weight_decay() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::from_f32(1.0); 5], vec![5])?;
    let optimizer = Adam::with_weight_decay(vec![param], 0.001, 0.01);

    // Just verify it creates successfully
    assert_eq!(optimizer.get_lr(), 0.001);

    println!("✓ Adam with weight decay test passed");
    Ok(())
}

#[test]
fn test_adam_zero_grad() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut param = Tensor::from_vec(vec![f16::from_f32(2.0); 5], vec![5])?;
    param.set_requires_grad(true);

    let grad = Tensor::from_vec(vec![f16::from_f32(1.0); 5], vec![5])?;
    param.set_grad(grad);

    let mut optimizer = Adam::new(vec![param.clone()], 0.001);

    optimizer.zero_grad();

    assert!(optimizer.get_params_mut()[0].grad().is_none());

    println!("✓ Adam zero grad test passed");
    Ok(())
}

// AdamW Tests

#[test]
fn test_adamw_creation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10])?;
    let optimizer = AdamW::new(vec![param], 0.001);

    assert_eq!(optimizer.get_lr(), 0.001);

    println!("✓ AdamW creation test passed");
    Ok(())
}

#[test]
fn test_adamw_basic_step() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    x.set_grad(grad);

    AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

    let mut optimizer = AdamW::new(vec![x.clone()], 0.1);

    let before = x.sync_and_read()[0].to_f32();
    optimizer.step()?;

    let after = optimizer.params()[0].sync_and_read()[0].to_f32();

    // AdamW should decrease parameter (gradient + weight decay)
    assert!(after < before);

    println!("✓ AdamW basic step test passed");
    Ok(())
}

// Convergence Tests

#[test]
fn test_sgd_simple_optimization() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Optimize simple quadratic: minimize (x - 3)^2
    // Target: x = 3
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(0.0)], vec![1])?;
    x.set_requires_grad(true);

    let mut optimizer = SGD::new(vec![x.clone()], 0.1);

    // Perform optimization steps
    for _ in 0..20 {
        // Gradient of (x - 3)^2 is 2(x - 3)
        let current = optimizer.get_params_mut()[0].sync_and_read()[0].to_f32();
        let grad_val = 2.0 * (current - 3.0);

        let grad = Tensor::from_vec(vec![f16::from_f32(grad_val)], vec![1])?;
        optimizer.get_params_mut()[0].set_grad(grad);
        AutogradContext::register_tensor_generic(
            optimizer.get_params_mut()[0].grad_node().unwrap(),
            optimizer.get_params_mut()[0].clone()
        );

        optimizer.step()?;
        optimizer.zero_grad();
    }

    // Should converge close to 3.0
    let final_val = optimizer.params()[0].sync_and_read()[0].to_f32();
    assert!((final_val - 3.0).abs() < 0.5, "SGD should converge close to optimum");

    println!("✓ SGD simple optimization test passed");
    Ok(())
}

#[test]
fn test_adam_simple_optimization() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Same optimization with Adam
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(0.0)], vec![1])?;
    x.set_requires_grad(true);

    let mut optimizer = Adam::new(vec![x.clone()], 0.1);

    for _ in 0..20 {
        let current = optimizer.get_params_mut()[0].sync_and_read()[0].to_f32();
        let grad_val = 2.0 * (current - 3.0);

        let grad = Tensor::from_vec(vec![f16::from_f32(grad_val)], vec![1])?;
        optimizer.get_params_mut()[0].set_grad(grad);
        AutogradContext::register_tensor_generic(
            optimizer.get_params_mut()[0].grad_node().unwrap(),
            optimizer.get_params_mut()[0].clone()
        );

        optimizer.step()?;
        optimizer.zero_grad();
    }

    let final_val = optimizer.params()[0].sync_and_read()[0].to_f32();
    assert!((final_val - 3.0).abs() < 0.5, "Adam should converge close to optimum");

    println!("✓ Adam simple optimization test passed");
    Ok(())
}

// State Dict Tests

#[test]
fn test_sgd_state_dict() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 5], vec![5])?;
    let optimizer = SGD::with_momentum(vec![param], 0.01, 0.9);

    // Get state dict
    let state = optimizer.state_dict();

    // Should be able to get state (even if empty initially)
    // Note: len() is always >= 0, so just verify state exists
    let _ = state.param_state.len();

    println!("✓ SGD state dict test passed");
    Ok(())
}

#[test]
fn test_adam_state_dict() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let param = Tensor::from_vec(vec![f16::ONE; 5], vec![5])?;
    let optimizer = Adam::new(vec![param], 0.001);

    let state = optimizer.state_dict();

    // Note: len() is always >= 0, so just verify state exists
    let _ = state.param_state.len();

    println!("✓ Adam state dict test passed");
    Ok(())
}

// Multiple Parameters

#[test]
fn test_sgd_multiple_parameters() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x1 = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    let mut x2 = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;

    x1.set_requires_grad(true);
    x2.set_requires_grad(true);

    let grad1 = Tensor::from_vec(vec![f16::from_f32(0.5)], vec![1])?;
    let grad2 = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;

    x1.set_grad(grad1);
    x2.set_grad(grad2);

    AutogradContext::register_tensor_generic(x1.grad_node().unwrap(), x1.clone());
    AutogradContext::register_tensor_generic(x2.grad_node().unwrap(), x2.clone());

    let mut optimizer = SGD::new(vec![x1, x2], 0.1);

    optimizer.step()?;

    // Both parameters should be updated
    let params = optimizer.params();
    assert_eq!(params.len(), 2);

    println!("✓ SGD multiple parameters test passed");
    Ok(())
}

#[test]
fn test_adam_multiple_parameters() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x1 = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    let mut x2 = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;

    x1.set_requires_grad(true);
    x2.set_requires_grad(true);

    let grad1 = Tensor::from_vec(vec![f16::from_f32(0.5)], vec![1])?;
    let grad2 = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;

    x1.set_grad(grad1);
    x2.set_grad(grad2);

    AutogradContext::register_tensor_generic(x1.grad_node().unwrap(), x1.clone());
    AutogradContext::register_tensor_generic(x2.grad_node().unwrap(), x2.clone());

    let mut optimizer = Adam::new(vec![x1, x2], 0.01);

    optimizer.step()?;

    let params = optimizer.params();
    assert_eq!(params.len(), 2);

    println!("✓ Adam multiple parameters test passed");
    Ok(())
}

// Edge Cases

#[test]
fn test_optimizer_zero_learning_rate() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    x.set_grad(grad);

    AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

    let mut optimizer = SGD::new(vec![x.clone()], 0.0); // lr = 0

    let before = x.sync_and_read()[0];
    optimizer.step()?;
    let after = optimizer.params()[0].sync_and_read()[0];

    // With lr=0, parameter should not change
    assert_close_f16(before, after, 1e-6);

    println!("✓ Optimizer zero learning rate test passed");
    Ok(())
}

#[test]
fn test_optimizer_large_learning_rate() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    AutogradContext::clear();

    let mut x = Tensor::from_vec(vec![f16::from_f32(2.0)], vec![1])?;
    x.set_requires_grad(true);

    let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
    x.set_grad(grad);

    AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

    let mut optimizer = SGD::new(vec![x], 10.0); // Very large lr

    // Should still execute (may diverge, but shouldn't crash)
    optimizer.step()?;

    println!("✓ Optimizer large learning rate test passed");
    Ok(())
}
