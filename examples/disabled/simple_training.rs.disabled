//! Simple Training Example
//!
//! Demonstrates optimization of a simple function: f(x) = (x - 5)²
//! Minimum at x = 5

use tensorlogic::{Tensor, TensorResult};
use tensorlogic::optim::{Optimizer, Adam};
use tensorlogic::autograd::AutogradContext;
use half::f16;

fn main() -> TensorResult<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  TensorLogic Simple Training Example                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("Problem: Minimize f(x) = (x - 5)²");
    println!("Minimum: x = 5, f(5) = 0\n");

    // Initialize parameter x = 0
    let init_value = vec![f16::ZERO];
    let mut x = Tensor::from_vec(init_value, vec![1])?;
    x.set_requires_grad(true);

    // Create Adam optimizer
    let mut optimizer = Adam::new(vec![x.clone()], 0.1);

    println!("Optimizer: Adam");
    println!("Learning rate: 0.1");
    println!("Initial x: {:.4}\n", x.to_vec()[0].to_f32());

    println!("{:<10} {:<15} {:<15}", "Iteration", "x", "Loss");
    println!("{}", "-".repeat(42));

    // Training loop
    for iteration in 0..50 {
        // Clear gradients
        optimizer.zero_grad();
        AutogradContext::clear();

        // Re-initialize x with requires_grad
        let current_value = optimizer.get_params_mut()[0].to_vec();
        let mut x = Tensor::from_vec(current_value, vec![1])?;
        x.set_requires_grad(true);

        // Compute loss: (x - 5)²
        let target = Tensor::from_vec(vec![f16::from_f32(5.0)], vec![1])?;
        let diff = x.sub(&target)?;
        let mut loss = diff.mul(&diff)?;

        let loss_value = loss.to_vec()[0].to_f32();
        let x_value = x.to_vec()[0].to_f32();

        // Print every 10 iterations
        if iteration % 10 == 0 || iteration == 49 {
            println!("{:<10} {:<15.6} {:<15.6}", iteration, x_value, loss_value);
        }

        // Backward pass
        loss.backward()?;

        // Update parameters
        // Set gradient for the parameter in optimizer
        optimizer.get_params_mut()[0].set_grad(x.grad().unwrap().clone());
        optimizer.get_params_mut()[0].set_requires_grad(true);
        AutogradContext::register_tensor(
            optimizer.get_params_mut()[0].grad_node().unwrap(),
            optimizer.get_params_mut()[0].clone()
        );

        optimizer.step()?;
    }

    let final_x = optimizer.get_params_mut()[0].to_vec()[0].to_f32();
    println!("\nFinal x: {:.6}", final_x);
    println!("Target x: 5.0");
    println!("Error: {:.6}", (final_x - 5.0).abs());

    println!("\n✅ Training completed successfully!");

    Ok(())
}
