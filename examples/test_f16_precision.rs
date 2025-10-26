//! Test f16 precision impact on basic operations
//!
//! Usage: cargo run --release --example test_f16_precision

use half::f16;

fn main() {
    println!("=== f16 Precision Impact Test ===\n");

    // Test 1: Basic arithmetic
    println!("[Test 1] Basic Arithmetic");
    let a_f32 = 1.0f32 / 3.0f32;
    let a_f16 = f16::from_f32(1.0) / f16::from_f32(3.0);
    println!("  1.0 / 3.0:");
    println!("    f32: {:.10}", a_f32);
    println!("    f16: {:.10}", a_f16.to_f32());
    println!("    Error: {:.10}", (a_f32 - a_f16.to_f32()).abs());
    println!();

    // Test 2: Softmax-like operations
    println!("[Test 2] Exp and Division (Softmax components)");
    let values_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let values_f16: Vec<f16> = values_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let exp_f32: Vec<f32> = values_f32.iter().map(|&v| v.exp()).collect();
    let exp_f16: Vec<f16> = values_f16.iter().map(|&v| {
        let f32_val = v.to_f32();
        f16::from_f32(f32_val.exp())
    }).collect();

    let sum_f32: f32 = exp_f32.iter().sum();
    let sum_f16 = exp_f16.iter()
        .fold(f16::ZERO, |acc, &v| acc + v)
        .to_f32();

    println!("  Exp([1,2,3,4,5]) sum:");
    println!("    f32: {:.6}", sum_f32);
    println!("    f16: {:.6}", sum_f16);
    println!("    Error: {:.6}", (sum_f32 - sum_f16).abs());
    println!();

    // Test 3: Accumulation (1000 additions)
    println!("[Test 3] Accumulation (sum of 1000 ones)");
    let sum_f32: f32 = (0..1000).map(|_| 1.0f32).sum();
    let sum_f16 = (0..1000)
        .map(|_| f16::ONE)
        .fold(f16::ZERO, |acc, v| acc + v)
        .to_f32();

    println!("  f32 sum: {:.6}", sum_f32);
    println!("  f16 sum: {:.6}", sum_f16);
    println!("  Error:   {:.6}", (sum_f32 - sum_f16).abs());
    println!();

    // Test 4: Deep computation simulation (22 layers)
    println!("[Test 4] Deep Computation (22 layers simulation)");
    let mut acc_f32 = 1.0f32;
    let mut acc_f16 = f16::ONE;

    for layer in 0..22 {
        // Simulate: x = x * 1.1 + 0.1 (simple layer operation)
        acc_f32 = acc_f32 * 1.1 + 0.1;
        acc_f16 = acc_f16 * f16::from_f32(1.1) + f16::from_f32(0.1);

        if layer == 0 || layer == 10 || layer == 21 {
            println!("  Layer {}: f32={:.6}, f16={:.6}, diff={:.6}",
                     layer, acc_f32, acc_f16.to_f32(),
                     (acc_f32 - acc_f16.to_f32()).abs());
        }
    }
    println!();

    // Summary
    println!("=== Summary ===");
    println!("f16 precision characteristics:");
    println!("  - Significant digits: ~3-4 decimal places");
    println!("  - Small errors in simple operations: <0.0001");
    println!("  - Accumulation errors increase with depth");
    println!("  - 22-layer model accumulates errors: ~{:.6}",
             (acc_f32 - acc_f16.to_f32()).abs());
    println!();
    println!("💡 Conclusion:");
    println!("  f16 errors are SMALL but CUMULATIVE in deep models.");
    println!("  This could partially explain differences vs llama.cpp.");
}
