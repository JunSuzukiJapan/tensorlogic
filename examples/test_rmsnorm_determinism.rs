//! Test RMSNorm determinism - run same operation 10 times
//! If Metal GPU RMSNorm is deterministic, all results should be identical

use tensorlogic::tensor::Tensor;
use tensorlogic::device::MetalDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RMSNorm Determinism Test ===\n");
    println!("Testing if Metal GPU RMSNorm produces identical results");
    println!("for the same input across multiple runs.\n");

    let device = MetalDevice::new()?;

    // Test 1: Attention normalization (hidden_dim=2048)
    println!("Test 1: Attention RMSNorm");
    let seq_len = 2;
    let hidden_dim = 2048;
    println!("  Input: x=[{}, {}], weight=[{}]", seq_len, hidden_dim, hidden_dim);
    println!("  Output: [{}, {}]\\n", seq_len, hidden_dim);

    // Create test data - realistic hidden states
    let x_data: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();

    // RMSNorm weight - typically ones or learned values close to 1
    let weight_data: Vec<f32> = (0..hidden_dim)
        .map(|i| 1.0 + ((i as f32 * 0.001) % 0.2) - 0.1)
        .collect();

    let x_data_f16: Vec<half::f16> = x_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let weight_data_f16: Vec<half::f16> = weight_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Run 10 times and collect results
    let mut all_results = Vec::new();

    for run in 0..10 {
        let x = Tensor::from_vec_metal(&device, x_data_f16.clone(), vec![seq_len, hidden_dim])?;
        let weight = Tensor::from_vec_metal(&device, weight_data_f16.clone(), vec![hidden_dim])?;

        let normed = x.rms_norm(vec![hidden_dim], &weight, 1e-6)?;
        let result_vec = normed.to_vec();

        all_results.push(result_vec);

        if run == 0 {
            println!("  Run {}: First 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
                run + 1,
                all_results[0][0].to_f32(),
                all_results[0][1].to_f32(),
                all_results[0][2].to_f32(),
                all_results[0][3].to_f32(),
                all_results[0][4].to_f32()
            );
        }
    }

    println!("\\nComparing all 10 runs...");

    let mut all_identical = true;
    let mut max_diff = 0.0f32;

    for run in 1..10 {
        for i in 0..all_results[0].len() {
            let diff = (all_results[0][i].to_f32() - all_results[run][i].to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.0 {
                all_identical = false;
                if run == 1 && i < 5 {
                    println!("  Difference at index {}: run1={:.6}, run{}={:.6}, diff={:.6}",
                        i, all_results[0][i].to_f32(), run + 1, all_results[run][i].to_f32(), diff);
                }
            }
        }
    }

    println!("\\nResults:");
    println!("  Max difference across all runs: {:.10}", max_diff);

    if all_identical {
        println!("  ✅ DETERMINISTIC: All 10 runs produced identical results");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     THIS IS THE BUG!");
    }

    // Test 2: With different input (larger values)
    println!("\\n\\nTest 2: RMSNorm with larger input values");

    let x_data2: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| ((i as f32 * 0.1) % 20.0) - 10.0)
        .collect();
    let x_data2_f16: Vec<half::f16> = x_data2.iter().map(|&x| half::f16::from_f32(x)).collect();

    let mut all_results2 = Vec::new();

    for run in 0..10 {
        let x = Tensor::from_vec_metal(&device, x_data2_f16.clone(), vec![seq_len, hidden_dim])?;
        let weight = Tensor::from_vec_metal(&device, weight_data_f16.clone(), vec![hidden_dim])?;

        let normed = x.rms_norm(vec![hidden_dim], &weight, 1e-6)?;
        let result_vec = normed.to_vec();

        all_results2.push(result_vec);

        if run == 0 {
            println!("  Run {}: First 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
                run + 1,
                all_results2[0][0].to_f32(),
                all_results2[0][1].to_f32(),
                all_results2[0][2].to_f32(),
                all_results2[0][3].to_f32(),
                all_results2[0][4].to_f32()
            );
        }
    }

    println!("\\nComparing all 10 runs...");

    let mut all_identical2 = true;
    let mut max_diff2 = 0.0f32;

    for run in 1..10 {
        for i in 0..all_results2[0].len() {
            let diff = (all_results2[0][i].to_f32() - all_results2[run][i].to_f32()).abs();
            if diff > max_diff2 {
                max_diff2 = diff;
            }
            if diff > 0.0 {
                all_identical2 = false;
                if run == 1 && i < 5 {
                    println!("  Difference at index {}: run1={:.6}, run{}={:.6}, diff={:.6}",
                        i, all_results2[0][i].to_f32(), run + 1, all_results2[run][i].to_f32(), diff);
                }
            }
        }
    }

    println!("\\nResults:");
    println!("  Max difference across all runs: {:.10}", max_diff2);

    if all_identical2 {
        println!("  ✅ DETERMINISTIC: All 10 runs produced identical results");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     THIS IS THE BUG!");
    }

    println!("\\n=== Test Complete ===");

    if all_identical && all_identical2 {
        println!("✅ RMSNorm is DETERMINISTIC - look elsewhere for non-determinism");
    } else {
        println!("❌ RMSNorm is NON-DETERMINISTIC - THIS IS THE BUG!");
        println!("   Need to fix Metal GPU RMSNorm implementation");
    }

    Ok(())
}
