//! Test Matmul/Linear determinism - run same operation 10 times
//! If Metal GPU matmul is deterministic, all results should be identical

use tensorlogic::tensor::Tensor;
use tensorlogic::device::MetalDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Matmul/Linear Determinism Test ===\n");
    println!("Testing if Metal GPU matmul produces identical results");
    println!("for the same input across multiple runs.\n");

    let device = MetalDevice::new()?;

    // Test 1: Query projection (used in attention)
    // Input: [seq_len, hidden_dim] @ [hidden_dim, hidden_dim]
    println!("Test 1: Query Projection linear(x, W_q)");
    let seq_len = 2;
    let hidden_dim = 2048;
    println!("  Input: x=[{}, {}], W_q=[{}, {}]", seq_len, hidden_dim, hidden_dim, hidden_dim);
    println!("  Output: [{}, {}]\\n", seq_len, hidden_dim);

    // Create test data
    let x_data: Vec<f32> = (0..(seq_len * hidden_dim))
        .map(|i| ((i as f32 * 0.01) % 2.0) - 1.0)
        .collect();
    let w_q_data: Vec<f32> = (0..(hidden_dim * hidden_dim))
        .map(|i| ((i as f32 * 0.001) % 1.0) - 0.5)
        .collect();

    let x_data_f16: Vec<half::f16> = x_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let w_q_data_f16: Vec<half::f16> = w_q_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Run 10 times and collect results
    let mut all_results = Vec::new();

    for run in 0..10 {
        let x = Tensor::from_vec_metal(&device, x_data_f16.clone(), vec![seq_len, hidden_dim])?;
        let w_q = Tensor::from_vec_metal(&device, w_q_data_f16.clone(), vec![hidden_dim, hidden_dim])?;

        let q = x.matmul(&w_q)?;
        let result_vec = q.to_vec();

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

    // Test 2: FFN Gate projection (larger matmul)
    println!("\\n\\nTest 2: FFN Gate Projection linear(x, W_gate)");
    let ffn_dim = 5632; // TinyLlama FFN intermediate dimension
    println!("  Input: x=[{}, {}], W_gate=[{}, {}]", seq_len, hidden_dim, hidden_dim, ffn_dim);
    println!("  Output: [{}, {}]\\n", seq_len, ffn_dim);

    let w_gate_data: Vec<f32> = (0..(hidden_dim * ffn_dim))
        .map(|i| ((i as f32 * 0.0001) % 0.5) - 0.25)
        .collect();
    let w_gate_data_f16: Vec<half::f16> = w_gate_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    let mut all_results2 = Vec::new();

    for run in 0..10 {
        let x = Tensor::from_vec_metal(&device, x_data_f16.clone(), vec![seq_len, hidden_dim])?;
        let w_gate = Tensor::from_vec_metal(&device, w_gate_data_f16.clone(), vec![hidden_dim, ffn_dim])?;

        let gate = x.matmul(&w_gate)?;
        let result_vec = gate.to_vec();

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

    // Test 3: Output projection (vocab_size dimension)
    println!("\\n\\nTest 3: Output Projection linear(x, output.weight)");
    let vocab_size = 32000;
    println!("  Input: x=[{}, {}], W_out=[{}, {}]", seq_len, hidden_dim, vocab_size, hidden_dim);
    println!("  Output: [{}, {}]\\n", seq_len, vocab_size);

    // Note: We transpose for linear() which expects [hidden_dim, vocab_size]
    let w_out_data: Vec<f32> = (0..(hidden_dim * vocab_size))
        .map(|i| ((i as f32 * 0.00001) % 0.2) - 0.1)
        .collect();
    let w_out_data_f16: Vec<half::f16> = w_out_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    let mut all_results3 = Vec::new();

    for run in 0..10 {
        let x = Tensor::from_vec_metal(&device, x_data_f16.clone(), vec![seq_len, hidden_dim])?;
        // linear() expects weight as [hidden_dim, output_dim] and transposes internally
        let w_out = Tensor::from_vec_metal(&device, w_out_data_f16.clone(), vec![vocab_size, hidden_dim])?;

        let logits = x.matmul(&w_out.transpose()?)?;
        let result_vec = logits.to_vec();

        all_results3.push(result_vec);

        if run == 0 {
            println!("  Run {}: First 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
                run + 1,
                all_results3[0][0].to_f32(),
                all_results3[0][1].to_f32(),
                all_results3[0][2].to_f32(),
                all_results3[0][3].to_f32(),
                all_results3[0][4].to_f32()
            );
        }
    }

    println!("\\nComparing all 10 runs...");

    let mut all_identical3 = true;
    let mut max_diff3 = 0.0f32;

    for run in 1..10 {
        for i in 0..all_results3[0].len() {
            let diff = (all_results3[0][i].to_f32() - all_results3[run][i].to_f32()).abs();
            if diff > max_diff3 {
                max_diff3 = diff;
            }
            if diff > 0.0 {
                all_identical3 = false;
                if run == 1 && i < 5 {
                    println!("  Difference at index {}: run1={:.6}, run{}={:.6}, diff={:.6}",
                        i, all_results3[0][i].to_f32(), run + 1, all_results3[run][i].to_f32(), diff);
                }
            }
        }
    }

    println!("\\nResults:");
    println!("  Max difference across all runs: {:.10}", max_diff3);

    if all_identical3 {
        println!("  ✅ DETERMINISTIC: All 10 runs produced identical results");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     THIS IS THE BUG!");
    }

    println!("\\n=== Test Complete ===");

    if all_identical && all_identical2 && all_identical3 {
        println!("✅ Matmul is DETERMINISTIC - look elsewhere for non-determinism");
    } else {
        println!("❌ Matmul is NON-DETERMINISTIC - THIS IS THE BUG!");
        println!("   Need to fix Metal GPU matmul implementation");
    }

    Ok(())
}
