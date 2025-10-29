//! Test Einsum determinism - run same operation 10 times
//! If Metal GPU einsum is deterministic, all results should be identical

use tensorlogic::tensor::Tensor;
use tensorlogic::device::MetalDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Einsum Determinism Test ===\n");
    println!("Testing if Metal GPU einsum produces identical results");
    println!("for the same input across multiple runs.\n");

    let device = MetalDevice::new()?;

    // Create test data - same as used in transformer attention
    let seq_len = 2;
    let n_heads = 4;
    let head_dim = 8;

    println!("Test 1: Attention Scores einsum(\"ihd,jhd->ihj\", Q, K)");
    println!("  Input: Q=[{}, {}, {}], K=[{}, {}, {}]\n", seq_len, n_heads, head_dim, seq_len, n_heads, head_dim);

    let q_data: Vec<f32> = (0..(seq_len * n_heads * head_dim))
        .map(|i| (i as f32 * 0.01) % 1.0)
        .collect();
    let k_data: Vec<f32> = (0..(seq_len * n_heads * head_dim))
        .map(|i| ((i as f32 * 0.02) % 1.0) + 0.1)
        .collect();

    let q_data_f16: Vec<half::f16> = q_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_data_f16: Vec<half::f16> = k_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Run 10 times and collect results
    let mut all_results = Vec::new();
    
    for run in 0..10 {
        let q = Tensor::from_vec_metal(&device, q_data_f16.clone(), vec![seq_len, n_heads, head_dim])?;
        let k = Tensor::from_vec_metal(&device, k_data_f16.clone(), vec![seq_len, n_heads, head_dim])?;
        
        let scores = Tensor::einsum("ihd,jhd->ihj", &[&q, &k])?;
        let result_vec = scores.to_vec();
        
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

    println!("\nComparing all 10 runs...");
    
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

    println!("\nResults:");
    println!("  Max difference across all runs: {:.10}", max_diff);
    
    if all_identical {
        println!("  ✅ DETERMINISTIC: All 10 runs produced identical results");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     This explains the non-deterministic token generation!");
    }

    // Test 2: Attention Output einsum
    println!("\n\nTest 2: Attention Output einsum(\"ihj,jhd->ihd\", attn_weights, V)");
    println!("  Input: attn_weights=[{}, {}, {}], V=[{}, {}, {}]\n", seq_len, n_heads, seq_len, seq_len, n_heads, head_dim);

    let attn_data: Vec<f32> = (0..(seq_len * n_heads * seq_len))
        .map(|i| (i as f32 * 0.05) % 1.0)
        .collect();
    let v_data: Vec<f32> = (0..(seq_len * n_heads * head_dim))
        .map(|i| ((i as f32 * 0.03) % 1.0) + 0.05)
        .collect();

    let attn_data_f16: Vec<half::f16> = attn_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_data_f16: Vec<half::f16> = v_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    let mut all_results2 = Vec::new();
    
    for run in 0..10 {
        let attn = Tensor::from_vec_metal(&device, attn_data_f16.clone(), vec![seq_len, n_heads, seq_len])?;
        let v = Tensor::from_vec_metal(&device, v_data_f16.clone(), vec![seq_len, n_heads, head_dim])?;
        
        let output = Tensor::einsum("ihj,jhd->ihd", &[&attn, &v])?;
        let result_vec = output.to_vec();
        
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

    println!("\nComparing all 10 runs...");
    
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

    println!("\nResults:");
    println!("  Max difference across all runs: {:.10}", max_diff2);
    
    if all_identical2 {
        println!("  ✅ DETERMINISTIC: All 10 runs produced identical results");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     This is the root cause of non-deterministic token generation!");
    }

    println!("\n=== Test Complete ===");
    
    if all_identical && all_identical2 {
        println!("✅ Einsum is DETERMINISTIC - look elsewhere for non-determinism");
    } else {
        println!("❌ Einsum is NON-DETERMINISTIC - THIS IS THE BUG!");
        println!("   Need to fix Metal GPU einsum implementation");
    }
    
    Ok(())
}
