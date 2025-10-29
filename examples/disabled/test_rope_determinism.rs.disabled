//! Test RoPE (Rotary Position Embedding) determinism - run same operation 10 times
//! If Metal GPU RoPE is deterministic, all results should be identical

use tensorlogic::tensor::Tensor;
use tensorlogic::device::MetalDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RoPE Determinism Test ===\n");
    println!("Testing if Metal GPU RoPE produces identical results");
    println!("for the same input across multiple runs.\n");

    let device = MetalDevice::new()?;

    // Create test data - same as used in transformer attention
    // Q/K shape after reshape: [seq_len, n_heads, head_dim]
    let seq_len = 2;
    let n_heads = 32;  // TinyLlama has 32 Q heads
    let head_dim = 64; // head_dim = hidden_dim / n_heads = 2048 / 32

    println!("Test: RoPE on Query/Key tensors");
    println!("  Input: Q=[{}, {}, {}]", seq_len, n_heads, head_dim);
    println!("  Operation: rope(Q, position_offset=0)\\n");

    // Create realistic Q tensor values
    let q_data: Vec<f32> = (0..(seq_len * n_heads * head_dim))
        .map(|i| {
            // Create varied values simulating Q projection output
            ((i as f32 * 0.01) % 2.0) - 1.0 // Range: -1.0 to 1.0
        })
        .collect();

    let q_data_f16: Vec<half::f16> = q_data.iter()
        .map(|&x| half::f16::from_f32(x))
        .collect();

    // Run 10 times and collect results
    let mut all_results = Vec::new();

    for run in 0..10 {
        let q = Tensor::from_vec_metal(
            &device,
            q_data_f16.clone(),
            vec![seq_len, n_heads, head_dim]
        )?;

        // Apply RoPE with position_offset=0 (start of sequence)
        let q_rope = q.rope(0)?;
        let result_vec = q_rope.to_vec();

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
        println!("     RoPE is NOT the source of non-determinism");
    } else {
        println!("  ❌ NON-DETERMINISTIC: Results vary across runs");
        println!("     THIS IS THE BUG! RoPE is the source of non-determinism!");
    }

    // Test 2: RoPE with different position_offset
    println!("\\n\\nTest 2: RoPE with position_offset=10 (KV cache scenario)");
    println!("  Input: Q=[{}, {}, {}]", seq_len, n_heads, head_dim);
    println!("  Operation: rope(Q, position_offset=10)\\n");

    let mut all_results2 = Vec::new();

    for run in 0..10 {
        let q = Tensor::from_vec_metal(
            &device,
            q_data_f16.clone(),
            vec![seq_len, n_heads, head_dim]
        )?;

        // Apply RoPE with position_offset=10
        let q_rope = q.rope(10)?;
        let result_vec = q_rope.to_vec();

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
    }

    println!("\\n=== Test Complete ===");

    if all_identical && all_identical2 {
        println!("✅ RoPE is DETERMINISTIC - look elsewhere for non-determinism");
    } else {
        println!("❌ RoPE is NON-DETERMINISTIC - THIS IS THE BUG!");
        println!("   Need to fix Metal GPU RoPE implementation");
    }

    Ok(())
}
