/// Test RoPE (Rotary Position Embedding) determinism
///
/// Verifies that RoPE produces identical outputs for identical inputs

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};
use half::f16;

#[test]
fn test_rope_determinism_simple() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Simple RoPE input: [seq_len=2, n_heads=4, head_dim=8]
    let data: Vec<f16> = (1..=64)
        .map(|i| f16::from_f32(i as f32))
        .collect();

    // Run RoPE 5 times with same input
    let mut results = Vec::new();
    for run in 0..5 {
        let input = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2, 4, 8])?;
        let output = input.rope(0)?;
        let output_data = output.sync_and_read();

        println!("Run {}: first 10 = {:?}", run, &output_data[0..10]);
        results.push(output_data);
    }

    // Verify all results identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, (&r, &f)) in result.iter().zip(first.iter()).enumerate() {
            let diff = (r.to_f32() - f.to_f32()).abs();
            assert!(
                diff < 1e-4,
                "Run {} differs at index {}: {} vs {} (diff={})",
                i, j, r.to_f32(), f.to_f32(), diff
            );
        }
    }

    println!("✓ RoPE is deterministic (5 runs identical)");
    Ok(())
}

#[test]
fn test_rope_determinism_transformer_size() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Realistic transformer size: [seq_len=46, n_heads=32, head_dim=64]
    let seq_len = 46;
    let n_heads = 32;
    let head_dim = 64;
    let total = seq_len * n_heads * head_dim;

    let data: Vec<f16> = (0..total)
        .map(|i| f16::from_f32((i % 100) as f32 * 0.1))
        .collect();

    // Run 3 times
    let mut results = Vec::new();
    for run in 0..3 {
        let input = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![seq_len, n_heads, head_dim])?;
        let output = input.rope(0)?;
        let output_data = output.sync_and_read();

        // Check last position (most important for next token prediction)
        let last_pos_start = (seq_len - 1) * n_heads * head_dim;
        let last_pos = &output_data[last_pos_start..last_pos_start + 10];

        println!("Run {}: last position first 10 = {:?}", run, last_pos);
        results.push(output_data);
    }

    // Verify identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(result.len(), first.len());

        // Check last position values
        let last_start = (seq_len - 1) * n_heads * head_dim;
        for j in last_start..last_start + 100 {
            let diff = (result[j].to_f32() - first[j].to_f32()).abs();
            assert!(
                diff < 1e-4,
                "Run {} differs at last position index {}: {} vs {}",
                i, j - last_start, result[j].to_f32(), first[j].to_f32()
            );
        }
    }

    println!("✓ RoPE transformer-size test passed");
    Ok(())
}

#[test]
fn test_rope_with_different_positions() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let data: Vec<f16> = (1..=64).map(|i| f16::from_f32(i as f32)).collect();

    // Apply RoPE with position_offset=0 twice
    let input1 = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2, 4, 8])?;
    let output1a = input1.rope(0)?;
    let data1a = output1a.sync_and_read();

    let input2 = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2, 4, 8])?;
    let output1b = input2.rope(0)?;
    let data1b = output1b.sync_and_read();

    // These should be identical
    for (i, (&a, &b)) in data1a.iter().zip(data1b.iter()).enumerate() {
        let diff = (a.to_f32() - b.to_f32()).abs();
        assert!(diff < 1e-4, "Position 0 differs at {}: {} vs {}", i, a.to_f32(), b.to_f32());
    }

    // Apply RoPE with position_offset=10 twice
    let input3 = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2, 4, 8])?;
    let output2a = input3.rope(10)?;
    let data2a = output2a.sync_and_read();

    let input4 = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![2, 4, 8])?;
    let output2b = input4.rope(10)?;
    let data2b = output2b.sync_and_read();

    // These should also be identical
    for (i, (&a, &b)) in data2a.iter().zip(data2b.iter()).enumerate() {
        let diff = (a.to_f32() - b.to_f32()).abs();
        assert!(diff < 1e-4, "Position 10 differs at {}: {} vs {}", i, a.to_f32(), b.to_f32());
    }

    // But position 0 and position 10 should differ
    let diff_01 = (data1a[0].to_f32() - data2a[0].to_f32()).abs();
    assert!(diff_01 > 0.01, "Position offset should change output");

    println!("✓ RoPE position offset works correctly and deterministically");
    Ok(())
}
