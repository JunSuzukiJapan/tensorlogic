// Mathematical verification of attention computation
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};

#[test]
fn test_attention_basic_math() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Simple test case: seq_len=2, n_heads=1, head_dim=4
    // This allows manual calculation verification

    // Q: [2, 4] - 2 tokens, 4 dimensions
    let q_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 1.0, 0.0, 0.0,  // token 1
    ];
    let q = Tensor::from_vec_gpu(&device, q_data, vec![2, 4]).unwrap();

    // K: [2, 4] - same shape as Q
    let k_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 1.0, 0.0, 0.0,  // token 1
    ];
    let k = Tensor::from_vec_gpu(&device, k_data, vec![2, 4]).unwrap();

    // V: [2, 4]
    let v_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 3.0, 0.0, 0.0,  // token 1
    ];
    let v = Tensor::from_vec_gpu(&device, v_data, vec![2, 4]).unwrap();

    // W_o: [4, 4] - identity for simplicity
    let w_o_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let w_o = Tensor::from_vec_gpu(&device, w_o_data, vec![4, 4]).unwrap();

    // Step 1: Q @ K^T
    // Q @ K^T = [2,4] @ [4,2] = [2,2]
    // Manual calculation:
    // scores[0,0] = q[0] · k[0] = 1*1 = 1.0
    // scores[0,1] = q[0] · k[1] = 1*0 = 0.0
    // scores[1,0] = q[1] · k[0] = 0*1 = 0.0
    // scores[1,1] = q[1] · k[1] = 1*1 = 1.0

    let scores = q.matmul_transposed_b(&k).unwrap();
    let scores_data = scores.sync_and_read();

    println!("\nScores (Q @ K^T):");
    println!("  [{:.4}, {:.4}]", scores_data[0], scores_data[1]);
    println!("  [{:.4}, {:.4}]", scores_data[2], scores_data[3]);

    assert!((scores_data[0] - 1.0).abs() < 1e-5, "scores[0,0] should be 1.0");
    assert!((scores_data[1] - 0.0).abs() < 1e-5, "scores[0,1] should be 0.0");
    assert!((scores_data[2] - 0.0).abs() < 1e-5, "scores[1,0] should be 0.0");
    assert!((scores_data[3] - 1.0).abs() < 1e-5, "scores[1,1] should be 1.0");

    // Step 2: Scale by 1/sqrt(head_dim) = 1/sqrt(4) = 0.5
    let scaled = scores.div_scalar(2.0).unwrap();  // sqrt(4) = 2
    let scaled_data = scaled.sync_and_read();

    println!("\nScaled scores (/ sqrt(4)):");
    println!("  [{:.4}, {:.4}]", scaled_data[0], scaled_data[1]);
    println!("  [{:.4}, {:.4}]", scaled_data[2], scaled_data[3]);

    assert!((scaled_data[0] - 0.5).abs() < 1e-5, "scaled[0,0] should be 0.5");
    assert!((scaled_data[3] - 0.5).abs() < 1e-5, "scaled[1,1] should be 0.5");

    // Step 3: Apply causal mask
    // mask[i,j] = 1 if j <= i else 0
    // For seq_len=2:
    // [[1, 0],
    //  [1, 1]]
    use tensorlogic::device::Device;
    let mask_data = vec![1.0f32, 0.0, 1.0, 1.0];
    let mask = Tensor::from_vec_gpu(&device, mask_data, vec![2, 2]).unwrap();

    let masked = scaled.apply_attention_mask(&mask).unwrap();
    let masked_data = masked.sync_and_read();

    println!("\nMasked scores:");
    println!("  [{:.4}, {:.4}]", masked_data[0], masked_data[1]);
    println!("  [{:.4}, {:.4}]", masked_data[2], masked_data[3]);

    // After masking:
    // [[0.5, -1e9],  -> softmax -> [1.0, 0.0]
    //  [0.0, 0.5]]   -> softmax -> [0.3775, 0.6225]  (not [0.5, 0.5]!)

    // Step 4: Softmax
    let attn_weights = masked.softmax().unwrap();
    let attn_data = attn_weights.sync_and_read();

    println!("\nAttention weights (after softmax):");
    println!("  [{:.4}, {:.4}]", attn_data[0], attn_data[1]);
    println!("  [{:.4}, {:.4}]", attn_data[2], attn_data[3]);

    // Row 0: [0.5, -1e9] -> softmax -> [~1.0, ~0.0]
    assert!(attn_data[0] > 0.99, "attn[0,0] should be ~1.0, got {}", attn_data[0]);
    assert!(attn_data[1] < 0.01, "attn[0,1] should be ~0.0, got {}", attn_data[1]);

    // Row 1: [0.0, 0.5] -> softmax([0.0, 0.5]) = [e^0/(e^0+e^0.5), e^0.5/(e^0+e^0.5)]
    //                                           = [1/2.649, 1.649/2.649] = [0.3775, 0.6225]
    assert!((attn_data[2] - 0.3775).abs() < 0.01, "attn[1,0] should be ~0.3775, got {}", attn_data[2]);
    assert!((attn_data[3] - 0.6225).abs() < 0.01, "attn[1,1] should be ~0.6225, got {}", attn_data[3]);

    // Step 5: attn_weights @ V
    // V = [[2, 0, 0, 0],
    //      [0, 3, 0, 0]]
    //
    // out[0] = 1.0 * v[0] + 0.0 * v[1] = [2, 0, 0, 0]
    // out[1] = 0.3775 * v[0] + 0.6225 * v[1] = [0.755, 1.8675, 0, 0]

    use tensorlogic::tensor::TensorTransform;
    let v_contig = v.contiguous().unwrap();
    let output = attn_weights.matmul(&v_contig).unwrap();
    let output_data = output.sync_and_read();

    println!("\nOutput (attn @ V):");
    println!("  [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[0], output_data[1], output_data[2], output_data[3]);
    println!("  [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[4], output_data[5], output_data[6], output_data[7]);

    // Verify output
    assert!((output_data[0] - 2.0).abs() < 0.01, "output[0,0] should be ~2.0");
    assert!((output_data[1] - 0.0).abs() < 0.01, "output[0,1] should be ~0.0");

    assert!((output_data[4] - 0.755).abs() < 0.01, "output[1,0] should be ~0.755");
    assert!((output_data[5] - 1.8675).abs() < 0.01, "output[1,1] should be ~1.8675");

    println!("\n✅ All attention math checks passed!");
}

#[test]
fn test_rope_position_encoding_math() {
    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test RoPE with known values
    // seq_len=1, n_heads=1, head_dim=4
    // Input: [1, 0, 0, 0] (simple vector)
    // Position 0: theta = 0, so cos=1, sin=0
    // Result should be unchanged for position 0

    let input_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let input = Tensor::from_vec_gpu(&device, input_data.clone(), vec![1, 1, 4]).unwrap();

    let output = input.rope(0).unwrap();
    let output_data = output.sync_and_read();

    println!("\nRoPE test (position=0):");
    println!("  Input:  [{:.4}, {:.4}, {:.4}, {:.4}]",
             input_data[0], input_data[1], input_data[2], input_data[3]);
    println!("  Output: [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_data[0], output_data[1], output_data[2], output_data[3]);

    // At position 0, all thetas are 0, so cos=1, sin=0
    // Rotation formula: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    //                             -> [x0*1 - x1*0, x0*0 + x1*1]
    //                             -> [x0, x1]
    // So output should equal input

    for i in 0..4 {
        assert!((output_data[i] - input_data[i]).abs() < 0.001,
                "RoPE at position 0 should preserve values");
    }

    // Test with position offset
    let output_pos5 = input.rope(5).unwrap();
    let output_pos5_data = output_pos5.sync_and_read();

    println!("\nRoPE test (position=5):");
    println!("  Output: [{:.4}, {:.4}, {:.4}, {:.4}]",
             output_pos5_data[0], output_pos5_data[1],
             output_pos5_data[2], output_pos5_data[3]);

    // Output should be different from input when position > 0
    let mut changed = false;
    for i in 0..4 {
        if (output_pos5_data[i] - input_data[i]).abs() > 0.001 {
            changed = true;
            break;
        }
    }
    assert!(changed, "RoPE should change values at non-zero positions");

    println!("\n✅ RoPE math checks passed!");
}
