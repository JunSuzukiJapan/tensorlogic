/// Compare Metal GPU softmax vs CPU softmax
///
/// This test verifies that Metal GPU implementation produces
/// the same numerical results as CPU implementation.

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

fn main() {
    println!("=== Metal GPU vs CPU Softmax Comparison ===\n");

    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test case 1: Simple 2D tensor
    println!("Test 1: Small 2D tensor");
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0,
        5.0, 4.0, 3.0, 2.0, 1.0,
    ];

    let cpu_input = data.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
    let gpu_input = cpu_input.clone();

    // CPU softmax (fallback implementation)
    let cpu_result = softmax_cpu(&cpu_input, &[2, 5], 1);

    // GPU softmax (Metal)
    let tensor = Tensor::from_vec_metal(&device, gpu_input, vec![2, 5])
        .expect("Failed to create tensor");
    let gpu_tensor = tensor.softmax()
        .expect("Failed to run GPU softmax");
    let gpu_result = gpu_tensor.to_vec();

    println!("  Input shape: [2, 5]");
    println!("  Softmax axis: 1 (last dimension)\n");

    println!("  Row 0 input:  {:?}", &data[0..5]);
    println!("  Row 0 CPU:    {:?}", cpu_result[0..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("  Row 0 GPU:    {:?}", gpu_result[0..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // Check sum = 1.0
    let cpu_sum_0: f32 = cpu_result[0..5].iter().map(|x| x.to_f32()).sum();
    let gpu_sum_0: f32 = gpu_result[0..5].iter().map(|x| x.to_f32()).sum();
    println!("  Row 0 sum - CPU: {:.6}, GPU: {:.6}", cpu_sum_0, gpu_sum_0);

    // Compare values
    let mut max_diff = 0.0f32;
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i].to_f32() - gpu_result[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
    }

    println!("\n  Maximum difference: {:.9}", max_diff);

    if max_diff < 1e-4 {
        println!("  ✅ CPU and GPU results match (within tolerance)");
    } else {
        println!("  ❌ WARNING: Significant difference detected!");
    }

    // Test case 2: Larger tensor (like attention scores)
    println!("\n\nTest 2: Larger tensor (simulating attention)");
    let seq_len = 32;
    let mut data2 = Vec::new();
    for i in 0..seq_len {
        data2.push((i as f32) * 0.1);
    }

    let cpu_input2 = data2.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
    let gpu_input2 = cpu_input2.clone();

    let cpu_result2 = softmax_cpu(&cpu_input2, &[1, seq_len], 1);

    let tensor2 = Tensor::from_vec_metal(&device, gpu_input2, vec![1, seq_len])
        .expect("Failed to create tensor");
    let gpu_tensor2 = tensor2.softmax()
        .expect("Failed to run GPU softmax");
    let gpu_result2 = gpu_tensor2.to_vec();

    let cpu_sum: f32 = cpu_result2.iter().map(|x| x.to_f32()).sum();
    let gpu_sum: f32 = gpu_result2.iter().map(|x| x.to_f32()).sum();

    println!("  Shape: [1, {}]", seq_len);
    println!("  CPU sum: {:.9}", cpu_sum);
    println!("  GPU sum: {:.9}", gpu_sum);

    let mut max_diff2 = 0.0f32;
    for i in 0..cpu_result2.len() {
        let diff = (cpu_result2[i].to_f32() - gpu_result2[i].to_f32()).abs();
        max_diff2 = max_diff2.max(diff);
    }

    println!("  Maximum difference: {:.9}", max_diff2);

    if max_diff2 < 1e-4 {
        println!("  ✅ CPU and GPU results match (within tolerance)");
    } else {
        println!("  ❌ WARNING: Significant difference detected!");
    }

    println!("\n=== Comparison Complete ===");
}

/// CPU softmax implementation for comparison
fn softmax_cpu(input: &[half::f16], shape: &[usize], axis: usize) -> Vec<half::f16> {
    assert_eq!(shape.len(), 2);
    assert_eq!(axis, 1, "Only last axis supported for simplicity");

    let batch_size = shape[0];
    let last_dim = shape[1];

    let mut output = vec![half::f16::ZERO; input.len()];

    for b in 0..batch_size {
        let offset = b * last_dim;
        let row = &input[offset..offset + last_dim];

        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for &val in row {
            let v = val.to_f32();
            if v.is_finite() {
                max_val = max_val.max(v);
            }
        }

        if !max_val.is_finite() {
            max_val = 0.0;
        }

        // Compute exp and sum
        let mut sum = 0.0f32;
        let mut exp_vals = vec![0.0f32; last_dim];
        for (i, &val) in row.iter().enumerate() {
            let v = val.to_f32();
            let exp_val = if v.is_finite() {
                (v - max_val).exp()
            } else {
                0.0
            };
            exp_vals[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        if sum > 0.0 && sum.is_finite() {
            for (i, &exp_val) in exp_vals.iter().enumerate() {
                output[offset + i] = half::f16::from_f32(exp_val / sum);
            }
        } else {
            let uniform = 1.0 / last_dim as f32;
            for i in 0..last_dim {
                output[offset + i] = half::f16::from_f32(uniform);
            }
        }
    }

    output
}
