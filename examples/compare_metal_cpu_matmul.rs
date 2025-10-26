/// Compare Metal GPU matmul vs CPU matmul
///
/// This is the most critical test because matmul is used extensively
/// in model inference (Q/K/V projections, output projection, FFN, etc.)

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

fn main() {
    println!("=== Metal GPU vs CPU Matmul Comparison ===\n");

    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test 1: Small matrix multiplication
    println!("Test 1: Small matmul (2x3) @ (3x2)");

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]

    // CPU version
    let cpu_result = matmul_cpu(&a_data, &b_data, &[2, 3], &[3, 2]);

    // GPU version
    let a_f16 = a_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b_f16 = b_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    let tensor_a = Tensor::from_vec_metal(&device, a_f16, vec![2, 3])
        .expect("Failed to create tensor A");
    let tensor_b = Tensor::from_vec_metal(&device, b_f16, vec![3, 2])
        .expect("Failed to create tensor B");

    let gpu_result_tensor = tensor_a.matmul(&tensor_b)
        .expect("Failed to run GPU matmul");
    let gpu_result = gpu_result_tensor.to_vec();

    println!("  A shape: [2, 3]");
    println!("  B shape: [3, 2]");
    println!("  Output shape: [2, 2]\n");

    println!("  CPU result:");
    println!("    [{:.2}, {:.2}]", cpu_result[0].to_f32(), cpu_result[1].to_f32());
    println!("    [{:.2}, {:.2}]", cpu_result[2].to_f32(), cpu_result[3].to_f32());

    println!("  GPU result:");
    println!("    [{:.2}, {:.2}]", gpu_result[0].to_f32(), gpu_result[1].to_f32());
    println!("    [{:.2}, {:.2}]", gpu_result[2].to_f32(), gpu_result[3].to_f32());

    let mut max_diff = 0.0f32;
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i].to_f32() - gpu_result[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
    }

    println!("\n  Maximum difference: {:.9}", max_diff);

    if max_diff < 0.01 {
        println!("  ✅ CPU and GPU results match");
    } else {
        println!("  ❌ WARNING: Significant difference!");
    }

    // Test 2: Larger matmul (simulating Q projection)
    println!("\n\nTest 2: Larger matmul (simulating model layer)");
    println!("  Shape: [1, 2048] @ [2048, 2048] -> [1, 2048]");

    let hidden_size = 2048;

    // Create simple test data
    let mut a2_data = Vec::new();
    for i in 0..hidden_size {
        a2_data.push((i as f32) * 0.001);
    }

    let mut b2_data = Vec::new();
    for i in 0..hidden_size * hidden_size {
        // Simple pattern for weight matrix
        b2_data.push(if i % 100 == 0 { 1.0 } else { 0.001 });
    }

    // CPU version
    let cpu_result2 = matmul_cpu(&a2_data, &b2_data, &[1, hidden_size], &[hidden_size, hidden_size]);

    // GPU version
    let a2_f16: Vec<half::f16> = a2_data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let b2_f16: Vec<half::f16> = b2_data.iter().map(|&x| half::f16::from_f32(x)).collect();

    let tensor_a2 = Tensor::from_vec_metal(&device, a2_f16, vec![1, hidden_size])
        .expect("Failed to create tensor A2");
    let tensor_b2 = Tensor::from_vec_metal(&device, b2_f16, vec![hidden_size, hidden_size])
        .expect("Failed to create tensor B2");

    let gpu_result2_tensor = tensor_a2.matmul(&tensor_b2)
        .expect("Failed to run GPU matmul");
    let gpu_result2 = gpu_result2_tensor.to_vec();

    println!("  First 5 values:");
    println!("    CPU: {:?}", cpu_result2[0..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("    GPU: {:?}", gpu_result2[0..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    let mut max_diff2 = 0.0f32;
    let mut avg_diff = 0.0f32;
    for i in 0..cpu_result2.len() {
        let diff = (cpu_result2[i].to_f32() - gpu_result2[i].to_f32()).abs();
        max_diff2 = max_diff2.max(diff);
        avg_diff += diff;
    }
    avg_diff /= cpu_result2.len() as f32;

    println!("\n  Maximum difference: {:.9}", max_diff2);
    println!("  Average difference: {:.9}", avg_diff);

    if max_diff2 < 0.1 {
        println!("  ✅ CPU and GPU results match (within tolerance)");
    } else {
        println!("  ❌ WARNING: Significant difference detected!");
        println!("      This could explain model prediction discrepancies");
    }

    println!("\n=== Comparison Complete ===");
}

/// CPU matmul implementation for comparison
fn matmul_cpu(a: &[f32], b: &[f32], a_shape: &[usize], b_shape: &[usize]) -> Vec<half::f16> {
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    assert_eq!(a_shape[1], b_shape[0], "Inner dimensions must match");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    result.iter().map(|&x| half::f16::from_f32(x)).collect()
}
