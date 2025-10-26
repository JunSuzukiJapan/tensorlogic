/// Compare Metal GPU vs CPU RMS Normalization
///
/// Verify that RMS Norm produces identical results on GPU vs CPU

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

fn rmsnorm_cpu(input: &[f32], weight: &[f32], hidden_size: usize) -> Vec<f32> {
    let eps = 1e-5;

    // Calculate RMS
    let sum_sq: f32 = input.iter().take(hidden_size).map(|x| x * x).sum();
    let rms = (sum_sq / hidden_size as f32 + eps).sqrt();

    // Normalize and scale
    input
        .iter()
        .take(hidden_size)
        .zip(weight.iter())
        .map(|(x, w)| (x / rms) * w)
        .collect()
}

fn main() {
    println!("=== Metal vs CPU RMS Norm Comparison ===\n");

    let device = MetalDevice::new().expect("Failed to create Metal device");

    // Test 1: Small input (1, 8)
    println!("Test 1: [1, 8]");
    let input1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight1 = vec![1.0f32; 8];

    let cpu_result1 = rmsnorm_cpu(&input1, &weight1, 8);

    let input1_f16: Vec<_> = input1.iter().map(|&x| half::f16::from_f32(x)).collect();
    let weight1_f16: Vec<_> = weight1.iter().map(|&x| half::f16::from_f32(x)).collect();

    let gpu_input = Tensor::from_vec_metal(&device, input1_f16, vec![1, 8])
        .expect("Failed to create GPU input");
    let gpu_weight = Tensor::from_vec_metal(&device, weight1_f16, vec![8])
        .expect("Failed to create GPU weight");

    let gpu_result_tensor = gpu_input.rms_norm(vec![8], &gpu_weight, 1e-6)
        .expect("Failed to run GPU RMS norm");
    let gpu_result1: Vec<f32> = gpu_result_tensor.to_vec().iter().map(|x| x.to_f32()).collect();

    println!("  Input: {:?}", &input1);
    println!("  CPU result: {:?}", &cpu_result1[..8]);
    println!("  GPU result: {:?}", &gpu_result1[..8]);

    let max_diff1 = cpu_result1.iter()
        .zip(gpu_result1.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let avg_diff1 = cpu_result1.iter()
        .zip(gpu_result1.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f32>() / cpu_result1.len() as f32;

    println!("  Maximum difference: {:.6}", max_diff1);
    println!("  Average difference: {:.6}", avg_diff1);

    if max_diff1 < 0.001 {
        println!("  ✅ Results match within tolerance\n");
    } else {
        println!("  ❌ Results differ significantly\n");
    }

    // Test 2: TinyLlama hidden size (1, 2048)
    println!("Test 2: [1, 2048] (TinyLlama hidden size)");
    let input2: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight2 = vec![1.0f32; 2048];

    let cpu_result2 = rmsnorm_cpu(&input2, &weight2, 2048);

    let input2_f16: Vec<_> = input2.iter().map(|&x| half::f16::from_f32(x)).collect();
    let weight2_f16: Vec<_> = weight2.iter().map(|&x| half::f16::from_f32(x)).collect();

    let gpu_input = Tensor::from_vec_metal(&device, input2_f16, vec![1, 2048])
        .expect("Failed to create GPU input");
    let gpu_weight = Tensor::from_vec_metal(&device, weight2_f16, vec![2048])
        .expect("Failed to create GPU weight");

    let gpu_result_tensor = gpu_input.rms_norm(vec![2048], &gpu_weight, 1e-6)
        .expect("Failed to run GPU RMS norm");
    let gpu_result2: Vec<f32> = gpu_result_tensor.to_vec().iter().map(|x| x.to_f32()).collect();

    println!("  First 5 CPU: {:?}", &cpu_result2[..5]);
    println!("  First 5 GPU: {:?}", &gpu_result2[..5]);

    let max_diff2 = cpu_result2.iter()
        .zip(gpu_result2.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let avg_diff2 = cpu_result2.iter()
        .zip(gpu_result2.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f32>() / cpu_result2.len() as f32;

    println!("  Maximum difference: {:.6}", max_diff2);
    println!("  Average difference: {:.6}", avg_diff2);

    if max_diff2 < 0.001 {
        println!("  ✅ Results match within tolerance\n");
    } else {
        println!("  ❌ Results differ significantly\n");
    }

    // Test 3: With non-uniform weights
    println!("Test 3: [1, 2048] with non-uniform weights");
    let input3: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).cos()).collect();
    let weight3: Vec<f32> = (0..2048).map(|i| 1.0 + (i as f32 * 0.0001)).collect();

    let cpu_result3 = rmsnorm_cpu(&input3, &weight3, 2048);

    let input3_f16: Vec<_> = input3.iter().map(|&x| half::f16::from_f32(x)).collect();
    let weight3_f16: Vec<_> = weight3.iter().map(|&x| half::f16::from_f32(x)).collect();

    let gpu_input = Tensor::from_vec_metal(&device, input3_f16, vec![1, 2048])
        .expect("Failed to create GPU input");
    let gpu_weight = Tensor::from_vec_metal(&device, weight3_f16, vec![2048])
        .expect("Failed to create GPU weight");

    let gpu_result_tensor = gpu_input.rms_norm(vec![2048], &gpu_weight, 1e-6)
        .expect("Failed to run GPU RMS norm");
    let gpu_result3: Vec<f32> = gpu_result_tensor.to_vec().iter().map(|x| x.to_f32()).collect();

    println!("  First 5 CPU: {:?}", &cpu_result3[..5]);
    println!("  First 5 GPU: {:?}", &gpu_result3[..5]);

    let max_diff3 = cpu_result3.iter()
        .zip(gpu_result3.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let avg_diff3 = cpu_result3.iter()
        .zip(gpu_result3.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f32>() / cpu_result3.len() as f32;

    println!("  Maximum difference: {:.6}", max_diff3);
    println!("  Average difference: {:.6}", avg_diff3);

    if max_diff3 < 0.001 {
        println!("  ✅ Results match within tolerance\n");
    } else {
        println!("  ❌ Results differ significantly\n");
    }

    println!("=== Summary ===");
    println!("All tests should show max difference < 0.001 for f16 precision");

    if max_diff1 < 0.001 && max_diff2 < 0.001 && max_diff3 < 0.001 {
        println!("✅ RMS Norm Metal implementation is correct");
    } else {
        println!("❌ RMS Norm Metal has precision issues");
    }
}
