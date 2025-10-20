//! CoreML vs Metal Performance Benchmark
//!
//! This benchmark compares inference performance between CoreML (Neural Engine)
//! and Metal GPU for various tensor operations.

use tensorlogic::coreml::CoreMLModel;
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;
use std::time::Instant;

fn benchmark_metal_matmul(device: &MetalDevice, size: usize, iterations: usize) -> f64 {
    let a = Tensor::ones(device, vec![size, size]).unwrap();
    let b = Tensor::ones(device, vec![size, size]).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b).unwrap();
    }
    let duration = start.elapsed();

    duration.as_secs_f64() / iterations as f64
}

fn benchmark_coreml_inference(model: &CoreMLModel, input: &Tensor, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.predict(input).unwrap();
    }
    let duration = start.elapsed();

    duration.as_secs_f64() / iterations as f64
}

fn main() {
    println!("=== TensorLogic Performance Benchmark ===\n");

    let device = MetalDevice::new().unwrap();

    // Test different matrix sizes
    let sizes = vec![64, 128, 256, 512];
    let iterations = 100;

    println!("Metal GPU Matrix Multiplication Benchmark");
    println!("----------------------------------------");
    for size in &sizes {
        let avg_time = benchmark_metal_matmul(&device, *size, iterations);
        let gflops = (2.0 * size.pow(3) as f64) / (avg_time * 1e9);
        println!(
            "Size: {}x{}, Avg time: {:.4}ms, Performance: {:.2} GFLOPS",
            size,
            size,
            avg_time * 1000.0,
            gflops
        );
    }

    println!("\nCoreML Neural Engine Inference Benchmark");
    println!("----------------------------------------");

    // Create dummy CoreML model
    let model = CoreMLModel::with_shapes(
        "benchmark_model".to_string(),
        "benchmark.mlmodelc".to_string(),
        vec![1, 3, 224, 224],
        vec![1, 1000],
    );

    let input = Tensor::ones(&device, vec![1, 3, 224, 224]).unwrap();
    let avg_time = benchmark_coreml_inference(&model, &input, iterations);

    println!(
        "ImageNet input (1x3x224x224), Avg time: {:.4}ms",
        avg_time * 1000.0
    );

    println!("\nPerformance Summary");
    println!("----------------------------------------");
    println!("Metal GPU: Optimized for general tensor operations");
    println!("CoreML: Optimized for neural network inference");
    println!("\nNote: CoreML benchmark uses placeholder implementation.");
    println!("Actual Neural Engine performance would be measured with real .mlmodelc files.");
}
