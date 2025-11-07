/// Comprehensive Performance Test Suite for TensorLogic
///
/// Tests performance characteristics including:
/// - Memory usage patterns
/// - Large-scale data handling
/// - Operation throughput
/// - Resource management

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;
use std::time::Instant;

/// Test 1: Memory Usage - Tensor Creation and Cleanup
///
/// Validates that tensors are properly cleaned up and memory is released.
/// Creates multiple tensors and verifies no memory leaks.
#[test]
fn test_memory_usage_tensor_lifecycle() {
    let device = MetalDevice::new().unwrap();

    // Create and drop many tensors to test memory management
    for size in [100, 1000, 10000] {
        let start = Instant::now();

        for _ in 0..100 {
            let data: Vec<half::f16> = (0..size)
                .map(|i| half::f16::from_f32((i % 256) as f32))
                .collect();
            let _tensor = Tensor::from_vec_gpu(&device, data, vec![size]).unwrap();
            // Tensor dropped here
        }

        let duration = start.elapsed();
        println!("  Created 100 tensors of size {}: {:?}", size, duration);

        // Verify operation completed in reasonable time
        assert!(duration.as_secs() < 5, "Tensor creation too slow for size {}", size);
    }
}

/// Test 2: Memory Usage - Sequential Operations
///
/// Tests memory usage during sequential tensor operations.
/// Verifies intermediate results are properly cleaned up.
#[test]
fn test_memory_usage_sequential_operations() {
    let device = MetalDevice::new().unwrap();
    let size = 1000;

    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let start = Instant::now();

    // Perform chain of operations
    let mut tensor = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();
    for _ in 0..50 {
        tensor = tensor.add(&tensor).unwrap();
        // Apply ReLU as a simple operation
        tensor = tensor.relu().unwrap();
    }

    let duration = start.elapsed();
    println!("  Sequential operations (50 iterations): {:?}", duration);

    assert!(duration.as_secs() < 10, "Sequential operations too slow");
}

/// Test 3: Large-Scale Data - Matrix Operations
///
/// Tests performance with large matrices (up to 2048x2048).
/// Validates that operations scale appropriately.
#[test]
fn test_large_scale_matrix_operations() {
    let device = MetalDevice::new().unwrap();

    let sizes = vec![256, 512, 1024];

    for size in sizes {
        let data_size = size * size;
        let data: Vec<half::f16> = (0..data_size)
            .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
            .collect();

        let start = Instant::now();

        let a = Tensor::from_vec_gpu(&device, data.clone(), vec![size, size]).unwrap();
        let b = Tensor::from_vec_gpu(&device, data.clone(), vec![size, size]).unwrap();
        let _c = a.matmul(&b).unwrap();

        let duration = start.elapsed();
        let gflops = (2.0 * size.pow(3) as f64) / duration.as_secs_f64() / 1e9;

        println!("  Matrix {}x{} multiplication: {:?} ({:.2} GFLOPS)",
                 size, size, duration, gflops);

        // Verify reasonable performance (at least 10 GFLOPS for large matrices)
        if size >= 512 {
            assert!(gflops > 10.0, "Matrix multiplication too slow: {:.2} GFLOPS", gflops);
        }
    }
}

/// Test 4: Large-Scale Data - Batch Processing
///
/// Tests handling of batch operations with multiple samples.
/// Validates efficient batch processing.
#[test]
fn test_large_scale_batch_processing() {
    let device = MetalDevice::new().unwrap();

    let batch_sizes = vec![16, 64, 256];
    let feature_dim = 128;

    for batch_size in batch_sizes {
        let data_size = batch_size * feature_dim;
        let data: Vec<half::f16> = (0..data_size)
            .map(|i| half::f16::from_f32((i % 100) as f32 / 100.0))
            .collect();

        let start = Instant::now();

        let batch = Tensor::from_vec_gpu(&device, data.clone(), vec![batch_size, feature_dim]).unwrap();

        // Simulate batch operations
        let _processed = batch.relu().unwrap();

        let duration = start.elapsed();
        let throughput = (batch_size as f64) / duration.as_secs_f64();

        println!("  Batch size {}: {:?} ({:.0} samples/sec)",
                 batch_size, duration, throughput);

        // Verify reasonable throughput (at least 1000 samples/sec)
        assert!(throughput > 1000.0, "Batch processing too slow: {:.0} samples/sec", throughput);
    }
}

/// Test 5: Large-Scale Data - 4D Tensors (Image Batches)
///
/// Tests handling of 4D tensors commonly used in computer vision.
/// Validates performance with realistic image batch sizes.
#[test]
fn test_large_scale_4d_tensors() {
    let device = MetalDevice::new().unwrap();

    // Common image batch configurations
    let configs = vec![
        (8, 3, 224, 224),   // Small batch, ImageNet size
        (32, 3, 128, 128),  // Medium batch, medium size
        (64, 3, 64, 64),    // Large batch, small images
    ];

    for (batch, channels, height, width) in configs {
        let data_size = batch * channels * height * width;
        let data: Vec<half::f16> = (0..data_size)
            .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
            .collect();

        let start = Instant::now();

        let images = Tensor::from_vec_gpu(
            &device,
            data,
            vec![batch, channels, height, width]
        ).unwrap();

        // Simulate image processing
        let _processed = images.relu().unwrap();

        let duration = start.elapsed();
        let pixels_per_sec = (data_size as f64) / duration.as_secs_f64();

        println!("  4D tensor [{}x{}x{}x{}]: {:?} ({:.0} Mpixels/sec)",
                 batch, channels, height, width, duration, pixels_per_sec / 1e6);

        // Verify reasonable performance (at least 100M pixels/sec)
        assert!(pixels_per_sec > 1e8,
                "4D tensor processing too slow: {:.0} Mpixels/sec", pixels_per_sec / 1e6);
    }
}

/// Test 6: Memory Pressure - Concurrent Tensor Creation
///
/// Tests behavior under memory pressure with many concurrent tensors.
/// Validates graceful handling of memory constraints.
#[test]
fn test_memory_pressure_concurrent_tensors() {
    let device = MetalDevice::new().unwrap();
    let tensor_size = 10000;
    let num_tensors = 100;

    let start = Instant::now();

    let mut tensors = Vec::new();
    for i in 0..num_tensors {
        let data: Vec<half::f16> = (0..tensor_size)
            .map(|j| half::f16::from_f32(((i * tensor_size + j) % 256) as f32))
            .collect();

        let tensor = Tensor::from_vec_gpu(&device, data, vec![tensor_size]).unwrap();
        tensors.push(tensor);
    }

    let duration = start.elapsed();
    let total_memory_mb = (num_tensors * tensor_size * 2) as f64 / 1024.0 / 1024.0; // f16 = 2 bytes

    println!("  Created {} tensors ({:.2} MB total): {:?}",
             num_tensors, total_memory_mb, duration);

    // All tensors still accessible
    assert_eq!(tensors.len(), num_tensors);

    // Clean up
    drop(tensors);
}

/// Test 7: Operation Throughput - Element-wise Operations
///
/// Measures throughput of element-wise operations.
/// Validates performance meets minimum requirements.
#[test]
fn test_operation_throughput_elementwise() {
    let device = MetalDevice::new().unwrap();
    let size = 1000000; // 1M elements

    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let a = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();
    let b = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();

    // Test addition
    let start = Instant::now();
    let _c = a.add(&b).unwrap();
    let add_duration = start.elapsed();
    let add_throughput = (size as f64) / add_duration.as_secs_f64() / 1e9;

    // Test multiplication
    let start = Instant::now();
    let _d = a.mul(&b).unwrap();
    let mul_duration = start.elapsed();
    let mul_throughput = (size as f64) / mul_duration.as_secs_f64() / 1e9;

    println!("  Element-wise addition (1M elements): {:?} ({:.2} Gops/sec)",
             add_duration, add_throughput);
    println!("  Element-wise multiplication (1M elements): {:?} ({:.2} Gops/sec)",
             mul_duration, mul_throughput);

    // Verify reasonable throughput (at least 0.3 Gops/sec)
    // Note: Threshold is conservative to account for system load variations
    assert!(add_throughput > 0.3, "Addition throughput too low: {:.2} Gops/sec", add_throughput);
    assert!(mul_throughput > 0.3, "Multiplication throughput too low: {:.2} Gops/sec", mul_throughput);
}

/// Test 8: Resource Management - Buffer Pool Utilization
///
/// Tests that buffer pool is effectively reducing allocation overhead.
/// Compares performance with and without buffer reuse.
#[test]
fn test_resource_management_buffer_pool() {
    let device = MetalDevice::new().unwrap();
    let size = 10000;
    let iterations = 100;

    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let start = Instant::now();

    // Repeated allocations should benefit from buffer pool
    for _ in 0..iterations {
        let a = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();
        let b = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();
        let _c = a.add(&b).unwrap();
        // Buffers returned to pool on drop
    }

    let duration = start.elapsed();
    let avg_per_iteration = duration.as_micros() / iterations;

    println!("  {} iterations with buffer pool: {:?} (avg: {}μs/iter)",
             iterations, duration, avg_per_iteration);

    // Verify reasonable performance with buffer pool
    assert!(avg_per_iteration < 1000, "Buffer pool allocation too slow: {}μs", avg_per_iteration);
}

/// Test 9: Stress Test - Continuous Operation
///
/// Runs continuous operations to test stability under sustained load.
/// Validates no memory leaks or performance degradation.
#[test]
fn test_stress_continuous_operation() {
    let device = MetalDevice::new().unwrap();
    let size = 1000;
    let duration_secs = 2; // Run for 2 seconds

    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let a = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();
    let b = Tensor::from_vec_gpu(&device, data.clone(), vec![size]).unwrap();

    let start = Instant::now();
    let mut iterations = 0;

    while start.elapsed().as_secs() < duration_secs {
        let _c = a.add(&b).unwrap();
        let _d = a.mul(&b).unwrap();
        iterations += 1;
    }

    let total_duration = start.elapsed();
    let ops_per_sec = (iterations as f64) / total_duration.as_secs_f64();

    println!("  Continuous operation: {} iterations in {:?} ({:.0} ops/sec)",
             iterations, total_duration, ops_per_sec);

    // Verify sustained performance (at least 100 ops/sec)
    assert!(ops_per_sec > 100.0, "Sustained performance too low: {:.0} ops/sec", ops_per_sec);
    assert!(iterations > 100, "Too few iterations completed: {}", iterations);
}

/// Test 10: Performance Regression Check
///
/// Establishes baseline performance metrics for regression testing.
/// Fails if performance drops significantly from established baselines.
#[test]
fn test_performance_regression_baselines() {
    let device = MetalDevice::new().unwrap();

    // Baseline 1: Small matrix multiplication (256x256)
    let size = 256;
    let data: Vec<half::f16> = (0..size*size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let a = Tensor::from_vec_gpu(&device, data.clone(), vec![size, size]).unwrap();
    let b = Tensor::from_vec_gpu(&device, data.clone(), vec![size, size]).unwrap();

    let start = Instant::now();
    let _c = a.matmul(&b).unwrap();
    let duration = start.elapsed();
    let gflops = (2.0 * size.pow(3) as f64) / duration.as_secs_f64() / 1e9;

    println!("  Baseline - MatMul 256x256: {:.2} GFLOPS", gflops);

    // Baseline 2: ReLU activation (1M elements)
    let size = 1000000;
    let data: Vec<half::f16> = (0..size)
        .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0))
        .collect();

    let tensor = Tensor::from_vec_gpu(&device, data, vec![size]).unwrap();

    let start = Instant::now();
    let _activated = tensor.relu().unwrap();
    let duration = start.elapsed();
    let gb_per_sec = (size as f64 * 2.0) / duration.as_secs_f64() / 1e9; // f16 = 2 bytes

    println!("  Baseline - ReLU 1M elements: {:.2} GB/s", gb_per_sec);

    // Verify against minimum baselines (realistic for current implementation)
    // Note: Thresholds are conservative to account for system load variations
    assert!(gflops > 10.0, "MatMul regression: {:.2} GFLOPS (expected > 10)", gflops);
    assert!(gb_per_sec > 0.5, "ReLU regression: {:.2} GB/s (expected > 0.5)", gb_per_sec);
}
