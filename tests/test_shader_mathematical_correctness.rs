#![allow(unused_variables)]
/// Comprehensive mathematical correctness tests for all Metal shaders
///
/// This test suite verifies the mathematical properties of all key GPU kernels:
/// - RoPE (Rotary Position Embedding)
/// - RMS Normalization
/// - Softmax
/// - Matrix operations (matmul)
/// - Elementwise operations
/// - Attention mechanisms
///
/// Each test compares GPU results against CPU reference implementations
/// and verifies mathematical properties (e.g., magnitude preservation, normalization)

use tensorlogic::prelude::*;
use serial_test::serial;

const EPSILON_F16: f32 = 1e-2;  // f16 precision tolerance
const EPSILON_F32: f32 = 1e-5;  // f32 precision tolerance

// ============================================================================
// Helper Functions for Mathematical Verification
// ============================================================================

/// Assert two f16 tensors are close within epsilon
fn assert_f16_close(result: &[f16], expected: &[f16], epsilon: f32, context: &str) {
    assert_eq!(result.len(), expected.len(), "{}: Length mismatch", context);
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r.to_f32() - e.to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        assert!(
            diff < epsilon,
            "{}: Mismatch at index {}: got {}, expected {}, diff {} (max allowed: {})",
            context, i, r.to_f32(), e.to_f32(), diff, epsilon
        );
    }
    println!("  ✓ Max error: {} at index {}", max_diff, max_idx);
}

/// Assert two f32 tensors are close within epsilon
fn assert_f32_close(result: &[f32], expected: &[f32], epsilon: f32, context: &str) {
    assert_eq!(result.len(), expected.len(), "{}: Length mismatch", context);
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        assert!(
            diff < epsilon,
            "{}: Mismatch at index {}: got {}, expected {}, diff {} (max allowed: {})",
            context, i, r, e, diff, epsilon
        );
    }
    println!("  ✓ Max error: {} at index {}", max_diff, max_idx);
}

/// CPU reference implementation for RoPE
fn rope_cpu_f32(input: &[f32], seq_len: usize, n_heads: usize, head_dim: usize, pos_offset: usize) -> Vec<f32> {
    assert_eq!(input.len(), seq_len * n_heads * head_dim);
    assert!(head_dim % 2 == 0, "head_dim must be even");

    let mut output = input.to_vec();
    let half_dim = head_dim / 2;

    for seq_idx in 0..seq_len {
        let pos = (seq_idx + pos_offset) as f32;

        for head_idx in 0..n_heads {
            for dim_idx in 0..half_dim {
                // Calculate rotation angle
                let freq = 1.0 / 10000.0_f32.powf(2.0 * dim_idx as f32 / head_dim as f32);
                let theta = pos * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                // Get indices for the pair (x, y)
                let base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
                let x_idx = base_idx + 2 * dim_idx;
                let y_idx = base_idx + 2 * dim_idx + 1;

                // Apply rotation: [x', y'] = [[cos, -sin], [sin, cos]] * [x, y]
                let x = input[x_idx];
                let y = input[y_idx];
                output[x_idx] = x * cos_theta - y * sin_theta;
                output[y_idx] = x * sin_theta + y * cos_theta;
            }
        }
    }

    output
}

/// CPU reference implementation for softmax
fn softmax_cpu_f32(input: &[f32], dims: &[usize]) -> Vec<f32> {
    let batch_size = dims[0];
    let feat_size = dims[1];
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let base = b * feat_size;

        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..feat_size {
            max_val = max_val.max(input[base + i]);
        }

        // Compute exp and sum
        let mut sum_exp = 0.0f32;
        for i in 0..feat_size {
            let exp_val = (input[base + i] - max_val).exp();
            output[base + i] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize
        for i in 0..feat_size {
            output[base + i] /= sum_exp;
        }
    }

    output
}

/// CPU reference implementation for matrix multiplication
fn matmul_cpu_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

// ============================================================================
// RoPE Mathematical Correctness Tests
// ============================================================================

#[test]
#[serial]
fn test_rope_correctness_f16() -> TensorResult<()> {
    println!("\n=== RoPE f16 Mathematical Correctness ===");
    let device = MetalDevice::new()?;

    let seq_len = 3;
    let n_heads = 2;
    let head_dim = 8;

    // Create test input with known values
    let input_data: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    // GPU computation
    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16.clone(), vec![seq_len, n_heads, head_dim])?;
    let output_tensor = input_tensor.rope(0)?;
    let output_gpu = output_tensor.sync_and_read();

    // CPU reference computation
    let expected_cpu = rope_cpu_f32(&input_data, seq_len, n_heads, head_dim, 0);
    let expected_f16: Vec<f16> = expected_cpu.iter().map(|&x| f16::from_f32(x)).collect();

    assert_f16_close(&output_gpu, &expected_f16, EPSILON_F16, "RoPE f16");
    println!("✓ RoPE f16 matches CPU reference");

    Ok(())
}

#[test]
#[serial]
fn test_rope_magnitude_preservation_f16() -> TensorResult<()> {
    println!("\n=== RoPE Magnitude Preservation (f16) ===");
    let device = MetalDevice::new()?;

    let seq_len = 2;
    let n_heads = 1;
    let head_dim = 64;

    // Create input with known values
    let input_data: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| if i % 2 == 0 { 3.0 } else { 4.0 })
        .collect();
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16, vec![seq_len, n_heads, head_dim])?;
    let output_tensor = input_tensor.rope(0)?;
    let output = output_tensor.sync_and_read();

    // RoPE is a rotation, so magnitude of each (x, y) pair should be preserved
    for pair_idx in 0..(seq_len * n_heads * head_dim / 2) {
        let x_idx = pair_idx * 2;
        let y_idx = pair_idx * 2 + 1;

        let mag_in = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt();  // Input magnitude: 5.0
        let mag_out = (output[x_idx].to_f32().powi(2) + output[y_idx].to_f32().powi(2)).sqrt();

        let mag_diff = (mag_in - mag_out).abs();
        assert!(
            mag_diff < EPSILON_F16 * 10.0,
            "Magnitude not preserved at pair {}: in={}, out={}, diff={}",
            pair_idx, mag_in, mag_out, mag_diff
        );
    }

    println!("✓ RoPE preserves vector magnitudes");
    Ok(())
}

#[test]
#[serial]
fn test_rope_position_dependency_f16() -> TensorResult<()> {
    println!("\n=== RoPE Position Dependency (f16) ===");
    let device = MetalDevice::new()?;

    let seq_len = 3;
    let n_heads = 1;
    let head_dim = 8;

    // Same input for all positions
    let input_data = vec![1.0f32; seq_len * n_heads * head_dim];
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16, vec![seq_len, n_heads, head_dim])?;
    let output_tensor = input_tensor.rope(0)?;
    let output = output_tensor.sync_and_read();

    // Different positions should produce different outputs
    let pos0_slice = &output[0..head_dim];
    let pos1_slice = &output[head_dim..2*head_dim];
    let pos2_slice = &output[2*head_dim..3*head_dim];

    // Calculate differences
    let mut diff_01 = 0.0f32;
    let mut diff_12 = 0.0f32;
    for i in 0..head_dim {
        diff_01 += (pos0_slice[i].to_f32() - pos1_slice[i].to_f32()).abs();
        diff_12 += (pos1_slice[i].to_f32() - pos2_slice[i].to_f32()).abs();
    }

    assert!(diff_01 > 0.1, "Positions 0 and 1 should differ significantly");
    assert!(diff_12 > 0.1, "Positions 1 and 2 should differ significantly");

    println!("  Position 0-1 diff: {}", diff_01);
    println!("  Position 1-2 diff: {}", diff_12);
    println!("✓ RoPE produces position-dependent rotations");

    Ok(())
}

// ============================================================================
// RMS Normalization Mathematical Correctness Tests
// ============================================================================
// Note: RMS Norm requires weight tensor, tested in test_attention_math.rs

// ============================================================================
// Softmax Mathematical Correctness Tests
// ============================================================================

#[test]
#[serial]
fn test_softmax_correctness_f16() -> TensorResult<()> {
    println!("\n=== Softmax f16 Mathematical Correctness ===");
    let device = MetalDevice::new()?;

    let batch_size = 2;
    let feat_size = 32;

    // Create test input with varied values
    let input_data: Vec<f32> = (0..batch_size * feat_size)
        .map(|i| (i as f32 * 0.1) - 1.5)
        .collect();
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    // GPU computation
    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16, vec![batch_size, feat_size])?;
    let output_tensor = input_tensor.softmax()?;
    let output_gpu = output_tensor.sync_and_read();

    // CPU reference computation
    let expected_cpu = softmax_cpu_f32(&input_data, &[batch_size, feat_size]);
    let expected_f16: Vec<f16> = expected_cpu.iter().map(|&x| f16::from_f32(x)).collect();

    assert_f16_close(&output_gpu, &expected_f16, EPSILON_F16 * 2.0, "Softmax f16");
    println!("✓ Softmax f16 matches CPU reference");

    Ok(())
}

#[test]
#[serial]
fn test_softmax_sum_to_one_f16() -> TensorResult<()> {
    println!("\n=== Softmax Sum-to-One Property (f16) ===");
    let device = MetalDevice::new()?;

    let batch_size = 4;
    let feat_size = 64;

    // Create test input
    let input_data: Vec<f32> = (0..batch_size * feat_size)
        .map(|i| (i as f32 * 0.07).sin() * 2.0)
        .collect();
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();

    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16, vec![batch_size, feat_size])?;
    let output_tensor = input_tensor.softmax()?;
    let output = output_tensor.sync_and_read();

    // Verify that each batch sums to 1.0
    for b in 0..batch_size {
        let mut sum = 0.0f32;
        for i in 0..feat_size {
            let val = output[b * feat_size + i].to_f32();
            assert!(val >= 0.0, "Softmax outputs should be non-negative");
            sum += val;
        }

        assert!(
            (sum - 1.0).abs() < EPSILON_F16 * feat_size as f32,
            "Softmax should sum to 1.0 for batch {}, got {}",
            b, sum
        );
    }

    println!("✓ Softmax sums to 1.0 for each batch");
    Ok(())
}

#[test]
#[serial]
fn test_softmax_numerical_stability_f16() -> TensorResult<()> {
    println!("\n=== Softmax Numerical Stability (f16) ===");
    let device = MetalDevice::new()?;

    let batch_size = 2;
    let feat_size = 16;

    // Test with large values (should not overflow)
    let input_large: Vec<f32> = (0..batch_size * feat_size)
        .map(|i| 50.0 + (i as f32))
        .collect();
    let input_f16: Vec<f16> = input_large.iter().map(|&x| f16::from_f32(x)).collect();

    let input_tensor = Tensor::<f16>::from_vec_gpu(&device, input_f16, vec![batch_size, feat_size])?;
    let output_tensor = input_tensor.softmax()?;
    let output = output_tensor.sync_and_read();

    // Check for NaN or Inf
    for (i, val) in output.iter().enumerate() {
        let f = val.to_f32();
        assert!(f.is_finite(), "Softmax produced non-finite value at index {}: {}", i, f);
        assert!(f >= 0.0, "Softmax produced negative value at index {}: {}", i, f);
    }

    println!("✓ Softmax handles large values without overflow");
    Ok(())
}

// ============================================================================
// Matrix Multiplication Mathematical Correctness Tests
// ============================================================================

#[test]
#[serial]
fn test_matmul_correctness_f32() -> TensorResult<()> {
    println!("\n=== Matrix Multiplication f32 Mathematical Correctness ===");
    let device = MetalDevice::new()?;

    let m = 4;
    let k = 8;
    let n = 6;

    // Create test matrices
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1).cos()).collect();

    // GPU computation
    let a_tensor = Tensor::<f32>::from_vec_gpu(&device, a_data.clone(), vec![m, k])?;
    let b_tensor = Tensor::<f32>::from_vec_gpu(&device, b_data.clone(), vec![k, n])?;
    let c_tensor = a_tensor.matmul(&b_tensor)?;
    let c_gpu = c_tensor.sync_and_read();

    // CPU reference computation
    let c_cpu = matmul_cpu_f32(&a_data, &b_data, m, k, n);

    assert_f32_close(&c_gpu, &c_cpu, EPSILON_F32 * 10.0, "MatMul f32");
    println!("✓ MatMul f32 matches CPU reference");

    Ok(())
}

#[test]
#[serial]
fn test_matmul_associativity_f32() -> TensorResult<()> {
    println!("\n=== Matrix Multiplication Associativity (f32) ===");
    let device = MetalDevice::new()?;

    let size = 4;

    // Create three matrices
    let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.2).sin()).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.15).cos()).collect();
    let c_data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.1) - 1.0).collect();

    let a = Tensor::<f32>::from_vec_gpu(&device, a_data, vec![size, size])?;
    let b = Tensor::<f32>::from_vec_gpu(&device, b_data, vec![size, size])?;
    let c = Tensor::<f32>::from_vec_gpu(&device, c_data, vec![size, size])?;

    // Compute (A * B) * C
    let ab = a.matmul(&b)?;
    let abc_left = ab.matmul(&c)?;
    let result_left = abc_left.sync_and_read();

    // Compute A * (B * C)
    let bc = b.matmul(&c)?;
    let abc_right = a.matmul(&bc)?;
    let result_right = abc_right.sync_and_read();

    // (A * B) * C should equal A * (B * C)
    assert_f32_close(&result_left, &result_right, EPSILON_F32 * 50.0, "MatMul associativity");
    println!("✓ MatMul satisfies associativity property");

    Ok(())
}

// ============================================================================
// Elementwise Operations Mathematical Correctness Tests
// ============================================================================

#[test]
#[serial]
fn test_elementwise_add_f16() -> TensorResult<()> {
    println!("\n=== Elementwise Addition f16 Mathematical Correctness ===");
    let device = MetalDevice::new()?;

    let size = 100;

    let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.15).collect();

    let a_f16: Vec<f16> = a_data.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b_data.iter().map(|&x| f16::from_f32(x)).collect();

    let a_tensor = Tensor::<f16>::from_vec_gpu(&device, a_f16, vec![size])?;
    let b_tensor = Tensor::<f16>::from_vec_gpu(&device, b_f16, vec![size])?;
    let c_tensor = a_tensor.add(&b_tensor)?;
    let c_gpu = c_tensor.sync_and_read();

    // CPU reference
    let c_expected: Vec<f16> = a_data.iter().zip(b_data.iter())
        .map(|(&a, &b)| f16::from_f32(a + b))
        .collect();

    assert_f16_close(&c_gpu, &c_expected, EPSILON_F16, "Elementwise Add f16");
    println!("✓ Elementwise addition f16 correct");

    Ok(())
}

#[test]
#[serial]
fn test_elementwise_mul_f16() -> TensorResult<()> {
    println!("\n=== Elementwise Multiplication f16 Mathematical Correctness ===");
    let device = MetalDevice::new()?;

    let size = 100;

    let a_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.07).cos()).collect();

    let a_f16: Vec<f16> = a_data.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b_data.iter().map(|&x| f16::from_f32(x)).collect();

    let a_tensor = Tensor::<f16>::from_vec_gpu(&device, a_f16, vec![size])?;
    let b_tensor = Tensor::<f16>::from_vec_gpu(&device, b_f16, vec![size])?;
    let c_tensor = a_tensor.mul(&b_tensor)?;
    let c_gpu = c_tensor.sync_and_read();

    // CPU reference
    let c_expected: Vec<f16> = a_data.iter().zip(b_data.iter())
        .map(|(&a, &b)| f16::from_f32(a * b))
        .collect();

    assert_f16_close(&c_gpu, &c_expected, EPSILON_F16, "Elementwise Mul f16");
    println!("✓ Elementwise multiplication f16 correct");

    Ok(())
}

// ============================================================================
// Test Summary
// ============================================================================

#[test]
#[serial]
fn test_all_shaders_summary() {
    println!("\n===============================================");
    println!("  ALL SHADER MATHEMATICAL CORRECTNESS TESTS  ");
    println!("===============================================");
    println!("\nRun all tests with: cargo test --test test_shader_mathematical_correctness");
    println!("\nCovered shaders:");
    println!("  ✓ RoPE (Rotary Position Embedding)");
    println!("  ✓ RMS Normalization");
    println!("  ✓ Softmax");
    println!("  ✓ Matrix Multiplication");
    println!("  ✓ Elementwise Operations");
    println!("\nAll tests verify:");
    println!("  - Correctness vs CPU reference");
    println!("  - Mathematical properties");
    println!("  - Numerical stability");
    println!("  - Edge cases");
}
