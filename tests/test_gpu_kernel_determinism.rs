/// Test GPU kernel determinism
///
/// Verifies that all GPU kernels produce deterministic results

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};
use tensorlogic::prelude::*;
use half::f16;

fn assert_tensors_equal(a: &[f16], b: &[f16], tolerance: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va.to_f32() - vb.to_f32()).abs();
        assert!(
            diff < tolerance,
            "{}: index {} differs: {} vs {} (diff={})",
            msg, i, va.to_f32(), vb.to_f32(), diff
        );
    }
}

#[test]
fn test_rms_norm_determinism() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let input_data: Vec<f16> = (1..=2048).map(|i| f16::from_f32(i as f32 * 0.001)).collect();
    let weight_data: Vec<f16> = (1..=2048).map(|i| f16::from_f32(1.0 + i as f32 * 0.0001)).collect();

    let weight = Tensor::<f16>::from_vec_gpu(&device, weight_data, vec![2048])?;

    // Run 5 times
    let mut results = Vec::new();
    for run in 0..5 {
        let input = Tensor::<f16>::from_vec_gpu(&device, input_data.clone(), vec![1, 2048])?;
        let output = input.rms_norm(vec![1], &weight, 1e-5)?;
        let data = output.sync_and_read();

        println!("Run {}: first 5 = {:?}", run, &data[0..5]);
        results.push(data);
    }

    // Verify all identical
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_tensors_equal(&results[0], result, 1e-4, &format!("RMS norm run {}", i));
    }

    println!("✓ RMS norm is deterministic");
    Ok(())
}

#[test]
fn test_softmax_determinism() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    // Attention scores shape [seq, seq]
    let data: Vec<f16> = (0..1024).map(|i| f16::from_f32((i as f32 - 512.0) * 0.01)).collect();

    let mut results = Vec::new();
    for run in 0..5 {
        let input = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![32, 32])?;
        // softmax() applies to last dimension by default
        let output = input.softmax()?;
        let out_data = output.sync_and_read();

        println!("Run {}: first row sum ≈ {}", run,
                 out_data[0..32].iter().map(|v| v.to_f32()).sum::<f32>());
        results.push(out_data);
    }

    for (i, result) in results.iter().enumerate().skip(1) {
        assert_tensors_equal(&results[0], result, 1e-4, &format!("Softmax run {}", i));
    }

    println!("✓ Softmax is deterministic");
    Ok(())
}

// SiLU and einsum tested implicitly through transformer tests

#[test]
fn test_slice_last_determinism() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let data: Vec<f16> = (0..9216).map(|i| f16::from_f32(i as f32)).collect();

    let mut results = Vec::new();
    for run in 0..5 {
        let input = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![46, 2, 100])?;
        let output = input.slice_last(0)?;
        let out_data = output.sync_and_read();

        println!("Run {}: shape {:?}, first 5 = {:?}",
                 run, output.shape().dims(), &out_data[0..5]);
        results.push(out_data);
    }

    for (i, result) in results.iter().enumerate().skip(1) {
        assert_tensors_equal(&results[0], result, 1e-5, &format!("slice_last run {}", i));
    }

    println!("✓ slice_last is deterministic");
    Ok(())
}

#[test]
fn test_reshape_determinism() -> TensorResult<()> {
    let device = MetalDevice::new()?;

    let data: Vec<f16> = (0..2048).map(|i| f16::from_f32(i as f32)).collect();

    let mut results = Vec::new();
    for run in 0..5 {
        let input = Tensor::<f16>::from_vec_gpu(&device, data.clone(), vec![1, 2048])?;
        let reshaped = input.reshape(vec![32, 64])?;
        let back = reshaped.reshape(vec![2048])?;
        let out_data = back.sync_and_read();

        println!("Run {}: first 5 = {:?}", run, &out_data[0..5]);
        results.push(out_data);
    }

    for (i, result) in results.iter().enumerate().skip(1) {
        assert_tensors_equal(&results[0], result, 1e-5, &format!("Reshape run {}", i));
    }

    println!("✓ Reshape is deterministic");
    Ok(())
}

// Embedding tested through full model tests
