/// Comprehensive tests for Einstein summation (einsum) operations
///
/// Tests cover:
/// - Matrix multiplication patterns
/// - Transpose operations
/// - Batch operations
/// - Attention score calculations
/// - Diagonal and trace operations
/// - Error cases

use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation, TensorIO};

// Helper function to assert tensors are close
fn assert_tensor_close_f32(result: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(result.len(), expected.len(), "Length mismatch");
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < epsilon,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i, r, e, (r - e).abs()
        );
    }
}

#[test]
fn test_einsum_matrix_multiplication() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij,jk->ik" (standard matrix multiplication)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;
    let b = Tensor::<f32>::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2]
    )?;

    let c = Tensor::einsum("ij,jk->ik", &[&a, &b])?;
    let result = c.sync_and_read();

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //         = [[19, 22], [43, 50]]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum matrix multiplication test passed");
    Ok(())
}

#[test]
fn test_einsum_transpose() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij->ji" (transpose)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    let b = Tensor::einsum("ij->ji", &[&a])?;
    let result = b.sync_and_read();

    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum transpose test passed");
    Ok(())
}

#[test]
fn test_einsum_batch_matrix_multiplication() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "bij,bjk->bik" (batch matrix multiplication)
    let a = Tensor::<f32>::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0,  // batch 0
            5.0, 6.0, 7.0, 8.0,  // batch 1
        ],
        vec![2, 2, 2]
    )?;
    let b = Tensor::<f32>::from_vec(
        vec![
            1.0, 0.0, 0.0, 1.0,  // batch 0 (identity)
            2.0, 0.0, 0.0, 2.0,  // batch 1 (2*identity)
        ],
        vec![2, 2, 2]
    )?;

    let c = Tensor::einsum("bij,bjk->bik", &[&a, &b])?;
    let result = c.sync_and_read();

    // Batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    // Batch 1: [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
    let expected = vec![
        1.0, 2.0, 3.0, 4.0,
        10.0, 12.0, 14.0, 16.0,
    ];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum batch matrix multiplication test passed");
    Ok(())
}

#[test]
fn test_einsum_attention_scores() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ihd,jhd->ihj" (attention score calculation)
    // Query: [seq_q, heads, head_dim]
    // Key:   [seq_k, heads, head_dim]
    // Output: [seq_q, heads, seq_k]

    let seq_q = 2;
    let seq_k = 3;
    let heads = 2;
    let head_dim = 4;

    let device = MetalDevice::new()?;
    let q = Tensor::<f32>::ones(&device, vec![seq_q, heads, head_dim])?;
    let k = Tensor::<f32>::ones(&device, vec![seq_k, heads, head_dim])?;

    let scores = Tensor::einsum("ihd,jhd->ihj", &[&q, &k])?;
    let result = scores.sync_and_read();
    let shape = scores.shape();

    // Shape should be [seq_q, heads, seq_k] = [2, 2, 3]
    assert_eq!(shape.dims(), &[seq_q, heads, seq_k]);

    // Each element should be head_dim (sum of 1*1 head_dim times)
    for &val in &result {
        assert!((val - head_dim as f32).abs() < 1e-5);
    }

    println!("✓ einsum attention scores test passed");
    Ok(())
}

#[test]
fn test_einsum_attention_output() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ihj,jhd->ihd" (attention output calculation)
    // Attention weights: [seq_q, heads, seq_k]
    // Values: [seq_k, heads, head_dim]
    // Output: [seq_q, heads, head_dim]

    let seq_q = 2;
    let seq_k = 3;
    let heads = 2;
    let head_dim = 4;

    // Uniform attention weights
    let attn_value = 1.0 / seq_k as f32;
    let attn_ones = Tensor::<f32>::ones(&device, vec![seq_q, heads, seq_k])?;
    let attn = attn_ones.mul_scalar(attn_value)?;
    let v = Tensor::<f32>::ones(&device, vec![seq_k, heads, head_dim])?;

    let output = Tensor::einsum("ihj,jhd->ihd", &[&attn, &v])?;
    let result = output.sync_and_read();
    let shape = output.shape();

    // Shape should be [seq_q, heads, head_dim] = [2, 2, 4]
    assert_eq!(shape.dims(), &[seq_q, heads, head_dim]);

    // Each element should be 1.0 (average of ones)
    for &val in &result {
        assert!((val - 1.0).abs() < 1e-5);
    }

    println!("✓ einsum attention output test passed");
    Ok(())
}

#[test]
fn test_einsum_element_wise_product() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij,ij->ij" (element-wise multiplication)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;
    let b = Tensor::<f32>::from_vec(
        vec![2.0, 3.0, 4.0, 5.0],
        vec![2, 2]
    )?;

    let c = Tensor::einsum("ij,ij->ij", &[&a, &b])?;
    let result = c.sync_and_read();

    let expected = vec![2.0, 6.0, 12.0, 20.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum element-wise product test passed");
    Ok(())
}

#[test]
fn test_einsum_outer_product() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "i,j->ij" (outer product)
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], vec![2])?;

    let c = Tensor::einsum("i,j->ij", &[&a, &b])?;
    let result = c.sync_and_read();

    // Expected: [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
    let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum outer product test passed");
    Ok(())
}

#[test]
fn test_einsum_matrix_vector() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij,j->i" (matrix-vector multiplication)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;
    let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;

    let c = Tensor::einsum("ij,j->i", &[&a, &b])?;
    let result = c.sync_and_read();

    // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    let expected = vec![14.0, 32.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum matrix-vector multiplication test passed");
    Ok(())
}

#[test]
fn test_einsum_trace() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ii->" (trace - sum of diagonal)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;

    let trace = Tensor::einsum("ii->", &[&a])?;
    let result = trace.sync_and_read();

    // Expected: 1 + 4 = 5
    assert_eq!(result.len(), 1);
    assert!((result[0] - 5.0).abs() < 1e-5);

    println!("✓ einsum trace test passed");
    Ok(())
}

#[test]
fn test_einsum_diagonal() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ii->i" (extract diagonal)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3]
    )?;

    let diag = Tensor::einsum("ii->i", &[&a])?;
    let result = diag.sync_and_read();

    // Expected diagonal: [1, 5, 9]
    let expected = vec![1.0, 5.0, 9.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum diagonal extraction test passed");
    Ok(())
}

#[test]
fn test_einsum_sum_all() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij->" (sum all elements)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;

    let sum = Tensor::einsum("ij->", &[&a])?;
    let result = sum.sync_and_read();

    // Expected: 1 + 2 + 3 + 4 = 10
    assert_eq!(result.len(), 1);
    assert!((result[0] - 10.0).abs() < 1e-5);

    println!("✓ einsum sum all test passed");
    Ok(())
}

#[test]
fn test_einsum_sum_axis() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij->i" (sum along axis 1)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;

    let sum = Tensor::einsum("ij->i", &[&a])?;
    let result = sum.sync_and_read();

    // Expected: [1+2+3, 4+5+6] = [6, 15]
    let expected = vec![6.0, 15.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum sum along axis test passed");
    Ok(())
}

#[test]
fn test_einsum_permute() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ijk->kji" (permute dimensions)
    let a = Tensor::<f32>::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2]
    )?;

    let b = Tensor::einsum("ijk->kji", &[&a])?;
    let shape = b.shape();

    // Shape should be [2, 2, 2] (permuted)
    assert_eq!(shape.dims(), &[2, 2, 2]);

    println!("✓ einsum permute dimensions test passed");
    Ok(())
}

#[test]
fn test_einsum_bilinear() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: "ij,jk,kl->il" (chained multiplication)
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], vec![1, 2])?;
    let b = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])?;
    let c = Tensor::<f32>::from_vec(vec![2.0, 3.0], vec![2, 1])?;

    let result = Tensor::einsum("ij,jk,kl->il", &[&a, &b, &c])?;
    let vec = result.sync_and_read();

    // [1,2] @ [[1,0],[0,1]] @ [[2],[3]] = [1,2] @ [[2],[3]] = [8]
    let expected = vec![8.0];
    assert_tensor_close_f32(&vec, &expected, 1e-5);

    println!("✓ einsum bilinear (chained) test passed");
    Ok(())
}

#[test]
fn test_einsum_f16_basic() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test einsum with f16 (half precision)
    use half::f16;

    let a = Tensor::<f16>::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(2.0),
             f16::from_f32(3.0), f16::from_f32(4.0)],
        vec![2, 2]
    )?;
    let b = Tensor::<f16>::from_vec(
        vec![f16::from_f32(5.0), f16::from_f32(6.0),
             f16::from_f32(7.0), f16::from_f32(8.0)],
        vec![2, 2]
    )?;

    let c = Tensor::einsum("ij,jk->ik", &[&a, &b])?;
    let result: Vec<f32> = c.sync_and_read().iter().map(|&x| x.to_f32()).collect();

    let expected = vec![19.0, 22.0, 43.0, 50.0];
    // f16 has lower precision
    assert_tensor_close_f32(&result, &expected, 1e-2);

    println!("✓ einsum f16 basic test passed");
    Ok(())
}

// Error handling tests

#[test]
#[should_panic(expected = "Invalid")]
fn test_einsum_invalid_equation() {
    // Test: Invalid equation format
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let _ = Tensor::einsum("ij,jk->ik->xyz", &[&a, &a]).unwrap();
}

#[test]
#[should_panic(expected = "Expected")]
fn test_einsum_operand_count_mismatch() {
    // Test: Wrong number of operands
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let _ = Tensor::einsum("ij,jk->ik", &[&a]).unwrap(); // Need 2, provided 1
}

#[test]
#[should_panic]
fn test_einsum_shape_mismatch() {
    // Test: Incompatible shapes for contraction
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let _ = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap(); // j dimensions don't match
}

#[test]
fn test_einsum_empty_output() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test: Empty equation (no output spec) - should infer output
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    // "ij,jk" should infer output as "ik"
    let c = Tensor::einsum("ij,jk", &[&a, &b])?;
    let result = c.sync_and_read();

    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_tensor_close_f32(&result, &expected, 1e-5);

    println!("✓ einsum implicit output test passed");
    Ok(())
}

#[test]
fn test_einsum_single_operand() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test various single-operand operations
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    // Identity (no-op)
    let identity = Tensor::einsum("ij->ij", &[&a])?;
    assert_tensor_close_f32(&identity.sync_and_read(), &a.sync_and_read(), 1e-5);

    println!("✓ einsum single operand test passed");
    Ok(())
}
