//! Integration tests for multi-operation pipelines
//! Tests combinations of operations that represent real-world usage patterns

use half::f16;
use tensorlogic::prelude::*;
use tensorlogic::tensor::TensorShape;

// ============================================================================
// Basic Pipeline Tests
// ============================================================================

#[test]
fn test_arithmetic_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Create tensors
    let a = Tensor::from_vec(
        (0..10).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10],
    )?;

    let b = Tensor::from_vec(
        vec![f16::from_f32(2.0); 10],
        vec![10],
    )?;

    // Pipeline: ((a + b) * 2) - 5
    let c = a.add(&b)?;
    let d = c.mul(&Tensor::from_vec(vec![f16::from_f32(2.0); 10], vec![10])?)?;
    let e = d.sub(&Tensor::from_vec(vec![f16::from_f32(5.0); 10], vec![10])?)?;

    let result = e.sync_and_read();
    // For i=0: ((0+2)*2)-5 = -1
    assert!((result[0].to_f32() - (-1.0)).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_matmul_activation_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Linear layer: y = xW + b, then ReLU
    let x = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ],
        vec![1, 3],
    )?;

    let w = Tensor::from_vec(
        vec![
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(-0.5),
            f16::from_f32(-1.0),
            f16::from_f32(0.3),
            f16::from_f32(0.7),
        ],
        vec![3, 2],
    )?;

    let b = Tensor::from_vec(
        vec![f16::from_f32(0.1), f16::from_f32(-0.2)],
        vec![1, 2],
    )?;

    // y = xW + b
    let linear = x.matmul(&w)?;
    let with_bias = linear.add(&b)?;

    // Apply ReLU
    let output = with_bias.relu()?;

    assert_eq!(output.shape().dims(), &[1, 2]);
    Ok(())
}

#[test]
fn test_reshape_matmul_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Reshape then matmul
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![12],
    )?;

    let a_reshaped = a.reshape(vec![3, 4])?;

    let b = Tensor::from_vec(
        vec![f16::ONE; 8],
        vec![4, 2],
    )?;

    let result = a_reshaped.matmul(&b)?;
    assert_eq!(result.shape().dims(), &[3, 2]);
    Ok(())
}

// ============================================================================
// Reduction Pipeline Tests
// ============================================================================

#[test]
fn test_sum_mean_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10, 10],
    )?;

    // Sum along dimension 0, then compute mean
    let sum_dim0 = a.sum_dim(0, false)?;
    let mean_of_sum = sum_dim0.mean()?;

    assert!(mean_of_sum.to_f32() > 0.0);
    Ok(())
}

#[test]
fn test_max_min_range_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32((i as f32) * 0.1)).collect(),
        vec![10, 10],
    )?;

    let max_val = a.max()?;
    let min_val = a.min()?;
    let range = f16::from_f32(max_val.to_f32() - min_val.to_f32());

    assert!(range.to_f32() > 0.0);
    Ok(())
}

#[test]
fn test_normalize_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Normalize: (x - mean) / std
    let x = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![100],
    )?;

    let mean = x.mean()?;
    let mean_tensor = Tensor::from_vec(vec![mean; 100], vec![100])?;

    let centered = x.sub(&mean_tensor)?;

    // For std, we'd need variance, but we can test the pipeline
    let sum_sq = centered.mul(&centered)?.sum()?;

    assert!(sum_sq.to_f32() > 0.0);
    Ok(())
}

// ============================================================================
// Broadcasting Pipeline Tests
// ============================================================================

#[test]
fn test_broadcast_add_mul_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let a = Tensor::from_vec(
        (0..12).map(|i| f16::from_f32(i as f32)).collect(),
        vec![3, 4],
    )?;

    let b = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![4],
    )?;

    // Broadcast b to [3, 4]
    let target_shape = TensorShape::new(vec![3, 4]);
    let b_broadcast = b.broadcast_to(&target_shape)?;

    // (a + b) * 2
    let added = a.add(&b_broadcast)?;
    let multiplied = added.mul(&Tensor::from_vec(vec![f16::from_f32(2.0); 12], vec![3, 4])?)?;

    assert_eq!(multiplied.shape().dims(), &[3, 4]);
    Ok(())
}

#[test]
fn test_batch_normalization_pattern() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate batch normalization: (x - mean) * gamma + beta
    let x = Tensor::from_vec(
        (0..20).map(|i| f16::from_f32(i as f32)).collect(),
        vec![4, 5],
    )?;

    let mean = x.mean_dim(0, true)?;
    let gamma = Tensor::<f16>::ones(&device, vec![1, 5])?;
    let beta = Tensor::<f16>::zeros(&device, vec![1, 5])?;

    // Broadcast mean to x's shape
    let target_shape = TensorShape::new(vec![4, 5]);
    let mean_broadcast = mean.broadcast_to(&target_shape)?;
    let gamma_broadcast = gamma.broadcast_to(&target_shape)?;
    let beta_broadcast = beta.broadcast_to(&target_shape)?;

    // (x - mean) * gamma + beta
    let centered = x.sub(&mean_broadcast)?;
    let scaled = centered.mul(&gamma_broadcast)?;
    let normalized = scaled.add(&beta_broadcast)?;

    assert_eq!(normalized.shape().dims(), &[4, 5]);
    Ok(())
}

// ============================================================================
// Indexing Pipeline Tests
// ============================================================================

#[test]
fn test_gather_process_scatter_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Gather, process, scatter back
    let x = Tensor::from_vec(
        (0..10).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10],
    )?;

    let indices = Tensor::from_vec(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(3.0),
            f16::from_f32(5.0),
        ],
        vec![3],
    )?;

    // Gather
    let gathered = x.gather(0, &indices)?;

    // Process: multiply by 10
    let processed = gathered.mul(&Tensor::from_vec(vec![f16::from_f32(10.0); 3], vec![3])?)?;

    // Scatter back
    let result = x.scatter(0, &indices, &processed)?;

    assert_eq!(result.shape().dims(), &[10]);
    Ok(())
}

#[test]
fn test_embedding_attention_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate simple embedding lookup + processing
    let vocab_size = 10;
    let d_model = 8;

    let embedding_weight = Tensor::from_vec(
        (0..vocab_size * d_model)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect(),
        vec![vocab_size, d_model],
    )?;

    let token_ids = Tensor::from_vec(
        vec![
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(5.0),
        ],
        vec![3],
    )?;

    // Embedding lookup
    let embeddings = embedding_weight.embedding(&token_ids)?;

    // Process: apply layer norm-like operation
    let mean = embeddings.mean_dim(1, true)?;
    let target_shape = TensorShape::new(vec![3, d_model]);
    let mean_broadcast = mean.broadcast_to(&target_shape)?;
    let centered = embeddings.sub(&mean_broadcast)?;

    assert_eq!(centered.shape().dims(), &[3, d_model]);
    Ok(())
}

// ============================================================================
// Complex Multi-Step Pipelines
// ============================================================================

#[test]
fn test_mlp_forward_pass() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simple 2-layer MLP: input -> hidden -> output
    let batch_size = 4;
    let input_dim = 10;
    let hidden_dim = 8;
    let output_dim = 3;

    // Input
    let x = Tensor::from_vec(
        (0..batch_size * input_dim)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect(),
        vec![batch_size, input_dim],
    )?;

    // Layer 1: Linear + ReLU
    let w1 = Tensor::<f16>::ones(&device, vec![input_dim, hidden_dim])?;
    let b1 = Tensor::zeros(&device, vec![1, hidden_dim])?;

    let h = x.matmul(&w1)?;
    let target_shape = TensorShape::new(vec![batch_size, hidden_dim]);
    let b1_broadcast = b1.broadcast_to(&target_shape)?;
    let h = h.add(&b1_broadcast)?;
    let h = h.relu()?;

    // Layer 2: Linear + Softmax
    let w2 = Tensor::<f16>::ones(&device, vec![hidden_dim, output_dim])?;
    let b2 = Tensor::zeros(&device, vec![1, output_dim])?;

    let output = h.matmul(&w2)?;
    let target_shape2 = TensorShape::new(vec![batch_size, output_dim]);
    let b2_broadcast = b2.broadcast_to(&target_shape2)?;
    let output = output.add(&b2_broadcast)?;
    let output = output.softmax()?;

    assert_eq!(output.shape().dims(), &[batch_size, output_dim]);
    Ok(())
}

#[test]
fn test_attention_mechanism_simple() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simplified attention: scores = Q @ K^T, attention = softmax(scores)
    let seq_len = 4;
    let d_k = 8;

    let q = Tensor::<f16>::ones(&device, vec![seq_len, d_k])?;
    let k = Tensor::<f16>::ones(&device, vec![seq_len, d_k])?;

    // Scores = Q @ K^T
    let k_t = k.transpose()?;
    let scores = q.matmul(&k_t)?;

    // Apply softmax
    let attention_weights = scores.softmax()?;

    assert_eq!(attention_weights.shape().dims(), &[seq_len, seq_len]);
    Ok(())
}

#[test]
fn test_residual_connection() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Residual connection: output = f(x) + x
    let x = Tensor::from_vec(
        (0..20).map(|i| f16::from_f32(i as f32 * 0.1)).collect(),
        vec![4, 5],
    )?;

    // Apply transformation
    let w = Tensor::<f16>::ones(&device, vec![5, 5])?;
    let transformed = x.matmul(&w)?;

    // Residual connection
    let output = transformed.add(&x)?;

    assert_eq!(output.shape().dims(), &[4, 5]);
    Ok(())
}

// ============================================================================
// Data Processing Pipelines
// ============================================================================

#[test]
fn test_data_augmentation_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate data augmentation: normalize, add noise, clip
    let data = Tensor::from_vec(
        (0..100).map(|i| f16::from_f32(i as f32)).collect(),
        vec![10, 10],
    )?;

    // Normalize to [0, 1]
    let max_val = data.max()?;
    let max_tensor = Tensor::from_vec(vec![max_val; 100], vec![10, 10])?;
    let normalized = data.div(&max_tensor)?;

    // Add "noise" (just add 0.1)
    let noise = Tensor::from_vec(vec![f16::from_f32(0.1); 100], vec![10, 10])?;
    let noisy = normalized.add(&noise)?;

    assert_eq!(noisy.shape().dims(), &[10, 10]);
    Ok(())
}

#[test]
fn test_feature_extraction_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Extract features: reshape, select, aggregate
    let data = Tensor::from_vec(
        (0..120).map(|i| f16::from_f32(i as f32)).collect(),
        vec![120],
    )?;

    // Reshape to [10, 12]
    let reshaped = data.reshape(vec![10, 12])?;

    // Aggregate along dimension 1 (mean)
    let features = reshaped.mean_dim(1, false)?;

    // Apply transformation
    let w = Tensor::<f16>::ones(&device, vec![10])?;
    let output = features.mul(&w)?;

    assert_eq!(output.shape().dims(), &[10]);
    Ok(())
}

// ============================================================================
// Conditional Logic Pipelines
// ============================================================================

#[test]
fn test_threshold_and_process() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let x = Tensor::from_vec(
        (0..20).map(|i| f16::from_f32((i as f32) - 10.0)).collect(),
        vec![20],
    )?;

    // Apply ReLU (threshold at 0)
    let positive_only = x.relu()?;

    // Process positive values
    let scaled = positive_only.mul(&Tensor::from_vec(vec![f16::from_f32(2.0); 20], vec![20])?)?;

    let sum = scaled.sum()?;
    assert!(sum.to_f32() >= 0.0);
    Ok(())
}

#[test]
fn test_multi_branch_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    let x = Tensor::from_vec(
        (0..20).map(|i| f16::from_f32(i as f32)).collect(),
        vec![4, 5],
    )?;

    // Branch 1: Simple linear
    let w1 = Tensor::<f16>::ones(&device, vec![5, 3])?;
    let branch1 = x.matmul(&w1)?;

    // Branch 2: Linear with activation
    let w2 = Tensor::<f16>::ones(&device, vec![5, 3])?;
    let branch2_linear = x.matmul(&w2)?;
    let branch2 = branch2_linear.relu()?;

    // Merge branches (add)
    let merged = branch1.add(&branch2)?;

    assert_eq!(merged.shape().dims(), &[4, 3]);
    Ok(())
}

// ============================================================================
// Performance Pattern Tests
// ============================================================================

#[test]
fn test_iterative_refinement() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Iterative processing (like gradient descent step)
    let mut x = Tensor::ones(&device, vec![10])?;

    for _iter in 0..5 {
        // x = x * 0.9 (decay)
        let decay = Tensor::from_vec(vec![f16::from_f32(0.9); 10], vec![10])?;
        x = x.mul(&decay)?;

        // Add small value
        let delta = Tensor::from_vec(vec![f16::from_f32(0.01); 10], vec![10])?;
        x = x.add(&delta)?;
    }

    let final_val = x.mean()?;
    assert!(final_val.to_f32() > 0.0);
    Ok(())
}

#[test]
fn test_batch_processing_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Process multiple batches
    let mut results = Vec::new();

    for batch_idx in 0..3 {
        let batch = Tensor::from_vec(
            (0..20).map(|i| f16::from_f32((batch_idx * 20 + i) as f32)).collect(),
            vec![4, 5],
        )?;

        // Process batch
        let w = Tensor::<f16>::ones(&device, vec![5, 2])?;
        let processed = batch.matmul(&w)?;
        let activated = processed.relu()?;
        let aggregated = activated.sum_dim(0, false)?;

        results.push(aggregated);
    }

    assert_eq!(results.len(), 3);
    Ok(())
}

// ============================================================================
// Real-World Scenario Tests
// ============================================================================

#[test]
fn test_inference_pipeline() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate model inference pipeline
    let input = Tensor::from_vec(
        vec![f16::from_f32(0.5); 10],
        vec![1, 10],
    )?;

    // Preprocess
    let mean = Tensor::from_vec(vec![f16::from_f32(0.3); 10], vec![1, 10])?;
    let std = Tensor::from_vec(vec![f16::from_f32(0.2); 10], vec![1, 10])?;
    let normalized = input.sub(&mean)?.div(&std)?;

    // Forward pass
    let w = Tensor::<f16>::ones(&device, vec![10, 5])?;
    let output = normalized.matmul(&w)?;
    let activated = output.relu()?;

    // Softmax for probabilities
    let probs = activated.softmax()?;

    assert_eq!(probs.shape().dims(), &[1, 5]);
    Ok(())
}

#[test]
fn test_training_step_simulation() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Simulate single training step (forward only, no autograd)
    let x = Tensor::<f16>::ones(&device, vec![8, 10])?; // batch_size=8, features=10
    let y_true = Tensor::<f16>::ones(&device, vec![8, 3])?; // 3 classes

    // Forward pass
    let w = Tensor::<f16>::ones(&device, vec![10, 3])?;
    let b = Tensor::<f16>::zeros(&device, vec![1, 3])?;

    let logits = x.matmul(&w)?;
    let target_shape = TensorShape::new(vec![8, 3]);
    let b_broadcast = b.broadcast_to(&target_shape)?;
    let logits = logits.add(&b_broadcast)?;

    // Softmax
    let probs = logits.softmax()?;

    // Loss (simplified)
    let diff = probs.sub(&y_true)?;
    let loss = diff.mul(&diff)?.sum()?;

    assert!(loss.to_f32() >= 0.0);
    Ok(())
}

#[test]
fn test_transformer_layer_components() -> TensorResult<()> {
    let device = MetalDevice::new()?;
    // Test components of transformer layer
    let seq_len = 4;
    let d_model = 8;

    let x = Tensor::<f16>::ones(&device, vec![seq_len, d_model])?;

    // Multi-head attention (simplified)
    let q_proj = Tensor::<f16>::ones(&device, vec![d_model, d_model])?;
    let k_proj = Tensor::<f16>::ones(&device, vec![d_model, d_model])?;
    let v_proj = Tensor::<f16>::ones(&device, vec![d_model, d_model])?;

    let q = x.matmul(&q_proj)?;
    let k = x.matmul(&k_proj)?;
    let v = x.matmul(&v_proj)?;

    // Attention scores
    let k_t = k.transpose()?;
    let scores = q.matmul(&k_t)?;
    let attn_weights = scores.softmax()?;

    // Attention output
    let attn_output = attn_weights.matmul(&v)?;

    // Residual connection
    let output = attn_output.add(&x)?;

    assert_eq!(output.shape().dims(), &[seq_len, d_model]);
    Ok(())
}
