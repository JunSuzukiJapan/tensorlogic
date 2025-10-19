//! Einstein summation (einsum) operations for tensors
//!
//! Einsum provides a concise notation for tensor operations:
//! - `ij,jk->ik`: Matrix multiplication
//! - `ii->i`: Diagonal extraction
//! - `ij->ji`: Transpose
//! - `ij,ij->ij`: Element-wise product
//! - `ij->`: Sum all elements
//! - `ij,j->i`: Matrix-vector product

use crate::device::Device;
use crate::error::{TensorError, TensorResult};
use crate::tensor::{Tensor, TensorShape};
use half::f16;
use std::collections::{HashMap, HashSet};

/// Parse einsum notation and execute the operation
impl Tensor {
    /// Einstein summation over tensors
    ///
    /// # Arguments
    /// - `equation`: Einsum notation (e.g., "ij,jk->ik")
    /// - `operands`: Input tensors
    ///
    /// # Examples
    /// ```ignore
    /// // Matrix multiplication
    /// let c = Tensor::einsum("ij,jk->ik", &[a, b])?;
    ///
    /// // Transpose
    /// let b = Tensor::einsum("ij->ji", &[a])?;
    ///
    /// // Trace (sum of diagonal)
    /// let trace = Tensor::einsum("ii->", &[a])?;
    /// ```
    pub fn einsum(equation: &str, operands: &[&Tensor]) -> TensorResult<Self> {
        // Parse equation
        let (input_specs, output_spec) = parse_einsum_equation(equation)?;

        // Validate operands
        if input_specs.len() != operands.len() {
            return Err(TensorError::InvalidOperation(format!(
                "Expected {} operands, got {}",
                input_specs.len(),
                operands.len()
            )));
        }

        // Check all tensors are on same device
        let device = operands[0].device();
        for op in &operands[1..] {
            if op.device() != device {
                return Err(TensorError::DeviceConversionError(
                    "All operands must be on same device".to_string(),
                ));
            }
        }

        match device {
            Device::CPU => einsum_cpu(equation, &input_specs, &output_spec, operands),
            Device::Metal(_) | Device::NeuralEngine => {
                // Fallback to CPU for now
                let cpu_ops: Vec<_> = operands.iter().map(|t| t.to_cpu()).collect::<Result<_, _>>()?;
                let cpu_refs: Vec<_> = cpu_ops.iter().collect();
                einsum_cpu(equation, &input_specs, &output_spec, &cpu_refs)
            }
        }
    }
}

/// Parse einsum equation into input and output specifications
fn parse_einsum_equation(equation: &str) -> TensorResult<(Vec<String>, String)> {
    let parts: Vec<&str> = equation.split("->").collect();

    if parts.len() > 2 {
        return Err(TensorError::InvalidOperation(
            "Invalid einsum equation: too many arrows".to_string(),
        ));
    }

    let inputs_str = parts[0];
    let output_str = if parts.len() == 2 {
        parts[1].to_string()
    } else {
        // Implicit output: all indices that appear once
        infer_output_spec(inputs_str)?
    };

    let input_specs: Vec<String> = inputs_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    Ok((input_specs, output_str))
}

/// Infer output specification for implicit einsum
fn infer_output_spec(inputs_str: &str) -> TensorResult<String> {
    let mut char_counts: HashMap<char, usize> = HashMap::new();

    for input in inputs_str.split(',') {
        for ch in input.trim().chars() {
            if ch.is_alphabetic() {
                *char_counts.entry(ch).or_insert(0) += 1;
            }
        }
    }

    // Output includes indices that appear exactly once (no summation)
    let mut output_chars: Vec<char> = char_counts
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&ch, _)| ch)
        .collect();

    output_chars.sort();
    Ok(output_chars.into_iter().collect())
}

/// Execute einsum on CPU
fn einsum_cpu(
    equation: &str,
    input_specs: &[String],
    output_spec: &str,
    operands: &[&Tensor],
) -> TensorResult<Tensor> {
    // Special case optimizations for common patterns
    if let Some(result) = try_optimized_einsum(equation, input_specs, output_spec, operands)? {
        return Ok(result);
    }

    // General einsum implementation
    general_einsum_cpu(input_specs, output_spec, operands)
}

/// Try to use optimized implementations for common patterns
fn try_optimized_einsum(
    equation: &str,
    input_specs: &[String],
    output_spec: &str,
    operands: &[&Tensor],
) -> TensorResult<Option<Tensor>> {
    // Matrix multiplication: ij,jk->ik
    if equation == "ij,jk->ik" || (input_specs.len() == 2 && input_specs[0] == "ij" && input_specs[1] == "jk" && output_spec == "ik") {
        if operands.len() == 2 {
            return Ok(Some(operands[0].matmul(operands[1])?));
        }
    }

    // Transpose: ij->ji
    if equation == "ij->ji" || (input_specs.len() == 1 && input_specs[0] == "ij" && output_spec == "ji") {
        if operands.len() == 1 && operands[0].shape().rank() == 2 {
            return Ok(Some(transpose_2d(operands[0])?));
        }
    }

    // Trace: ii->
    if equation == "ii->" || (input_specs.len() == 1 && input_specs[0] == "ii" && output_spec.is_empty()) {
        if operands.len() == 1 && operands[0].shape().rank() == 2 {
            return Ok(Some(trace(operands[0])?));
        }
    }

    // Element-wise product: ij,ij->ij
    if input_specs.len() == 2 && input_specs[0] == input_specs[1] && output_spec == input_specs[0] {
        if operands.len() == 2 {
            return Ok(Some(operands[0].mul(operands[1])?));
        }
    }

    // Batch matrix multiplication: bij,bjk->bik
    if equation == "bij,bjk->bik" {
        if operands.len() == 2 && operands[0].shape().rank() == 3 && operands[1].shape().rank() == 3 {
            return Ok(Some(batch_matmul(operands[0], operands[1])?));
        }
    }

    Ok(None)
}

/// General einsum implementation
fn general_einsum_cpu(
    input_specs: &[String],
    output_spec: &str,
    operands: &[&Tensor],
) -> TensorResult<Tensor> {
    // Collect all indices
    let mut all_indices: HashSet<char> = HashSet::new();
    for spec in input_specs {
        for ch in spec.chars() {
            all_indices.insert(ch);
        }
    }

    // Determine which indices to sum over (not in output)
    let output_indices: HashSet<char> = output_spec.chars().collect();
    let sum_indices: Vec<char> = all_indices
        .difference(&output_indices)
        .copied()
        .collect();

    // Build index-to-dimension mapping for each operand
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (op, spec) in operands.iter().zip(input_specs.iter()) {
        let dims = op.shape().dims();
        for (i, ch) in spec.chars().enumerate() {
            if let Some(&existing_dim) = index_dims.get(&ch) {
                if existing_dim != dims[i] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![existing_dim],
                        actual: vec![dims[i]],
                    });
                }
            } else {
                index_dims.insert(ch, dims[i]);
            }
        }
    }

    // Compute output shape
    let output_dims: Vec<usize> = output_spec
        .chars()
        .map(|ch| {
            index_dims.get(&ch).copied().ok_or_else(|| {
                TensorError::InvalidOperation(format!("Unknown index in output: {}", ch))
            })
        })
        .collect::<TensorResult<_>>()?;

    let output_shape = TensorShape::new(output_dims.clone());
    let output_numel = output_shape.numel();

    // Compute result using nested loops
    let mut output = vec![f16::ZERO; output_numel];

    // Iterate over all output positions
    for out_idx in 0..output_numel {
        // Compute output coordinates
        let out_coords = index_to_coords(out_idx, &output_dims);
        let mut index_values: HashMap<char, usize> = HashMap::new();

        for (i, ch) in output_spec.chars().enumerate() {
            index_values.insert(ch, out_coords[i]);
        }

        // Sum over contraction indices
        let sum_value = sum_over_indices(&sum_indices, &index_dims, &index_values, input_specs, operands)?;
        output[out_idx] = sum_value;
    }

    Tensor::from_vec(output, output_dims)
}

/// Recursively sum over contraction indices
fn sum_over_indices(
    remaining_indices: &[char],
    index_dims: &HashMap<char, usize>,
    current_values: &HashMap<char, usize>,
    input_specs: &[String],
    operands: &[&Tensor],
) -> TensorResult<f16> {
    if remaining_indices.is_empty() {
        // Base case: compute product of all operands
        let mut product = f16::ONE;

        for (op, spec) in operands.iter().zip(input_specs.iter()) {
            let coords: Vec<usize> = spec
                .chars()
                .map(|ch| current_values[&ch])
                .collect();

            let linear_idx = coords_to_index(&coords, op.shape().dims());
            let data = op.to_vec();
            product *= data[linear_idx];
        }

        Ok(product)
    } else {
        // Recursive case: sum over next index
        let idx = remaining_indices[0];
        let dim_size = index_dims[&idx];
        let mut sum = f16::ZERO;

        for i in 0..dim_size {
            let mut new_values = current_values.clone();
            new_values.insert(idx, i);

            sum += sum_over_indices(
                &remaining_indices[1..],
                index_dims,
                &new_values,
                input_specs,
                operands,
            )?;
        }

        Ok(sum)
    }
}

/// Convert linear index to multi-dimensional coordinates
fn index_to_coords(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    let mut strides = vec![1; dims.len()];

    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    for i in 0..dims.len() {
        coords[i] = index / strides[i];
        index %= strides[i];
    }

    coords
}

/// Convert multi-dimensional coordinates to linear index
fn coords_to_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;

    for i in (0..dims.len()).rev() {
        index += coords[i] * stride;
        stride *= dims[i];
    }

    index
}

/// Transpose a 2D tensor
fn transpose_2d(tensor: &Tensor) -> TensorResult<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(TensorError::InvalidOperation(
            "transpose_2d requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (dims[0], dims[1]);
    let data = tensor.to_vec();
    let mut transposed = vec![f16::ZERO; m * n];

    for i in 0..m {
        for j in 0..n {
            transposed[j * m + i] = data[i * n + j];
        }
    }

    Tensor::from_vec(transposed, vec![n, m])
}

/// Compute trace of a 2D tensor
fn trace(tensor: &Tensor) -> TensorResult<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Err(TensorError::InvalidOperation(
            "trace requires square matrix".to_string(),
        ));
    }

    let n = dims[0];
    let data = tensor.to_vec();
    let mut sum = f16::ZERO;

    for i in 0..n {
        sum += data[i * n + i];
    }

    Tensor::from_vec(vec![sum], vec![1])
}

/// Batch matrix multiplication
fn batch_matmul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    if a_dims.len() != 3 || b_dims.len() != 3 {
        return Err(TensorError::InvalidOperation(
            "batch_matmul requires 3D tensors".to_string(),
        ));
    }

    if a_dims[0] != b_dims[0] || a_dims[2] != b_dims[1] {
        return Err(TensorError::ShapeMismatch {
            expected: a_dims.to_vec(),
            actual: b_dims.to_vec(),
        });
    }

    let (batch, m, k) = (a_dims[0], a_dims[1], a_dims[2]);
    let n = b_dims[2];

    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut c_data = vec![f16::ZERO; batch * m * n];

    for b_idx in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = f16::ZERO;
                for p in 0..k {
                    let a_idx = b_idx * m * k + i * k + p;
                    let b_idx_calc = b_idx * k * n + p * n + j;
                    sum += a_data[a_idx] * b_data[b_idx_calc];
                }
                c_data[b_idx * m * n + i * n + j] = sum;
            }
        }
    }

    Tensor::from_vec(c_data, vec![batch, m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_matmul() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let c = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();
        let result = c.to_vec();

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(result[0], f16::from_f32(19.0));
        assert_eq!(result[1], f16::from_f32(22.0));
        assert_eq!(result[2], f16::from_f32(43.0));
        assert_eq!(result[3], f16::from_f32(50.0));
    }

    #[test]
    fn test_einsum_transpose() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        )
        .unwrap();

        let b = Tensor::einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(b.shape().dims(), &[3, 2]);

        let result = b.to_vec();
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(4.0));
        assert_eq!(result[2], f16::from_f32(2.0));
        assert_eq!(result[3], f16::from_f32(5.0));
    }

    #[test]
    fn test_einsum_trace() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let trace = Tensor::einsum("ii->", &[&a]).unwrap();
        let result = trace.to_vec();

        // trace = 1 + 4 = 5
        assert_eq!(result[0], f16::from_f32(5.0));
    }

    #[test]
    fn test_einsum_element_wise() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let c = Tensor::einsum("ij,ij->ij", &[&a, &b]).unwrap();
        let result = c.to_vec();

        assert_eq!(result[0], f16::from_f32(2.0));
        assert_eq!(result[1], f16::from_f32(6.0));
        assert_eq!(result[2], f16::from_f32(12.0));
        assert_eq!(result[3], f16::from_f32(20.0));
    }

    #[test]
    fn test_einsum_outer_product() {
        let a = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            vec![2],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![f16::from_f32(3.0), f16::from_f32(4.0)],
            vec![2],
        )
        .unwrap();

        let c = Tensor::einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);

        let result = c.to_vec();
        assert_eq!(result[0], f16::from_f32(3.0)); // 1*3
        assert_eq!(result[1], f16::from_f32(4.0)); // 1*4
        assert_eq!(result[2], f16::from_f32(6.0)); // 2*3
        assert_eq!(result[3], f16::from_f32(8.0)); // 2*4
    }

    #[test]
    fn test_einsum_batch_matmul() {
        let a = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                // Second batch
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2, 2],
        )
        .unwrap();

        let b = Tensor::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                // Second batch
                f16::from_f32(1.0),
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
            ],
            vec![2, 2, 2],
        )
        .unwrap();

        let c = Tensor::einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2, 2]);
    }
}
