//! Neural Engine operations using CoreML

use crate::device::NeuralEngineBuffer;
use crate::error::{TensorError, TensorResult};

/// Neural Engine operation executor
///
/// Note: This is a simplified implementation for Phase 4.
/// Full CoreML model integration will be added in future phases.
pub struct NeuralEngineOps;

impl NeuralEngineOps {
    /// Perform matrix multiplication on Neural Engine
    ///
    /// This is a placeholder implementation that delegates to CPU.
    /// Future versions will use CoreML models for actual Neural Engine execution.
    pub fn matmul(
        a: &NeuralEngineBuffer,
        b: &NeuralEngineBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> TensorResult<NeuralEngineBuffer> {
        // Validate shapes
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape != vec![m, k] || b_shape != vec![k, n] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![m, k],
                actual: a_shape,
            });
        }

        // Get data and perform CPU matmul
        let a_data = a.to_f16_vec();
        let b_data = b.to_f16_vec();

        let mut c_data = vec![half::f16::ZERO; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = half::f16::ZERO;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        NeuralEngineBuffer::from_f16_slice(&c_data, &[m, n])
    }

    /// Perform ReLU activation on Neural Engine
    ///
    /// This is a placeholder implementation that delegates to CPU.
    pub fn relu(input: &NeuralEngineBuffer) -> TensorResult<NeuralEngineBuffer> {
        let data = input.to_f16_vec();
        let shape = input.shape();

        let output_data: Vec<half::f16> = data
            .iter()
            .map(|&x| {
                if x > half::f16::ZERO {
                    x
                } else {
                    half::f16::ZERO
                }
            })
            .collect();

        NeuralEngineBuffer::from_f16_slice(&output_data, &shape)
    }

    /// Check if Neural Engine is available
    pub fn is_available() -> bool {
        // On macOS/iOS, CoreML with Neural Engine is available
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }

    /// Get Neural Engine info
    pub fn info() -> String {
        if Self::is_available() {
            "Neural Engine available via CoreML (placeholder implementation)".to_string()
        } else {
            "Neural Engine not available on this platform".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_engine_availability() {
        let available = NeuralEngineOps::is_available();
        println!("Neural Engine available: {}", available);
        println!("Info: {}", NeuralEngineOps::info());
    }

    #[test]
    fn test_neural_engine_matmul() {
        // Create test matrices
        let a_data = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let b_data = vec![
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
            half::f16::from_f32(7.0),
            half::f16::from_f32(8.0),
        ];

        let a = NeuralEngineBuffer::from_f16_slice(&a_data, &[2, 2]).unwrap();
        let b = NeuralEngineBuffer::from_f16_slice(&b_data, &[2, 2]).unwrap();

        // Perform matmul
        let c = NeuralEngineOps::matmul(&a, &b, 2, 2, 2).unwrap();

        // Verify result
        assert_eq!(c.shape(), vec![2, 2]);
        let c_data = c.to_f16_vec();

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c_data[0].to_f32(), 19.0);
        assert_eq!(c_data[1].to_f32(), 22.0);
        assert_eq!(c_data[2].to_f32(), 43.0);
        assert_eq!(c_data[3].to_f32(), 50.0);
    }

    #[test]
    fn test_neural_engine_relu() {
        let data = vec![
            half::f16::from_f32(-2.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
        ];

        let input = NeuralEngineBuffer::from_f16_slice(&data, &[5]).unwrap();
        let output = NeuralEngineOps::relu(&input).unwrap();

        let result = output.to_f16_vec();
        assert_eq!(result[0].to_f32(), 0.0);
        assert_eq!(result[1].to_f32(), 0.0);
        assert_eq!(result[2].to_f32(), 0.0);
        assert_eq!(result[3].to_f32(), 1.0);
        assert_eq!(result[4].to_f32(), 2.0);
    }
}
