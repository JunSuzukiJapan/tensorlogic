//! Comprehensive numerical correctness tests for tensor operations
//!
//! This test suite verifies the mathematical correctness of all tensor operations
//! by comparing computed results against known expected values.

#[cfg(test)]
mod tensor_ops_tests {
    use crate::device::{Device, MetalDevice};
    use crate::tensor::{Tensor, TensorAccessors, TensorCreation, TensorIO, TensorTransform};
    use crate::error::TensorResult;
    use half::f16;

    /// Helper to create test device (Metal for GPU acceleration)
    fn test_device() -> Device {
        Device::Metal(MetalDevice::new().expect("Failed to create Metal device"))
    }

    /// Helper to create GPU tensor (sync is now automatic in from_vec_gpu)
    fn create_gpu_tensor(device: &MetalDevice, data: Vec<f16>, shape: Vec<usize>) -> Tensor<f16> {
        Tensor::from_vec_gpu(device, data, shape).unwrap()
    }

    /// Helper to compare tensor values with tolerance
    /// Note: This takes ownership of the actual values to avoid multiple sync_and_read calls
    fn assert_tensor_eq_values(actual_vec: &[f16], expected: &[f32], tolerance: f32) {
        assert_eq!(
            actual_vec.len(),
            expected.len(),
            "Tensor length mismatch: {} vs {}",
            actual_vec.len(),
            expected.len()
        );

        for (i, (&a, &e)) in actual_vec.iter().zip(expected.iter()).enumerate() {
            let diff = (a.to_f32() - e).abs();
            assert!(
                diff < tolerance,
                "Value mismatch at index {}: expected {}, got {} (diff: {})",
                i, e, a.to_f32(), diff
            );
        }
    }

    /// Helper to compare tensors with tolerance (reads values once)
    fn assert_tensor_eq(actual: &Tensor<f16>, expected: &[f32], tolerance: f32) {
        let actual_vec = actual.sync_and_read();
        assert_tensor_eq_values(&actual_vec, expected, tolerance);
    }

    // ============================================================================
    // Element-wise Operations
    // ============================================================================

    #[test]
    fn test_add_values() {
        let device = test_device();

        // Test: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
                vec![3],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.add(&b).unwrap();
        assert_tensor_eq(&result, &[5.0, 7.0, 9.0], 0.01);
    }

    #[test]
    fn test_add_2d() {
        let device = test_device();

        // Test: [[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0),
                         f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0),
                     f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(5.0), f16::from_f32(6.0),
                         f16::from_f32(7.0), f16::from_f32(8.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(5.0), f16::from_f32(6.0),
                     f16::from_f32(7.0), f16::from_f32(8.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let result = a.add(&b).unwrap();
        assert_tensor_eq(&result, &[6.0, 8.0, 10.0, 12.0], 0.01);
    }

    #[test]
    fn test_sub_values() {
        let device = test_device();

        // Test: [10, 20, 30] - [1, 2, 3] = [9, 18, 27]
        let a = match &device {
            Device::Metal(dev) => {
                create_gpu_tensor(
                    dev,
                    vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
                    vec![3],
                )
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
                vec![3],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                create_gpu_tensor(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
                    vec![3],
                )
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.sub(&b).unwrap();
        assert_tensor_eq(&result, &[9.0, 18.0, 27.0], 0.01);
    }

    #[test]
    fn test_mul_values() {
        let device = test_device();

        // Test: [2, 3, 4] * [5, 6, 7] = [10, 18, 28]
        let a = match &device {
            Device::Metal(dev) => {
                create_gpu_tensor(
                    dev,
                    vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![3],
                )
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![3],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                create_gpu_tensor(
                    dev,
                    vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)],
                    vec![3],
                )
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.mul(&b).unwrap();
        assert_tensor_eq(&result, &[10.0, 18.0, 28.0], 0.01);
    }

    #[test]
    fn test_div_values() {
        let device = test_device();

        // Test: [10, 20, 30] / [2, 4, 5] = [5, 5, 6]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)],
                vec![3],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(5.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(5.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.div(&b).unwrap();
        assert_tensor_eq(&result, &[5.0, 5.0, 6.0], 0.01);
    }

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    #[test]
    fn test_matmul_2x2() {
        let device = test_device();

        // Test: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Calculation:
        // [0,0] = 1*5 + 2*7 = 5 + 14 = 19
        // [0,1] = 1*6 + 2*8 = 6 + 16 = 22
        // [1,0] = 3*5 + 4*7 = 15 + 28 = 43
        // [1,1] = 3*6 + 4*8 = 18 + 32 = 50

        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0),
                         f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0),
                     f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(5.0), f16::from_f32(6.0),
                         f16::from_f32(7.0), f16::from_f32(8.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(5.0), f16::from_f32(6.0),
                     f16::from_f32(7.0), f16::from_f32(8.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        assert_tensor_eq(&result, &[19.0, 22.0, 43.0, 50.0], 0.1);
    }

    #[test]
    fn test_matmul_identity() {
        let device = test_device();

        // Test: [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0),
                         f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0),
                     f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let identity = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(0.0),
                         f16::from_f32(0.0), f16::from_f32(1.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(0.0),
                     f16::from_f32(0.0), f16::from_f32(1.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let result = a.matmul(&identity).unwrap();
        assert_tensor_eq(&result, &[1.0, 2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    fn test_matmul_rectangular() {
        let device = test_device();

        // Test: [[1, 2, 3], [4, 5, 6]] @ [[7], [8], [9]]
        // = [[1*7 + 2*8 + 3*9], [4*7 + 5*8 + 6*9]]
        // = [[7 + 16 + 27], [28 + 40 + 54]]
        // = [[50], [122]]

        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                         f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                    vec![2, 3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                     f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                vec![2, 3],
            ).unwrap(),
        };

        let b = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(7.0), f16::from_f32(8.0), f16::from_f32(9.0)],
                    vec![3, 1],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(7.0), f16::from_f32(8.0), f16::from_f32(9.0)],
                vec![3, 1],
            ).unwrap(),
        };

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.dims(), &[2, 1]);
        assert_tensor_eq(&result, &[50.0, 122.0], 0.1);
    }

    // ============================================================================
    // Shape Operations
    // ============================================================================

    #[test]
    fn test_reshape_values() {
        let device = test_device();

        // Test: [1, 2, 3, 4, 5, 6] reshaped to [2, 3]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                         f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                    vec![6],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                     f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                vec![6],
            ).unwrap(),
        };

        let result = a.reshape(vec![2, 3]).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
        // Values should remain in same order: [[1, 2, 3], [4, 5, 6]]
        assert_tensor_eq(&result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.01);
    }

    #[test]
    fn test_flatten_values() {
        let device = test_device();

        // Test: [[1, 2], [3, 4]] flattened to [1, 2, 3, 4]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0),
                         f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![2, 2],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0),
                     f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![2, 2],
            ).unwrap(),
        };

        let result = a.flatten().unwrap();
        assert_eq!(result.dims(), &[4]);
        assert_tensor_eq(&result, &[1.0, 2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    fn test_transpose_values() {
        let device = test_device();

        // Test: [[1, 2, 3], [4, 5, 6]] transposed to [[1, 4], [2, 5], [3, 6]]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                         f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                    vec![2, 3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0),
                     f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)],
                vec![2, 3],
            ).unwrap(),
        };

        let result = a.transpose().unwrap();
        assert_eq!(result.dims(), &[3, 2]);
        // Row-major layout: [[1, 4], [2, 5], [3, 6]] = [1, 4, 2, 5, 3, 6]
        let result_contiguous = result.contiguous().unwrap();
        assert_tensor_eq(&result_contiguous, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0.01);
    }

    // ============================================================================
    // Creation Operations
    // ============================================================================

    #[test]
    fn test_zeros_values() {
        // Create zeros using from_vec since TensorCreation requires MetalDevice
        let result = Tensor::<f16>::from_vec(
            vec![f16::from_f32(0.0); 6],
            vec![2, 3],
        ).unwrap();

        assert_eq!(result.dims(), &[2, 3]);
        assert_tensor_eq(&result, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01);
    }

    #[test]
    fn test_ones_values() {
        // Create ones using from_vec since TensorCreation requires MetalDevice
        let result = Tensor::<f16>::from_vec(
            vec![f16::from_f32(1.0); 4],
            vec![2, 2],
        ).unwrap();

        assert_eq!(result.dims(), &[2, 2]);
        assert_tensor_eq(&result, &[1.0, 1.0, 1.0, 1.0], 0.01);
    }

    // ============================================================================
    // Mathematical Functions
    // ============================================================================

    #[test]
    fn test_exp_values() {
        let device = test_device();

        // Test: exp([0, 1, 2]) ≈ [1, 2.718, 7.389]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.exp().unwrap();
        assert_tensor_eq(&result, &[1.0, 2.718281828, 7.389056099], 0.01);
    }

    #[test]
    fn test_log_values() {
        let device = test_device();

        // Test: log([1, e, e²]) = [0, 1, 2]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(1.0), f16::from_f32(2.718281828), f16::from_f32(7.389056099)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(1.0), f16::from_f32(2.718281828), f16::from_f32(7.389056099)],
                vec![3],
            ).unwrap(),
        };

        let result = a.log().unwrap();
        assert_tensor_eq(&result, &[0.0, 1.0, 2.0], 0.01);
    }

    #[test]
    fn test_sqrt_values() {
        let device = test_device();

        // Test: sqrt([4, 9, 16]) = [2, 3, 4]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(4.0), f16::from_f32(9.0), f16::from_f32(16.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(4.0), f16::from_f32(9.0), f16::from_f32(16.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.sqrt().unwrap();
        assert_tensor_eq(&result, &[2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    fn test_pow_values() {
        let device = test_device();

        // Test: pow([2, 3, 4], 2) = [4, 9, 16]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
                vec![3],
            ).unwrap(),
        };

        let result = a.pow(2.0).unwrap();
        assert_tensor_eq(&result, &[4.0, 9.0, 16.0], 0.01);
    }

    #[test]
    fn test_sin_values() {
        let device = test_device();

        // Test: sin([0, π/2, π]) ≈ [0, 1, 0]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0),
                         f16::from_f32(std::f32::consts::PI)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0),
                     f16::from_f32(std::f32::consts::PI)],
                vec![3],
            ).unwrap(),
        };

        let result = a.sin().unwrap();
        assert_tensor_eq(&result, &[0.0, 1.0, 0.0], 0.01);
    }

    #[test]
    fn test_cos_values() {
        let device = test_device();

        // Test: cos([0, π/2, π]) ≈ [1, 0, -1]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0),
                         f16::from_f32(std::f32::consts::PI)],
                    vec![3],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(0.0), f16::from_f32(std::f32::consts::PI / 2.0),
                     f16::from_f32(std::f32::consts::PI)],
                vec![3],
            ).unwrap(),
        };

        let result = a.cos().unwrap();
        assert_tensor_eq(&result, &[1.0, 0.0, -1.0], 0.01);
    }

    #[test]
    fn test_sigmoid_values() {
        let device = test_device();

        // Test: sigmoid([0]) = [0.5]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.0)],
                    vec![1],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(0.0)],
                vec![1],
            ).unwrap(),
        };

        let result = a.sigmoid().unwrap();
        assert_tensor_eq(&result, &[0.5], 0.01);
    }

    #[test]
    fn test_tanh_values() {
        let device = test_device();

        // Test: tanh([0]) = [0]
        let a = match &device {
            Device::Metal(dev) => {
                Tensor::from_vec_gpu(
                    dev,
                    vec![f16::from_f32(0.0)],
                    vec![1],
                ).unwrap()
            }
            _ => Tensor::from_vec(
                vec![f16::from_f32(0.0)],
                vec![1],
            ).unwrap(),
        };

        let result = a.tanh().unwrap();
        assert_tensor_eq(&result, &[0.0], 0.01);
    }
}
