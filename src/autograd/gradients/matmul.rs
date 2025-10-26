use crate::autograd::GradientFunction;
use super::prelude::*;
use crate::error::TensorResult;
use crate::tensor::Tensor;

/// MatMul演算の勾配関数
///
/// C = A @ B の場合 (A: [M, K], B: [K, N], C: [M, N]):
/// ∂L/∂A = ∂L/∂C @ B^T  ([M, N] @ [N, K] = [M, K])
/// ∂L/∂B = A^T @ ∂L/∂C  ([K, M] @ [M, N] = [K, N])
pub struct MatMulBackward {
    a: Tensor,
    b: Tensor,
}

impl MatMulBackward {
    pub fn new(a: Tensor, b: Tensor) -> Self {
        Self { a, b }
    }
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor, _inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>> {
        // 転置を計算（einsum経由）
        // Note: einsumの結果はCPUテンソルになる可能性があるため、
        // grad_outputをCPUに移動してから計算し、最後に元のデバイスに戻す
        let grad_output_cpu = grad_output.to_cpu()?;
        let a_cpu = self.a.to_cpu()?;
        let b_cpu = self.b.to_cpu()?;

        // B^T を計算
        let b_t = Tensor::einsum("ij->ji", &[&b_cpu])?;
        let grad_a_cpu = grad_output_cpu.matmul(&b_t)?;

        // A^T を計算
        let a_t = Tensor::einsum("ij->ji", &[&a_cpu])?;
        let grad_b_cpu = a_t.matmul(&grad_output_cpu)?;

        // 元のデバイスがMetalの場合は戻す（ここでは簡略化のためCPUのまま）
        Ok(vec![grad_a_cpu, grad_b_cpu])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use half::f16;

    fn get_test_device() -> MetalDevice {
        MetalDevice::new().expect("No Metal device available")
    }

    #[test]
    fn test_matmul_backward() {
        let device = get_test_device();

        // A = [[1, 2], [3, 4]] (2x2)
        let a = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        // B = [[5, 6], [7, 8]] (2x2)
        let b = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        // C = A @ B = [[19, 22], [43, 50]]
        // grad_output = [[1, 1], [1, 1]]
        let grad_output = Tensor::from_vec_metal(
            &device,
            vec![
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                f16::from_f32(1.0),
            ],
            vec![2, 2],
        )
        .unwrap();

        let backward = MatMulBackward::new(a.clone(), b.clone());
        let grads = backward.backward(&grad_output, &[]).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].dims(), &[2, 2]); // grad_a
        assert_eq!(grads[1].dims(), &[2, 2]); // grad_b

        // grad_a = grad_output @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]]
        //        = [[11, 15], [11, 15]]
        let grad_a_expected = vec![
            f16::from_f32(11.0),
            f16::from_f32(15.0),
            f16::from_f32(11.0),
            f16::from_f32(15.0),
        ];
        assert_eq!(grads[0].to_vec(), grad_a_expected);

        // grad_b = A^T @ grad_output = [[1,3],[2,4]] @ [[1,1],[1,1]]
        //        = [[4, 4], [6, 6]]
        let grad_b_expected = vec![
            f16::from_f32(4.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
            f16::from_f32(6.0),
        ];
        assert_eq!(grads[1].to_vec(), grad_b_expected);
    }
}
