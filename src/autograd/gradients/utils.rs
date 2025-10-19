use crate::error::TensorResult;
use crate::tensor::{Tensor, TensorShape};

/// ブロードキャストされた勾配を元の形状に縮約
///
/// # Arguments
/// * `grad` - ブロードキャストされた形状の勾配
/// * `original_shape` - 元の形状
///
/// # Returns
/// 元の形状に縮約された勾配
pub fn reduce_grad_for_broadcast(grad: &Tensor, original_shape: &TensorShape) -> TensorResult<Tensor> {
    if grad.shape() == original_shape {
        return Ok(grad.clone());
    }

    let grad_dims = grad.dims();
    let orig_dims = original_shape.dims();

    // 勾配の次元数が元の次元数より大きい場合（先頭に次元が追加された場合）
    let rank_diff = grad_dims.len().saturating_sub(orig_dims.len());

    let mut result = grad.clone();

    // 先頭の追加次元を縮約（sum）
    for _ in 0..rank_diff {
        result = result.sum_dim(0, false)?;
    }

    // サイズ1だった次元を縮約
    let current_dims = result.dims().to_vec();
    for (i, (&orig_size, &current_size)) in orig_dims.iter().zip(current_dims.iter()).enumerate() {
        if orig_size == 1 && current_size > 1 {
            result = result.sum_dim(i, true)?;
        }
    }

    Ok(result)
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
    fn test_reduce_grad_no_broadcast() {
        let device = get_test_device();
        let grad = Tensor::from_vec_metal(
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

        let original_shape = TensorShape::new(vec![2, 2]);
        let result = reduce_grad_for_broadcast(&grad, &original_shape).unwrap();

        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.to_vec(), grad.to_vec());
    }

    #[test]
    fn test_reduce_grad_broadcast_scalar() {
        let device = get_test_device();
        // 勾配は [2, 2] だが、元の形状は [1] (スカラー)
        let grad = Tensor::from_vec_metal(
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

        let original_shape = TensorShape::new(vec![1]);
        let result = reduce_grad_for_broadcast(&grad, &original_shape).unwrap();

        assert_eq!(result.dims(), &[1]);
        // sum = 1 + 2 + 3 + 4 = 10
        assert_eq!(result.to_vec()[0], f16::from_f32(10.0));
    }
}
