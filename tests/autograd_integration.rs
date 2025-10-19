use half::f16;
use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

fn get_test_device() -> MetalDevice {
    MetalDevice::new().expect("No Metal device available")
}

#[test]
fn test_requires_grad_api() {
    let device = get_test_device();

    let mut x = Tensor::from_vec_metal(
        &device,
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![2],
    )
    .unwrap();

    // デフォルトはrequires_grad = false
    assert!(!x.requires_grad());

    // requires_gradを有効化
    x.set_requires_grad(true);
    assert!(x.requires_grad());

    // 初期状態では勾配はNone
    assert!(x.grad().is_none());
}

#[test]
fn test_zero_grad() {
    let device = get_test_device();

    let mut x = Tensor::from_vec_metal(
        &device,
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![2],
    )
    .unwrap();

    x.set_requires_grad(true);

    // 勾配をクリア
    x.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
fn test_backward_requires_scalar() {
    let device = get_test_device();

    let mut x = Tensor::from_vec_metal(
        &device,
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![2],
    )
    .unwrap();

    x.set_requires_grad(true);

    // 非スカラーテンソルでbackward()を呼ぶとエラー
    let result = x.backward();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("can only be called on scalar tensors"));
}

#[test]
fn test_backward_scalar() {
    let device = get_test_device();

    let mut x = Tensor::from_vec_metal(&device, vec![f16::from_f32(3.0)], vec![1]).unwrap();

    x.set_requires_grad(true);

    // スカラーテンソルでbackward()を呼ぶと成功（実装はまだ空）
    let result = x.backward();
    assert!(result.is_ok());
}

#[test]
fn test_backward_requires_grad_false() {
    let device = get_test_device();

    let mut x = Tensor::from_vec_metal(&device, vec![f16::from_f32(3.0)], vec![1]).unwrap();

    // requires_grad = false でbackward()を呼ぶとエラー
    let result = x.backward();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("requires_grad=False"));
}

// 将来の統合テスト用サンプル（現在は実装待ち）
#[test]
#[ignore] // 完全なautograd実装後に有効化
fn test_simple_autodiff() {
    let device = get_test_device();

    // y = x^2 の微分を計算
    // dy/dx = 2x
    let mut x = Tensor::from_vec_metal(&device, vec![f16::from_f32(3.0)], vec![1]).unwrap();
    x.set_requires_grad(true);

    let y = x.mul(&x).unwrap(); // y = x^2 = 9.0

    // backward()で勾配を計算
    let mut y_scalar = y.clone();
    y_scalar.backward().unwrap();

    // x.grad() = dy/dx = 2x = 6.0
    let grad = x.grad().unwrap();
    assert_eq!(grad.to_vec()[0], f16::from_f32(6.0));
}

#[test]
#[ignore] // 完全なautograd実装後に有効化
fn test_chain_rule() {
    let device = get_test_device();

    // z = (x + y)^2
    // dz/dx = 2(x + y)
    // dz/dy = 2(x + y)
    let mut x = Tensor::from_vec_metal(&device, vec![f16::from_f32(2.0)], vec![1]).unwrap();
    let mut y = Tensor::from_vec_metal(&device, vec![f16::from_f32(3.0)], vec![1]).unwrap();

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    let sum = x.add(&y).unwrap(); // sum = 5.0
    let z = sum.mul(&sum).unwrap(); // z = 25.0

    let mut z_scalar = z.clone();
    z_scalar.backward().unwrap();

    // dz/dx = 2(x + y) = 10.0
    assert_eq!(x.grad().unwrap().to_vec()[0], f16::from_f32(10.0));

    // dz/dy = 2(x + y) = 10.0
    assert_eq!(y.grad().unwrap().to_vec()[0], f16::from_f32(10.0));
}
