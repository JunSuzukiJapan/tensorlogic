use crate::error::TensorResult;
use crate::tensor::{FloatType, Tensor};
use crate::autograd::TensorVariant;
use half::f16;

/// 勾配計算関数トレイト (f16専用、後方互換性のため残す)
pub trait GradientFunction: Send + Sync {
    /// 逆伝播を実行し、各入力に対する勾配を計算
    ///
    /// # Arguments
    /// * `grad_output` - この演算の出力に対する勾配
    /// * `inputs` - forward時の入力テンソル（勾配計算に必要な場合）
    ///
    /// # Returns
    /// 各入力テンソルに対する勾配のベクトル
    fn backward(&self, grad_output: &Tensor<f16>, inputs: &[&Tensor<f16>]) -> TensorResult<Vec<Tensor<f16>>>;
}

/// 勾配計算関数トレイト (TensorVariant版)
pub trait GradientFunctionVariant: Send + Sync {
    /// 逆伝播を実行し、各入力に対する勾配を計算
    ///
    /// # Arguments
    /// * `grad_output` - この演算の出力に対する勾配
    /// * `inputs` - forward時の入力テンソル（勾配計算に必要な場合）
    ///
    /// # Returns
    /// 各入力テンソルに対する勾配のベクトル
    fn backward_variant(&self, grad_output: &TensorVariant, inputs: &[&TensorVariant]) -> TensorResult<Vec<TensorVariant>>;
}

/// 勾配計算関数トレイト (ジェネリック版)
pub trait GradientFunctionGeneric<T: FloatType>: Send + Sync {
    /// 逆伝播を実行し、各入力に対する勾配を計算
    ///
    /// # Arguments
    /// * `grad_output` - この演算の出力に対する勾配
    /// * `inputs` - forward時の入力テンソル（勾配計算に必要な場合）
    ///
    /// # Returns
    /// 各入力テンソルに対する勾配のベクトル
    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> TensorResult<Vec<Tensor<T>>>;
}
