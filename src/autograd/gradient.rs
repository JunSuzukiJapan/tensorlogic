use crate::error::TensorResult;
use crate::tensor::Tensor;

/// 勾配計算関数トレイト
pub trait GradientFunction: Send + Sync {
    /// 逆伝播を実行し、各入力に対する勾配を計算
    ///
    /// # Arguments
    /// * `grad_output` - この演算の出力に対する勾配
    /// * `inputs` - forward時の入力テンソル（勾配計算に必要な場合）
    ///
    /// # Returns
    /// 各入力テンソルに対する勾配のベクトル
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> TensorResult<Vec<Tensor>>;
}
