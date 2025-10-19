//! Adam optimizer implementation
//!
//! Adam (Adaptive Moment Estimation) uses adaptive learning rates
//! by computing first and second moment estimates.

use crate::autograd::AutogradContext;
use crate::tensor::Tensor;
use crate::{TensorError, TensorResult};
use std::collections::HashMap;

use super::optimizer::{
    mul_scalar, register_param, update_param_from_registry, Optimizer, OptimizerState, ParamGroup,
};

/// Adam optimizer
///
/// Algorithm:
/// ```text
/// m_{t+1} = β₁ m_t + (1-β₁) ∇L(θ_t)           # First moment
/// v_{t+1} = β₂ v_t + (1-β₂) [∇L(θ_t)]²        # Second moment
/// m̂_{t+1} = m_{t+1} / (1 - β₁^{t+1})          # Bias correction
/// v̂_{t+1} = v_{t+1} / (1 - β₂^{t+1})          # Bias correction
/// θ_{t+1} = θ_t - η m̂_{t+1} / (√v̂_{t+1} + ε)
/// ```
pub struct Adam {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,

    /// Coefficients for computing running averages (beta1, beta2)
    betas: (f32, f32),

    /// Term added to denominator for numerical stability
    eps: f32,

    /// Weight decay (L2 penalty)
    weight_decay: f32,

    /// Whether to use AMSGrad variant
    amsgrad: bool,

    /// First moment estimates (m_t)
    exp_avg: HashMap<usize, Tensor>,

    /// Second moment estimates (v_t)
    exp_avg_sq: HashMap<usize, Tensor>,

    /// Maximum second moment for AMSGrad
    max_exp_avg_sq: HashMap<usize, Tensor>,

    /// Step counter
    step_count: usize,

    /// Parameter counter for indexing
    param_counter: usize,
}

impl Adam {
    /// Get mutable access to parameters (for direct manipulation in training loop)
    pub fn get_params_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.param_groups[0].params
    }

    /// Create new Adam optimizer with default parameters
    ///
    /// Default values:
    /// - betas: (0.9, 0.999)
    /// - eps: 1e-8 (adjusted for f16: 1e-3)
    /// - weight_decay: 0.0
    /// - amsgrad: false
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-3, 0.0, false)
    }

    /// Create Adam optimizer with custom options
    pub fn with_options(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            param_groups: vec![ParamGroup::new(params, lr)],
            betas,
            eps,
            weight_decay,
            amsgrad,
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            max_exp_avg_sq: HashMap::new(),
            step_count: 0,
            param_counter: 0,
        }
    }

    /// Create Adam with weight decay
    pub fn with_weight_decay(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-3, weight_decay, false)
    }

    /// Create Adam with AMSGrad variant
    pub fn with_amsgrad(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_options(params, lr, (0.9, 0.999), 1e-3, 0.0, true)
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> TensorResult<()> {
        self.step_count += 1;
        let mut param_idx = 0;

        let beta1 = self.betas.0;
        let beta2 = self.betas.1;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let amsgrad = self.amsgrad;

        // Bias correction terms
        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for group in &mut self.param_groups {
            let lr = group.lr;

            for param in &mut group.params {
                if !param.requires_grad() {
                    continue;
                }

                // Get gradient from registry
                update_param_from_registry(param)?;

                let grad = param.grad().ok_or_else(|| {
                    TensorError::InvalidOperation(
                        "No gradient available for parameter".to_string(),
                    )
                })?;

                // Apply weight decay to gradient if needed
                let mut d_p = grad.clone();
                if weight_decay != 0.0 {
                    let wd = mul_scalar(param, weight_decay)?;
                    d_p = d_p.add(&wd)?;
                }

                // Initialize or update first moment (m_t)
                let m_t = if let Some(m) = self.exp_avg.get_mut(&param_idx) {
                    // m_{t+1} = β₁ m_t + (1-β₁) ∇L(θ_t)
                    let m_scaled = mul_scalar(m, beta1)?;
                    let grad_scaled = mul_scalar(&d_p, 1.0 - beta1)?;
                    let new_m = m_scaled.add(&grad_scaled)?;
                    *m = new_m.clone();
                    new_m
                } else {
                    // First time: initialize with gradient
                    let init_m = mul_scalar(&d_p, 1.0 - beta1)?;
                    self.exp_avg.insert(param_idx, init_m.clone());
                    init_m
                };

                // Initialize or update second moment (v_t)
                let v_t = if let Some(v) = self.exp_avg_sq.get_mut(&param_idx) {
                    // v_{t+1} = β₂ v_t + (1-β₂) [∇L(θ_t)]²
                    let v_scaled = mul_scalar(v, beta2)?;
                    let grad_sq = d_p.mul(&d_p)?; // Element-wise square
                    let grad_sq_scaled = mul_scalar(&grad_sq, 1.0 - beta2)?;
                    let new_v = v_scaled.add(&grad_sq_scaled)?;
                    *v = new_v.clone();
                    new_v
                } else {
                    // First time: initialize with squared gradient
                    let grad_sq = d_p.mul(&d_p)?;
                    let init_v = mul_scalar(&grad_sq, 1.0 - beta2)?;
                    self.exp_avg_sq.insert(param_idx, init_v.clone());
                    init_v
                };

                // Compute denominator
                let denom = if amsgrad {
                    // AMSGrad: max(v_t, v_t_max)
                    let v_t_max = if let Some(max_v) = self.max_exp_avg_sq.get_mut(&param_idx) {
                        // Element-wise max
                        let v_data = v_t.to_vec();
                        let max_v_data = max_v.to_vec();
                        let new_max_data: Vec<_> = v_data
                            .iter()
                            .zip(max_v_data.iter())
                            .map(|(a, b)| if a.to_f32() > b.to_f32() { *a } else { *b })
                            .collect();
                        let new_max = Tensor::from_vec(new_max_data, v_t.dims().to_vec())?;
                        *max_v = new_max.clone();
                        new_max
                    } else {
                        self.max_exp_avg_sq.insert(param_idx, v_t.clone());
                        v_t.clone()
                    };

                    // sqrt(v_t_max) + eps
                    let sqrt_v = sqrt_tensor(&v_t_max)?;
                    add_scalar(&sqrt_v, eps)?
                } else {
                    // Standard Adam: sqrt(v_t) + eps
                    let sqrt_v = sqrt_tensor(&v_t)?;
                    add_scalar(&sqrt_v, eps)?
                };

                // Bias correction
                // m̂_{t+1} = m_{t+1} / (1 - β₁^{t+1})
                let m_hat = mul_scalar(&m_t, 1.0 / bias_correction1)?;

                // v̂_{t+1} = v_{t+1} / (1 - β₂^{t+1})
                let denom_corrected = mul_scalar(&denom, bias_correction2)?;

                // Update: θ_{t+1} = θ_t - η m̂_{t+1} / denom
                let update = div_tensors(&m_hat, &denom_corrected)?;
                let scaled_update = mul_scalar(&update, lr)?;
                param.sub_(&scaled_update)?;

                // Re-register parameter in AutogradContext
                register_param(param)?;

                param_idx += 1;
            }
        }

        self.param_counter = param_idx;
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.zero_grad();
                if let Some(node_id) = param.grad_node() {
                    AutogradContext::register_tensor(node_id, param.clone());
                }
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.param_groups[0].lr
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn state_dict(&self) -> OptimizerState {
        OptimizerState {
            step: self.step_count,
            param_state: HashMap::new(), // TODO: Serialize exp_avg, exp_avg_sq
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> TensorResult<()> {
        self.step_count = state.step;
        Ok(())
    }

    fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }

    fn num_param_groups(&self) -> usize {
        self.param_groups.len()
    }
}

/// Helper function to compute element-wise square root
fn sqrt_tensor(tensor: &Tensor) -> TensorResult<Tensor> {
    let data = tensor.to_vec();
    let sqrt_data: Vec<_> = data
        .iter()
        .map(|x| half::f16::from_f32(x.to_f32().sqrt()))
        .collect();
    Tensor::from_vec(sqrt_data, tensor.dims().to_vec())
}

/// Helper function to add scalar to tensor
fn add_scalar(tensor: &Tensor, scalar: f32) -> TensorResult<Tensor> {
    let data = tensor.to_vec();
    let result_data: Vec<_> = data
        .iter()
        .map(|x| half::f16::from_f32(x.to_f32() + scalar))
        .collect();
    Tensor::from_vec(result_data, tensor.dims().to_vec())
}

/// Helper function to divide tensors element-wise
fn div_tensors(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    a.div(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_creation() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let adam = Adam::new(vec![x], 0.001);
        assert_eq!(adam.get_lr(), 0.001);
        assert_eq!(adam.betas, (0.9, 0.999));
        assert_eq!(adam.eps, 1e-3);
    }

    #[test]
    fn test_adam_with_weight_decay() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let adam = Adam::with_weight_decay(vec![x], 0.001, 0.01);
        assert_eq!(adam.weight_decay, 0.01);
    }

    #[test]
    fn test_adam_with_amsgrad() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let adam = Adam::with_amsgrad(vec![x], 0.001);
        assert!(adam.amsgrad);
    }

    #[test]
    fn test_adam_zero_grad() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(2.0)], vec![1]).unwrap();
        let mut adam = Adam::new(vec![x.clone()], 0.01);

        // Zero gradients
        adam.zero_grad();

        // Verify optimizer state is initialized correctly
        assert_eq!(adam.step_count, 0);
        assert_eq!(adam.exp_avg.len(), 0);
    }

    #[test]
    fn test_adam_lr_management() {
        let x = Tensor::from_vec(vec![half::f16::from_f32(1.0)], vec![1]).unwrap();
        let mut adam = Adam::new(vec![x], 0.001);
        assert_eq!(adam.get_lr(), 0.001);

        adam.set_lr(0.0001);
        assert_eq!(adam.get_lr(), 0.0001);
    }
}
