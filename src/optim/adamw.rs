//! AdamW optimizer (Adam with decoupled weight decay)

use crate::error::TensorResult;
use crate::tensor::Tensor;
use super::optimizer::{Optimizer, ParamGroup, OptimizerState};

/// AdamW optimizer (to be implemented)
pub struct AdamW {
    param_groups: Vec<ParamGroup>,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self {
            param_groups: vec![ParamGroup::new(params, lr)],
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> TensorResult<()> {
        unimplemented!("AdamW optimizer not yet implemented")
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.zero_grad();
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.param_groups.first().map(|g| g.lr).unwrap_or(0.0)
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn state_dict(&self) -> OptimizerState {
        OptimizerState::new()
    }

    fn load_state_dict(&mut self, _state: OptimizerState) -> TensorResult<()> {
        Ok(())
    }

    fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }

    fn num_param_groups(&self) -> usize {
        self.param_groups.len()
    }
}
