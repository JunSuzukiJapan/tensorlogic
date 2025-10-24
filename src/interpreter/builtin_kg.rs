//! Knowledge Graph embedding builtin functions for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;
use half::f16;

impl Interpreter {
    /// Evaluate Knowledge Graph embedding function
    pub(super) fn eval_kg_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "entity_onehot" => Some(self.eval_entity_onehot(args)),
            "entity_dim" => Some(self.eval_entity_dim(args)),
            "transe_score" => Some(self.eval_transe_score(args)),
            "distmult_score" => Some(self.eval_distmult_score(args)),
            "complex_score" => Some(self.eval_complex_score(args)),
            "margin_ranking_loss" => Some(self.eval_margin_ranking_loss(args)),
            "binary_cross_entropy" => Some(self.eval_binary_cross_entropy(args)),
            "predict_tail_transe" => Some(self.eval_predict_tail_transe(args)),
            "predict_head_transe" => Some(self.eval_predict_head_transe(args)),
            "predict_tail_distmult" => Some(self.eval_predict_tail_distmult(args)),
            "predict_head_distmult" => Some(self.eval_predict_head_distmult(args)),
            "predict_tail_complex" => Some(self.eval_predict_tail_complex(args)),
            "predict_head_complex" => Some(self.eval_predict_head_complex(args)),
            "compute_rank" => Some(self.eval_compute_rank(args)),
            "compute_mrr" => Some(self.eval_compute_mrr(args)),
            "compute_hits_at_k" => Some(self.eval_compute_hits_at_k(args)),
            "compute_mean_rank" => Some(self.eval_compute_mean_rank(args)),
            _ => None,
        }
    }

    fn eval_entity_onehot(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_entity_dim(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_transe_score(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_distmult_score(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_complex_score(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_margin_ranking_loss(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_binary_cross_entropy(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_tail_transe(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_head_transe(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_tail_distmult(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_head_distmult(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_tail_complex(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_predict_head_complex(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_compute_rank(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_compute_mrr(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_compute_hits_at_k(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

    fn eval_compute_mean_rank(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        // TODO: Extract implementation from mod.rs
        Err(RuntimeError::NotImplemented("Extracting...".to_string()))
    }

}
