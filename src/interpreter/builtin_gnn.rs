//! Graph Neural Network operations for TensorLogic interpreter

use super::*;

impl Interpreter {
    pub(super) fn eval_gnn_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "aggregate_neighbors" | "relational_aggregate" | "graph_attention" | "normalize_features" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("GNN function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }
}
