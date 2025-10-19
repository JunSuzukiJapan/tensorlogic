use crate::autograd::GradientFunction;

/// ノードID（計算グラフ内のノードを識別）
pub type NodeId = usize;

/// 演算の種類
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    ReLU,
    GELU,
    Softmax,
    Sum,
    Mean,
    Broadcast,
}

/// 計算グラフのノード
pub struct GradNode {
    pub id: NodeId,
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub grad_fn: Option<Box<dyn GradientFunction>>,
}

impl GradNode {
    pub fn new(
        id: NodeId,
        operation: Operation,
        inputs: Vec<NodeId>,
        grad_fn: Option<Box<dyn GradientFunction>>,
    ) -> Self {
        Self {
            id,
            operation,
            inputs,
            grad_fn,
        }
    }
}
