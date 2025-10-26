mod context;
mod graph;
mod gradient;
pub mod gradients;
mod node;
mod fusion;
mod gradcheck;
mod tensor_variant;

pub use context::AutogradContext;
pub use tensor_variant::TensorVariant;
pub use graph::ComputationGraph;
pub use gradient::GradientFunction;
pub use node::{GradNode, NodeId, Operation};
pub use fusion::{
    FusionOptimizer, FusionPattern, FusionOpportunity, FusionConfig,
    FusionStats, FusionStatsSummary, BinaryOp, ScalarOp,
};
pub use gradcheck::{GradientChecker, GradCheckConfig, GradCheckResult};
