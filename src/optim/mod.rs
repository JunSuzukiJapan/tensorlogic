//! Optimizers for training neural networks
//!
//! This module provides various optimization algorithms for updating model parameters
//! during training, including SGD, Momentum, Adam, and AdamW.
//!
//! Also includes learning rate schedulers for adjusting learning rates during training.

mod optimizer;
mod sgd;
mod adam;
mod adamw;
pub mod scheduler;

pub use optimizer::{Optimizer, ParamGroup, OptimizerState};
pub use sgd::SGD;
pub use adam::Adam;
pub use adamw::AdamW;
pub use scheduler::{LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR, ConstantLR};
