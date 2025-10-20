/// Learning rate scheduler implementations
///
/// Provides various strategies for adjusting learning rate during training:
/// - StepLR: Decay learning rate by gamma every step_size epochs
/// - ExponentialLR: Decay learning rate by gamma each epoch
/// - CosineAnnealingLR: Cosine annealing schedule
/// - ReduceLROnPlateau: Reduce LR when metric plateaus (future)

/// Learning rate scheduler interface
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Update learning rate (called after each epoch)
    fn step(&mut self);

    /// Reset scheduler to initial state
    fn reset(&mut self);
}

/// Step learning rate scheduler
/// Decays the learning rate by gamma every step_size epochs
///
/// # Example
/// ```
/// let scheduler = StepLR::new(0.1, 10, 0.1);
/// // epoch 0-9: lr = 0.1
/// // epoch 10-19: lr = 0.01
/// // epoch 20-29: lr = 0.001
/// ```
pub struct StepLR {
    initial_lr: f32,
    current_lr: f32,
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
}

impl StepLR {
    /// Create a new StepLR scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `step_size` - Period of learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay (default: 0.1)
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;

        // Decay every step_size epochs
        if self.current_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_epoch = 0;
    }
}

/// Exponential learning rate scheduler
/// Decays the learning rate by gamma each epoch
///
/// # Example
/// ```
/// let scheduler = ExponentialLR::new(0.1, 0.95);
/// // epoch 0: lr = 0.1
/// // epoch 1: lr = 0.095
/// // epoch 2: lr = 0.09025
/// ```
pub struct ExponentialLR {
    initial_lr: f32,
    current_lr: f32,
    gamma: f32,
    current_epoch: usize,
}

impl ExponentialLR {
    /// Create a new ExponentialLR scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;
        self.current_lr *= self.gamma;
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_epoch = 0;
    }
}

/// Cosine annealing learning rate scheduler
/// Anneals the learning rate using a cosine curve
///
/// # Example
/// ```
/// let scheduler = CosineAnnealingLR::new(0.1, 100, 0.001);
/// // Smoothly decreases from 0.1 to 0.001 over 100 epochs
/// ```
pub struct CosineAnnealingLR {
    initial_lr: f32,
    current_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    /// Create a new CosineAnnealingLR scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - Initial learning rate (maximum)
    /// * `t_max` - Maximum number of epochs (half period)
    /// * `eta_min` - Minimum learning rate (default: 0.0)
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            t_max,
            eta_min,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;

        // Cosine annealing formula
        let t = (self.current_epoch % self.t_max) as f32;
        let t_max = self.t_max as f32;
        let cos_inner = std::f32::consts::PI * t / t_max;

        self.current_lr = self.eta_min +
            (self.initial_lr - self.eta_min) * (1.0 + cos_inner.cos()) / 2.0;
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.current_epoch = 0;
    }
}

/// Constant learning rate (no scheduling)
pub struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn step(&mut self) {
        // No change
    }

    fn reset(&mut self) {
        // No change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLR::new(0.1, 2, 0.1);

        // Epoch 0-1: lr = 0.1
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // Epoch 2-3: lr = 0.01
        scheduler.step();
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);

        // Epoch 4-5: lr = 0.001
        scheduler.step();
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let mut scheduler = ExponentialLR::new(0.1, 0.9);

        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.09).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 10, 0.0);

        // Initial LR
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        // After 5 steps (epoch 5), should be at halfway point (cos(pi/2) = 0)
        for _ in 0..5 {
            scheduler.step();
        }
        let mid_lr = scheduler.get_lr();
        assert!((mid_lr - 0.05).abs() < 0.01); // Should be at 50% of range

        // After 9 steps total (epoch 9), should be near minimum (cos(9*pi/10) ≈ -0.95)
        for _ in 0..4 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.01); // Near eta_min (0.0)

        // At epoch 10 (after one more step), cycle restarts
        scheduler.step();
        assert!((scheduler.get_lr() - 0.1).abs() < 0.01); // Back to initial
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = StepLR::new(0.1, 2, 0.5);

        // Step a few times
        scheduler.step();
        scheduler.step();
        scheduler.step();

        // LR should have changed
        assert!((scheduler.get_lr() - 0.1).abs() > 1e-6);

        // Reset
        scheduler.reset();
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_constant_lr() {
        let mut scheduler = ConstantLR::new(0.01);

        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
        scheduler.step();
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-6);
    }
}
