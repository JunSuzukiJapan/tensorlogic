// Tutorial 02: Multi-Parameter Optimization with TensorLogic
//
// This tutorial demonstrates training multiple parameters simultaneously
// with different target values using gradient descent.
//
// Problem: Learn two parameters to minimize a combined loss function

// Declare learnable parameters with different initial values
tensor w1: float16[1] learnable = [1.0]
tensor w2: float16[1] learnable = [0.5]

main {
    // Multi-parameter optimization
    // Loss = w1^2 + (w2 - 1)*(w2 - 1) + w2^2
    // This creates different optimization pressures on each parameter

    learn {
        objective: w1 * w1 + w2 * w2,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
