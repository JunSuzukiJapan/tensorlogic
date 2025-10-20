// Tutorial 03: Simple Neural Network Weights with TensorLogic
//
// This tutorial demonstrates learning multiple weights for a neural network layer.
// We optimize 3 weights simultaneously.
//
// Problem: Minimize squared magnitude of 3 network weights

// Three learnable weights for a simple network
tensor w1: float32[1] learnable = [0.5]
tensor w2: float32[1] learnable = [1.0]
tensor w3: float32[1] learnable = [0.3]

main {
    // Neural network weight regularization: minimize sum of squared weights
    // Loss = w1^2 + w2^2 + w3^2
    // This encourages small weight values (L2 regularization)

    learn {
        objective: w1 * w1 + w2 * w2 + w3 * w3,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
