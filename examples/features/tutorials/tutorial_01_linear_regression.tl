// Tutorial 01: Linear Regression with TensorLogic
//
// This tutorial demonstrates how to implement a simple linear regression
// model using gradient descent to minimize the loss function.
//
// Problem: Learn parameters w and b to minimize loss

// Declare learnable parameters
tensor w: float16[1] learnable = [0.5]
tensor b: float16[1] learnable = [0.5]

main {
    // Training: minimize a simple loss function
    // Loss = w^2 + b^2 (converges to w=0, b=0)

    learn {
        objective: w * w + b * b,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
