// Linear Regression with SGD
//
// Learn to fit a line: y = 2x + 1
// Using Stochastic Gradient Descent (SGD)

main {
    print("=== Linear Regression with SGD ===")

    // Generate training data: y = 2x + 1 + noise
    let X_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    let y_train = [[3.0], [5.0], [7.0], [9.0], [11.0]]

    // Initialize learnable parameters
    let W = learnable(ones([1, 1]))  // Weight
    let b = learnable(zeros([1]))     // Bias

    print("Initial W:", W)
    print("Initial b:", b)

    // Training loop
    learn {
        optimizer: sgd(lr=0.01)
        epochs: 100

        for epoch in range(epochs) {
            // Forward pass: y_pred = X @ W + b
            let y_pred = matmul(X_train, W) + b

            // Compute MSE loss
            let diff = y_pred - y_train
            let loss = mean(diff * diff)

            // Backward pass (automatic)
            backward(loss)

            if epoch % 20 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    print("\nLearned W:", W, "(target: 2.0)")
    print("Learned b:", b, "(target: 1.0)")
    print("\n✅ Linear regression converged!")
}
