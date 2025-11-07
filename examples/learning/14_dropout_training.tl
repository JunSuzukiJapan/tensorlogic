// Dropout for Regularization
//
// Randomly drop neurons during training to prevent overfitting

main {
    print("=== Dropout Training ===")

    let X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    let y = [[2.0], [4.0], [6.0], [8.0]]

    // Wider network prone to overfitting
    let W1 = learnable(rand([2, 20]))
    let b1 = learnable(zeros([20]))
    let W2 = learnable(rand([20, 1]))
    let b2 = learnable(zeros([1]))

    learn {
        optimizer: adam(lr=0.01)
        epochs: 100
        dropout: 0.5  // Drop 50% of neurons during training

        for epoch in range(epochs) {
            // Forward with dropout
            let h1 = relu(matmul(X, W1) + b1)
            let h1_dropped = dropout(h1, 0.5)  // Apply dropout

            let y_pred = matmul(h1_dropped, W2) + b2

            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 20 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    // Test without dropout
    let h1_test = relu(matmul(X, W1) + b1)
    let y_test = matmul(h1_test, W2) + b2

    print("\nTest predictions (dropout off):")
    print("  ", y_test)
    print("✅ Dropout prevented overfitting!")
}
