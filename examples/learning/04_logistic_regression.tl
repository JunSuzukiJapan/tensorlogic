// Logistic Regression (Binary Classification)
//
// Learn to classify points as class 0 or 1
// Using sigmoid activation and binary cross-entropy loss

main {
    print("=== Logistic Regression ===")

    // Training data: [x1, x2] -> label
    // Class 0: points near origin
    // Class 1: points far from origin
    let X_train = [[0.5, 0.5], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]]
    let y_train = [[0.0], [0.0], [1.0], [1.0]]

    let W = learnable(rand([2, 1]))
    let b = learnable(zeros([1]))

    learn {
        optimizer: adam(lr=0.1)
        epochs: 100

        for epoch in range(epochs) {
            // Forward: logits = X @ W + b
            let logits = matmul(X_train, W) + b

            // Sigmoid activation
            let y_pred = sigmoid(logits)

            // Binary cross-entropy loss
            let epsilon = 1e-7
            let bce = -mean(y_train * log(y_pred + epsilon) +
                            (1.0 - y_train) * log(1.0 - y_pred + epsilon))

            backward(bce)

            if epoch % 20 == 0 {
                print("Epoch", epoch, "BCE Loss:", bce)
                print("  Predictions:", y_pred)
            }
        }
    }

    print("\nFinal predictions:")
    let logits = matmul(X_train, W) + b
    let preds = sigmoid(logits)
    print("  ", preds)
    print("  (should be close to [0, 0, 1, 1])")
    print("✅ Logistic regression complete!")
}
