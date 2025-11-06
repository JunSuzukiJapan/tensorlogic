// Overfitting Demonstration
//
// Compare training loss vs validation loss

main {
    print("=== Overfitting Demo ===")

    // Small training set
    let X_train = [[1.0], [2.0], [3.0]]
    let y_train = [[2.0], [4.0], [6.0]]

    // Validation set
    let X_val = [[1.5], [2.5], [3.5]]
    let y_val = [[3.0], [5.0], [7.0]]

    // Over-parameterized model (more params than data)
    let W1 = learnable(rand([1, 10]))
    let b1 = learnable(zeros([10]))
    let W2 = learnable(rand([10, 1]))
    let b2 = learnable(zeros([1]))

    learn {
        optimizer: adam(lr=0.1)
        epochs: 200

        for epoch in range(epochs) {
            // Training
            let h1_train = relu(matmul(X_train, W1) + b1)
            let y_pred_train = matmul(h1_train, W2) + b2
            let diff_train = y_pred_train - y_train
            let loss_train = mean(diff_train * diff_train)

            backward(loss_train)

            if epoch % 50 == 0 {
                // Validation (no backprop)
                let h1_val = relu(matmul(X_val, W1) + b1)
                let y_pred_val = matmul(h1_val, W2) + b2
                let diff_val = y_pred_val - y_val
                let loss_val = mean(diff_val * diff_val)

                print("Epoch", epoch)
                print("  Train Loss:", loss_train)
                print("  Val Loss:  ", loss_val)

                if loss_val > loss_train * 2.0 {
                    print("  ⚠️ Overfitting detected!")
                }
            }
        }
    }

    print("\n⚠️ Model overfit to training data")
    print("Use regularization or more data!")
}
