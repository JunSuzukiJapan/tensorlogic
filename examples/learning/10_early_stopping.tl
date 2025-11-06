// Early Stopping
//
// Stop training when validation loss stops improving

main {
    print("=== Early Stopping ===")

    let X_train = [[1.0], [2.0], [3.0], [4.0]]
    let y_train = [[3.0], [5.0], [7.0], [9.0]]

    let X_val = [[2.5], [3.5]]
    let y_val = [[6.0], [8.0]]

    let W = learnable(ones([1, 1]))
    let b = learnable(zeros([1]))

    let best_val_loss = 999999.0
    let patience = 10
    let patience_counter = 0

    learn {
        optimizer: sgd(lr=0.01)
        epochs: 1000

        for epoch in range(epochs) {
            // Training step
            let y_pred = matmul(X_train, W) + b
            let diff = y_pred - y_train
            let loss = mean(diff * diff)

            backward(loss)

            // Validation every 5 epochs
            if epoch % 5 == 0 {
                let y_pred_val = matmul(X_val, W) + b
                let diff_val = y_pred_val - y_val
                let val_loss = mean(diff_val * diff_val)

                if val_loss < best_val_loss {
                    best_val_loss := val_loss
                    patience_counter := 0
                    print("Epoch", epoch, "✓ New best val loss:", val_loss)
                } else {
                    patience_counter := patience_counter + 1

                    if patience_counter >= patience {
                        print("Epoch", epoch, "Early stopping triggered")
                        break
                    }
                }
            }
        }
    }

    print("\nTraining stopped early to prevent overfitting")
    print("Best val loss:", best_val_loss)
    print("✅ Early stopping applied!")
}
