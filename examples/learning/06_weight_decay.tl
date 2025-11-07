// Weight Decay (L2 Regularization)
//
// Prevents overfitting by penalizing large weights

main {
    print("=== Weight Decay Example ===")

    let X = [[1.0], [2.0], [3.0], [4.0]]
    let y = [[2.0], [4.0], [6.0], [8.0]]

    let W = learnable(rand([1, 1]))
    let b = learnable(zeros([1]))

    print("Training with weight decay...")

    learn {
        optimizer: adamw(lr=0.1, weight_decay=0.01)
        epochs: 100

        for epoch in range(epochs) {
            let y_pred = matmul(X, W) + b
            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 25 == 0 {
                print("Epoch", epoch, "Loss:", loss, "W:", W)
            }
        }
    }

    print("\nWeight decay keeps W from growing too large")
    print("Final W:", W)
    print("Final b:", b)
    print("✅ Weight decay applied!")
}
