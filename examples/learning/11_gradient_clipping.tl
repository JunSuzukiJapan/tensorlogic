// Gradient Clipping
//
// Prevent exploding gradients by clipping to maximum norm

main {
    print("=== Gradient Clipping ===")

    let X = [[1.0], [2.0], [3.0], [4.0]]
    let y = [[10.0], [20.0], [30.0], [40.0]]

    let W = learnable(rand([1, 1]) * 0.01)
    let b = learnable(zeros([1]))

    learn {
        optimizer: sgd(lr=1.0)  // High LR could cause exploding gradients
        epochs: 50
        clip_grad_norm: 1.0  // Clip gradients to max norm of 1.0

        for epoch in range(epochs) {
            let y_pred = matmul(X, W) + b
            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 10 == 0 {
                print("Epoch", epoch, "Loss:", loss)
                print("  W:", W)
            }
        }
    }

    print("\nGradient clipping prevented instability")
    print("Final W:", W)
    print("✅ Gradients clipped successfully!")
}
