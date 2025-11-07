// Learning Rate Scheduling
//
// Decrease learning rate over time for better convergence

main {
    print("=== Learning Rate Schedule ===")

    let X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    let y = [[3.0], [5.0], [7.0], [9.0], [11.0]]

    let W = learnable(rand([1, 1]))
    let b = learnable(zeros([1]))

    let initial_lr = 0.1

    learn {
        optimizer: sgd(lr=initial_lr)
        epochs: 100

        for epoch in range(epochs) {
            // Decay learning rate every 25 epochs
            if epoch % 25 == 0 && epoch > 0 {
                let new_lr = initial_lr / (1.0 + epoch / 25.0)
                set_lr(optimizer, new_lr)
                print("Epoch", epoch, "New LR:", new_lr)
            }

            let y_pred = matmul(X, W) + b
            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 25 == 0 {
                print("  Loss:", loss)
            }
        }
    }

    print("\nLR scheduling improves convergence")
    print("Final W:", W)
    print("Final b:", b)
    print("✅ LR schedule applied!")
}
