// Convergence Monitoring
//
// Track loss history and detect convergence

main {
    print("=== Convergence Monitoring ===")

    let X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    let y = [[2.0], [4.0], [6.0], [8.0], [10.0]]

    let W = learnable(rand([1, 1]))
    let b = learnable(zeros([1]))

    let loss_history = []
    let convergence_threshold = 0.01

    learn {
        optimizer: adam(lr=0.1)
        epochs: 100

        for epoch in range(epochs) {
            let y_pred = matmul(X, W) + b
            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            // Store loss
            loss_history.append(loss)

            if epoch % 10 == 0 {
                print("Epoch", epoch, "Loss:", loss)

                // Check convergence (loss change < threshold)
                if epoch > 10 {
                    let prev_loss = loss_history[epoch - 10]
                    let loss_change = abs(loss - prev_loss)

                    if loss_change < convergence_threshold {
                        print("✓ Converged at epoch", epoch)
                        break
                    }
                }
            }
        }
    }

    print("\nFinal W:", W, "(target: 2.0)")
    print("Final b:", b, "(target: 0.0)")
    print("Loss decreased from", loss_history[0], "to", loss_history[-1])
    print("✅ Convergence monitored!")
}
