// Candle関数のデモプログラム
// すべてのcndl_関数を実際に呼び出して動作を確認

main {
    print("=== Candle Functions Demo ===")
    print("")

    // ============================================================================
    // 1. テンソル操作
    // ============================================================================
    print("1. Tensor Operations")
    print("--------------------")

    // Matrix multiplication
    let a = f32::ones([2, 3])
    let b = f32::ones([3, 4])
    let matmul_result = cndl_matmul(a, b)
    print("cndl_matmul([2,3] @ [3,4]):", shape(matmul_result))

    // Transpose
    let x = f32::ones([2, 3, 4])
    let transposed = cndl_transpose(x, 0, 2)
    print("cndl_transpose([2,3,4], dim0=0, dim1=2):", shape(transposed))

    // Reshape
    let flat = f32::ones([12])
    let reshaped = cndl_reshape(flat, f32::from_array([3.0, 4.0]))
    print("cndl_reshape([12] -> [3,4]):", shape(reshaped))
    print("")

    // ============================================================================
    // 2. 数学操作
    // ============================================================================
    print("2. Math Operations")
    print("------------------")

    // Softmax
    let logits = f32::from_array([1.0, 2.0, 3.0, 4.0])
    let softmax_result = cndl_softmax(logits, 0)
    print("cndl_softmax([1,2,3,4]):", softmax_result)

    // Log Softmax
    let log_softmax_result = cndl_log_softmax(logits, 0)
    print("cndl_log_softmax([1,2,3,4]):", log_softmax_result)
    print("")

    // ============================================================================
    // 3. アクティベーション関数
    // ============================================================================
    print("3. Activation Functions")
    print("-----------------------")

    let input = f32::from_array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Input:", input)

    // GELU
    let gelu_result = cndl_gelu(input)
    print("cndl_gelu:", gelu_result)

    // SiLU (Swish)
    let silu_result = cndl_silu(input)
    print("cndl_silu:", silu_result)

    // ReLU
    let relu_result = cndl_relu(input)
    print("cndl_relu:", relu_result)

    // Tanh
    let tanh_result = cndl_tanh(input)
    print("cndl_tanh:", tanh_result)
    print("")

    // ============================================================================
    // 4. 正規化
    // ============================================================================
    print("4. Normalization")
    print("----------------")

    // RMS Norm
    let norm_input = f32::ones([2, 4])
    let rms_result = cndl_rms_norm(norm_input)
    print("cndl_rms_norm(ones[2,4]):", rms_result)

    // Layer Norm
    let layer_norm_result = cndl_layer_norm(norm_input, 4)
    print("cndl_layer_norm(ones[2,4]):", layer_norm_result)
    print("")

    // ============================================================================
    // 5. モデルの保存と読み込み（Safetensors）
    // ============================================================================
    print("5. Model Save/Load (Safetensors)")
    print("---------------------------------")

    // テンソルを作成
    let weights = f32::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print("Original tensor:", weights)

    // Safetensors形式で保存
    cndl_save_safetensor(weights, "/tmp/demo_weights.safetensors", "my_weights")

    // Safetensors形式で読み込み
    let loaded_weights = cndl_load_safetensor("/tmp/demo_weights.safetensors", "my_weights")
    print("Loaded tensor:", loaded_weights)

    // ファイルの内容を一覧表示
    cndl_list_safetensors("/tmp/demo_weights.safetensors")
    print("")

    // ============================================================================
    // 6. 複数テンソルの保存
    // ============================================================================
    print("6. Multiple Tensors")
    print("-------------------")

    // 複数のテンソルを別々に保存
    let tensor1 = f32::from_array([1.0, 2.0])
    let tensor2 = f32::from_array([3.0, 4.0, 5.0])

    cndl_save_safetensor(tensor1, "/tmp/multi_tensors.safetensors", "tensor1")
    // Note: 同じファイルに複数保存する場合は、既存ファイルを読み込んでマージする必要があります
    // 現在の実装では上書きされます

    print("")
    print("=== Demo Complete ===")
}
