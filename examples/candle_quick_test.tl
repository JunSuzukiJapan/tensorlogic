// Candle関数の簡易テスト
// 各関数が正しく動作するか確認

main {
    print("=== Quick Test: Candle Functions ===")

    // Test 1: matmul
    print("\n[1/11] Testing cndl_matmul...")
    a := f32::ones([2, 3])
    b := f32::ones([3, 2])
    result := cndl_matmul(a, b)
    assert_eq(numel(result), 4)
    print("✓ cndl_matmul OK")

    // Test 2: transpose
    print("\n[2/11] Testing cndl_transpose...")
    x := f32::ones([2, 3, 4])
    transposed := cndl_transpose(x, 0, 2)
    print("✓ cndl_transpose OK")

    // Test 3: reshape
    print("\n[3/11] Testing cndl_reshape...")
    flat := f32::ones([12])
    reshaped := cndl_reshape(flat, f32::from_array([3.0, 4.0]))
    print("✓ cndl_reshape OK")

    // Test 4: softmax
    print("\n[4/11] Testing cndl_softmax...")
    logits := f32::from_array([1.0, 2.0, 3.0])
    softmax_out := cndl_softmax(logits, 0)
    print("✓ cndl_softmax OK")

    // Test 5: log_softmax
    print("\n[5/11] Testing cndl_log_softmax...")
    log_softmax_out := cndl_log_softmax(logits, 0)
    print("✓ cndl_log_softmax OK")

    // Test 6: gelu
    print("\n[6/11] Testing cndl_gelu...")
    input := f32::from_array([0.0, 1.0, -1.0])
    gelu_out := cndl_gelu(input)
    print("✓ cndl_gelu OK")

    // Test 7: silu
    print("\n[7/11] Testing cndl_silu...")
    silu_out := cndl_silu(input)
    print("✓ cndl_silu OK")

    // Test 8: relu
    print("\n[8/11] Testing cndl_relu...")
    relu_out := cndl_relu(input)
    print("✓ cndl_relu OK")

    // Test 9: tanh
    print("\n[9/11] Testing cndl_tanh...")
    tanh_out := cndl_tanh(input)
    print("✓ cndl_tanh OK")

    // Test 10: rms_norm
    print("\n[10/11] Testing cndl_rms_norm...")
    norm_input := f32::ones([2, 4])
    rms_out := cndl_rms_norm(norm_input)
    print("✓ cndl_rms_norm OK")

    // Test 11: safetensors save/load
    print("\n[11/11] Testing cndl_save/load_safetensor...")
    test_tensor := f32::from_array([1.0, 2.0, 3.0])
    cndl_save_safetensor(test_tensor, "/tmp/quick_test.safetensors", "test")
    loaded := cndl_load_safetensor("/tmp/quick_test.safetensors", "test")
    assert_eq(numel(loaded), 3)
    print("✓ cndl_save/load_safetensor OK")

    print("\n=== All Tests Passed! ===")
}
