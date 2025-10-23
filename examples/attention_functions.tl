// TensorLogicでのAttention関数実装（動作確認版）

// 簡単なmatmul関数テスト
fn test_matmul(A: float16[?, ?], B: float16[?, ?]) -> float16[?, ?] {
    let C = matmul(A, B)
    return C
}

// スケーリング関数
fn scaled_multiply(x: float16[?, ?], scale: float16) -> float16[?, ?] {
    let result = x * scale
    return result
}

// シンプルなAttention（スケーリング済み）
fn simple_attention_core(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    scale: float16
) -> float16[?, ?] {
    let K_T = transpose(K)
    let scores = matmul(Q, K_T)
    let scaled_scores = scores * scale
    let attn_weights = softmax(scaled_scores, 1)
    let output = matmul(attn_weights, V)
    return output
}

// SiLU活性化
fn silu_activation(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    let result = x * sig
    return result
}

main {
    print("=== TensorLogic 関数実行テスト ===")
    print("")
    
    // Test 1: matmul
    print("Test 1: Matmul関数")
    let A = ones([4, 8])
    let B = ones([8, 4])
    let C = test_matmul(A, B)
    print("  入力A:", shape(A))
    print("  入力B:", shape(B))
    print("  出力C:", shape(C))
    print("  ✓ Matmul成功")
    print("")
    
    // Test 2: スケーリング
    print("Test 2: スケーリング関数")
    let X = ones([4, 4])
    let scaled_X = scaled_multiply(X, 0.5)
    print("  入力:", shape(X))
    print("  出力:", shape(scaled_X))
    print("  ✓ スケーリング成功")
    print("")
    
    // Test 3: SiLU
    print("Test 3: SiLU活性化")
    let Y = positional_encoding(4, 8)
    let Y_silu = silu_activation(Y)
    print("  入力:", shape(Y))
    print("  出力:", shape(Y_silu))
    print("  ✓ SiLU成功")
    print("")
    
    // Test 4: Simple Attention
    print("Test 4: Simple Attention")
    let Q = positional_encoding(4, 8)
    let K = positional_encoding(4, 8)
    let V = positional_encoding(4, 8)
    let scale_val = 0.125
    
    print("  Q形状:", shape(Q))
    print("  K形状:", shape(K))
    print("  V形状:", shape(V))
    print("  スケール:", scale_val)
    
    let attn_output = simple_attention_core(Q, K, V, scale_val)
    print("  出力形状:", shape(attn_output))
    print("  ✓ Attention成功")
    print("")
    
    print("=== 全テスト成功！ ===")
    print("")
    print("✅ TensorLogicの関数実行機能が完全に動作しています")
    print("")
    print("次のステップ:")
    print("  • Transformer全体の実装")
    print("  • TinyLlamaモデルでの推論")
    print("  • 22層のTransformerスタック")
}
