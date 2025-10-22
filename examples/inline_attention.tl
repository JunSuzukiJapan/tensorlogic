// ============================================
// インラインAttention実装（関数なし）
// ============================================

main {
    print("=== インラインTransformer実装 ===")
    print("")
    
    // パラメータ設定
    let seq_len = 4
    let d_model = 8
    let d_ff = 16
    
    print("設定:")
    print("  seq_len:", seq_len)
    print("  d_model:", d_model)
    print("  d_ff:", d_ff)
    print("")
    
    // 入力作成
    print("Step 1: 入力作成")
    let x = positional_encoding(seq_len, d_model)
    print("  入力形状:", shape(x))
    print("")
    
    // 重み作成（実際にはモデルから読み込む）
    print("Step 2: 重み初期化")
    let attn_norm = ones([d_model])
    let W_q = positional_encoding(d_model, d_model)
    let W_k = positional_encoding(d_model, d_model)
    let W_v = positional_encoding(d_model, d_model)
    let W_o = positional_encoding(d_model, d_model)
    let ffn_norm = ones([d_model])
    let W_gate = positional_encoding(d_model, d_ff)
    let W_up = positional_encoding(d_model, d_ff)
    let W_down = positional_encoding(d_ff, d_model)
    print("  ✓ 重み準備完了")
    print("")
    
    // Attention計算
    print("Step 3: Self-Attention")
    let x_norm = rms_norm(x, attn_norm)
    let Q = matmul(x_norm, W_q)
    let K = matmul(x_norm, W_k)
    let V = matmul(x_norm, W_v)
    print("  Q形状:", shape(Q))
    print("  K形状:", shape(K))
    print("  V形状:", shape(V))
    
    // Attention scores
    let K_T = transpose(K)
    let scores = matmul(Q, K_T)
    print("  Scores形状:", shape(scores))
    
    // Scale
    let scale = 0.125
    let scaled_scores = scores * scale
    
    // Softmax
    let attn_weights = softmax(scaled_scores, -1)
    print("  Attention weights:", shape(attn_weights))
    
    // Value加重和
    let attn_out = matmul(attn_weights, V)
    let attn_proj = matmul(attn_out, W_o)
    print("  Attention出力:", shape(attn_proj))
    
    // 残差接続
    let x_after_attn = x + attn_proj
    print("  ✓ Attention完了")
    print("")
    
    // Feed-Forward Network (SwiGLU)
    print("Step 4: SwiGLU FFN")
    let x_norm2 = rms_norm(x_after_attn, ffn_norm)
    
    // Gate分岐
    let gate = matmul(x_norm2, W_gate)
    let gate_sig = sigmoid(gate)
    let gate_silu = gate * gate_sig
    print("  Gate形状:", shape(gate_silu))
    
    // Up分岐
    let up = matmul(x_norm2, W_up)
    print("  Up形状:", shape(up))
    
    // 要素積
    let intermediate = gate_silu * up
    print("  Intermediate形状:", shape(intermediate))
    
    // Down投影
    let ffn_out = matmul(intermediate, W_down)
    print("  FFN出力:", shape(ffn_out))
    
    // 残差接続
    let final_output = x_after_attn + ffn_out
    print("  ✓ FFN完了")
    print("")
    
    print("Step 5: 最終出力")
    print("  出力形状:", shape(final_output))
    print("  期待値: [4.0, 8.0]")
    print("")
    
    print("=== サマリー ===")
    print("✅ 実装した処理:")
    print("  1. RMSNorm (Pre-Attention)")
    print("  2. Q/K/V投影")
    print("  3. Scaled Dot-Product Attention")
    print("  4. 出力投影 + 残差接続")
    print("  5. RMSNorm (Pre-FFN)")
    print("  6. SwiGLU FFN (Gate + Up + Down)")
    print("  7. 残差接続")
    print("")
    print("🎯 これは完全なTransformerレイヤーです！")
    print("")
    print("次のステップ:")
    print("  • 実際のTinyLlamaモデル重みで実行")
    print("  • マルチヘッド/GQAの実装")
    print("  • 22層のスタッキング")
    print("")
    print("✅ TensorLogicでTransformer実装成功！")
}
