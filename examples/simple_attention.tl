// ============================================
// シンプルなAttention実装（TensorLogic）
// ============================================

// SiLU活性化関数 (SwiGLU用)
// SiLU(x) = x * sigmoid(x)
function silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    let result = x * sig
    result := result
}

// スケーリングされたドット積Attention (簡易版)
function simple_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?]
) -> float16[?, ?] {
    let K_T = transpose(K)
    let scores = matmul(Q, K_T)
    let scale = 0.125
    let scaled_scores = scores * scale
    let attn_weights = softmax(scaled_scores, -1)
    let output = matmul(attn_weights, V)
    output := output
}

// SwiGLU Feed-Forward Network
function simple_swiglu(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let gate_act = silu(gate)
    let up = matmul(x, W_up)
    let intermediate = gate_act * up
    let output = matmul(intermediate, W_down)
    output := output
}

// シンプルなTransformerレイヤー
function simple_transformer_layer(
    x: float16[?, ?],
    attn_norm_weight: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    ffn_norm_weight: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let x_norm = rms_norm(x, attn_norm_weight)
    let Q = matmul(x_norm, W_q)
    let K = matmul(x_norm, W_k)
    let V = matmul(x_norm, W_v)
    let attn_out = simple_attention(Q, K, V)
    let attn_proj = matmul(attn_out, W_o)
    let x_after_attn = x + attn_proj
    let x_norm2 = rms_norm(x_after_attn, ffn_norm_weight)
    let ffn_out = simple_swiglu(x_norm2, W_gate, W_up, W_down)
    let final_output = x_after_attn + ffn_out
    final_output := final_output
}

main {
    print("=== TensorLogic Transformer実装 ===")
    print("")
    
    print("✅ 定義した関数:")
    print("  • silu(x) - SiLU活性化関数")
    print("  • simple_attention(Q, K, V)")
    print("  • simple_swiglu(x, W_gate, W_up, W_down)")
    print("  • simple_transformer_layer(...)")
    print("")
    
    print("🔧 テスト実行:")
    let seq_len = 4
    let d_model = 8
    let d_ff = 16
    
    print("  シーケンス長:", seq_len)
    print("  モデル次元:", d_model)
    print("")
    
    let x = positional_encoding(seq_len, d_model)
    print("  入力:", shape(x))
    
    let attn_norm = ones([d_model])
    let W_q = positional_encoding(d_model, d_model)
    let W_k = positional_encoding(d_model, d_model)
    let W_v = positional_encoding(d_model, d_model)
    let W_o = positional_encoding(d_model, d_model)
    let ffn_norm = ones([d_model])
    let W_gate = positional_encoding(d_model, d_ff)
    let W_up = positional_encoding(d_model, d_ff)
    let W_down = positional_encoding(d_ff, d_model)
    
    print("  Transformerレイヤー実行...")
    let output = simple_transformer_layer(
        x,
        attn_norm,
        W_q, W_k, W_v, W_o,
        ffn_norm,
        W_gate, W_up, W_down
    )
    
    print("  ✓ 完了!")
    print("  出力:", shape(output))
    print("")
    print("✅ 基本Transformer実装成功!")
}
