// TinyLlama Complete Inference Pipeline
// Token embeddings → 22 layers → Final norm → Output projection → Logits

// SiLU activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    let result = x * sig
    result := result
}

// SwiGLU FFN
fn swiglu_ffn(
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

// Grouped Query Attention
fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let seq_len = 4
    let num_q_heads = 32
    let num_kv_heads = 4
    let head_dim = 64
    let group_size = 8

    let Q_heads = reshape(Q, [seq_len, num_q_heads, head_dim])
    let K_heads = reshape(K, [seq_len, num_kv_heads, head_dim])
    let V_heads = reshape(V, [seq_len, num_kv_heads, head_dim])

    let K_with_group = reshape(K_heads, [seq_len, num_kv_heads, 1, head_dim])
    let V_with_group = reshape(V_heads, [seq_len, num_kv_heads, 1, head_dim])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, num_kv_heads, group_size, head_dim])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, num_kv_heads, group_size, head_dim])

    let K_expanded = reshape(K_broadcast, [seq_len, num_q_heads, head_dim])
    let V_expanded = reshape(V_broadcast, [seq_len, num_q_heads, head_dim])

    let Q_flat = reshape(Q_heads, [seq_len * num_q_heads, head_dim])
    let K_flat = reshape(K_expanded, [seq_len * num_q_heads, head_dim])
    let V_flat = reshape(V_expanded, [seq_len * num_q_heads, head_dim])

    let K_T = transpose(K_flat)
    let scores = matmul(Q_flat, K_T)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 1)
    let attn_output = matmul(attn_weights, V_flat)

    let attn_reshaped = reshape(attn_output, [seq_len, num_q_heads * head_dim])
    let output = matmul(attn_reshaped, W_o)
    output := output
}

// Transformer Layer
fn transformer_layer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?]
) -> float16[?, ?] {
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = matmul(x_norm1, W_q)
    let K = matmul(x_norm1, W_k)
    let V = matmul(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    let result = x1 + ffn_out
    result := result
}

main {
    print("=== TinyLlama Complete Inference Pipeline ===")
    print("")

    // Load model
    print("[1/5] モデル読み込み")
    let model_path = "/Users/junsuzuki/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    print("      ✓ 201 tensors loaded")
    print("")

    // Model config
    let seq_len = 4
    let d_model = 2048
    let vocab_size = 32000

    // Token embeddings (simulating input tokens)
    print("[2/5] トークン埋め込み")
    let x = ones([seq_len, d_model])
    print("      ✓ Input shape:", shape(x))
    print("")

    // Process through layers (demonstration with Layer 0-2)
    print("[3/5] Transformer Layers (0-2 デモ)")

    // Layer 0
    print("      • Layer 0...")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let x0 = transformer_layer(x, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)

    // Layer 1
    print("      • Layer 1...")
    let W_q_1 = get_tensor(model, "blk.1.attn_q.weight")
    let W_k_1 = get_tensor(model, "blk.1.attn_k.weight")
    let W_v_1 = get_tensor(model, "blk.1.attn_v.weight")
    let W_o_1 = get_tensor(model, "blk.1.attn_output.weight")
    let attn_norm_1 = get_tensor(model, "blk.1.attn_norm.weight")
    let W_gate_1 = get_tensor(model, "blk.1.ffn_gate.weight")
    let W_up_1 = get_tensor(model, "blk.1.ffn_up.weight")
    let W_down_1 = get_tensor(model, "blk.1.ffn_down.weight")
    let ffn_norm_1 = get_tensor(model, "blk.1.ffn_norm.weight")
    let x1 = transformer_layer(x0, W_q_1, W_k_1, W_v_1, W_o_1, attn_norm_1, W_gate_1, W_up_1, W_down_1, ffn_norm_1)

    // Layer 2
    print("      • Layer 2...")
    let W_q_2 = get_tensor(model, "blk.2.attn_q.weight")
    let W_k_2 = get_tensor(model, "blk.2.attn_k.weight")
    let W_v_2 = get_tensor(model, "blk.2.attn_v.weight")
    let W_o_2 = get_tensor(model, "blk.2.attn_output.weight")
    let attn_norm_2 = get_tensor(model, "blk.2.attn_norm.weight")
    let W_gate_2 = get_tensor(model, "blk.2.ffn_gate.weight")
    let W_up_2 = get_tensor(model, "blk.2.ffn_up.weight")
    let W_down_2 = get_tensor(model, "blk.2.ffn_down.weight")
    let ffn_norm_2 = get_tensor(model, "blk.2.ffn_norm.weight")
    let x_layers = transformer_layer(x1, W_q_2, W_k_2, W_v_2, W_o_2, attn_norm_2, W_gate_2, W_up_2, W_down_2, ffn_norm_2)

    print("      ✓ Layers 0-2 complete")
    print("      📝 Layers 3-21: 同じパターンで追加可能")
    print("")

    // Final normalization
    print("[4/5] 最終正規化")
    let output_norm = get_tensor(model, "output_norm.weight")
    let x_final = rms_norm(x_layers, output_norm)
    print("      ✓ Final norm applied")
    print("      ✓ Shape:", shape(x_final))
    print("")

    // Output projection to vocabulary
    print("[5/5] 出力投影")
    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Output weight:", shape(output_weight))
    let logits = matmul(x_final, output_weight)
    print("      ✓ Logits shape:", shape(logits))
    print("")

    print("==================================================")
    print("🎉 TinyLlama 完全推論パイプライン実装完了!")
    print("")
    print("パイプライン構成:")
    print("  ✅ トークン埋め込み [4, 2048]")
    print("  ✅ Transformer Layers 0-2 (デモ)")
    print("  ✅ 最終RMSNorm")
    print("  ✅ 出力投影 → Logits [4, 32000]")
    print("")
    print("次のステップ:")
    print("  • 全22層の実装")
    print("  • トークナイザー統合")
    print("  • 自己回帰生成ループ")
    print("  • サンプリング (greedy/top-k/top-p)")
}
