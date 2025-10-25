// Debug: Simplified attention check (avoiding 3D tensor slice)

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    result := linear(intermediate, W_down)
}

main {
    print("=== Simplified Attention NaN Debug ===")
    print("")

    let home = env("HOME")
    let model_path = home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    print("Loading layer 0 weights...")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let embed_table = get_tensor(model, "token_embd.weight")
    print("   ✓ Loaded")
    print("")

    print("[Test 1: Single token (should work)]")
    let tokens1 = [1.0]
    let e1 = embedding(embed_table, tokens1)
    print("   Embedding:", shape(e1))

    let x_norm1 = rms_norm(e1, attn_norm_0)
    print("   After norm:", shape(x_norm1))

    let Q1 = linear(x_norm1, W_q_0)
    let K1 = linear(x_norm1, W_k_0)
    let V1 = linear(x_norm1, W_v_0)
    print("   Q, K, V created")

    let seq_len1 = 1.0
    let Q_heads1 = reshape(Q1, [seq_len1, 32.0, 64.0])
    let K_heads1 = reshape(K1, [seq_len1, 4.0, 64.0])
    let V_heads1 = reshape(V1, [seq_len1, 4.0, 64.0])
    print("   Reshaped to heads")

    let K_with_group1 = reshape(K_heads1, [seq_len1, 4.0, 1.0, 64.0])
    let V_with_group1 = reshape(V_heads1, [seq_len1, 4.0, 1.0, 64.0])
    let K_broadcast1 = broadcast_to(K_with_group1, [seq_len1, 4.0, 8.0, 64.0])
    let V_broadcast1 = broadcast_to(V_with_group1, [seq_len1, 4.0, 8.0, 64.0])
    let K_expanded1 = reshape(K_broadcast1, [seq_len1, 32.0, 64.0])
    let V_expanded1 = reshape(V_broadcast1, [seq_len1, 32.0, 64.0])
    print("   Broadcasted for GQA")

    let scores1 = einsum("ihd,jhd->ihj", Q_heads1, K_expanded1)
    print("   Scores shape:", shape(scores1))

    let scaled_scores1 = scores1 * 0.125
    print("   Scaled scores computed")

    let attn_weights1 = softmax(scaled_scores1, 2)
    print("   Softmax applied")

    let attn_output1 = einsum("ihj,jhd->ihd", attn_weights1, V_expanded1)
    print("   Attention output shape:", shape(attn_output1))

    let attn_reshaped1 = reshape(attn_output1, [seq_len1, 2048.0])
    let attn_final1 = linear(attn_reshaped1, W_o_0)
    print("   Final attention:", shape(attn_final1))
    print("   Values[0:3]:", slice(attn_final1, 0, 0, 3))

    let x1 = e1 + attn_final1
    print("   After residual:", shape(x1))
    print("   Values[0:3]:", slice(x1, 0, 0, 3))

    let x_norm2_1 = rms_norm(x1, ffn_norm_0)
    print("   After FFN norm:", shape(x_norm2_1))

    let ffn_out1 = swiglu_ffn(x_norm2_1, W_gate_0, W_up_0, W_down_0)
    print("   FFN output:", shape(ffn_out1))
    print("   Values[0:3]:", slice(ffn_out1, 0, 0, 3))

    let final1 = x1 + ffn_out1
    print("   ✅ Final layer 0 output:", shape(final1))
    print("   Values[0:3]:", slice(final1, 0, 0, 3))
    print("")

    print("[Test 2: Two tokens (should fail with NaN)]")
    let tokens2 = [1.0, 2580.0]
    let e2 = embedding(embed_table, tokens2)
    print("   Embedding:", shape(e2))
    print("   E[0,0:3]:", slice(e2, 0, 0, 3))
    print("   E[1,0:3]:", slice(e2, 1, 0, 3))

    let x_norm2 = rms_norm(e2, attn_norm_0)
    print("   After norm:", shape(x_norm2))
    print("   Norm[0,0:3]:", slice(x_norm2, 0, 0, 3))
    print("   Norm[1,0:3]:", slice(x_norm2, 1, 0, 3))

    let Q2 = linear(x_norm2, W_q_0)
    let K2 = linear(x_norm2, W_k_0)
    let V2 = linear(x_norm2, W_v_0)
    print("   Q, K, V created")
    print("   Q[0,0:3]:", slice(Q2, 0, 0, 3))
    print("   Q[1,0:3]:", slice(Q2, 1, 0, 3))

    let seq_len2 = 2.0
    let Q_heads2 = reshape(Q2, [seq_len2, 32.0, 64.0])
    let K_heads2 = reshape(K2, [seq_len2, 4.0, 64.0])
    let V_heads2 = reshape(V2, [seq_len2, 4.0, 64.0])
    print("   Reshaped to heads")

    let K_with_group2 = reshape(K_heads2, [seq_len2, 4.0, 1.0, 64.0])
    let V_with_group2 = reshape(V_heads2, [seq_len2, 4.0, 1.0, 64.0])
    let K_broadcast2 = broadcast_to(K_with_group2, [seq_len2, 4.0, 8.0, 64.0])
    let V_broadcast2 = broadcast_to(V_with_group2, [seq_len2, 4.0, 8.0, 64.0])
    let K_expanded2 = reshape(K_broadcast2, [seq_len2, 32.0, 64.0])
    let V_expanded2 = reshape(V_broadcast2, [seq_len2, 32.0, 64.0])
    print("   Broadcasted for GQA")

    let scores2 = einsum("ihd,jhd->ihj", Q_heads2, K_expanded2)
    print("   Scores shape:", shape(scores2))

    let scaled_scores2 = scores2 * 0.125
    print("   Scaled scores computed")

    let attn_weights2 = softmax(scaled_scores2, 2)
    print("   Softmax shape:", shape(attn_weights2))

    let attn_output2 = einsum("ihj,jhd->ihd", attn_weights2, V_expanded2)
    print("   Attention output shape:", shape(attn_output2))

    let attn_reshaped2 = reshape(attn_output2, [seq_len2, 2048.0])
    print("   Reshaped shape:", shape(attn_reshaped2))
    print("   Reshaped[0,0:3]:", slice(attn_reshaped2, 0, 0, 3))
    print("   Reshaped[1,0:3]:", slice(attn_reshaped2, 1, 0, 3))

    let attn_final2 = linear(attn_reshaped2, W_o_0)
    print("   Final attention shape:", shape(attn_final2))
    print("   Attn final[0,0:3]:", slice(attn_final2, 0, 0, 3))
    print("   Attn final[1,0:3]:", slice(attn_final2, 1, 0, 3))

    let x2 = e2 + attn_final2
    print("   After residual:", shape(x2))
    print("   Residual[0,0:3]:", slice(x2, 0, 0, 3))
    print("   Residual[1,0:3]:", slice(x2, 1, 0, 3))

    print("")
    print("✅ Debug complete!")
}
