// Debug: Investigate NaN in attention mechanism

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result = x * sig
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
    result = linear(intermediate, W_down)
}

main {
    print("=== Attention Mechanism NaN Debug ===")
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
    let embed_table = get_tensor(model, "token_embd.weight")
    print("   ✓ Loaded")
    print("")

    print("[Test: 2-token attention]")
    let tokens = [1.0, 2580.0]

    let e = embedding(embed_table, tokens)
    print("   1. Embedding shape:", shape(e))
    print("   2. Embedding[0,0:3]:", slice(e, 0, 0, 3))
    print("   3. Embedding[1,0:3]:", slice(e, 1, 0, 3))
    print("")

    print("   4. Applying RMS norm...")
    let x_norm = rms_norm(e, attn_norm_0)
    print("   5. Norm output shape:", shape(x_norm))
    print("   6. Norm[0,0:3]:", slice(x_norm, 0, 0, 3))
    print("   7. Norm[1,0:3]:", slice(x_norm, 1, 0, 3))
    print("")

    print("   8. Computing Q, K, V...")
    let Q = linear(x_norm, W_q_0)
    let K = linear(x_norm, W_k_0)
    let V = linear(x_norm, W_v_0)

    print("   9. Q shape:", shape(Q))
    print("   10. Q[0,0:3]:", slice(Q, 0, 0, 3))
    print("   11. Q[1,0:3]:", slice(Q, 1, 0, 3))
    print("")

    print("   12. K shape:", shape(K))
    print("   13. K[0,0:3]:", slice(K, 0, 0, 3))
    print("   14. K[1,0:3]:", slice(K, 1, 0, 3))
    print("")

    print("   15. V shape:", shape(V))
    print("   16. V[0,0:3]:", slice(V, 0, 0, 3))
    print("   17. V[1,0:3]:", slice(V, 1, 0, 3))
    print("")

    print("   18. Reshaping to multi-head format...")
    let seq_len_f = 2.0
    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    print("   19. Q_heads shape:", shape(Q_heads))
    print("   20. K_heads shape:", shape(K_heads))
    print("   21. V_heads shape:", shape(V_heads))
    print("")

    print("   22. Broadcasting K, V for GQA...")
    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_heads, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    print("   23. K_expanded shape:", shape(K_expanded))
    print("   24. V_expanded shape:", shape(V_expanded))
    print("")

    print("   25. Computing attention scores...")
    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    print("   26. Scores shape:", shape(scores))
    print("   27. Scores[0,0,0:3]:", slice(scores, 0, 0, 3))
    print("")

    print("   28. Scaling scores...")
    let scaled_scores = scores * 0.125
    print("   29. Scaled scores shape:", shape(scaled_scores))
    print("   30. Scaled scores[0,0,0:3]:", slice(scaled_scores, 0, 0, 3))
    print("")

    print("   31. Applying softmax...")
    let attn_weights = softmax(scaled_scores, 2)
    print("   32. Attention weights shape:", shape(attn_weights))
    print("   33. Attn weights[0,0,0:3]:", slice(attn_weights, 0, 0, 3))
    print("")

    print("   34. Computing attention output...")
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    print("   35. Attention output shape:", shape(attn_output))
    print("   36. Attn output[0,0,0:3]:", slice(attn_output, 0, 0, 3))
    print("")

    print("   37. Reshaping and projecting...")
    let d_model = 2048.0
    let shape_output = [seq_len_f, d_model]
    let attn_reshaped = reshape(attn_output, shape_output)
    print("   38. Reshaped shape:", shape(attn_reshaped))

    let attn_final = linear(attn_reshaped, W_o_0)
    print("   39. Final projection shape:", shape(attn_final))
    print("   40. Final[0,0:3]:", slice(attn_final, 0, 0, 3))
    print("   41. Final[1,0:3]:", slice(attn_final, 1, 0, 3))
    print("")

    print("✅ Attention debug complete!")
}
