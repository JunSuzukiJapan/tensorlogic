// Debug Token 1 NaN - Check Layer 0 output

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
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
    linear(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len_f = Q_shape_tensor[0]

    let shape_Q_heads = [seq_len_f, 32.0, 64.0]
    let shape_K_heads = [seq_len_f, 4.0, 64.0]
    let shape_V_heads = [seq_len_f, 4.0, 64.0]

    let Q_heads = reshape(Q, shape_Q_heads)
    let K_heads = reshape(K, shape_K_heads)
    let V_heads = reshape(V, shape_V_heads)

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let shape_with_group = [seq_len_f, 4.0, 1.0, 64.0]
    let K_with_group = reshape(K_rope, shape_with_group)
    let V_with_group = reshape(V_heads, shape_with_group)

    let shape_broadcast = [seq_len_f, 4.0, 8.0, 64.0]
    let K_broadcast = broadcast_to(K_with_group, shape_broadcast)
    let V_broadcast = broadcast_to(V_with_group, shape_broadcast)

    let shape_expanded = [seq_len_f, 32.0, 64.0]
    let K_expanded = reshape(K_broadcast, shape_expanded)
    let V_expanded = reshape(V_broadcast, shape_expanded)

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let d_model = 2048.0
    let shape_flat = [seq_len_f, d_model]
    let attn_flat = reshape(attn_output, shape_flat)
    linear(attn_flat, W_o)
}

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
    let x_norm = rms_norm(x, attn_norm)
    let Q = linear(x_norm, W_q)
    let K = linear(x_norm, W_k)
    let V = linear(x_norm, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x_attn = x + attn_out

    let x_ffn_norm = rms_norm(x_attn, ffn_norm)
    let ffn_out = swiglu_ffn(x_ffn_norm, W_gate, W_up, W_down)
    x_attn + ffn_out
}

main {
    print("================================================================================")
    print("🔍 Token 1 NaN Debug - Layer 0 Output Check")
    print("================================================================================")
    print("")

    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = "/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json"

    print("[1/5] Loading model and tokenizer...")
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Loaded")
    print("")

    print("[2/5] Loading Layer 0 weights...")
    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    print("      ✓ Layer 0 weights loaded")
    print("")

    print("[3/5] Creating prompt and tokenizing...")
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let chat_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n" + user_message + "</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, chat_prompt, false)
    print("      Tokens:", len(tokens))
    print("")

    print("[4/5] Running forward pass through Layer 0...")
    print("")

    print("  Step 1: Embedding")
    let e = embedding(embed_table, tokens)
    print("    ✓ Embedding shape:", shape(e))

    print("")
    print("  Step 2: Layer 0 forward pass")
    let h0 = transformer_layer(e, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    print("    ✓ Layer 0 output shape:", shape(h0))

    print("")
    print("[5/5] Checking for NaN...")
    print("")

    // Check if any values are NaN
    let sum_val = sum(h0)
    print("  Layer 0 output sum:", sum_val)

    if sum_val != sum_val {
        print("  ❌ Layer 0 output contains NaN!")
        print("")
        print("  The NaN originates in or before Layer 0.")
        print("  Need to check:")
        print("    1. Embedding output")
        print("    2. RMS norm in attention")
        print("    3. Q/K/V projections")
        print("    4. RoPE application")
        print("    5. Attention scores")
        print("    6. Softmax")
        print("    7. FFN operations")
    }

    if sum_val == sum_val {
        print("  ✅ Layer 0 output is valid (no NaN)")
        print("")
        print("  The NaN originates AFTER Layer 0.")
        print("  Layers 1-21, final norm, or logits calculation has the issue.")
    }

    print("")
    print("================================================================================")
    print("✅ Layer 0 check complete")
    print("================================================================================")
}
