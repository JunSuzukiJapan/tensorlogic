// Test sigmoid with actual FFN gate values
// Testing the intermediate sigmoid computation in FFN

fn test_sigmoid_direct(x: float16[?, ?]) -> float16[?, ?] {
    sigmoid(x)
}

fn test_silu_direct(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("🔍 Sigmoid Test with Actual FFN Gate Values")
    print("================================================================================")
    print("")

    print("[1] Loading model...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)
    let emb_weight = get_tensor(model, "token_embd.weight")
    print("    ✓ Loaded")
    print("")

    print("[2] Creating test input (46 tokens)...")
    let prompt = "<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    let embeddings = embedding(emb_weight, tokens)
    print("    ✓ Input created")
    print("")

    print("[3] Computing first layer FFN to get real gate values...")
    let layer_idx = 0

    // Attention
    let attn_norm = model.blk[layer_idx].attn_norm.weight
    let W_q = model.blk[layer_idx].attn_q.weight
    let W_k = model.blk[layer_idx].attn_k.weight
    let W_v = model.blk[layer_idx].attn_v.weight
    let W_o = model.blk[layer_idx].attn_output.weight

    let x_norm1 = rms_norm(embeddings, attn_norm)
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)

    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]
    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])
    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)
    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125
    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)
    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)
    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    let attn_out = linear(attn_reshaped, W_o)
    let x1 = embeddings + attn_out

    // FFN
    let ffn_norm = model.blk[layer_idx].ffn_norm.weight
    let W_gate = model.blk[layer_idx].ffn_gate.weight
    let W_up = model.blk[layer_idx].ffn_up.weight

    let x_norm2 = rms_norm(x1, ffn_norm)
    print("    x_norm2 (FFN input) sum:", sum(x_norm2))
    print("")

    print("──────────────────────────────────────────────────────────────────────────")
    print("TESTING SIGMOID ON ACTUAL GATE VALUES")
    print("──────────────────────────────────────────────────────────────────────────")
    print("")

    let gate = linear(x_norm2, W_gate)
    print("  [1] gate = linear(x_norm2, W_gate)")
    print("      gate sum:", sum(gate))
    print("      gate shape:", shape(gate))
    print("")

    let sig = sigmoid(gate)
    print("  [2] sigmoid(gate)")
    print("      sigmoid sum:", sum(sig))
    print("      sigmoid shape:", shape(sig))
    print("")

    let silu_result = gate * sig
    print("  [3] silu = gate * sigmoid(gate)")
    print("      silu sum:", sum(silu_result))
    print("      silu shape:", shape(silu_result))
    print("")

    // Test silu function directly
    let silu_fn = test_silu_direct(gate)
    print("  [4] silu_direct(gate)")
    print("      silu_direct sum:", sum(silu_fn))
    print("")

    // Test on smaller subset
    print("──────────────────────────────────────────────────────────────────────────")
    print("TESTING ON FIRST ROW ONLY")
    print("──────────────────────────────────────────────────────────────────────────")
    print("")

    let gate_first = slice(gate, 0, 0, 1)
    print("  gate[0] sum:", sum(gate_first))

    let sig_first = sigmoid(gate_first)
    print("  sigmoid(gate[0]) sum:", sum(sig_first))

    let silu_first = gate_first * sig_first
    print("  silu(gate[0]) sum:", sum(silu_first))
    print("")

    print("================================================================================")
    print("✅ Sigmoid Test Completed")
    print("================================================================================")
}
