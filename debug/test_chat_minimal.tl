// Minimal test to identify the error

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len_int = Q_shape_tensor[0]
    let seq_len_f = seq_len_int * 1.0

    print("[DEBUG] seq_len_int:", seq_len_int)
    print("[DEBUG] seq_len_f:", seq_len_f)
    print("[DEBUG] seq_len_f type test")

    let Q_heads = reshape(Q, [seq_len_f, 32.0, 64.0])
    let K_heads = reshape(K, [seq_len_f, 4.0, 64.0])
    let V_heads = reshape(V, [seq_len_f, 4.0, 64.0])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_with_group = reshape(K_rope, [seq_len_f, 4.0, 1.0, 64.0])
    let V_with_group = reshape(V_heads, [seq_len_f, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_with_group, [seq_len_f, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_with_group, [seq_len_f, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [seq_len_f, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len_f, 32.0, 64.0])

    print("[DEBUG] Before einsum")
    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    print("[DEBUG] After einsum, before scaling")
    let scaled_scores = scores * 0.125

    print("[DEBUG] Before causal mask")
    print("[DEBUG] Using seq_len_int:", seq_len_int)

    print("[DEBUG] Creating causal mask")
    let mask_2d = causal_mask(seq_len_int)
    print("[DEBUG] Causal mask created")

    let mask_3d = reshape(mask_2d, [seq_len_f, 1.0, seq_len_f])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len_f, 32.0, seq_len_f])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let attn_reshaped = reshape(attn_output, [seq_len_f, 2048.0])
    linear(attn_reshaped, W_o)
}

main {
    print("=== Minimal Chat Demo Test ===")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")

    let formatted_prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let prompt_tokens = tokenize(tokenizer, formatted_prompt, false)
    print("Prompt tokens:", len(prompt_tokens))

    let current_emb = embedding(emb_weight, prompt_tokens)
    print("Embedding shape:", shape(current_emb))

    // Test single layer
    print("")
    print("Testing Layer 0...")
    let attn_norm = model.blk[0].attn_norm.weight
    let W_q = model.blk[0].attn_q.weight
    let W_k = model.blk[0].attn_k.weight
    let W_v = model.blk[0].attn_v.weight
    let W_o = model.blk[0].attn_output.weight

    print("[DEBUG] Before RMSNorm")
    let x_norm1 = rms_norm(current_emb, attn_norm)
    print("[DEBUG] After RMSNorm")

    let Q = linear(x_norm1, W_q)
    print("[DEBUG] Q computed")
    let K = linear(x_norm1, W_k)
    print("[DEBUG] K computed")
    let V = linear(x_norm1, W_v)
    print("[DEBUG] V computed")

    print("[DEBUG] Calling GQA attention")
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    print("[DEBUG] GQA attention completed")

    print("")
    print("=== Test Complete ===")
}
