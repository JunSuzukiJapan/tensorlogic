// Simple forward pass test with "Hello" input
//
// Simplified version without KV cache to debug token 0 issue

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    result = x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let up = linear(x, W_up)
    let intermediate = silu(gate) * up
    result = linear(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q = linear(x, W_q)
    let K = linear(x, W_k)
    let V = linear(x, W_v)

    let Q_heads = reshape(Q, [shape(Q)[0], 32.0, 64.0])
    let K_heads = reshape(K, [shape(K)[0], 4.0, 64.0])
    let V_heads = reshape(V, [shape(V)[0], 4.0, 64.0])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_expanded = broadcast_to(
        reshape(K_rope, [shape(K_rope)[0], 4.0, 1.0, 64.0]),
        [shape(K_rope)[0], 4.0, 8.0, 64.0]
    )
    let K_final = reshape(K_expanded, [shape(K_rope)[0], 32.0, 64.0])

    let V_expanded = broadcast_to(
        reshape(V_heads, [shape(V_heads)[0], 4.0, 1.0, 64.0]),
        [shape(V_heads)[0], 4.0, 8.0, 64.0]
    )
    let V_final = reshape(V_expanded, [shape(V_heads)[0], 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_final)
    let scaled = scores * 0.125
    let attn_weights = softmax(scaled, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_final)

    let attn_reshaped = reshape(attn_output, [shape(attn_output)[0], 2048.0])
    result = linear(attn_reshaped, W_o)
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
    let x_norm1 = rms_norm(x, attn_norm)
    let attn_out = tinyllama_gqa_attention(x_norm1, W_q, W_k, W_v, W_o)
    let h1 = x + attn_out
    let h1_norm = rms_norm(h1, ffn_norm)
    let ffn_out = swiglu_ffn(h1_norm, W_gate, W_up, W_down)
    result = h1 + ffn_out
}

main {
    print("=== Simple Forward Pass Test ===")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"

    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    print("✓ Model and tokenizer loaded")

    // Tokenize "Hello"
    let prompt = "Hello"
    let tokens = tokenize(tokenizer, prompt, false)

    print("Prompt:", prompt)
    print("Tokens:", tokens)

    // Load weights
    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load first layer weights
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    print("✓ Weights loaded")

    // Forward pass
    let embeddings = embedding(embed_table, tokens)
    print("\nEmbeddings shape:", shape(embeddings))

    let h0 = transformer_layer(
        embeddings,
        W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0,
        W_gate_0, W_up_0, W_down_0, ffn_norm_0
    )

    print("After layer 0 shape:", shape(h0))

    // Get logits (h0 should be [1, 2048] since we only have 1 token)
    let h0_norm = rms_norm(h0, output_norm)
    let logits = linear(h0_norm, output_weight)

    print("Logits shape:", shape(logits))

    // Sample 5 tokens
    print("\n=== Token Generation ===")
    let token_1 = temperature_sample(logits, 0.8)
    let token_2 = temperature_sample(logits, 0.8)
    let token_3 = temperature_sample(logits, 0.8)
    let token_4 = temperature_sample(logits, 0.8)
    let token_5 = temperature_sample(logits, 0.8)

    print("Token 1:", token_1)
    print("Token 2:", token_2)
    print("Token 3:", token_3)
    print("Token 4:", token_4)
    print("Token 5:", token_5)

    print("\n✅ Test complete")
}
