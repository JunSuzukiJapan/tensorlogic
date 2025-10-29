// TinyLlama Chat Demo using Object-Oriented Model Access
// This demonstrates the new hierarchical object access: model.blk[i].feature.property

main {
    print("=== TinyLlama 1.1B Chat (Object-Oriented Architecture) ===")

    // Load model
    let model = load_model("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer("/Users/suzukijun/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    // Input
    let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenizer.encode(prompt)
    print("Input tokens:", tokens)

    // Embedding
    let x = embedding_lookup(model.token_embd.weight, tokens)
    print("After embedding: shape =", shape(x))

    // Process through all 22 layers using object-oriented access
    for i in 0..22 {
        print("  Layer", i, "...")
        let layer = model.blk[i]

        // Attention
        let attn_norm_out = rms_norm(x, layer.attn_norm.weight, 1e-5)
        let q = linear(attn_norm_out, layer.attn_q.weight)
        let k = linear(attn_norm_out, layer.attn_k.weight)
        let v = linear(attn_norm_out, layer.attn_v.weight)

        let q_rope = apply_rope(q, 0)
        let k_rope = apply_rope(k, 0)

        let attn_out = scaled_dot_product_attention(q_rope, k_rope, v)
        let attn_proj = linear(attn_out, layer.attn_output.weight)
        let x_attn = x + attn_proj

        // FFN
        let ffn_norm_out = rms_norm(x_attn, layer.ffn_norm.weight, 1e-5)
        let ffn_out = swiglu_ffn(
            ffn_norm_out,
            layer.ffn_gate.weight,
            layer.ffn_down.weight,
            layer.ffn_up.weight
        )
        x = x_attn + ffn_out
    }

    // Output
    let final_norm = rms_norm(x, model.output_norm.weight, 1e-5)
    let logits = linear(final_norm, model.output.weight)

    // Sample
    let next_token = sample_top_p(logits, 0.9, 0.8)
    let output = tokenizer.decode(next_token)

    print("\n=== Output ===")
    print(output)
    print("\n✓ Demo complete using object-oriented model access!")
}
