// Profile full transformer layer

fn transformer_layer(x, W_attn_norm, W_q, W_k, W_v, W_o, W_ffn_norm, W_gate, W_up, W_down, K_cache, V_cache) -> result {
    let normed = rms_norm(x, W_attn_norm)
    let Q = linear(normed, W_q)
    let attn_out = attention_with_cache(Q, K_cache, V_cache, W_o)
    let after_attn = x + attn_out
    let normed2 = rms_norm(after_attn, W_ffn_norm)
    let ffn_out = swiglu_ffn(normed2, W_gate, W_up, W_down)
    result = after_attn + ffn_out
}

main {
    print("=== Transformer Layer Performance Profiling ===")
    print("")

    let home = env("HOME")
    print("Loading model...")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    print("Preparing input...")
    let tokens = tokenizer.tokenize("Hello", false)
    let tok_embd = model.token_embd.weight
    let x = embedding(tok_embd, tokens)
    
    print("Loading layer 0 weights...")
    let L0 = model.blk[0]
    
    print("Building initial KV cache...")
    let K0 = linear(x, L0.attn_k.weight)
    let V0 = linear(x, L0.attn_v.weight)
    
    print("")
    print("=== Running Full Transformer Layer ===")
    let h0 = transformer_layer(x, L0.attn_norm.weight, L0.attn_q.weight, L0.attn_k.weight, L0.attn_v.weight, L0.attn_output.weight, L0.ffn_norm.weight, L0.ffn_gate.weight, L0.ffn_up.weight, L0.ffn_down.weight, K0, V0)
    print("  Completed layer 0")
    
    print("")
    print("=== Profile Complete ===")
}
