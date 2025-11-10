// Test: Call a simple function first to see if it primes the system

fn simple_add(x: float16[?, ?]) -> float16[?, ?] {
    x + x
}

fn silu_buggy(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Primer Test")
    print("================================================================================"
)
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_table = get_tensor(model, "token_embd.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    let tokens = tokenize(tokenizer, "Hello", false)
    let e = embedding(embed_table, tokens)
    let x_norm = rms_norm(e, ffn_norm_0)
    let gate = linear(x_norm, W_gate_0)

    print("Input gate sum:", sum(gate))
    print("")

    print("Primer: Call simple_add first")
    let primed = simple_add(gate)
    print("  Primer result sum:", sum(primed))
    print("")

    print("Now call silu_buggy (should be primed)")
    let result = silu_buggy(gate)
    print("  silu_buggy result sum:", sum(result))
    print("")

    print("================================================================================")
    if sum(result) == 0.0 {
        print("❌ Still returns 0 even after priming")
    }
    if sum(result) != 0.0 {
        print("✅ Priming worked! Returns non-zero")
    }
    print("================================================================================")
}
