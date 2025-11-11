// Test: Does calling a working version first affect the buggy version?

fn silu_working(x: float16[?, ?]) -> float16[?, ?] {
    let result = x * sigmoid(x)
    result
}

fn silu_buggy(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

main {
    print("================================================================================")
    print("Call Order Test")
    print("================================================================================")
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

    print("Test 1: Call working version FIRST")
    let result_working = silu_working(gate)
    print("Working version result sum:", sum(result_working))
    print("")

    print("Test 2: Then call buggy version SECOND")
    let result_buggy = silu_buggy(gate)
    print("Buggy version result sum:", sum(result_buggy))
    print("")

    print("================================================================================")
    if sum(result_buggy) == 0.0 {
        print("❌ Buggy version still returns 0 even when called second")
    }
    if sum(result_buggy) != 0.0 {
        print("✅ Buggy version works when called after working version!")
    }
    print("================================================================================")
}
