// Test: Detailed tracing of silu bug to understand buffer lifecycle

fn silu_buggy(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn silu_working(x: float16[?, ?]) -> float16[?, ?] {
    let result = x * sigmoid(x)
    result
}

main {
    print("================================================================================")
    print("Detailed SiLU Bug Trace")
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

    print("Testing buggy version with TL_BUFFER_DEBUG=1...")
    print("Call silu_buggy(gate)...")
    let result_buggy = silu_buggy(gate)
    print("Returned from silu_buggy")
    let sum_buggy = sum(result_buggy)
    print("Result sum:", sum_buggy)
    print("")

    print("Testing working version with TL_BUFFER_DEBUG=1...")
    print("Call silu_working(gate)...")
    let result_working = silu_working(gate)
    print("Returned from silu_working")
    let sum_working = sum(result_working)
    print("Result sum:", sum_working)
    print("")

    print("================================================================================")
    if sum_buggy == 0.0 {
        print("❌ Buggy version returns 0")
    }
    if sum_buggy != 0.0 {
        print("✅ Buggy version fixed!")
    }
    print("================================================================================")
}
