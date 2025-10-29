// Profile linear layer performance

main {
    print("=== Linear Layer Performance Profiling ===")
    print("")

    let home = env("HOME")
    print("Loading model...")
    let model = load_model_f32(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    
    print("Loading tokenizer...")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    print("Tokenizing test input...")
    let tokens = tokenizer.tokenize("Hello world", false)
    
    print("Getting embeddings...")
    let tok_embd = model.token_embd.weight
    let x = embedding(tok_embd, tokens)
    
    print("Getting layer 0 weights...")
    let L0_q = model.blk[0].attn_q.weight
    let L0_k = model.blk[0].attn_k.weight
    
    print("")
    print("=== Performance Test: Linear Operations ===")
    print("Matrix shapes: x=[seq, 2048], weight=[2048, N]")
    print("")
    
    print("Test 1: Q projection (typical decode: [1, 2048] @ [2048, 2048])")
    let Q = linear(x, L0_q)
    print("  Completed Q projection")
    
    print("")
    print("Test 2: K projection")
    let K = linear(x, L0_k)
    print("  Completed K projection")
    
    print("")
    print("=== Profile Complete ===")
}
