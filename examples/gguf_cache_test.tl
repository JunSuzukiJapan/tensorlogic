// Simple GGUF cache test

let home = "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
print("Loading GGUF weight cache from: {}", home)

let cache = load_weight_cache_gguf_f16(home, 10)
print("Cache loaded: {}", cache)

let weight = get_weight(cache, "blk.0.attn_q.weight")
print("Weight shape: {}", shape(weight))
print("Cache after loading: {}", cache)
