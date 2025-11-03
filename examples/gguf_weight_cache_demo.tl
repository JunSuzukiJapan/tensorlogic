// GGUF Weight Cache Demo
// Tests lazy loading of quantized GGUF weights with LRU cache

print("GGUF Weight Cache Demo")
print("")

// Load GGUF weight cache (f16 precision)
print("Loading GGUF weight cache...")
let cache = load_weight_cache_gguf_f16(
    "~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf",
    10
)

print("Cache loaded successfully!")
print(cache)
print("")

// Test 1: Load first weight
print("Test 1: First access (cache miss)")
let q_weight = get_weight(cache, "blk.0.attn_q.weight")
print("  blk.0.attn_q.weight shape:")
print(shape(q_weight))
print("  Cache status:")
print(cache)
print("")

// Test 2: Load same weight again  
print("Test 2: Second access (cache hit)")
let q_weight2 = get_weight(cache, "blk.0.attn_q.weight")
print("  blk.0.attn_q.weight shape:")
print(shape(q_weight2))
print("  Cache status:")
print(cache)
print("")

// Test 3: Load multiple weights
print("Test 3: Fill cache with multiple weights")
let k_weight = get_weight(cache, "blk.0.attn_k.weight")
print("  blk.0.attn_k.weight loaded")

let v_weight = get_weight(cache, "blk.0.attn_v.weight")
print("  blk.0.attn_v.weight loaded")

let attn_output = get_weight(cache, "blk.0.attn_output.weight")
print("  blk.0.attn_output.weight loaded")

print("  Cache status:")
print(cache)
print("")

print("Demo completed successfully!")
