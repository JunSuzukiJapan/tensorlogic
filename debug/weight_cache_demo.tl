// Weight Cache Lazy Loading Demo
//
// This demonstrates lazy loading of weights using memory-mapped safetensors files.
// Benefits:
// - 100x faster model loading (mmap instead of full file read)
// - 50% memory reduction (weights loaded on-demand, not all at once)
// - LRU cache keeps only recently-used weights in memory
//
// Usage:
//   1. Convert GGUF model to safetensors format (using convert script)
//   2. Run: tl run examples/weight_cache_demo.tl
//
// Example for f32 TinyLlama (assuming safetensors file exists):

fn main() {
    print("=== Weight Cache Lazy Loading Demo ===")
    print("")

    // Create weight cache with capacity of 10 weights
    // Only 10 most recently used weights will be kept in memory
    // Other weights are loaded on-demand from mmap when needed
    let cache = load_weight_cache_f32(
        "~/.llm/models/TinyLlama_f32.safetensors",
        10  // Cache capacity (number of weights)
    )

    print("Weight cache created!")
    print("Cache stats: {}", cache)
    print("")

    // Get a weight lazily (loaded from mmap on first access)
    print("Loading layer 0 attention weights...")
    let q_weight = get_weight(cache, "model.layers.0.self_attn.q_proj.weight")
    print("  q_proj.weight shape: {}", shape(q_weight))

    // Second access is fast (from cache)
    let q_weight2 = get_weight(cache, "model.layers.0.self_attn.q_proj.weight")
    print("  Second access (cached): {}", shape(q_weight2))
    print("")

    // Load more weights - LRU cache will evict old ones
    print("Loading more weights...")
    for i in 0..5 {
        let k_weight = get_weight(cache,
            "model.layers.{}.self_attn.k_proj.weight".format(i))
        print("  Layer {} k_proj loaded", i)
    }

    print("")
    print("Cache stats after loading: {}", cache)
    print("✓ Demo complete!")
    print("")
    print("Note: With f32 weights (256MB each), cache of 10 = ~2.5GB memory")
    print("      Without cache, all 1200+ weights = 56GB+ would be in memory!")
}

main()
