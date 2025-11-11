// Debug: Check cos/sin array values

main {
    print("================================================================================")
    print("Debug: cos/sin Array Values")
    print("================================================================================")
    print("")

    let head_dim = 64.0
    let max_seq_len = 10.0  // Small for debugging
    let rope_base = 10000.0

    print("Computing cos/sin arrays...")
    let cos_array = precompute_rope_cos(head_dim, max_seq_len, rope_base)
    let sin_array = precompute_rope_sin(head_dim, max_seq_len, rope_base)

    print("cos_array shape:", shape(cos_array))
    print("sin_array shape:", shape(sin_array))
    print("")


    print("Total sums:")
    print("  cos sum:", sum(cos_array))
    print("  sin sum:", sum(sin_array))
    print("================================================================================")
}
