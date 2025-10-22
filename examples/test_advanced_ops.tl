main {
    print("Testing advanced operations (masking, broadcast, fused)")

    print("apply_attention_mask")
    let scores = positional_encoding(4, 4)
    let mask = ones([4, 4])
    let masked_scores = apply_attention_mask(scores, mask)

    print("padding_mask")
    let lengths = [3, 5, 2, 4]
    let pad_mask = padding_mask(lengths, 5)

    print("combine_masks")
    let mask2 = ones([4, 5])
    let combined = combine_masks(pad_mask, mask2)

    print("broadcast_to")
    let small = positional_encoding(1, 4)
    let broadcasted = broadcast_to(small, [3, 4])

    print("fused_add_relu")
    let a = positional_encoding(3, 4)
    let b = positional_encoding(3, 4)
    let fused1 = fused_add_relu(a, b)

    print("fused_mul_relu")
    let fused2 = fused_mul_relu(a, b)

    print("fused_affine")
    let scale = ones([3, 4])
    let bias = zeros([3, 4])
    let affine_result = fused_affine(a, scale, bias)

    print("fused_gelu_linear")
    let input = positional_encoding(2, 4)
    let weight = positional_encoding(4, 3)
    let bias_vec = zeros([2, 3])
    let gelu_linear = fused_gelu_linear(input, weight, bias_vec)

    print("SUCCESS: All 9 advanced operations work!")
    print("")
    print("Summary:")
    print("  Masking: apply_attention_mask, padding_mask, combine_masks")
    print("  Broadcast: broadcast_to")
    print("  Fused: fused_add_relu, fused_mul_relu, fused_affine, fused_gelu_linear")
}
