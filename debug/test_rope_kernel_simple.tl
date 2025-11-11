// Test: Simple Metal kernel verification for RoPE

main {
    print("================================================================================")
    print("RoPE Metal Kernel Verification")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Test 1: Position 0 should preserve values (cos=1, sin=0 at position 0)
    print("Test 1: Position 0 Identity")
    print("------------------------------------------------------------")

    // Create simple test data: pair (3,4) repeated 32 times for head_dim=64
    let test_vec = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                    3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]

    let Q_test = from_vec(test_vec, [1, 1, 64])
    let Q_rope = rope(Q_test)
    let out_vec = to_vec(Q_rope)

    print("Input  pair 0: [", test_vec[0], test_vec[1], "]")
    print("Output pair 0: [", out_vec[0], out_vec[1], "]")

    let error0 = test_vec[0] - out_vec[0]
    let error1 = test_vec[1] - out_vec[1]
    print("Error pair 0: [", error0, error1, "]")
    print("")

    // Test 2: Magnitude check - for (3,4), magnitude should be 5
    print("Test 2: Magnitude Preservation")
    print("------------------------------------------------------------")

    let x0 = out_vec[0]
    let x1 = out_vec[1]
    let mag_sq = x0 * x0 + x1 * x1

    print("Expected magnitude squared: 25.0")
    print("Actual magnitude squared:", mag_sq)
    print("Error:", 25.0 - mag_sq)
    print("")

    // Test 3: Multiple positions should give different results
    print("Test 3: Position Dependency (3 positions)")
    print("------------------------------------------------------------")

    // Shape [3, 1, 64] = 3 positions, 1 head, 64 dimensions
    let multi_vec = [
        // Position 0
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        // Position 1
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        // Position 2
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0
    ]

    let Q_multi = from_vec(multi_vec, [3, 1, 64])
    let Q_multi_rope = rope(Q_multi)
    let multi_out = to_vec(Q_multi_rope)

    print("Position 0, pair 0: [", multi_out[0], multi_out[1], "]")
    print("Position 1, pair 0: [", multi_out[64], multi_out[65], "]")
    print("Position 2, pair 0: [", multi_out[128], multi_out[129], "]")

    let diff_01_x = multi_out[0] - multi_out[64]
    let diff_01_y = multi_out[1] - multi_out[65]
    let diff_12_x = multi_out[64] - multi_out[128]
    let diff_12_y = multi_out[65] - multi_out[129]

    print("Difference pos0-pos1: [", diff_01_x, diff_01_y, "]")
    print("Difference pos1-pos2: [", diff_12_x, diff_12_y, "]")
    print("")

    // Test 4: Check magnitude for multiple pairs at position 0
    print("Test 4: Multiple Pairs Magnitude Check (Position 0)")
    print("------------------------------------------------------------")

    let pair0_x = out_vec[0]
    let pair0_y = out_vec[1]
    let pair0_mag_sq = pair0_x * pair0_x + pair0_y * pair0_y

    let pair10_x = out_vec[20]
    let pair10_y = out_vec[21]
    let pair10_mag_sq = pair10_x * pair10_x + pair10_y * pair10_y

    let pair20_x = out_vec[40]
    let pair20_y = out_vec[41]
    let pair20_mag_sq = pair20_x * pair20_x + pair20_y * pair20_y

    let pair30_x = out_vec[60]
    let pair30_y = out_vec[61]
    let pair30_mag_sq = pair30_x * pair30_x + pair30_y * pair30_y

    print("Pair 0  magnitude squared:", pair0_mag_sq, " (expected: 25.0)")
    print("Pair 10 magnitude squared:", pair10_mag_sq, " (expected: 25.0)")
    print("Pair 20 magnitude squared:", pair20_mag_sq, " (expected: 25.0)")
    print("Pair 30 magnitude squared:", pair30_mag_sq, " (expected: 25.0)")
    print("")

    print("================================================================================")
    print("Verification Complete")
    print("================================================================================")
    print("Expected:")
    print("  - Test 1: Error pair 0 should be close to [0, 0]")
    print("  - Test 2: Magnitude error should be close to 0")
    print("  - Test 3: Position differences should be non-zero")
    print("  - Test 4: All magnitudes should be close to 25.0")
    print("================================================================================")
}
