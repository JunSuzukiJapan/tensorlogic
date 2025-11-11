// Test: Comprehensive Metal kernel verification for RoPE
// Tests each aspect of the RoPE calculation independently

main {
    print("================================================================================")
    print("RoPE Metal Kernel Comprehensive Verification")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let model = load_model(model_path)

    // Test 1: Position 0 should be identity (cos=1, sin=0)
    print("Test 1: Position 0 (Identity Transformation)")
    print("=" * 60)
    let test1_data = [3.0, 4.0, 5.0, 12.0]  // Two pairs: (3,4) and (5,12)
    let i = 0
    while i < 60 {
        test1_data = test1_data + [3.0, 4.0]
        i = i + 2
    }

    let Q1 = from_vec(test1_data, [1, 1, 64])
    let Q1_rope = rope(Q1)
    let Q1_data_out = to_vec(Q1_rope)

    let error1 = 0.0
    let i = 0
    while i < 4 {
        let diff = abs(test1_data[i] - Q1_data_out[i])
        error1 = error1 + diff
        i = i + 1
    }
    print("First 4 elements - Input:", test1_data[0], test1_data[1], test1_data[2], test1_data[3])
    print("First 4 elements - Output:", Q1_data_out[0], Q1_data_out[1], Q1_data_out[2], Q1_data_out[3])
    print("Total error:", error1)
    print("Status:", if error1 < 0.1 then "PASS" else "FAIL")
    print("")

    // Test 2: Magnitude preservation for single pair
    print("Test 2: Magnitude Preservation (Single Position)")
    print("=" * 60)
    let test2_data = []
    let i = 0
    while i < 32 {
        test2_data = test2_data + [3.0, 4.0]  // magnitude = 5.0
        i = i + 1
    }

    let Q2 = from_vec(test2_data, [1, 1, 64])
    let Q2_rope = rope(Q2)
    let Q2_data_out = to_vec(Q2_rope)

    // Check first pair magnitude
    let mag_in = sqrt(3.0 * 3.0 + 4.0 * 4.0)
    let mag_out = sqrt(Q2_data_out[0] * Q2_data_out[0] + Q2_data_out[1] * Q2_data_out[1])
    print("Input magnitude:", mag_in)
    print("Output magnitude:", mag_out)
    print("Magnitude error:", abs(mag_in - mag_out))
    print("Status:", if abs(mag_in - mag_out) < 0.1 then "PASS" else "FAIL")
    print("")

    // Test 3: Different positions should give different results
    print("Test 3: Position Dependency")
    print("=" * 60)
    let test3_data = []
    let i = 0
    while i < 32 {
        test3_data = test3_data + [3.0, 4.0]
        i = i + 1
    }

    // Create 3 positions with same data
    let test3_multi = test3_data + test3_data + test3_data
    let Q3 = from_vec(test3_multi, [3, 1, 64])
    let Q3_rope = rope(Q3)
    let Q3_out = to_vec(Q3_rope)

    // Position 0 (elements 0-63)
    let pos0_x0 = Q3_out[0]
    let pos0_x1 = Q3_out[1]

    // Position 1 (elements 64-127)
    let pos1_x0 = Q3_out[64]
    let pos1_x1 = Q3_out[65]

    // Position 2 (elements 128-191)
    let pos2_x0 = Q3_out[128]
    let pos2_x1 = Q3_out[129]

    print("Position 0, pair 0:", pos0_x0, pos0_x1)
    print("Position 1, pair 0:", pos1_x0, pos1_x1)
    print("Position 2, pair 0:", pos2_x0, pos2_x1)

    let diff_01 = abs(pos0_x0 - pos1_x0) + abs(pos0_x1 - pos1_x1)
    let diff_12 = abs(pos1_x0 - pos2_x0) + abs(pos1_x1 - pos2_x1)

    print("Difference 0-1:", diff_01)
    print("Difference 1-2:", diff_12)
    print("Status:", if diff_01 > 0.5 && diff_12 > 0.5 then "PASS" else "FAIL")
    print("")

    // Test 4: Consistency check - running twice should give same result
    print("Test 4: Determinism")
    print("=" * 60)
    let test4_data = []
    let i = 0
    while i < 32 {
        test4_data = test4_data + [3.0, 4.0]
        i = i + 1
    }

    let Q4a = from_vec(test4_data, [1, 1, 64])
    let Q4a_rope = rope(Q4a)
    let Q4a_out = to_vec(Q4a_rope)

    let Q4b = from_vec(test4_data, [1, 1, 64])
    let Q4b_rope = rope(Q4b)
    let Q4b_out = to_vec(Q4b_rope)

    let error4 = 0.0
    let i = 0
    while i < 4 {
        error4 = error4 + abs(Q4a_out[i] - Q4b_out[i])
        i = i + 1
    }

    print("Run 1 first 4:", Q4a_out[0], Q4a_out[1], Q4a_out[2], Q4a_out[3])
    print("Run 2 first 4:", Q4b_out[0], Q4b_out[1], Q4b_out[2], Q4b_out[3])
    print("Total error:", error4)
    print("Status:", if error4 < 0.001 then "PASS" else "FAIL")
    print("")

    // Test 5: Check all dimension pairs preserve magnitude
    print("Test 5: All Dimension Pairs Magnitude Preservation")
    print("=" * 60)
    let test5_data = []
    let i = 0
    while i < 32 {
        test5_data = test5_data + [3.0, 4.0]
        i = i + 1
    }

    let Q5 = from_vec(test5_data, [1, 1, 64])
    let Q5_rope = rope(Q5)
    let Q5_out = to_vec(Q5_rope)

    let max_mag_error = 0.0
    let pair_idx = 0
    while pair_idx < 32 {
        let idx0 = pair_idx * 2
        let idx1 = pair_idx * 2 + 1

        let mag_in = 5.0  // sqrt(3^2 + 4^2)
        let mag_out = sqrt(Q5_out[idx0] * Q5_out[idx0] + Q5_out[idx1] * Q5_out[idx1])
        let mag_error = abs(mag_in - mag_out)

        if mag_error > max_mag_error {
            max_mag_error = mag_error
        }

        pair_idx = pair_idx + 1
    }

    print("Maximum magnitude error across all 32 pairs:", max_mag_error)
    print("Status:", if max_mag_error < 0.2 then "PASS" else "FAIL")
    print("")

    print("================================================================================")
    print("Summary")
    print("================================================================================")
    print("All tests should PASS for correct RoPE implementation")
    print("If any test fails, there is a bug in the Metal kernel")
    print("================================================================================")
}
