// Positional Encoding for Transformer
// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

// Parameters
tensor max_len: float16[1] = [10.0]    // Maximum sequence length
tensor d_model: float16[1] = [8.0]     // Model dimension (must be even)

main {
    // Create position indices: [0, 1, 2, ..., max_len-1]
    // For simplicity, we'll create a few positions manually
    tensor pos_0: float16[1] = [0.0]
    tensor pos_1: float16[1] = [1.0]
    tensor pos_2: float16[1] = [2.0]
    tensor pos_3: float16[1] = [3.0]

    // Calculate positional encoding for position 0
    // Even dimensions (0, 2, 4, 6): use sin
    // Odd dimensions (1, 3, 5, 7): use cos

    // For dimension 0 (i=0): div_term = 10000^(0/8) = 1.0
    tensor div_0: float16[1] = [1.0]
    tensor angle_0_0: float16[1] = pos_0 / div_0  // 0.0
    tensor pe_0_0: float16[1] = sin(angle_0_0)    // sin(0) = 0
    tensor pe_0_1: float16[1] = cos(angle_0_0)    // cos(0) = 1

    // For dimension 2 (i=1): div_term = 10000^(2/8) = 10000^0.25 ≈ 10.0
    tensor div_1: float16[1] = [10.0]
    tensor angle_0_2: float16[1] = pos_0 / div_1  // 0.0
    tensor pe_0_2: float16[1] = sin(angle_0_2)    // sin(0) = 0
    tensor pe_0_3: float16[1] = cos(angle_0_2)    // cos(0) = 1

    // For dimension 4 (i=2): div_term = 10000^(4/8) = 100.0
    tensor div_2: float16[1] = [100.0]
    tensor angle_0_4: float16[1] = pos_0 / div_2  // 0.0
    tensor pe_0_4: float16[1] = sin(angle_0_4)    // sin(0) = 0
    tensor pe_0_5: float16[1] = cos(angle_0_4)    // cos(0) = 1

    // For dimension 6 (i=3): div_term = 10000^(6/8) = 1000.0
    tensor div_3: float16[1] = [1000.0]
    tensor angle_0_6: float16[1] = pos_0 / div_3  // 0.0
    tensor pe_0_6: float16[1] = sin(angle_0_6)    // sin(0) = 0
    tensor pe_0_7: float16[1] = cos(angle_0_6)    // cos(0) = 1

    print("Position 0 Encoding:")
    print("  PE[0,0] (sin):", pe_0_0)
    print("  PE[0,1] (cos):", pe_0_1)
    print("  PE[0,2] (sin):", pe_0_2)
    print("  PE[0,3] (cos):", pe_0_3)
    print("  PE[0,4] (sin):", pe_0_4)
    print("  PE[0,5] (cos):", pe_0_5)
    print("  PE[0,6] (sin):", pe_0_6)
    print("  PE[0,7] (cos):", pe_0_7)

    // Calculate positional encoding for position 1
    tensor angle_1_0: float16[1] = pos_1 / div_0  // 1.0
    tensor pe_1_0: float16[1] = sin(angle_1_0)    // sin(1.0) ≈ 0.841
    tensor pe_1_1: float16[1] = cos(angle_1_0)    // cos(1.0) ≈ 0.540

    tensor angle_1_2: float16[1] = pos_1 / div_1  // 0.1
    tensor pe_1_2: float16[1] = sin(angle_1_2)    // sin(0.1) ≈ 0.0998
    tensor pe_1_3: float16[1] = cos(angle_1_2)    // cos(0.1) ≈ 0.995

    print("Position 1 Encoding:")
    print("  PE[1,0] (sin):", pe_1_0)
    print("  PE[1,1] (cos):", pe_1_1)
    print("  PE[1,2] (sin):", pe_1_2)
    print("  PE[1,3] (cos):", pe_1_3)

    // Demonstrate different frequencies
    // Lower dimensions change quickly (high frequency)
    // Higher dimensions change slowly (low frequency)
    tensor angle_2_0: float16[1] = pos_2 / div_0  // 2.0
    tensor pe_2_0: float16[1] = sin(angle_2_0)    // sin(2.0) ≈ 0.909

    tensor angle_2_6: float16[1] = pos_2 / div_3  // 0.002
    tensor pe_2_6: float16[1] = sin(angle_2_6)    // sin(0.002) ≈ 0.002

    print("Position 2 Encoding (frequency comparison):")
    print("  PE[2,0] (high freq):", pe_2_0)
    print("  PE[2,6] (low freq):", pe_2_6)
}
