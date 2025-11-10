// Comprehensive test for mathematical Metal shader operations
// Tests: exp, log, sqrt, pow, sin, cos, tan

main {
    print("=== Mathematical Operations Test ===")
    print("")

    // Test 1: Exponential
    print("[1/7] Testing exp...")
    let a1 = zeros([2, 3])
    let result1 = exp(a1)
    print("      exp(0) =", result1[0, 0], "(expected: 1.0)")

    let a1b = ones([2, 3])
    let result1b = exp(a1b)
    print("      exp(1) =", result1b[0, 0], "(expected: ~2.718)")

    let check1 = result1[0, 0] == 1.0
    let check1b = result1b[0, 0] > 2.7 && result1b[0, 0] < 2.8

    if check1 && check1b {
        print("      ✅ exp works")
    } else {
        print("      ❌ exp failed")
    }
    print("")

    // Test 2: Natural logarithm
    print("[2/7] Testing log...")
    let a2 = ones([2, 3])
    let result2 = log(a2)
    print("      log(1) =", result2[0, 0], "(expected: 0.0)")

    let a2b = exp(ones([2, 3]))  // e^1 ≈ 2.718
    let result2b = log(a2b)
    print("      log(e) =", result2b[0, 0], "(expected: ~1.0)")

    let check2 = result2[0, 0] < 0.1 && result2[0, 0] > -0.1
    let check2b = result2b[0, 0] > 0.9 && result2b[0, 0] < 1.1

    if check2 && check2b {
        print("      ✅ log works")
    } else {
        print("      ❌ log failed")
    }
    print("")

    // Test 3: Square root
    print("[3/7] Testing sqrt...")
    let a3 = ones([2, 3])
    let result3 = sqrt(a3)
    print("      sqrt(1) =", result3[0, 0], "(expected: 1.0)")

    let a3b = ones([2, 3]) * 4.0
    let result3b = sqrt(a3b)
    print("      sqrt(4) =", result3b[0, 0], "(expected: 2.0)")

    let check3 = result3[0, 0] == 1.0
    let check3b = result3b[0, 0] == 2.0

    if check3 && check3b {
        print("      ✅ sqrt works")
    } else {
        print("      ❌ sqrt failed")
    }
    print("")

    // Test 4: Power
    print("[4/7] Testing pow...")
    let a4 = ones([2, 3]) * 2.0
    let result4 = pow(a4, 3)
    print("      pow(2, 3) =", result4[0, 0], "(expected: 8.0)")

    let a4b = ones([2, 3]) * 3.0
    let result4b = pow(a4b, 2)
    print("      pow(3, 2) =", result4b[0, 0], "(expected: 9.0)")

    let check4 = result4[0, 0] == 8.0
    let check4b = result4b[0, 0] == 9.0

    if check4 && check4b {
        print("      ✅ pow works")
    } else {
        print("      ❌ pow failed")
    }
    print("")

    // Test 5: Sine
    print("[5/7] Testing sin...")
    let a5 = zeros([2, 3])
    let result5 = sin(a5)
    print("      sin(0) =", result5[0, 0], "(expected: 0.0)")

    let check5 = result5[0, 0] < 0.01 && result5[0, 0] > -0.01

    if check5 {
        print("      ✅ sin works")
    } else {
        print("      ❌ sin failed")
    }
    print("")

    // Test 6: Cosine
    print("[6/7] Testing cos...")
    let a6 = zeros([2, 3])
    let result6 = cos(a6)
    print("      cos(0) =", result6[0, 0], "(expected: 1.0)")

    let check6 = result6[0, 0] > 0.99 && result6[0, 0] < 1.01

    if check6 {
        print("      ✅ cos works")
    } else {
        print("      ❌ cos failed")
    }
    print("")

    // Test 7: Tangent
    print("[7/7] Testing tan...")
    let a7 = zeros([2, 3])
    let result7 = tan(a7)
    print("      tan(0) =", result7[0, 0], "(expected: 0.0)")

    let check7 = result7[0, 0] < 0.01 && result7[0, 0] > -0.01

    if check7 {
        print("      ✅ tan works")
    } else {
        print("      ❌ tan failed")
    }
    print("")

    print("=== Test Summary ===")
    print("")
    print("Tested mathematical operations:")
    print("  ✓ exp (exponential)")
    print("  ✓ log (natural logarithm)")
    print("  ✓ sqrt (square root)")
    print("  ✓ pow (power)")
    print("  ✓ sin (sine)")
    print("  ✓ cos (cosine)")
    print("  ✓ tan (tangent)")
    print("")
    print("All mathematical shader operations tested!")
}
