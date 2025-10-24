// Test ComplEx Scoring Function
// Phase 12.4: ComplEx algorithm implementation

main {
    print("=== Testing ComplEx Scoring ===")
    print("")

    // Create complex-valued embeddings (real and imaginary parts)
    // Dimension: 4 for simplicity

    print("Creating entity embeddings (complex):")
    let alice_re = zeros([4])
    let alice_im = zeros([4])
    let bob_re = zeros([4])
    let bob_im = zeros([4])
    print("  alice = (alice_re, alice_im)")
    print("  bob = (bob_re, bob_im)")
    print("")

    print("Creating relation embedding (complex):")
    let friend_of_re = zeros([4])
    let friend_of_im = zeros([4])
    print("  friend_of = (friend_of_re, friend_of_im)")
    print("")

    // Test ComplEx scoring function
    print("--- Test 1: ComplEx Score Computation ---")
    print("")
    print("Computing: complex_score(alice, friend_of, bob)")
    print("Formula: Re(<alice, friend_of, conj(bob)>)")
    print("")

    let score = complex_score(alice_re, alice_im, friend_of_re, friend_of_im, bob_re, bob_im)
    print("Score:", score)
    print("")

    print("Explanation:")
    print("  ComplEx uses complex numbers to model both:")
    print("  - Symmetric relations (like DistMult)")
    print("  - Antisymmetric relations (like TransE)")
    print("  via the complex conjugate operation")
    print("")

    // Test with different embeddings
    print("--- Test 2: Different Embeddings ---")
    print("")

    let tokyo_re = zeros([4])
    let tokyo_im = zeros([4])
    let lives_in_re = zeros([4])
    let lives_in_im = zeros([4])

    let score2 = complex_score(alice_re, alice_im, lives_in_re, lives_in_im, tokyo_re, tokyo_im)
    print("complex_score(alice, lives_in, tokyo):", score2)
    print("")

    print("--- Test 3: Tail Prediction ---")
    print("")
    print("Question: alice friend_of ?")
    print("")

    print("Candidate 1: bob")
    let score_bob = predict_tail_complex(alice_re, alice_im, friend_of_re, friend_of_im, bob_re, bob_im)
    print("  Score:", score_bob)
    print("")

    print("Candidate 2: tokyo")
    let score_tokyo = predict_tail_complex(alice_re, alice_im, friend_of_re, friend_of_im, tokyo_re, tokyo_im)
    print("  Score:", score_tokyo)
    print("")

    print("Higher score = more likely")
    print("(Would rank candidates by score)")
    print("")

    print("--- Test 4: Head Prediction ---")
    print("")
    print("Question: ? lives_in tokyo")
    print("")

    print("Candidate 1: alice")
    let score_alice = predict_head_complex(alice_re, alice_im, lives_in_re, lives_in_im, tokyo_re, tokyo_im)
    print("  Score:", score_alice)
    print("")

    print("Candidate 2: bob")
    let score_bob_head = predict_head_complex(bob_re, bob_im, lives_in_re, lives_in_im, tokyo_re, tokyo_im)
    print("  Score:", score_bob_head)
    print("")

    print("--- ComplEx Properties ---")
    print("")
    print("1. COMPLEX CONJUGATE")
    print("   For z = a + bi, conj(z) = a - bi")
    print("   This makes ComplEx asymmetric (unlike DistMult)")
    print("")
    print("2. GENERALIZES DISTMULT")
    print("   When imaginary parts = 0:")
    print("   ComplEx reduces to DistMult scoring")
    print("")
    print("3. DOUBLE PARAMETERS")
    print("   Each embedding needs 2x parameters")
    print("   (real and imaginary parts)")
    print("")
    print("4. EXPRESSIVENESS")
    print("   Can model:")
    print("   - Symmetric relations (family, colleague)")
    print("   - Antisymmetric relations (parent, boss)")
    print("   - Composition (friend of friend)")
    print("")

    print("✅ ComplEx scoring tests completed!")
}
