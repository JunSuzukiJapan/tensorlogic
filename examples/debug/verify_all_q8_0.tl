// Comprehensive Q8_0 model verification against Candle reference
// Verifies multiple tensors with statistics

main {
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q8_0.gguf"
    let model = load_model(model_path)

    let total_checks = 0
    let passed_checks = 0
    let failed_checks = 0

    print("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
    print("Q8_0 Model Comprehensive Verification")
    print("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
    print("")

    // 1. token_embd.weight - BOS Token Sum
    print("1. token_embd.weight - BOS Token Sum")
    let emb_weight = get_tensor(model, "token_embd.weight")
    let bos_emb = embedding(emb_weight, 1)
    let bos_sum = sum(bos_emb)
    let bos_expected = 0.0528666377
    let bos_diff = abs(bos_sum - bos_expected)
    let bos_rel_err = bos_diff / abs(bos_expected) * 100.0
    print("   TensorLogic: " + str(bos_sum))
    print("   Candle Ref:  " + str(bos_expected))
    print("   Rel Error:   " + str(bos_rel_err) + "%")
    if bos_diff < 0.0001 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 2. token_embd.weight - Full tensor sum
    print("2. token_embd.weight - Full Tensor Sum")
    let emb_sum = sum(emb_weight)
    let emb_sum_expected = -24.2523326874
    let emb_sum_diff = abs(emb_sum - emb_sum_expected)
    let emb_sum_rel_err = emb_sum_diff / abs(emb_sum_expected) * 100.0
    print("   TensorLogic: " + str(emb_sum))
    print("   Candle Ref:  " + str(emb_sum_expected))
    print("   Rel Error:   " + str(emb_sum_rel_err) + "%")
    if emb_sum_diff < 0.1 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 3. blk.0.attn_norm.weight - Sum
    print("3. blk.0.attn_norm.weight - Sum")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let attn_norm_sum = sum(attn_norm)
    let attn_norm_expected = 11.8375186920
    let attn_norm_diff = abs(attn_norm_sum - attn_norm_expected)
    let attn_norm_rel_err = attn_norm_diff / abs(attn_norm_expected) * 100.0
    print("   TensorLogic: " + str(attn_norm_sum))
    print("   Candle Ref:  " + str(attn_norm_expected))
    print("   Rel Error:   " + str(attn_norm_rel_err) + "%")
    if attn_norm_diff < 0.01 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 4. blk.0.attn_q.weight - Sum
    print("4. blk.0.attn_q.weight - Sum")
    let attn_q = get_tensor(model, "blk.0.attn_q.weight")
    let attn_q_sum = sum(attn_q)
    let attn_q_expected = -31.2397632599
    let attn_q_diff = abs(attn_q_sum - attn_q_expected)
    let attn_q_rel_err = attn_q_diff / abs(attn_q_expected) * 100.0
    print("   TensorLogic: " + str(attn_q_sum))
    print("   Candle Ref:  " + str(attn_q_expected))
    print("   Rel Error:   " + str(attn_q_rel_err) + "%")
    if attn_q_diff < 0.1 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 5. blk.0.attn_k.weight - Sum
    print("5. blk.0.attn_k.weight - Sum")
    let attn_k = get_tensor(model, "blk.0.attn_k.weight")
    let attn_k_sum = sum(attn_k)
    let attn_k_expected = -52.1654663086
    let attn_k_diff = abs(attn_k_sum - attn_k_expected)
    let attn_k_rel_err = attn_k_diff / abs(attn_k_expected) * 100.0
    print("   TensorLogic: " + str(attn_k_sum))
    print("   Candle Ref:  " + str(attn_k_expected))
    print("   Rel Error:   " + str(attn_k_rel_err) + "%")
    if attn_k_diff < 0.1 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 6. blk.0.attn_v.weight - Sum
    print("6. blk.0.attn_v.weight - Sum")
    let attn_v = get_tensor(model, "blk.0.attn_v.weight")
    let attn_v_sum = sum(attn_v)
    let attn_v_expected = 5.9289512634
    let attn_v_diff = abs(attn_v_sum - attn_v_expected)
    let attn_v_rel_err = attn_v_diff / abs(attn_v_expected) * 100.0
    print("   TensorLogic: " + str(attn_v_sum))
    print("   Candle Ref:  " + str(attn_v_expected))
    print("   Rel Error:   " + str(attn_v_rel_err) + "%")
    if attn_v_diff < 0.01 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // 7. output_norm.weight - Sum
    print("7. output_norm.weight - Sum")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_norm_sum = sum(output_norm)
    let output_norm_expected = 3921.6875
    let output_norm_diff = abs(output_norm_sum - output_norm_expected)
    let output_norm_rel_err = output_norm_diff / abs(output_norm_expected) * 100.0
    print("   TensorLogic: " + str(output_norm_sum))
    print("   Candle Ref:  " + str(output_norm_expected))
    print("   Rel Error:   " + str(output_norm_rel_err) + "%")
    if output_norm_diff < 1.0 {
        print("   ✅ PASS")
        passed_checks = passed_checks + 1
    } else {
        print("   ❌ FAIL")
        failed_checks = failed_checks + 1
    }
    total_checks = total_checks + 1
    print("")

    // Summary
    print("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
    print("Summary")
    print("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
    print("Total Checks:  " + str(total_checks))
    print("Passed:        " + str(passed_checks))
    print("Failed:        " + str(failed_checks))

    if failed_checks == 0 {
        print("")
        print("✅ ALL CHECKS PASSED!")
    } else {
        print("")
        print("❌ SOME CHECKS FAILED")
    }
}
