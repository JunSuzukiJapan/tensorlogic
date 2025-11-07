// モデル全体の保存と読み込みデモ
// cndl_save_model_safetensor と cndl_load_model_safetensor の使用例

main {
    print("=== Model Save/Load Demo ===")
    print("")

    // ============================================================================
    // 1. 既存モデルのロードと保存
    // ============================================================================
    print("1. Load and Save Existing Model")
    print("--------------------------------")

    // 既存のGGUFモデルをロード
    // model := load_model_f16("path/to/model.gguf")

    // Safetensors形式で保存
    // cndl_save_model_safetensor(model, "model_converted.safetensors")

    print("(Skipped - requires actual model file)")
    print("")

    // ============================================================================
    // 2. 個別テンソルの保存と、モデルとしての読み込み
    // ============================================================================
    print("2. Save Tensors, Load as Model")
    print("-------------------------------")

    // 複数のテンソルを個別に作成
    layer1_weight := f32::from_array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
    layer1_bias := f32::from_array([0.1, 0.2, 0.3])

    layer2_weight := f32::from_array([[7.0, 8.0],
                                      [9.0, 10.0],
                                      [11.0, 12.0]])
    layer2_bias := f32::from_array([0.5, 0.5])

    print("Created tensors:")
    print("  layer1.weight:", shape(layer1_weight))
    print("  layer1.bias:", shape(layer1_bias))
    print("  layer2.weight:", shape(layer2_weight))
    print("  layer2.bias:", shape(layer2_bias))
    print("")

    // 各テンソルを同じファイルに保存
    // Note: 現在の実装では1つずつ保存すると上書きされます
    // 複数テンソルを1ファイルに保存するには、モデルとして保存する必要があります
    print("Saving individual tensors...")
    cndl_save_safetensor(layer1_weight, "/tmp/demo_layer1_weight.safetensors", "weight")
    cndl_save_safetensor(layer1_bias, "/tmp/demo_layer1_bias.safetensors", "bias")
    cndl_save_safetensor(layer2_weight, "/tmp/demo_layer2_weight.safetensors", "weight")
    cndl_save_safetensor(layer2_bias, "/tmp/demo_layer2_bias.safetensors", "bias")
    print("✓ Saved 4 tensors to separate files")
    print("")

    // ============================================================================
    // 3. モデル全体のロードと検証
    // ============================================================================
    print("3. Load Full Model")
    print("------------------")

    // 保存されたモデル全体を読み込み
    // loaded_model := cndl_load_model_safetensor("/tmp/demo_model.safetensors")
    // print("Loaded model with", num_tensors(loaded_model), "tensors")

    print("(Requires pre-saved model file)")
    print("")

    // ============================================================================
    // 4. モデルの変換（GGUF -> Safetensors）
    // ============================================================================
    print("4. Model Format Conversion")
    print("---------------------------")

    // GGUFモデルを読み込み
    // gguf_model := load_model_f16("model.gguf")

    // Safetensors形式で保存（HuggingFace互換）
    // cndl_save_model_safetensor(gguf_model, "model.safetensors")

    // 保存されたSafetensorsモデルを読み込み
    // safetensors_model := cndl_load_model_safetensor("model.safetensors")

    print("(Example workflow - requires actual files)")
    print("")

    // ============================================================================
    // 5. 保存されたモデルの内容確認
    // ============================================================================
    print("5. Inspect Saved Model")
    print("----------------------")

    // Safetensorsファイルの内容を一覧表示
    cndl_list_safetensors("/tmp/demo_layer1_weight.safetensors")
    print("")

    print("=== Demo Complete ===")
    print("")
    print("Use cases:")
    print("  - Convert GGUF models to Safetensors")
    print("  - Save trained models in HuggingFace format")
    print("  - Share models across frameworks (PyTorch, Candle, TensorLogic)")
    print("  - Archive model checkpoints")
}
