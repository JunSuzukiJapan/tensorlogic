// Model Information Demo
// Shows that model loading works

main {
    print("=======================================================================")
    print("TensorLogic Model Loading Demo")
    print("=======================================================================")
    print("")
    
    let home = env("HOME")
    let model_path = home + "/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf"
    
    print("Attempting to load model from:")
    print("  " + model_path)
    print("")
    
    print("Loading model...")
    let model = load_model(model_path)
    print("")
    print("Model loaded successfully!")
    print("")
    
    print("Model Information:")
    print("  Format: GGUF (Quantized)")
    print("  Type: TinyLlama 1.1B Chat")
    print("")
    
    print("Note:")
    print("  Full LLM text generation requires additional implementation.")
    print("  Current status: Model loading ✓ | Text generation ✗")
    print("")
    print("=======================================================================")
}
