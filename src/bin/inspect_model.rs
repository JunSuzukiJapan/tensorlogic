use tensorlogic::model::Model;
use tensorlogic::device::MetalDevice;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        eprintln!("Example: {} ~/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from: {}", model_path);
    println!();

    // Create Metal device for model loading
    let device = MetalDevice::new().unwrap_or_else(|e| {
        eprintln!("Error creating Metal device: {}", e);
        std::process::exit(1);
    });

    let model = Model::load(model_path, &device).unwrap_or_else(|e| {
        eprintln!("Error loading model: {}", e);
        std::process::exit(1);
    });

    println!("=== Model Information ===");
    println!("Name: {}", model.metadata.name);
    println!("Format: {:?}", model.metadata.format);
    println!("Quantization: {:?}", model.metadata.quantization);
    println!("Total tensors: {}", model.num_tensors());
    println!();

    println!("=== Tensor Organization ===");
    println!();

    // Collect and sort tensor names
    let mut names: Vec<&String> = model.tensor_names();
    names.sort();

    // Group tensors by prefix
    let mut current_prefix = String::new();
    for name in &names {
        // Extract prefix (e.g., "layers.0", "token_embd", "output")
        let parts: Vec<&str> = name.split('.').collect();
        let prefix = if parts.len() > 2 && parts[0] == "layers" {
            // For layers, use "layers.X"
            format!("{}.{}", parts[0], parts[1])
        } else if parts.len() > 1 {
            // For other multi-part names, use first part
            parts[0].to_string()
        } else {
            // Single part name
            name.to_string()
        };

        if prefix != current_prefix {
            println!("\n[{}]", prefix);
            current_prefix = prefix.clone();
        }

        // Get tensor and print info
        if let Some(tensor) = model.get_tensor(name) {
            let shape_str: Vec<String> = tensor.shape().dims().iter().map(|&d| d.to_string()).collect();
            let num_elements: usize = tensor.shape().dims().iter().product();
            println!("  {} : [{}] ({} elements)", name, shape_str.join(", "), num_elements);
        }
    }

    println!();
    println!("=== Architecture Analysis ===");
    println!();

    // Analyze architecture
    let num_layers = names.iter()
        .filter(|n| n.starts_with("layers."))
        .filter_map(|n| {
            let parts: Vec<&str> = n.split('.').collect();
            if parts.len() > 1 {
                parts[1].parse::<usize>().ok()
            } else {
                None
            }
        })
        .max()
        .map(|max| max + 1)
        .unwrap_or(0);

    println!("Number of layers: {}", num_layers);

    // Find embedding dimension
    if let Some(embd_tensor) = model.get_tensor("token_embd.weight") {
        let dims = embd_tensor.shape().dims();
        if dims.len() == 2 {
            println!("Vocabulary size: {}", dims[0]);
            println!("Embedding dimension (d_model): {}", dims[1]);
        }
    }

    // Find attention dimensions
    if let Some(wq_tensor) = model.get_tensor("layers.0.attention.wq.weight") {
        let dims = wq_tensor.shape().dims();
        if dims.len() == 2 {
            println!("Attention query dimension: {}", dims[0]);
            println!("Attention hidden dimension: {}", dims[1]);
        }
    }

    // Find FFN dimensions
    if let Some(w1_tensor) = model.get_tensor("layers.0.feed_forward.w1.weight") {
        let dims = w1_tensor.shape().dims();
        if dims.len() == 2 {
            println!("FFN intermediate dimension (d_ff): {}", dims[0]);
        }
    }

    println!();
    println!("=== Sample Tensors ===");
    println!();

    // Show first few tensors in detail
    for name in names.iter().take(5) {
        if let Some(tensor) = model.get_tensor(name) {
            println!("{}", name);
            println!("  Shape: {:?}", tensor.shape().dims());
            println!("  Elements: {}", tensor.shape().dims().iter().product::<usize>());
            println!();
        }
    }

    println!("✅ Model inspection complete!");
}
