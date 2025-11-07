use tensorlogic::model::formats::MmapGGUFLoader;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let model_path = format!("{}/.llm/models/tinyllama-1.1b-chat-q4_0.gguf", home);

    let loader = MmapGGUFLoader::new(&model_path)?;

    println!("Analyzing GGUF types in model...");
    println!("{}", "─".repeat(60));

    let mut type_counts: HashMap<String, usize> = HashMap::new();

    for name in loader.tensor_names() {
        if let Some(info) = loader.tensor_info(name) {
            let type_str = format!("{:?}", info.gguf_type);
            *type_counts.entry(type_str).or_insert(0) += 1;
        }
    }

    println!("\nGGUF Type Distribution:");
    for (gguf_type, count) in type_counts {
        println!("  {}: {} tensors", gguf_type, count);
    }

    println!("\n{}", "─".repeat(60));
    Ok(())
}
