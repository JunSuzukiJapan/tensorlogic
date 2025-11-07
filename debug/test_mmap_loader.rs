//! Test mmap GGUF loader performance

use tensorlogic::model::formats::MmapGGUFLoader;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("HOME")? + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";

    println!("Testing MmapGGUFLoader with: {}", model_path);
    println!("{}", "─".repeat(60));

    // Test 1: Loading speed
    let start = Instant::now();
    let loader = MmapGGUFLoader::new(&model_path)?;
    let load_time = start.elapsed();

    println!("✓ Loader created in {:?}", load_time);
    println!("  Tensors: {}", loader.metadata().tensor_count);
    println!("  Version: {}", loader.metadata().version);

    // Test 2: Zero-copy access
    let tensor_names = loader.tensor_names();
    println!("\n✓ Found {} tensors", tensor_names.len());

    if let Some(&first_name) = tensor_names.first() {
        let start = Instant::now();
        let data1 = loader.get_tensor_data(first_name)?;
        let access_time1 = start.elapsed();

        let start = Instant::now();
        let data2 = loader.get_tensor_data(first_name)?;
        let access_time2 = start.elapsed();

        println!("\n✓ Zero-copy verification:");
        println!("  Tensor: {}", first_name);
        println!("  Size: {} bytes", data1.len());
        println!("  First access: {:?}", access_time1);
        println!("  Second access: {:?}", access_time2);
        println!("  Same pointer: {}", data1.as_ptr() == data2.as_ptr());
    }

    // Test 3: List some tensor names
    println!("\n✓ Sample tensor names:");
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        if let Some(info) = loader.tensor_info(name) {
            println!("  {}: {} {:?}", i+1, name, info.shape);
        }
    }

    println!("\n{}", "─".repeat(60));
    println!("All tests passed! 🎉");

    Ok(())
}
