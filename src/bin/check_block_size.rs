use gguf_rs_lib::tensor::quantization::blocks::Q4_0Block;

fn main() {
    let block_size = std::mem::size_of::<Q4_0Block>();
    println!("Q4_0Block size: {} bytes", block_size);
    println!("Expected: 18 bytes (2 for f16 scale + 16 for 32 4-bit values)");

    if block_size == 18 {
        println!("✅ Block size is correct!");
    } else {
        println!("❌ Block size is WRONG! This will cause incorrect dequantization!");
    }
}
