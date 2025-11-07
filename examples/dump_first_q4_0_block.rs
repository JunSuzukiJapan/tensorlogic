// Dump first Q4_0 block of token_embd to verify dequantization
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() {
    let model_path = std::env::var("HOME").unwrap() + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let mut file = BufReader::new(File::open(&model_path).unwrap());

    // Skip to data section (this is a simplified version, actual offset needs GGUF header parsing)
    // For now, let's just show the first 100 bytes of the file to see structure
    let mut buffer = vec![0u8; 200];
    file.read_exact(&mut buffer).unwrap();

    println!("First 200 bytes of GGUF file:");
    for (i, chunk) in buffer.chunks(20).enumerate() {
        print!("{:04x}: ", i * 20);
        for byte in chunk {
            print!("{:02x} ", byte);
        }
        println!();
    }
}
