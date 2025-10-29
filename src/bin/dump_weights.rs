use std::env;
use tensorlogic::device::MetalDevice;
use tensorlogic::model::Model;
use tensorlogic::tensor::{TensorAccessors, TensorIO};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from: {}", model_path);

    // Create Metal device
    let device = MetalDevice::new().unwrap_or_else(|e| {
        eprintln!("Error creating Metal device: {}", e);
        std::process::exit(1);
    });

    let model = Model::<half::f16>::load(model_path, &device).unwrap_or_else(|e| {
        eprintln!("Error loading model: {}", e);
        std::process::exit(1);
    });

    println!("\n=== Dumping blk.0.attn_q.weight ===\n");

    let w_q = model.get_tensor("blk.0.attn_q.weight").unwrap_or_else(|| {
        eprintln!("Tensor not found: blk.0.attn_q.weight");
        std::process::exit(1);
    });

    println!("Shape: {:?}", w_q.shape().dims());

    // Get raw data as f16 values
    let data = w_q.to_vec();
    let dims = w_q.shape().dims();

    if dims.len() != 2 {
        eprintln!("Expected 2D tensor, got {}D", dims.len());
        std::process::exit(1);
    }

    let rows = dims[0];
    let cols = dims[1];

    println!("Rows: {}, Cols: {}", rows, cols);
    println!();

    // 1. First 10 values (linear)
    println!("1. First 10 values (linear):");
    for i in 0..10.min(data.len()) {
        print!("{:.8} ", data[i].to_f32());
    }
    println!("\n");

    // 2. First row W_q[0, :10]
    println!("2. First row W_q[0, :10]:");
    for j in 0..10.min(cols) {
        let idx = 0 * cols + j; // Row-major: row * ncols + col
        print!("{:.8} ", data[idx].to_f32());
    }
    println!("\n");

    // 3. First column W_q[:10, 0]
    println!("3. First column W_q[:10, 0]:");
    for i in 0..10.min(rows) {
        let idx = i * cols + 0; // Row-major: row * ncols + col
        print!("{:.8} ", data[idx].to_f32());
    }
    println!("\n");

    // 4. Second row W_q[1, :10]
    println!("4. Second row W_q[1, :10]:");
    for j in 0..10.min(cols) {
        let idx = 1 * cols + j;
        print!("{:.8} ", data[idx].to_f32());
    }
    println!("\n");

    // 5. Element W_q[100, 200]
    if rows > 100 && cols > 200 {
        let idx = 100 * cols + 200;
        println!("5. Element W_q[100, 200]: {:.8}", data[idx].to_f32());
    } else {
        println!("5. Element W_q[100, 200]: Out of bounds");
    }

    println!("\n✅ Dump complete!");
}
