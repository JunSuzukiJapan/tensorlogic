use half::f16;

fn main() {
    let original = 20358_i64;
    let as_f32 = original as f32;
    let as_f16 = f16::from_f32(as_f32);
    let back_to_f32 = as_f16.to_f32();

    println!("Original integer: {}", original);
    println!("As f32:          {}", as_f32);
    println!("As f16:          {}", as_f16);
    println!("Back to f32:     {}", back_to_f32);
    println!("Lost precision:  {}", as_f32 - back_to_f32);
    println!("");

    // Test the problematic value
    let test_val = 20352.0_f32;
    println!("Expected: 20358");
    println!("Got:      {}", f16::from_f32(test_val).to_f32());
}
