//! Test that interpreter uses Metal GPU for tensor operations

use tensorlogic::parser::TensorLogicParser;
use tensorlogic::interpreter::Interpreter;

#[test]
fn test_interpreter_uses_metal_gpu() {
    let source = r#"
        tensor x: float16[3] = [1.0, 2.0, 3.0]
        tensor y: float16[3] = [4.0, 5.0, 6.0]

        main {
            result := x + y
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let mut interpreter = Interpreter::new();

    // Execute program
    interpreter.execute(&program).unwrap();

    // Get the result tensor
    let result = interpreter.get_variable("result").unwrap();
    let tensor = result.as_tensor().unwrap();

    // Verify the tensor is on Metal device
    println!("Tensor device: {:?}", tensor.device());

    match tensor.device() {
        tensorlogic::device::Device::Metal(_) => {
            println!("✅ Tensor is on Metal GPU");
        }
        tensorlogic::device::Device::CPU => {
            panic!("❌ Tensor is on CPU, should be on Metal GPU");
        }
        _ => {
            panic!("❌ Unexpected device type");
        }
    }

    // Verify the computation result
    let values = tensor.to_vec();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0].to_f32(), 5.0);
    assert_eq!(values[1].to_f32(), 7.0);
    assert_eq!(values[2].to_f32(), 9.0);

    println!("✅ Computation result correct: {:?}", values.iter().map(|x| x.to_f32()).collect::<Vec<_>>());
}

#[test]
fn test_matrix_multiply_uses_metal_gpu() {
    let source = r#"
        tensor A: float16[2, 3] = [[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]]
        tensor B: float16[3, 2] = [[7.0, 8.0],
                                    [9.0, 10.0],
                                    [11.0, 12.0]]

        main {
            result := A @ B
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let mut interpreter = Interpreter::new();

    // Execute program
    interpreter.execute(&program).unwrap();

    // Get the result tensor
    let result = interpreter.get_variable("result").unwrap();
    let tensor = result.as_tensor().unwrap();

    // Verify the tensor is on Metal device
    println!("Matrix multiply result device: {:?}", tensor.device());

    match tensor.device() {
        tensorlogic::device::Device::Metal(_) => {
            println!("✅ Matrix multiply result is on Metal GPU");
        }
        tensorlogic::device::Device::CPU => {
            panic!("❌ Matrix multiply result is on CPU, should be on Metal GPU");
        }
        _ => {
            panic!("❌ Unexpected device type");
        }
    }

    // Verify the result dimensions [2, 2]
    assert_eq!(tensor.dims(), &[2, 2]);
    println!("✅ Matrix multiply dimensions correct: {:?}", tensor.dims());
}

#[test]
fn test_learnable_tensor_uses_metal_gpu() {
    let source = r#"
        tensor w: float16[1] learnable = [0.5]

        main {
            result := w * w
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let mut interpreter = Interpreter::new();

    // Execute program
    interpreter.execute(&program).unwrap();

    // Get the learnable tensor
    let w = interpreter.get_variable("w").unwrap();
    let tensor = w.as_tensor().unwrap();

    // Verify the tensor is on Metal device
    println!("Learnable tensor device: {:?}", tensor.device());

    match tensor.device() {
        tensorlogic::device::Device::Metal(_) => {
            println!("✅ Learnable tensor is on Metal GPU");
        }
        tensorlogic::device::Device::CPU => {
            panic!("❌ Learnable tensor is on CPU, should be on Metal GPU");
        }
        _ => {
            panic!("❌ Unexpected device type");
        }
    }

    // Verify the value
    let values = tensor.to_vec();
    assert_eq!(values[0].to_f32(), 0.5);
    println!("✅ Learnable tensor value correct: {}", values[0].to_f32());
}
