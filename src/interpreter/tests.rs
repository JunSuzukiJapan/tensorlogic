use super::*;
use crate::parser::TensorLogicParser;
use crate::device::MetalDevice;

#[test]
fn test_interpreter_creation() {
    let interpreter = Interpreter::new();
    assert!(interpreter.env.variables.is_empty());
}

#[test]
fn test_simple_tensor_decl() {
    let source = "tensor w: float32[10, 20]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    assert!(interpreter.execute(&program).is_ok());

    // Verify tensor was created
    let w = interpreter.get_variable("w").unwrap();
    assert!(matches!(w, Value::Tensor(_)));
}

#[test]
fn test_learnable_tensor() {
    let source = "tensor w: float32[10] learnable";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    let w = interpreter.get_variable("w").unwrap();
    if let Value::Tensor(tensor) = w {
        assert!(tensor.requires_grad());
    } else {
        panic!("Expected tensor");
    }
}

#[test]
fn test_tensor_with_init() {
    let source = "tensor w: float32[3] = [1.0, 2.0, 3.0]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    let w = interpreter.get_variable("w").unwrap();
    if let Value::Tensor(tensor) = w {
        assert_eq!(tensor.shape().dims(), &[3]);
    } else {
        panic!("Expected tensor");
    }
}

#[test]
fn test_assignment_statement() {
    let source = r#"
        tensor x: float32[5]
        main {
            y := x
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    // Both x and y should exist
    assert!(interpreter.get_variable("x").is_ok());
    assert!(interpreter.get_variable("y").is_ok());
}

#[test]
fn test_undefined_variable_error() {
    let source = r#"
        main {
            y := x
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    let result = interpreter.execute(&program);

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RuntimeError::UndefinedVariable(_)
    ));
}

#[test]
fn test_scalar_literal_float() {
    let interpreter = Interpreter::new();

    let lit = TensorLiteral::Scalar(ScalarLiteral::Float(3.14));
    let value = interpreter.eval_literal(&lit).unwrap();

    if let Value::Float(f) = value {
        assert!((f - 3.14).abs() < 0.01);
    } else {
        panic!("Expected float value");
    }
}

#[test]
fn test_scalar_literal_integer() {
    let interpreter = Interpreter::new();

    let lit = TensorLiteral::Scalar(ScalarLiteral::Integer(42));
    let value = interpreter.eval_literal(&lit).unwrap();

    if let Value::Integer(i) = value {
        assert_eq!(i, 42);
    } else {
        panic!("Expected integer value");
    }
}

#[test]
fn test_scalar_literal_boolean() {
    let interpreter = Interpreter::new();

    let lit = TensorLiteral::Scalar(ScalarLiteral::Boolean(true));
    let value = interpreter.eval_literal(&lit).unwrap();

    if let Value::Boolean(b) = value {
        assert!(b);
    } else {
        panic!("Expected boolean value");
    }
}

#[test]
fn test_array_literal_1d() {
    let interpreter = Interpreter::new();

    let lit = TensorLiteral::Array(vec![
        TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(3.0)),
    ]);

    let value = interpreter.eval_literal(&lit).unwrap();

    if let Value::Tensor(tensor) = value {
        assert_eq!(tensor.shape().dims(), &[3]);
    } else {
        panic!("Expected tensor");
    }
}

#[test]
fn test_array_literal_2d() {
    let interpreter = Interpreter::new();

    let lit = TensorLiteral::Array(vec![
        TensorLiteral::Array(vec![
            TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
            TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        ]),
        TensorLiteral::Array(vec![
            TensorLiteral::Scalar(ScalarLiteral::Float(3.0)),
            TensorLiteral::Scalar(ScalarLiteral::Float(4.0)),
        ]),
    ]);

    let value = interpreter.eval_literal(&lit).unwrap();

    if let Value::Tensor(tensor) = value {
        assert_eq!(tensor.shape().dims(), &[2, 2]);
    } else {
        panic!("Expected tensor");
    }
}

#[test]
fn test_binary_op_add_tensors() {
    let source = r#"
        tensor a: float32[3] = [1.0, 2.0, 3.0]
        tensor b: float32[3] = [4.0, 5.0, 6.0]
        main {
            c := a + b
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    let c = interpreter.get_variable("c").unwrap();
    if let Value::Tensor(tensor) = c {
        assert_eq!(tensor.shape().dims(), &[3]);
    } else {
        panic!("Expected tensor");
    }
}

#[test]
fn test_binary_op_add_floats() {
    let interpreter = Interpreter::new();

    let left = Value::Float(3.0);
    let right = Value::Float(4.0);

    let result = interpreter.eval_binary_op(&BinaryOp::Add, left, right).unwrap();

    if let Value::Float(f) = result {
        assert!((f - 7.0).abs() < 0.01);
    } else {
        panic!("Expected float");
    }
}

#[test]
fn test_binary_op_sub_floats() {
    let interpreter = Interpreter::new();

    let left = Value::Float(10.0);
    let right = Value::Float(3.0);

    let result = interpreter.eval_binary_op(&BinaryOp::Sub, left, right).unwrap();

    if let Value::Float(f) = result {
        assert!((f - 7.0).abs() < 0.01);
    } else {
        panic!("Expected float");
    }
}

#[test]
fn test_binary_op_mul_floats() {
    let interpreter = Interpreter::new();

    let left = Value::Float(3.0);
    let right = Value::Float(4.0);

    let result = interpreter.eval_binary_op(&BinaryOp::Mul, left, right).unwrap();

    if let Value::Float(f) = result {
        assert!((f - 12.0).abs() < 0.01);
    } else {
        panic!("Expected float");
    }
}

#[test]
fn test_binary_op_div_floats() {
    let interpreter = Interpreter::new();

    let left = Value::Float(12.0);
    let right = Value::Float(3.0);

    let result = interpreter.eval_binary_op(&BinaryOp::Div, left, right).unwrap();

    if let Value::Float(f) = result {
        assert!((f - 4.0).abs() < 0.01);
    } else {
        panic!("Expected float");
    }
}

#[test]
fn test_division_by_zero_error() {
    let interpreter = Interpreter::new();

    let left = Value::Float(10.0);
    let right = Value::Float(0.0);

    let result = interpreter.eval_binary_op(&BinaryOp::Div, left, right);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), RuntimeError::DivisionByZero));
}

#[test]
fn test_unary_op_neg_float() {
    let interpreter = Interpreter::new();

    let operand = Value::Float(5.0);
    let result = interpreter.eval_unary_op(&UnaryOp::Neg, operand).unwrap();

    if let Value::Float(f) = result {
        assert!((f + 5.0).abs() < 0.01);
    } else {
        panic!("Expected float");
    }
}

#[test]
fn test_unary_op_not_boolean() {
    let interpreter = Interpreter::new();

    let operand = Value::Boolean(true);
    let result = interpreter.eval_unary_op(&UnaryOp::Not, operand).unwrap();

    if let Value::Boolean(b) = result {
        assert!(!b);
    } else {
        panic!("Expected boolean");
    }
}

#[test]
fn test_multiple_declarations() {
    let source = r#"
        tensor w: float32[10]
        tensor b: float32[10]
        tensor x: float32[5, 10]
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    assert!(interpreter.get_variable("w").is_ok());
    assert!(interpreter.get_variable("b").is_ok());
    assert!(interpreter.get_variable("x").is_ok());
}

#[test]
fn test_main_block_execution() {
    let source = r#"
        tensor a: float32[2] = [1.0, 2.0]
        tensor b: float32[2] = [3.0, 4.0]

        main {
            sum := a + b
            diff := a - b
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter.execute(&program).unwrap();

    assert!(interpreter.get_variable("sum").is_ok());
    assert!(interpreter.get_variable("diff").is_ok());
}

#[test]
fn test_collect_scalars() {
    let interpreter = Interpreter::new();

    let elements = vec![
        TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        TensorLiteral::Scalar(ScalarLiteral::Integer(3)),
    ];

    let values = interpreter.collect_scalars(&elements).unwrap();

    assert_eq!(values.len(), 3);
    assert!((values[0] - 1.0).abs() < 0.01);
    assert!((values[1] - 2.0).abs() < 0.01);
    assert!((values[2] - 3.0).abs() < 0.01);
}

#[test]
fn test_infer_shape_1d() {
    let interpreter = Interpreter::new();

    let elements = vec![
        TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(3.0)),
    ];

    let shape = interpreter.infer_shape(&elements).unwrap();

    assert_eq!(shape, vec![3]);
}

#[test]
fn test_infer_shape_2d() {
    let interpreter = Interpreter::new();

    let elements = vec![
        TensorLiteral::Array(vec![
            TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
            TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        ]),
        TensorLiteral::Array(vec![
            TensorLiteral::Scalar(ScalarLiteral::Float(3.0)),
            TensorLiteral::Scalar(ScalarLiteral::Float(4.0)),
        ]),
    ];

    let shape = interpreter.infer_shape(&elements).unwrap();

    assert_eq!(shape, vec![2, 2]);
}

#[test]
fn test_value_as_tensor() {
    let device = MetalDevice::new().unwrap();
    let tensor = Tensor::zeros(&device, vec![2, 3]).unwrap();
    let value = Value::Tensor(tensor);

    assert!(value.as_tensor().is_ok());
}

#[test]
fn test_value_as_float() {
    let value = Value::Float(3.14);
    assert!((value.as_float().unwrap() - 3.14).abs() < 0.01);

    let value = Value::Integer(42);
    assert!((value.as_float().unwrap() - 42.0).abs() < 0.01);
}

#[test]
fn test_value_as_bool() {
    let value = Value::Boolean(true);
    assert!(value.as_bool().unwrap());
}
