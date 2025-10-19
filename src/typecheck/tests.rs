use super::*;
use crate::parser::TensorLogicParser;

#[test]
fn test_type_checker_creation() {
    let checker = TypeChecker::new();
    assert!(checker.env.variables.is_empty());
}

#[test]
fn test_simple_tensor_decl() {
    let source = "tensor w: float32[10, 20]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    // Verify variable was added
    let w_type = checker.env.get_variable("w").unwrap();
    assert_eq!(w_type.base_type, BaseType::Float32);
    assert_eq!(w_type.dimensions.len(), 2);
}

#[test]
fn test_learnable_tensor_decl() {
    let source = "tensor w: float32[10, 20] learnable";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    let w_type = checker.env.get_variable("w").unwrap();
    assert_eq!(w_type.learnable, LearnableStatus::Learnable);
}

#[test]
fn test_duplicate_declaration() {
    let source = r#"
        tensor w: float32[10]
        tensor w: float32[20]
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    let result = checker.check_program(&program);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TypeError::DuplicateDeclaration(_)));
}

#[test]
fn test_variable_dimension() {
    let source = "tensor w: float32[n, m]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    // Verify dimension variables were added
    assert!(checker.env.has_dimension_var("n"));
    assert!(checker.env.has_dimension_var("m"));
}

#[test]
fn test_dynamic_dimension() {
    let source = "tensor w: float32[?]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    let w_type = checker.env.get_variable("w").unwrap();
    assert_eq!(w_type.dimensions.len(), 1);
    assert!(matches!(w_type.dimensions[0], Dimension::Dynamic));
}

#[test]
fn test_relation_decl() {
    let source = "relation Parent(x: entity, y: entity)";
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    let parent_params = checker.env.get_relation("Parent").unwrap();
    assert_eq!(parent_params.len(), 2);
    assert!(matches!(parent_params[0], EntityType::Entity));
}

#[test]
fn test_function_decl() {
    let source = r#"
        function sigmoid(x: float32[?]) -> float32[?] {
            y := x
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    let (params, return_type) = checker.env.get_function("sigmoid").unwrap();
    assert_eq!(params.len(), 1);
    assert!(return_type.is_some());
}

#[test]
fn test_assignment_statement() {
    let source = r#"
        tensor x: float32[10]
        main {
            y := x
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    assert!(checker.check_program(&program).is_ok());

    // Verify both variables exist
    assert!(checker.env.get_variable("x").is_ok());
    assert!(checker.env.get_variable("y").is_ok());
}

#[test]
fn test_undefined_variable() {
    let source = r#"
        main {
            y := x
        }
    "#;
    let program = TensorLogicParser::parse_program(source).unwrap();

    let mut checker = TypeChecker::new();
    let result = checker.check_program(&program);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TypeError::UndefinedVariable(_)));
}

#[test]
fn test_literal_type_inference_scalar() {
    let checker = TypeChecker::new();

    let float_lit = TensorLiteral::Scalar(ScalarLiteral::Float(3.14));
    let float_type = checker.infer_literal_type(&float_lit).unwrap();
    assert_eq!(float_type.base_type, BaseType::Float32);
    assert_eq!(float_type.rank(), 0);

    let int_lit = TensorLiteral::Scalar(ScalarLiteral::Integer(42));
    let int_type = checker.infer_literal_type(&int_lit).unwrap();
    assert_eq!(int_type.base_type, BaseType::Int32);

    let bool_lit = TensorLiteral::Scalar(ScalarLiteral::Boolean(true));
    let bool_type = checker.infer_literal_type(&bool_lit).unwrap();
    assert_eq!(bool_type.base_type, BaseType::Bool);
}

#[test]
fn test_literal_type_inference_array() {
    let checker = TypeChecker::new();

    // [1.0, 2.0, 3.0]
    let array_lit = TensorLiteral::Array(vec![
        TensorLiteral::Scalar(ScalarLiteral::Float(1.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(2.0)),
        TensorLiteral::Scalar(ScalarLiteral::Float(3.0)),
    ]);

    let array_type = checker.infer_literal_type(&array_lit).unwrap();
    assert_eq!(array_type.base_type, BaseType::Float32);
    assert_eq!(array_type.rank(), 1);
    assert_eq!(array_type.dimensions[0], Dimension::Fixed(3));
}

#[test]
fn test_binary_op_add() {
    let mut checker = TypeChecker::new();

    // Add two float32[10] tensors
    checker
        .env
        .add_variable(
            "x".to_string(),
            TensorTypeInfo::new(BaseType::Float32, vec![Dimension::Fixed(10)]),
        )
        .unwrap();

    checker
        .env
        .add_variable(
            "y".to_string(),
            TensorTypeInfo::new(BaseType::Float32, vec![Dimension::Fixed(10)]),
        )
        .unwrap();

    let x_type = checker.env.get_variable("x").unwrap().clone();
    let y_type = checker.env.get_variable("y").unwrap().clone();

    let result_type = checker
        .infer_binary_op_type(&BinaryOp::Add, &x_type, &y_type)
        .unwrap();

    assert_eq!(result_type.base_type, BaseType::Float32);
    assert_eq!(result_type.dimensions, vec![Dimension::Fixed(10)]);
}

#[test]
fn test_binary_op_dimension_mismatch() {
    let checker = TypeChecker::new();

    let x_type = TensorTypeInfo::new(BaseType::Float32, vec![Dimension::Fixed(10)]);
    let y_type = TensorTypeInfo::new(BaseType::Float32, vec![Dimension::Fixed(20)]);

    let result = checker.infer_binary_op_type(&BinaryOp::Add, &x_type, &y_type);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TypeError::DimensionMismatch { .. }));
}

#[test]
fn test_binary_op_base_type_mismatch() {
    let checker = TypeChecker::new();

    let x_type = TensorTypeInfo::new(BaseType::Float32, vec![Dimension::Fixed(10)]);
    let y_type = TensorTypeInfo::new(BaseType::Int32, vec![Dimension::Fixed(10)]);

    let result = checker.infer_binary_op_type(&BinaryOp::Add, &x_type, &y_type);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TypeError::BaseTypeMismatch { .. }));
}

#[test]
fn test_matmul_type_inference() {
    let checker = TypeChecker::new();

    // [M, K] @ [K, N] -> [M, N]
    let left_type = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Fixed(10), Dimension::Fixed(20)],
    );
    let right_type = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Fixed(20), Dimension::Fixed(30)],
    );

    let result_type = checker
        .infer_binary_op_type(&BinaryOp::MatMul, &left_type, &right_type)
        .unwrap();

    assert_eq!(result_type.base_type, BaseType::Float32);
    assert_eq!(result_type.rank(), 2);
    assert_eq!(result_type.dimensions[0], Dimension::Fixed(10));
    assert_eq!(result_type.dimensions[1], Dimension::Fixed(30));
}

#[test]
fn test_unary_op_transpose() {
    let checker = TypeChecker::new();

    let operand_type = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Fixed(10), Dimension::Fixed(20)],
    );

    let result_type = checker
        .infer_unary_op_type(&UnaryOp::Transpose, &operand_type)
        .unwrap();

    assert_eq!(result_type.base_type, BaseType::Float32);
    assert_eq!(result_type.dimensions[0], Dimension::Fixed(20));
    assert_eq!(result_type.dimensions[1], Dimension::Fixed(10));
}

#[test]
fn test_dimension_matching() {
    let type1 = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Fixed(10), Dimension::Dynamic],
    );

    let type2 = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Fixed(10), Dimension::Fixed(20)],
    );

    // Dynamic matches any
    assert!(type1.dimensions_match(&type2.dimensions));
    assert!(type2.dimensions_match(&type1.dimensions));
}

#[test]
fn test_variable_dimension_matching() {
    let type1 = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Variable(Identifier::new("n"))],
    );

    let type2 = TensorTypeInfo::new(
        BaseType::Float32,
        vec![Dimension::Variable(Identifier::new("n"))],
    );

    assert!(type1.dimensions_match(&type2.dimensions));
}

#[test]
fn test_multiple_base_types() {
    let test_cases = vec![
        ("tensor a: float32[1]", BaseType::Float32),
        ("tensor b: float64[1]", BaseType::Float64),
        ("tensor c: int32[1]", BaseType::Int32),
        ("tensor d: int64[1]", BaseType::Int64),
        ("tensor e: bool[1]", BaseType::Bool),
        ("tensor f: complex64[1]", BaseType::Complex64),
    ];

    for (source, expected_type) in test_cases {
        let program = TensorLogicParser::parse_program(source).unwrap();
        let mut checker = TypeChecker::new();
        checker.check_program(&program).unwrap();

        if let Declaration::Tensor(decl) = &program.declarations[0] {
            let var_type = checker.env.get_variable(decl.name.as_str()).unwrap();
            assert_eq!(var_type.base_type, expected_type);
        }
    }
}
