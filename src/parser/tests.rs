use super::*;

#[test]
fn test_parse_tensor_decl_simple() {
    let source = "tensor w: float32[10, 20]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Tensor(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "w");
        assert_eq!(decl.tensor_type.base_type, BaseType::Float32);
        assert_eq!(decl.tensor_type.dimensions.len(), 2);
        assert_eq!(decl.tensor_type.learnable, LearnableStatus::Default);
        assert!(decl.init_expr.is_none());
    } else {
        panic!("Expected tensor declaration");
    }
}

#[test]
fn test_parse_tensor_decl_learnable() {
    let source = "tensor w: float32[10, 20] learnable";
    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Tensor(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "w");
        assert_eq!(decl.tensor_type.learnable, LearnableStatus::Learnable);
    } else {
        panic!("Expected tensor declaration");
    }
}

#[test]
fn test_parse_relation_decl() {
    let source = "relation Parent(x: entity, y: entity)";
    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Relation(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "Parent");
        assert_eq!(decl.params.len(), 2);
        assert_eq!(decl.params[0].name.as_str(), "x");
        assert_eq!(decl.params[1].name.as_str(), "y");
    } else {
        panic!("Expected relation declaration");
    }
}

#[test]
fn test_parse_relation_with_embed() {
    let source = "relation Parent(x: entity, y: entity) embed float32[64]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Relation(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "Parent");
        assert!(decl.embedding_spec.is_some());

        if let Some(embed_type) = &decl.embedding_spec {
            assert_eq!(embed_type.base_type, BaseType::Float32);
            assert_eq!(embed_type.dimensions.len(), 1);
        }
    } else {
        panic!("Expected relation declaration");
    }
}

#[test]
fn test_parse_embedding_decl() {
    let source = r#"
        embedding person_embed {
            entities: {alice, bob, charlie}
            dimension: 64
            init: xavier
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Embedding(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "person_embed");
        assert_eq!(decl.dimension, 64);
        assert_eq!(decl.init_method, InitMethod::Xavier);

        if let EntitySet::Explicit(entities) = &decl.entities {
            assert_eq!(entities.len(), 3);
            assert_eq!(entities[0].as_str(), "alice");
            assert_eq!(entities[1].as_str(), "bob");
            assert_eq!(entities[2].as_str(), "charlie");
        } else {
            panic!("Expected explicit entity set");
        }
    } else {
        panic!("Expected embedding declaration");
    }
}

#[test]
fn test_parse_function_decl() {
    let source = r#"
        function sigmoid(x: float32[?]) -> float32[?] {
            x := x
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 1);

    if let Declaration::Function(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "sigmoid");
        assert_eq!(decl.params.len(), 1);
        assert_eq!(decl.params[0].name.as_str(), "x");

        if let ReturnType::Tensor(ret_type) = &decl.return_type {
            assert_eq!(ret_type.base_type, BaseType::Float32);
        } else {
            panic!("Expected tensor return type");
        }

        assert_eq!(decl.body.len(), 1);
    } else {
        panic!("Expected function declaration");
    }
}

#[test]
fn test_parse_main_block() {
    let source = r#"
        main {
            x := y
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    assert!(program.main_block.is_some());

    if let Some(main) = program.main_block {
        assert_eq!(main.statements.len(), 1);

        if let Statement::Assignment { target, .. } = &main.statements[0] {
            assert_eq!(target.as_str(), "x");
        } else {
            panic!("Expected assignment statement");
        }
    }
}

#[test]
fn test_parse_variable_dimension() {
    let source = "tensor w: float32[n, m]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Declaration::Tensor(decl) = &program.declarations[0] {
        assert_eq!(decl.tensor_type.dimensions.len(), 2);

        if let Dimension::Variable(id) = &decl.tensor_type.dimensions[0] {
            assert_eq!(id.as_str(), "n");
        } else {
            panic!("Expected variable dimension");
        }

        if let Dimension::Variable(id) = &decl.tensor_type.dimensions[1] {
            assert_eq!(id.as_str(), "m");
        } else {
            panic!("Expected variable dimension");
        }
    }
}

#[test]
fn test_parse_dynamic_dimension() {
    let source = "tensor w: float32[?]";
    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Declaration::Tensor(decl) = &program.declarations[0] {
        assert_eq!(decl.tensor_type.dimensions.len(), 1);
        assert_eq!(decl.tensor_type.dimensions[0], Dimension::Dynamic);
    }
}

#[test]
fn test_parse_multiple_declarations() {
    let source = r#"
        tensor w: float32[10] learnable
        tensor b: float32[10] learnable
        relation Parent(x: entity, y: entity)
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    assert_eq!(program.declarations.len(), 3);
    assert!(matches!(program.declarations[0], Declaration::Tensor(_)));
    assert!(matches!(program.declarations[1], Declaration::Tensor(_)));
    assert!(matches!(program.declarations[2], Declaration::Relation(_)));
}

#[test]
fn test_parse_base_types() {
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

        if let Declaration::Tensor(decl) = &program.declarations[0] {
            assert_eq!(decl.tensor_type.base_type, expected_type);
        }
    }
}

#[test]
fn test_parse_embedding_auto() {
    let source = r#"
        embedding person_set {
            entities: auto
            dimension: 128
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Declaration::Embedding(decl) = &program.declarations[0] {
        assert_eq!(decl.name.as_str(), "person_set");
        assert_eq!(decl.dimension, 128);
        assert!(matches!(decl.entities, EntitySet::Auto));
        assert_eq!(decl.init_method, InitMethod::Random); // default
    }
}

#[test]
fn test_parse_init_methods() {
    let test_cases = vec![
        ("random", InitMethod::Random),
        ("xavier", InitMethod::Xavier),
        ("he", InitMethod::He),
        ("zeros", InitMethod::Zeros),
        ("ones", InitMethod::Ones),
    ];

    for (method, expected) in test_cases {
        let source = format!(
            r#"
            embedding e {{
                entities: auto
                dimension: 64
                init: {}
            }}
            "#,
            method
        );

        let program = TensorLogicParser::parse_program(&source).unwrap();

        if let Declaration::Embedding(decl) = &program.declarations[0] {
            assert_eq!(decl.init_method, expected);
        }
    }
}

#[test]
fn test_parse_tensor_literal_scalar() {
    let source = r#"
        tensor w: float32[1] = 3.14
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Declaration::Tensor(decl) = &program.declarations[0] {
        assert!(decl.init_expr.is_some());

        if let Some(TensorExpr::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(val)))) =
            &decl.init_expr
        {
            assert!((val - 3.14).abs() < 0.01);
        } else {
            panic!("Expected scalar literal");
        }
    }
}

#[test]
fn test_parse_assignment_statement() {
    let source = r#"
        main {
            result := x
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Some(main) = program.main_block {
        assert_eq!(main.statements.len(), 1);

        if let Statement::Assignment { target, value } = &main.statements[0] {
            assert_eq!(target.as_str(), "result");
            assert!(matches!(value, TensorExpr::Variable(_)));
        }
    }
}

#[test]
fn test_parse_binary_expression() {
    let source = r#"
        main {
            result := a + b
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Some(main) = program.main_block {
        if let Statement::Assignment { value, .. } = &main.statements[0] {
            if let TensorExpr::BinaryOp { op, left, right } = value {
                assert_eq!(*op, BinaryOp::Add);
                assert!(matches!(**left, TensorExpr::Variable(_)));
                assert!(matches!(**right, TensorExpr::Variable(_)));
            } else {
                panic!("Expected binary operation");
            }
        }
    }
}

#[test]
fn test_parse_chained_expression() {
    let source = r#"
        main {
            result := a + b * c
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Some(main) = program.main_block {
        if let Statement::Assignment { value, .. } = &main.statements[0] {
            // Parse as: (a + b) * c due to left-to-right parsing
            // Note: This is simplified precedence, not full operator precedence
            if let TensorExpr::BinaryOp { op, .. } = value {
                assert_eq!(*op, BinaryOp::Mul);
            } else {
                panic!("Expected binary operation");
            }
        }
    }
}

#[test]
fn test_parse_matmul_expression() {
    let source = r#"
        main {
            result := A @ B
        }
    "#;

    let program = TensorLogicParser::parse_program(source).unwrap();

    if let Some(main) = program.main_block {
        if let Statement::Assignment { value, .. } = &main.statements[0] {
            if let TensorExpr::BinaryOp { op, .. } = value {
                assert_eq!(*op, BinaryOp::MatMul);
            } else {
                panic!("Expected binary operation");
            }
        }
    }
}
