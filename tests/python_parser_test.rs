use tensorlogic::parser::TensorLogicParser;
use tensorlogic::ast::{Statement, TensorExpr};

#[test]
fn test_parse_python_import() {
    let source = r#"
main {
    python import numpy as np
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    assert!(program.main_block.is_some());

    let main_block = program.main_block.unwrap();
    assert_eq!(main_block.statements.len(), 1);

    match &main_block.statements[0] {
        Statement::PythonImport { module, alias } => {
            assert_eq!(module, "numpy");
            assert_eq!(alias.as_ref().unwrap(), "np");
        }
        _ => panic!("Expected PythonImport statement"),
    }
}

#[test]
fn test_parse_python_import_without_alias() {
    let source = r#"
main {
    python import torch
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let main_block = program.main_block.unwrap();

    match &main_block.statements[0] {
        Statement::PythonImport { module, alias } => {
            assert_eq!(module, "torch");
            assert!(alias.is_none());
        }
        _ => panic!("Expected PythonImport statement"),
    }
}

#[test]
fn test_parse_python_call() {
    let source = r#"
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[1] = python.call("np.sum", x)
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let main_block = program.main_block.unwrap();
    assert_eq!(main_block.statements.len(), 2);

    // Check second statement (tensor y declaration)
    match &main_block.statements[1] {
        Statement::TensorDecl(decl) => {
            assert_eq!(decl.name.as_str(), "y");

            // Check init expression is a Python call
            if let Some(TensorExpr::PythonCall { function, args }) = &decl.init_expr {
                assert_eq!(function, "np.sum");
                assert_eq!(args.len(), 1);

                // First argument should be variable x
                match &args[0] {
                    TensorExpr::Variable(id) => assert_eq!(id.as_str(), "x"),
                    _ => panic!("Expected Variable expression"),
                }
            } else {
                panic!("Expected PythonCall expression");
            }
        }
        _ => panic!("Expected TensorDecl statement"),
    }
}

#[test]
fn test_parse_python_call_multiple_args() {
    let source = r#"
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor y: float16[3] = [4.0, 5.0, 6.0]
    tensor z: float16[3] = python.call("np.add", x, y)
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let main_block = program.main_block.unwrap();

    match &main_block.statements[2] {
        Statement::TensorDecl(decl) => {
            if let Some(TensorExpr::PythonCall { function, args }) = &decl.init_expr {
                assert_eq!(function, "np.add");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected PythonCall expression");
            }
        }
        _ => panic!("Expected TensorDecl statement"),
    }
}

#[test]
fn test_parse_combined_python_integration() {
    let source = r#"
main {
    python import numpy as np
    python import torch

    tensor x: float16[3] = [1.0, 2.0, 3.0]
    tensor sum_x: float16[1] = python.call("np.sum", x)
    tensor doubled: float16[3] = python.call("torch.mul", x, 2.0)

    print("Python integration complete")
}
"#;

    let program = TensorLogicParser::parse_program(source).unwrap();
    let main_block = program.main_block.unwrap();

    // Should have 6 statements
    assert_eq!(main_block.statements.len(), 6);

    // First two should be Python imports
    matches!(&main_block.statements[0], Statement::PythonImport { .. });
    matches!(&main_block.statements[1], Statement::PythonImport { .. });
}
