//! Tests for AST construction and visitor pattern

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_tensor_decl_creation() {
        let decl = TensorDecl {
            name: Identifier::new("w"),
            tensor_type: TensorType::learnable_float16(vec![10, 20]),
            init_expr: None,
        };

        assert_eq!(decl.name.as_str(), "w");
        assert_eq!(decl.tensor_type.dimensions.len(), 2);
        assert_eq!(decl.tensor_type.learnable, LearnableStatus::Learnable);
    }

    #[test]
    fn test_binary_expr_creation() {
        let expr = TensorExpr::binary(
            BinaryOp::Add,
            TensorExpr::var("x"),
            TensorExpr::var("y"),
        );

        match expr {
            TensorExpr::BinaryOp { op, .. } => {
                assert_eq!(op, BinaryOp::Add);
            }
            _ => panic!("Expected BinaryOp"),
        }
    }

    #[test]
    fn test_atom_creation() {
        let atom = Atom::new(
            "Parent",
            vec![
                Term::Variable(Identifier::new("x")),
                Term::Variable(Identifier::new("y")),
            ],
        );

        assert_eq!(atom.predicate.as_str(), "Parent");
        assert_eq!(atom.terms.len(), 2);
    }

    #[test]
    fn test_rule_creation() {
        let rule = RuleDecl {
            head: RuleHead::Atom(Atom::new(
                "Ancestor",
                vec![
                    Term::Variable(Identifier::new("x")),
                    Term::Variable(Identifier::new("z")),
                ],
            )),
            body: vec![
                BodyTerm::Atom(Atom::new(
                    "Parent",
                    vec![
                        Term::Variable(Identifier::new("x")),
                        Term::Variable(Identifier::new("y")),
                    ],
                )),
                BodyTerm::Atom(Atom::new(
                    "Ancestor",
                    vec![
                        Term::Variable(Identifier::new("y")),
                        Term::Variable(Identifier::new("z")),
                    ],
                )),
            ],
        };

        assert_eq!(rule.body.len(), 2);
    }

    #[test]
    fn test_embedding_decl() {
        let decl = EmbeddingDecl {
            name: Identifier::new("person"),
            entities: EntitySet::Explicit(vec![
                Identifier::new("alice"),
                Identifier::new("bob"),
                Identifier::new("charlie"),
            ]),
            dimension: 64,
            init_method: InitMethod::Xavier,
        };

        assert_eq!(decl.dimension, 64);
        assert_eq!(decl.init_method, InitMethod::Xavier);

        if let EntitySet::Explicit(entities) = &decl.entities {
            assert_eq!(entities.len(), 3);
        } else {
            panic!("Expected Explicit entity set");
        }
    }

    #[test]
    fn test_learning_spec() {
        let spec = LearningSpec {
            objective: TensorExpr::binary(
                BinaryOp::Add,
                TensorExpr::var("loss1"),
                TensorExpr::var("loss2"),
            ),
            optimizer: OptimizerSpec {
                name: "adam".to_string(),
                params: vec![("lr".to_string(), 0.001)],
            },
            epochs: 1000,
            scheduler: None,
        };

        assert_eq!(spec.epochs, 1000);
        assert_eq!(spec.optimizer.name, "adam");
        assert_eq!(spec.optimizer.params[0].0, "lr");
    }

    #[test]
    fn test_constraint_creation() {
        let constraint = Constraint::Comparison {
            op: CompOp::Gt,
            left: TensorExpr::var("x"),
            right: TensorExpr::scalar(0.0),
        };

        match constraint {
            Constraint::Comparison { op, .. } => {
                assert_eq!(op, CompOp::Gt);
            }
            _ => panic!("Expected Comparison constraint"),
        }
    }

    #[test]
    fn test_tensor_equation() {
        let eq = TensorEquation {
            left: TensorExpr::var("y"),
            right: TensorExpr::binary(
                BinaryOp::MatMul,
                TensorExpr::var("x"),
                TensorExpr::var("w"),
            ),
            eq_type: EquationType::Assign,
        };

        assert_eq!(eq.eq_type, EquationType::Assign);
    }

    #[test]
    fn test_function_decl() {
        let func = FunctionDecl {
            name: Identifier::new("sigmoid"),
            params: vec![Param {
                name: Identifier::new("x"),
                entity_type: EntityType::Tensor(TensorType::float16(vec![])),
            }],
            return_type: ReturnType::Tensor(TensorType::float16(vec![])),
            body: vec![Statement::Assignment {
                target: Identifier::new("result"),
                value: TensorExpr::var("x"),
            }],
        };

        assert_eq!(func.name.as_str(), "sigmoid");
        assert_eq!(func.params.len(), 1);
    }

    #[test]
    fn test_control_flow_if() {
        let stmt = Statement::ControlFlow(ControlFlow::If {
            condition: Condition::Constraint(Constraint::Comparison {
                op: CompOp::Gt,
                left: TensorExpr::var("x"),
                right: TensorExpr::scalar(0.0),
            }),
            then_block: vec![Statement::Assignment {
                target: Identifier::new("y"),
                value: TensorExpr::var("x"),
            }],
            else_block: None,
        });

        match stmt {
            Statement::ControlFlow(ControlFlow::If { then_block, .. }) => {
                assert_eq!(then_block.len(), 1);
            }
            _ => panic!("Expected If statement"),
        }
    }

    // Visitor pattern tests
    struct IdentifierCounter {
        count: usize,
    }

    impl Visitor for IdentifierCounter {
        type Error = ();

        fn visit_identifier(&mut self, _id: &Identifier) -> Result<(), ()> {
            self.count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_visitor_pattern() {
        let mut counter = IdentifierCounter { count: 0 };

        let expr = TensorExpr::binary(
            BinaryOp::Add,
            TensorExpr::var("x"),
            TensorExpr::var("y"),
        );

        counter.visit_tensor_expr(&expr).unwrap();
        assert_eq!(counter.count, 2); // x and y
    }

    #[test]
    fn test_program_visitor() {
        let mut counter = IdentifierCounter { count: 0 };

        let program = Program {
            declarations: vec![Declaration::Tensor(TensorDecl {
                name: Identifier::new("w"),
                tensor_type: TensorType::float16(vec![10]),
                init_expr: Some(TensorExpr::var("x")),
            })],
            main_block: None,
        };

        counter.visit_program(&program).unwrap();
        assert_eq!(counter.count, 2); // w and x
    }
}
