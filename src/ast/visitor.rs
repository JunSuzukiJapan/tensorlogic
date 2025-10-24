//! Visitor pattern for traversing AST
//!
//! Provides a flexible way to implement AST transformations,
//! analysis passes, and code generation.

use super::*;

/// AST visitor trait
pub trait Visitor: Sized {
    type Error;

    // Program structure
    fn visit_program(&mut self, program: &Program) -> Result<(), Self::Error> {
        walk_program(self, program)
    }

    fn visit_main_block(&mut self, block: &MainBlock) -> Result<(), Self::Error> {
        walk_main_block(self, block)
    }

    // Declarations
    fn visit_declaration(&mut self, decl: &Declaration) -> Result<(), Self::Error> {
        walk_declaration(self, decl)
    }

    fn visit_import_decl(&mut self, decl: &ImportDecl) -> Result<(), Self::Error> {
        walk_import_decl(self, decl)
    }

    fn visit_tensor_decl(&mut self, decl: &TensorDecl) -> Result<(), Self::Error> {
        walk_tensor_decl(self, decl)
    }

    fn visit_relation_decl(&mut self, decl: &RelationDecl) -> Result<(), Self::Error> {
        walk_relation_decl(self, decl)
    }

    fn visit_rule_decl(&mut self, decl: &RuleDecl) -> Result<(), Self::Error> {
        walk_rule_decl(self, decl)
    }

    fn visit_embedding_decl(&mut self, decl: &EmbeddingDecl) -> Result<(), Self::Error> {
        walk_embedding_decl(self, decl)
    }

    fn visit_function_decl(&mut self, decl: &FunctionDecl) -> Result<(), Self::Error> {
        walk_function_decl(self, decl)
    }

    // Expressions
    fn visit_tensor_expr(&mut self, expr: &TensorExpr) -> Result<(), Self::Error> {
        walk_tensor_expr(self, expr)
    }

    fn visit_atom(&mut self, atom: &Atom) -> Result<(), Self::Error> {
        walk_atom(self, atom)
    }

    // Statements
    fn visit_statement(&mut self, stmt: &Statement) -> Result<(), Self::Error> {
        walk_statement(self, stmt)
    }

    // Constraints
    fn visit_constraint(&mut self, constraint: &Constraint) -> Result<(), Self::Error> {
        walk_constraint(self, constraint)
    }

    // Identifiers
    fn visit_identifier(&mut self, _id: &Identifier) -> Result<(), Self::Error> {
        Ok(())
    }
}

// Default walking functions

pub fn walk_program<V: Visitor>(visitor: &mut V, program: &Program) -> Result<(), V::Error> {
    for decl in &program.declarations {
        visitor.visit_declaration(decl)?;
    }
    if let Some(main) = &program.main_block {
        visitor.visit_main_block(main)?;
    }
    Ok(())
}

pub fn walk_main_block<V: Visitor>(visitor: &mut V, block: &MainBlock) -> Result<(), V::Error> {
    for stmt in &block.statements {
        visitor.visit_statement(stmt)?;
    }
    Ok(())
}

pub fn walk_declaration<V: Visitor>(visitor: &mut V, decl: &Declaration) -> Result<(), V::Error> {
    match decl {
        Declaration::Import(d) => visitor.visit_import_decl(d),
        Declaration::Entity(_d) => {
            // Entity declarations don't need visiting for now
            Ok(())
        },
        Declaration::Tensor(d) => visitor.visit_tensor_decl(d),
        Declaration::Relation(d) => visitor.visit_relation_decl(d),
        Declaration::Rule(d) => visitor.visit_rule_decl(d),
        Declaration::Embedding(d) => visitor.visit_embedding_decl(d),
        Declaration::RelationEmbedding(_d) => {
            // Relation embedding declarations don't need visiting for now
            Ok(())
        },
        Declaration::Function(d) => visitor.visit_function_decl(d),
    }
}

pub fn walk_import_decl<V: Visitor>(_visitor: &mut V, _decl: &ImportDecl) -> Result<(), V::Error> {
    // Import declarations have no sub-nodes to visit
    Ok(())
}

pub fn walk_tensor_decl<V: Visitor>(visitor: &mut V, decl: &TensorDecl) -> Result<(), V::Error> {
    visitor.visit_identifier(&decl.name)?;
    if let Some(expr) = &decl.init_expr {
        visitor.visit_tensor_expr(expr)?;
    }
    Ok(())
}

pub fn walk_relation_decl<V: Visitor>(
    visitor: &mut V,
    decl: &RelationDecl,
) -> Result<(), V::Error> {
    visitor.visit_identifier(&decl.name)?;
    for param in &decl.params {
        visitor.visit_identifier(&param.name)?;
    }
    Ok(())
}

pub fn walk_rule_decl<V: Visitor>(visitor: &mut V, decl: &RuleDecl) -> Result<(), V::Error> {
    match &decl.head {
        RuleHead::Atom(atom) => visitor.visit_atom(atom)?,
        RuleHead::Equation(eq) => {
            visitor.visit_tensor_expr(&eq.left)?;
            visitor.visit_tensor_expr(&eq.right)?;
        }
    }

    for term in &decl.body {
        match term {
            BodyTerm::Atom(atom) => visitor.visit_atom(atom)?,
            BodyTerm::Equation(eq) => {
                visitor.visit_tensor_expr(&eq.left)?;
                visitor.visit_tensor_expr(&eq.right)?;
            }
            BodyTerm::Constraint(c) => visitor.visit_constraint(c)?,
        }
    }

    Ok(())
}

pub fn walk_embedding_decl<V: Visitor>(
    visitor: &mut V,
    decl: &EmbeddingDecl,
) -> Result<(), V::Error> {
    visitor.visit_identifier(&decl.name)?;
    if let EntitySet::Explicit(ids) = &decl.entities {
        for id in ids {
            visitor.visit_identifier(id)?;
        }
    }
    Ok(())
}

pub fn walk_function_decl<V: Visitor>(
    visitor: &mut V,
    decl: &FunctionDecl,
) -> Result<(), V::Error> {
    visitor.visit_identifier(&decl.name)?;
    for param in &decl.params {
        visitor.visit_identifier(&param.name)?;
    }
    for stmt in &decl.body {
        visitor.visit_statement(stmt)?;
    }
    Ok(())
}

pub fn walk_tensor_expr<V: Visitor>(visitor: &mut V, expr: &TensorExpr) -> Result<(), V::Error> {
    match expr {
        TensorExpr::Variable(id) => visitor.visit_identifier(id),
        TensorExpr::Literal(_) => Ok(()),
        TensorExpr::BinaryOp { left, right, .. } => {
            visitor.visit_tensor_expr(left)?;
            visitor.visit_tensor_expr(right)
        }
        TensorExpr::UnaryOp { operand, .. } => visitor.visit_tensor_expr(operand),
        TensorExpr::EinSum { tensors, .. } => {
            for tensor in tensors {
                visitor.visit_tensor_expr(tensor)?;
            }
            Ok(())
        }
        TensorExpr::FunctionCall { name, args } => {
            visitor.visit_identifier(name)?;
            for arg in args {
                visitor.visit_tensor_expr(arg)?;
            }
            Ok(())
        }
        TensorExpr::TensorIndex { tensor, indices } => {
            visitor.visit_tensor_expr(tensor)?;
            for idx_expr in indices {
                if let crate::ast::IndexExpr::Var(var) = idx_expr {
                    visitor.visit_identifier(var)?;
                }
            }
            Ok(())
        }
        TensorExpr::EmbeddingLookup { embedding, entity } => {
            visitor.visit_identifier(embedding)?;
            if let EntityRef::Variable(id) = entity {
                visitor.visit_identifier(id)?;
            }
            Ok(())
        }
        TensorExpr::PythonCall { args, .. } => {
            for arg in args {
                visitor.visit_tensor_expr(arg)?;
            }
            Ok(())
        }
    }
}

pub fn walk_atom<V: Visitor>(visitor: &mut V, atom: &Atom) -> Result<(), V::Error> {
    visitor.visit_identifier(&atom.predicate)?;
    for term in &atom.terms {
        match term {
            Term::Variable(id) => visitor.visit_identifier(id)?,
            Term::Constant(_) => {}
            Term::Tensor(expr) => visitor.visit_tensor_expr(expr)?,
        }
    }
    Ok(())
}

pub fn walk_statement<V: Visitor>(visitor: &mut V, stmt: &Statement) -> Result<(), V::Error> {
    match stmt {
        Statement::TensorDecl(decl) => {
            visitor.visit_identifier(&decl.name)?;
            if let Some(init_expr) = &decl.init_expr {
                visitor.visit_tensor_expr(init_expr)?;
            }
            Ok(())
        }
        Statement::Let { target, value } | Statement::Assignment { target, value } => {
            visitor.visit_identifier(target)?;
            visitor.visit_tensor_expr(value)
        }
        Statement::Equation(eq) => {
            visitor.visit_tensor_expr(&eq.left)?;
            visitor.visit_tensor_expr(&eq.right)
        }
        Statement::FunctionCall { name, args } => {
            visitor.visit_identifier(name)?;
            for arg in args {
                visitor.visit_tensor_expr(arg)?;
            }
            Ok(())
        }
        Statement::FactAssertion { atom } => {
            visitor.visit_atom(atom)?;
            Ok(())
        }
        Statement::Query { atom, constraints } => {
            visitor.visit_atom(atom)?;
            for constraint in constraints {
                visitor.visit_constraint(constraint)?;
            }
            Ok(())
        }
        Statement::Inference { query, .. } => visitor.visit_statement(query),
        Statement::InferenceBlock { items } => {
            for (_, query) in items {
                visitor.visit_statement(query)?;
            }
            Ok(())
        }
        Statement::Learning(spec) => {
            visitor.visit_tensor_expr(&spec.objective)?;
            Ok(())
        }
        Statement::WithBlock { statements, .. } => {
            for stmt in statements {
                visitor.visit_statement(stmt)?;
            }
            Ok(())
        }
        Statement::ControlFlow(cf) => match cf {
            ControlFlow::If {
                condition,
                then_block,
                else_block,
            } => {
                if let Condition::Constraint(c) = condition {
                    visitor.visit_constraint(c)?;
                } else if let Condition::Tensor(expr) = condition {
                    visitor.visit_tensor_expr(expr)?;
                }
                for stmt in then_block {
                    visitor.visit_statement(stmt)?;
                }
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        visitor.visit_statement(stmt)?;
                    }
                }
                Ok(())
            }
            ControlFlow::For {
                variable,
                iterable,
                body,
            } => {
                visitor.visit_identifier(variable)?;
                if let Iterable::Tensor(expr) = iterable {
                    visitor.visit_tensor_expr(expr)?;
                }
                for stmt in body {
                    visitor.visit_statement(stmt)?;
                }
                Ok(())
            }
            ControlFlow::While { condition, body } => {
                if let Condition::Constraint(c) = condition {
                    visitor.visit_constraint(c)?;
                } else if let Condition::Tensor(expr) = condition {
                    visitor.visit_tensor_expr(expr)?;
                }
                for stmt in body {
                    visitor.visit_statement(stmt)?;
                }
                Ok(())
            }
            ControlFlow::Loop { body } => {
                for stmt in body {
                    visitor.visit_statement(stmt)?;
                }
                Ok(())
            }
        },
        Statement::PythonImport { .. } => {
            // No sub-expressions to visit
            Ok(())
        }
        Statement::Break => {
            // No sub-expressions to visit
            Ok(())
        }
        Statement::Return { value } => {
            // Visit return value if present
            if let Some(expr) = value {
                visitor.visit_tensor_expr(expr)?;
            }
            Ok(())
        }
    }
}

pub fn walk_constraint<V: Visitor>(
    visitor: &mut V,
    constraint: &Constraint,
) -> Result<(), V::Error> {
    match constraint {
        Constraint::Comparison { left, right, .. } => {
            visitor.visit_tensor_expr(left)?;
            visitor.visit_tensor_expr(right)
        }
        Constraint::Shape { tensor, .. } => visitor.visit_tensor_expr(tensor),
        Constraint::Rank { tensor, .. } => visitor.visit_tensor_expr(tensor),
        Constraint::Norm { tensor, .. } => visitor.visit_tensor_expr(tensor),
        Constraint::Not(c) => visitor.visit_constraint(c),
        Constraint::And(left, right) | Constraint::Or(left, right) => {
            visitor.visit_constraint(left)?;
            visitor.visit_constraint(right)
        }
    }
}

/// Mutable visitor for AST transformations
pub trait VisitorMut: Sized {
    type Error;

    fn visit_program_mut(&mut self, program: &mut Program) -> Result<(), Self::Error> {
        walk_program_mut(self, program)
    }

    fn visit_tensor_expr_mut(&mut self, expr: &mut TensorExpr) -> Result<(), Self::Error> {
        walk_tensor_expr_mut(self, expr)
    }

    fn visit_statement_mut(&mut self, stmt: &mut Statement) -> Result<(), Self::Error> {
        walk_statement_mut(self, stmt)
    }
}

pub fn walk_program_mut<V: VisitorMut>(
    visitor: &mut V,
    program: &mut Program,
) -> Result<(), V::Error> {
    for decl in &mut program.declarations {
        if let Declaration::Function(func) = decl {
            for stmt in &mut func.body {
                visitor.visit_statement_mut(stmt)?;
            }
        }
    }
    if let Some(main) = &mut program.main_block {
        for stmt in &mut main.statements {
            visitor.visit_statement_mut(stmt)?;
        }
    }
    Ok(())
}

pub fn walk_tensor_expr_mut<V: VisitorMut>(
    visitor: &mut V,
    expr: &mut TensorExpr,
) -> Result<(), V::Error> {
    match expr {
        TensorExpr::BinaryOp { left, right, .. } => {
            visitor.visit_tensor_expr_mut(left)?;
            visitor.visit_tensor_expr_mut(right)
        }
        TensorExpr::UnaryOp { operand, .. } => visitor.visit_tensor_expr_mut(operand),
        TensorExpr::EinSum { tensors, .. } => {
            for tensor in tensors {
                visitor.visit_tensor_expr_mut(tensor)?;
            }
            Ok(())
        }
        TensorExpr::FunctionCall { args, .. } => {
            for arg in args {
                visitor.visit_tensor_expr_mut(arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub fn walk_statement_mut<V: VisitorMut>(
    visitor: &mut V,
    stmt: &mut Statement,
) -> Result<(), V::Error> {
    match stmt {
        Statement::Let { value, .. } | Statement::Assignment { value, .. } => visitor.visit_tensor_expr_mut(value),
        Statement::Equation(eq) => {
            visitor.visit_tensor_expr_mut(&mut eq.left)?;
            visitor.visit_tensor_expr_mut(&mut eq.right)
        }
        Statement::Learning(spec) => visitor.visit_tensor_expr_mut(&mut spec.objective),
        Statement::WithBlock { statements, .. } => {
            for stmt in statements {
                visitor.visit_statement_mut(stmt)?;
            }
            Ok(())
        }
        Statement::ControlFlow(cf) => match cf {
            ControlFlow::If {
                then_block,
                else_block,
                ..
            } => {
                for stmt in then_block {
                    visitor.visit_statement_mut(stmt)?;
                }
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        visitor.visit_statement_mut(stmt)?;
                    }
                }
                Ok(())
            }
            ControlFlow::For { body, .. } | ControlFlow::While { body, .. } | ControlFlow::Loop { body } => {
                for stmt in body {
                    visitor.visit_statement_mut(stmt)?;
                }
                Ok(())
            }
        },
        _ => Ok(()),
    }
}
