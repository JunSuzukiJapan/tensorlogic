//! Expression and statement evaluation logic for TensorLogic interpreter
//!
//! This module contains the core evaluation methods that execute TensorLogic code.

use super::*;
use crate::ast::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn execute_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
        match stmt {
            Statement::TensorDecl(decl) => {
                // Handle tensor declaration in main block
                if let Some(init_expr) = &decl.init_expr {
                    let value = self.eval_expr(init_expr)?;
                    self.env
                        .set_variable(decl.name.as_str().to_string(), value);
                } else {
                    // No initializer - create uninitialized tensor (would need default value)
                    return Err(RuntimeError::TypeError(
                        "Tensor declarations in main block must have initializers".to_string(),
                    ));
                }
                Ok(())
            }
            Statement::Let { target, value } => {
                let evaluated_value = self.eval_expr(value)?;

                // Check if we're inside a function call
                if let Some(frame) = self.call_stack.last_mut() {
                    // Inside function: declare in local scope
                    frame.local_vars.insert(target.as_str().to_string(), evaluated_value);
                    Ok(())
                } else {
                    // Global scope: declare in environment
                    self.env
                        .declare_variable(target.as_str().to_string(), evaluated_value)?;
                    Ok(())
                }
            }
            Statement::Assignment { target, value } => {
                let evaluated_value = self.eval_expr(value)?;

                // Check if we're inside a function call
                if let Some(frame) = self.call_stack.last_mut() {
                    // Inside function: set in local scope
                    frame.local_vars.insert(target.as_str().to_string(), evaluated_value);
                } else {
                    // Global scope: assignment (:=) auto-declares if variable doesn't exist
                    if self.env.has_variable(target.as_str()) {
                        self.env.set_variable(target.as_str().to_string(), evaluated_value)?;
                    } else {
                        self.env.declare_variable(target.as_str().to_string(), evaluated_value)?;
                    }
                }
                Ok(())
            }
            Statement::Equation(eq) => {
                use crate::ast::EquationType;

                match eq.eq_type {
                    EquationType::Assign => {
                        // := is assignment with auto-declaration
                        // Left side must be a simple identifier
                        if let TensorExpr::Variable(var_name) = &eq.left {
                            let value = self.eval_expr(&eq.right)?;

                            // Check if we're inside a function call
                            if let Some(frame) = self.call_stack.last_mut() {
                                // Inside function: update local variable
                                frame.local_vars.insert(var_name.as_str().to_string(), value);
                            } else {
                                // Global scope: try to set existing variable, if it doesn't exist, declare it
                                if self.env.has_variable(var_name.as_str()) {
                                    self.env.set_variable(var_name.as_str().to_string(), value)?;
                                } else {
                                    self.env.declare_variable(var_name.as_str().to_string(), value)?;
                                }
                            }
                            Ok(())
                        } else {
                            Err(RuntimeError::TypeError(
                                "Left side of := must be a variable name".to_string()
                            ))
                        }
                    }
                    _ => {
                        // For other equation types (=, ~), just execute both sides
                        let _left = self.eval_expr(&eq.left)?;
                        let _right = self.eval_expr(&eq.right)?;
                        // In a full implementation, this would perform unification or constraint solving
                        Ok(())
                    }
                }
            }
            Statement::FunctionCall { name, args } => {
                // Handle function calls as statements (e.g., print)
                if name.as_str() == "print" {
                    // Special handling for print
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            print!(" ");
                        }
                        let val = self.eval_expr(arg)?;
                        print!("{}", val);
                    }
                    println!();
                    Ok(())
                } else {
                    // Other function calls - evaluate and discard result
                    self.eval_function_call(name, args)?;
                    Ok(())
                }
            }
            Statement::ControlFlow(cf) => match cf {
                ControlFlow::If {
                    condition,
                    then_block,
                    else_block,
                } => {
                    // Evaluate condition
                    let condition_result = match condition {
                        Condition::Constraint(c) => self.eval_constraint(c)?,
                        Condition::Tensor(expr) => {
                            let val = self.eval_expr(expr)?;
                            val.as_bool()?
                        }
                    };

                    // Execute appropriate block
                    if condition_result {
                        for stmt in then_block {
                            // Propagate return statements upward
                            let result = self.execute_statement(stmt);
                            if let Err(RuntimeError::ReturnValue(_)) = result {
                                return result;
                            }
                            result?;
                        }
                    } else if let Some(else_stmts) = else_block {
                        for stmt in else_stmts {
                            // Propagate return statements upward
                            let result = self.execute_statement(stmt);
                            if let Err(RuntimeError::ReturnValue(_)) = result {
                                return result;
                            }
                            result?;
                        }
                    }
                    Ok(())
                }
                ControlFlow::For {
                    variable,
                    iterable,
                    body,
                } => {
                    // Evaluate iterable
                    let items = match iterable {
                        Iterable::Range(n) => {
                            // Create range 0..n
                            (0..*n).map(|i| Value::Integer(i as i64)).collect::<Vec<_>>()
                        }
                        Iterable::Tensor(expr) => {
                            // Iterate over tensor elements
                            let tensor_val = self.eval_expr(expr)?;
                            let tensor = tensor_val.as_tensor()?;
                            let data = tensor.to_vec();
                            data.iter().map(|&v| Value::Float(v.to_f32() as f64)).collect()
                        }
                        Iterable::EntitySet(entity_set) => {
                            // Get entities from set
                            match entity_set {
                                EntitySet::Auto => {
                                    return Err(RuntimeError::InvalidOperation(
                                        "Cannot iterate over 'auto' entity set".to_string()
                                    ));
                                }
                                EntitySet::Explicit(entities) => {
                                    entities.iter()
                                        .map(|id| Value::String(id.as_str().to_string()))
                                        .collect()
                                }
                                EntitySet::Type(type_name) => {
                                    // Get entities from entity registry
                                    let type_name_str = type_name.as_str();
                                    if let Some(type_info) = self.entity_registry.get_type_info(type_name_str) {
                                        type_info.all_entities().iter()
                                            .map(|name| Value::String(name.clone()))
                                            .collect()
                                    } else {
                                        return Err(RuntimeError::InvalidOperation(
                                            format!("Entity type '{}' not found", type_name_str)
                                        ));
                                    }
                                }
                            }
                        }
                    };

                    // Execute body for each item
                    let mut should_break = false;
                    for item in items {
                        // For loop variable - directly set without checking
                        self.env.variables.insert(variable.as_str().to_string(), item);
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }

                    Ok(())
                }
                ControlFlow::While {
                    condition,
                    body,
                } => {
                    // Execute while condition is true
                    loop {
                        let condition_result = match condition {
                            Condition::Constraint(c) => self.eval_constraint(c)?,
                            Condition::Tensor(expr) => {
                                let val = self.eval_expr(expr)?;
                                val.as_bool()?
                            }
                        };

                        if !condition_result {
                            break;
                        }

                        let mut should_break = false;
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }

                    Ok(())
                }
                ControlFlow::Loop { body } => {
                    loop {
                        let mut should_break = false;
                        for stmt in body {
                            let result = self.execute_statement(stmt);
                            match result {
                                Err(RuntimeError::BreakOutsideLoop) => {
                                    should_break = true;
                                    break;
                                }
                                Err(RuntimeError::ReturnValue(_)) => {
                                    // Propagate return upward
                                    return result;
                                }
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        if should_break {
                            break;
                        }
                    }
                    Ok(())
                }
            },
            Statement::FactAssertion { atom } => {
                // Check if this is actually a built-in function call
                // (since fact_assertion and function_call are now syntactically identical)
                let predicate_name = atom.predicate.as_str();

                if predicate_name == "print" {
                    // Handle as print function
                    for (i, term) in atom.terms.iter().enumerate() {
                        if i > 0 {
                            print!(" ");
                        }
                        // Convert term to expression and evaluate
                        let expr = self.term_to_expr(term);
                        let val = self.eval_expr(&expr)?;
                        print!("{}", val);
                    }
                    println!();
                    return Ok(());
                }

                // Otherwise, treat as a fact assertion
                println!("Adding fact: {}", predicate_name);

                // Collect entities from fact (for data-driven entity types)
                self.collect_entities_from_fact(atom)?;

                // Convert atom terms based on relation variable definitions
                let converted_atom = self.convert_atom_terms(atom);

                self.logic_engine.add_fact(converted_atom);
                println!("  ✓ Fact added to knowledge base");
                Ok(())
            }
            Statement::Query { atom, constraints } => {
                // Query execution with logic engine
                println!("Query: {}", formatter::format_atom(atom));

                // Convert atom terms based on relation variable definitions
                let converted_atom = self.convert_atom_terms(atom);

                // Query the logic engine
                let results = self.logic_engine.query(&converted_atom)?;

                if results.is_empty() {
                    println!("  No solutions found");
                } else {
                    println!("  Found {} solution(s)", results.len());

                    // Apply constraints if any
                    if !constraints.is_empty() {
                        println!("  Applying {} constraint(s)", constraints.len());
                        // TODO: Filter results based on constraints
                    }

                    // Display results
                    for (i, sub) in results.iter().enumerate() {
                        if sub.is_empty() {
                            println!("  Solution {}: Yes", i + 1);
                        } else {
                            println!("  Solution {}:", i + 1);
                            for (var, term) in sub {
                                println!("    {} = {}", var, formatter::format_term(term));
                            }
                        }
                    }
                }

                Ok(())
            }
            Statement::Inference { method, query } => {
                // Inference execution with logic engine integration
                match method {
                    InferenceMethod::Forward => {
                        // Forward inference: Logic → Tensor conversion
                        println!("Forward inference: Logic → Tensor");

                        // Execute query to get logic results
                        if let Statement::Query { atom, .. } = &**query {
                            let results = self.logic_engine.query(atom)?;
                            println!("  Logic results: {} solution(s)", results.len());

                            // Convert logic results to tensor representation
                            if !results.is_empty() {
                                println!("  Converting to tensor representation...");
                                for sub in &results {
                                    let _tensor = self.logic_to_tensor(sub)?;
                                    // In a full implementation, these tensors would be:
                                    // 1. Passed through Neural Engine for inference
                                    // 2. Combined for batch processing
                                    // 3. Stored for further computation
                                }
                                println!("  ✓ Tensor conversion completed");
                            }
                        }
                        Ok(())
                    }
                    InferenceMethod::Backward => {
                        // Backward inference: Tensor → Logic conversion
                        println!("Backward inference: Tensor → Logic");

                        // Get tensor from Neural Engine prediction (placeholder)
                        let device = self.env.metal_device();
                        let prediction_tensor = Tensor::zeros(device, vec![1, 10])?;

                        // Convert tensor predictions to logic facts
                        if let Statement::Query { atom, .. } = &**query {
                            let predicate = atom.predicate.as_str();
                            self.tensor_to_logic(&prediction_tensor, predicate)?;
                            println!("  ✓ Tensor to logic conversion completed");
                        }

                        Ok(())
                    }
                    InferenceMethod::Gradient => {
                        // Gradient inference: propagate differential information
                        println!("Gradient inference: Differentiable logic");

                        // Execute query and track gradient flow
                        if let Statement::Query { atom, .. } = &**query {
                            let _results = self.logic_engine.query(atom)?;

                            // Propagate gradients through logic operations
                            self.propagate_gradient_through_logic(atom)?;

                            println!("  ✓ Gradient propagation through logic completed");
                        }

                        Ok(())
                    }
                    InferenceMethod::Symbolic => {
                        // Symbolic inference: symbolic reasoning
                        println!("Symbolic inference: Symbolic reasoning");

                        // This would:
                        // 1. Use logic engine for symbolic manipulation
                        // 2. Apply symbolic rules and transformations
                        // 3. Return symbolic results

                        self.execute_statement(query)?;

                        println!("  Symbolic reasoning completed");
                        Ok(())
                    }
                }
            }
            Statement::InferenceBlock { items } => {
                // Execute multiple inference operations in sequence
                println!("\n=== Inference Block Started ===");
                println!("Total inference operations: {}", items.len());
                println!();

                for (index, (method, query)) in items.iter().enumerate() {
                    println!("--- Inference {}/{} ---", index + 1, items.len());

                    // Execute single inference by creating a temporary Inference statement
                    let inference_stmt = Statement::Inference {
                        method: *method,
                        query: query.clone(),
                    };

                    self.execute_statement(&inference_stmt)?;
                    println!();
                }

                println!("=== Inference Block Completed ===\n");
                Ok(())
            }
            Statement::Learning(spec) => {
                // Learning execution with detailed progress display
                self.execute_learning(spec)
            }
            Statement::Break => {
                Err(RuntimeError::BreakOutsideLoop)
            }
            Statement::Return { value } => {
                // Evaluate return value (if any) and signal early return
                let return_val = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    Value::Void
                };
                Err(RuntimeError::ReturnValue(return_val))
            }
            Statement::PythonImport { module, alias } => {
                #[cfg(any(feature = "python", feature = "python-extension"))]
                {
                    // Initialize Python environment if needed
                    if self.python_env.is_none() {
                        self.python_env = Some(crate::python::environment::PythonEnvironment::new());
                    }

                    // Import the module
                    let name = alias.as_deref();
                    self.python_env.as_mut().unwrap()
                        .import_module(module, name)
                        .map_err(|e| RuntimeError::InvalidOperation(e))?;

                    let display_name = alias.as_ref().unwrap_or(module);
                    println!("✓ Python import: {} (as {})", module, display_name);
                    Ok(())
                }
                #[cfg(not(any(feature = "python", feature = "python-extension")))]
                {
                    Err(RuntimeError::NotImplemented(
                        "Python integration not enabled (compile with --features python)".to_string()
                    ))
                }
            }
            Statement::WithBlock { entity_type, statements } => {
                // Execute with-block for entity collection
                let type_name = entity_type.as_str();
                println!("\n=== With block: {} ===", type_name);

                // Get entity count before execution
                let count_before = self.entity_registry.get_entity_count(type_name);

                // Execute all statements in the with block
                for stmt in statements {
                    self.execute_statement(stmt)?;
                }

                // Get entity count after execution
                let count_after = self.entity_registry.get_entity_count(type_name);

                // Display statistics
                if let (Some(before), Some(after)) = (count_before, count_after) {
                    if after > before {
                        println!("\n  📊 Entity Statistics:");
                        println!("     • Initial count: {}", before);
                        println!("     • Final count: {}", after);
                        println!("     • New entities: {}", after - before);

                        // Show newly added entities (if reasonable number)
                        if after - before <= 10 && after > before {
                            if let Some(type_info) = self.entity_registry.get_type_info(type_name) {
                                let all_entities = type_info.all_entities();
                                let new_entities: Vec<&String> = all_entities.iter().skip(before).collect();
                                if !new_entities.is_empty() {
                                    println!("     • Added: {:?}", new_entities);
                                }
                            }
                        }
                    } else {
                        println!("\n  📊 Total entities in {}: {}", type_name, after);

                        // Display entity list if not too many
                        if after > 0 {
                            if let Some(type_info) = self.entity_registry.get_type_info(type_name) {
                                let all_entities = type_info.all_entities();

                                if all_entities.len() <= 10 {
                                    // Show all entities
                                    println!("     • Entities: {:?}", all_entities);
                                } else {
                                    // Show first 10 + count
                                    let sample: Vec<&String> = all_entities.iter().take(10).collect();
                                    println!("     • Entities (first 10): {:?}", sample);
                                    println!("     • ... and {} more", all_entities.len() - 10);
                                }
                            }
                        }
                    }
                } else {
                    println!("\n  ℹ️  Entity type '{}' not found in registry", type_name);
                }

                println!("=== With block completed ===\n");
                Ok(())
            }
        }
    }

    /// Evaluate an expression
    pub(super) fn eval_expr(&mut self, expr: &TensorExpr) -> RuntimeResult<Value> {
        match expr {
            TensorExpr::Variable(id) => {
                // Use self.get_variable() to check local scope first
                if let Some(value) = self.get_variable(id.as_str()) {
                    return Ok(value);
                }

                // Check if it's an entity type (meta-type)
                if self.entity_registry.get_type_info(id.as_str()).is_some() {
                    return Ok(Value::Type(id.as_str().to_string()));
                }

                // Not found as variable or type
                Err(RuntimeError::UndefinedVariable(id.as_str().to_string()))
            }

            TensorExpr::Literal(lit) => self.eval_literal(lit),

            TensorExpr::BinaryOp { op, left, right } => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                self.eval_binary_op(op, left_val, right_val)
            }

            TensorExpr::UnaryOp { op, operand } => {
                let operand_val = self.eval_expr(operand)?;
                self.eval_unary_op(op, operand_val)
            }

            TensorExpr::FunctionCall { name, args } => {
                self.eval_function_call(name, args)
            }

            TensorExpr::TensorIndex { tensor, indices } => {
                self.eval_tensor_index(tensor, indices)
            }

            TensorExpr::EinSum { spec, tensors } => {
                self.eval_einsum(spec, tensors)
            }

            TensorExpr::EmbeddingLookup { embedding, entity } => {
                self.eval_embedding_lookup(embedding, entity)
            }

            TensorExpr::PythonCall { function, args } => {
                #[cfg(any(feature = "python", feature = "python-extension"))]
                {
                    // Ensure Python environment is initialized
                    if self.python_env.is_none() {
                        return Err(RuntimeError::InvalidOperation(
                            "Python environment not initialized. Import a module first with 'python import'".to_string()
                        ));
                    }

                    // Evaluate all arguments
                    let tensor_args: Result<Vec<_>, _> = args.iter()
                        .map(|arg| {
                            let val = self.eval_expr(arg)?;
                            val.as_tensor().map(|t| t.clone())
                        })
                        .collect();
                    let tensor_args = tensor_args?;

                    // Create references for the call
                    let tensor_refs: Vec<&Tensor> = tensor_args.iter().collect();

                    // Call Python function
                    let result = self.python_env.as_ref().unwrap()
                        .call_function(function, tensor_refs)
                        .map_err(|e| RuntimeError::InvalidOperation(e))?;

                    println!("✓ Python call: {}({} args)", function, args.len());
                    Ok(Value::Tensor(result))
                }
                #[cfg(not(any(feature = "python", feature = "python-extension")))]
                {
                    Err(RuntimeError::NotImplemented(
                        "Python integration not enabled (compile with --features python)".to_string()
                    ))
                }
            }

            TensorExpr::PropertyAccess { object, property } => {
                // Evaluate the object expression
                let obj_value = self.eval_expr(object)?;

                // Access the property based on object type
                match obj_value {
                    Value::Model(ref model) => {
                        // Access model tensors by name
                        // Common property names: tok_embeddings, output, norm, etc.
                        let tensor_name = property.as_str();

                        if let Some(tensor) = model.get_tensor(tensor_name) {
                            Ok(Value::Tensor(tensor.clone()))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Model does not have tensor '{}'", tensor_name)
                            ))
                        }
                    }
                    _ => Err(RuntimeError::TypeError(
                        format!("Cannot access property '{}' on {:?}", property.as_str(), obj_value)
                    ))
                }
            }

            TensorExpr::MethodCall { object, method, args } => {
                // Evaluate the object expression
                let obj_value = self.eval_expr(object)?;

                // Call method based on object type
                match method.as_str() {
                    "shape" => {
                        // Call shape() method - returns Tensor
                        match obj_value {
                            Value::Tensor(ref t) => {
                                // Create shape tensor from dimensions
                                use half::f16;
                                let shape_data: Vec<f16> = t.shape().dims().iter()
                                    .map(|&d| f16::from_f32(d as f32))
                                    .collect();
                                let device = t.device().clone();
                                let shape_tensor = match &device {
                                    crate::device::Device::Metal(metal_device) => {
                                        crate::tensor::Tensor::from_vec_metal(metal_device, shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::CPU => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::NeuralEngine => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                }.map_err(|e| RuntimeError::TensorError(e))?;
                                Ok(Value::Tensor(shape_tensor))
                            }
                            _ => Err(RuntimeError::TypeError(
                                format!("Cannot call shape() on {:?}", obj_value)
                            ))
                        }
                    }
                    _ => {
                        // For other methods, try calling as a regular function with object as first argument
                        // Build argument list by prepending the object to the method arguments
                        let mut final_args = vec![(**object).clone()];
                        final_args.extend_from_slice(args);

                        self.eval_function_call(&Identifier::new(method.as_str()), &final_args)
                    }
                }
            }
        }
    }

    /// Evaluate a literal
    pub(super) fn eval_literal(&mut self, lit: &TensorLiteral) -> RuntimeResult<Value> {
        match lit {
            TensorLiteral::Scalar(scalar) => match scalar {
                ScalarLiteral::Float(f) => Ok(Value::Float(*f)),
                ScalarLiteral::Integer(i) => Ok(Value::Integer(*i)),
                ScalarLiteral::Boolean(b) => Ok(Value::Boolean(*b)),
                ScalarLiteral::Complex { .. } => {
                    Err(RuntimeError::NotImplemented("Complex numbers not yet supported".to_string()))
                }
                ScalarLiteral::String(s) => Ok(Value::String(s.clone())),
            },
            TensorLiteral::Array(elements) => {
                // Convert array to tensor
                self.eval_array_literal(elements)
            }
        }
    }

    /// Evaluate an array literal to a tensor
    pub(super) fn eval_array_literal(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Value> {
        // Support empty arrays - return empty Tensor
        if elements.is_empty() {
            use half::f16;
            let tensor = Tensor::from_vec_metal(self.env.metal_device(), vec![], vec![0])
                .map_err(|e| RuntimeError::TensorError(e))?;
            return Ok(Value::Tensor(tensor));
        }

        // Recursively collect all scalar values
        let values = self.collect_scalars(elements)?;

        // Determine shape
        let shape = self.infer_shape(elements)?;

        // Always create Tensor for array literals (not TokenIdArray)
        // Convert f32 to f16 for all values
        let f16_values: Vec<f16> = values.into_iter().map(f16::from_f32).collect();
        let tensor = Tensor::from_vec_metal(self.env.metal_device(), f16_values, shape)
            .map_err(|e| RuntimeError::TensorError(e))?;
        Ok(Value::Tensor(tensor))
    }

    /// Collect all scalar values from nested arrays
    pub(super) fn collect_scalars(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Vec<f32>> {
        let mut values = Vec::new();

        for elem in elements {
            match elem {
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(f))) => {
                    values.push(*f as f32);
                }
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Integer(i))) => {
                    values.push(*i as f32);
                }
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    values.extend(self.collect_scalars(nested)?);
                }
                ArrayElement::Expression(expr) => {
                    // Evaluate the expression (e.g., variable reference like seq_len, d_model)
                    let value = self.eval_expr(expr)?;
                    match value {
                        Value::Float(f) => {
                            values.push(f as f32);
                        }
                        Value::Integer(i) => {
                            values.push(i as f32);
                        }
                        _ => {
                            return Err(RuntimeError::TypeError(
                                "Array element expression must evaluate to a scalar number".to_string(),
                            ));
                        }
                    }
                }
                _ => {
                    return Err(RuntimeError::NotImplemented(
                        "Only float/int arrays supported".to_string(),
                    ));
                }
            }
        }

        Ok(values)
    }

    /// Infer shape from nested array structure
    pub(super) fn infer_shape(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Vec<usize>> {
        let mut shape = vec![elements.len()];

        if let Some(first) = elements.first() {
            match first {
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    let nested_shape = self.infer_shape(nested)?;
                    shape.extend(nested_shape);
                }
                _ => {}
            }
        }

        Ok(shape)
    }

    /// Evaluate a binary operation
    pub(super) fn eval_binary_op(&self, op: &BinaryOp, left: Value, right: Value) -> RuntimeResult<Value> {
        match (left, right) {
            (Value::Tensor(l), Value::Tensor(r)) => {
                let result = match op {
                    BinaryOp::Add => l.add(&r),
                    BinaryOp::Sub => l.sub(&r),
                    BinaryOp::Mul => l.mul(&r),
                    BinaryOp::Div => l.div(&r),
                    BinaryOp::MatMul => l.matmul(&r),
                    BinaryOp::Power => {
                        return Err(RuntimeError::NotImplemented("Power not yet implemented".to_string()));
                    }
                    BinaryOp::TensorProd => {
                        return Err(RuntimeError::NotImplemented("Tensor product not yet implemented".to_string()));
                    }
                    BinaryOp::Hadamard => {
                        // Hadamard is element-wise multiplication (same as mul)
                        l.mul(&r)
                    }
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        return Err(RuntimeError::NotImplemented(format!("Comparison {:?} not yet implemented for tensors", op)));
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        return Err(RuntimeError::NotImplemented(format!("Logical {:?} not yet implemented for tensors", op)));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }
            (Value::Float(l), Value::Float(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::Float(l + r)),
                    BinaryOp::Sub => Ok(Value::Float(l - r)),
                    BinaryOp::Mul => Ok(Value::Float(l * r)),
                    BinaryOp::Div => {
                        if r == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Float(l / r))
                    }
                    BinaryOp::Power => Ok(Value::Float(l.powf(r))),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for floats",
                            op
                        )));
                    }
                }
            }
            (Value::Boolean(l), Value::Boolean(r)) => {
                match op {
                    BinaryOp::And => Ok(Value::Boolean(l && r)),
                    BinaryOp::Or => Ok(Value::Boolean(l || r)),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for booleans",
                        op
                    ))),
                }
            }
            (Value::String(l), Value::String(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::String(format!("{}{}", l, r))),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for strings",
                        op
                    ))),
                }
            }
            (Value::Integer(l), Value::Integer(r)) => {
                match op {
                    BinaryOp::Add => Ok(Value::Integer(l + r)),
                    BinaryOp::Sub => Ok(Value::Integer(l - r)),
                    BinaryOp::Mul => Ok(Value::Integer(l * r)),
                    BinaryOp::Div => {
                        if r == 0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Integer(l / r))
                    }
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => Err(RuntimeError::InvalidOperation(format!(
                        "Operation {:?} not supported for integers",
                        op
                    ))),
                }
            }
            // Tensor-Float operations (e.g., tensor * 0.5)
            (Value::Tensor(t), Value::Float(s)) => {
                let scalar_f16 = half::f16::from_f32(s as f32);
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f16),
                    BinaryOp::Sub => t.sub_scalar(scalar_f16),
                    BinaryOp::Mul => t.mul_scalar(scalar_f16),
                    BinaryOp::Div => {
                        if s == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        t.div_scalar(scalar_f16)
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for Tensor-Float",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Tensor(result))
            }
            // Float-Tensor operations (e.g., 0.5 * tensor)
            (Value::Float(s), Value::Tensor(t)) => {
                let scalar_f16 = half::f16::from_f32(s as f32);
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f16),
                    BinaryOp::Mul => t.mul_scalar(scalar_f16),
                    BinaryOp::Sub => {
                        // s - tensor = -(tensor - s)
                        let temp = t.sub_scalar(scalar_f16)
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        let zero = Tensor::zeros(self.env.metal_device(), t.shape().dims().to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        zero.sub(&temp)
                    }
                    BinaryOp::Div => {
                        // s / tensor = s * (1/tensor)
                        return Err(RuntimeError::InvalidOperation(
                            "Scalar / Tensor not yet supported".to_string()
                        ));
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for Float-Tensor",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::Tensor(result))
            }
            _ => Err(RuntimeError::TypeError(
                "Binary operation requires compatible types".to_string(),
            )),
        }
    }

    /// Evaluate a unary operation
    pub(super) fn eval_unary_op(&self, op: &UnaryOp, operand: Value) -> RuntimeResult<Value> {
        match operand {
            Value::Tensor(t) => {
                let result = match op {
                    UnaryOp::Neg => {
                        // Negate tensor: -t
                        let zero = Tensor::zeros(self.env.metal_device(), t.shape().dims().to_vec())
                            .map_err(|e| RuntimeError::TensorError(e))?;
                        zero.sub(&t)
                    }
                    UnaryOp::Transpose => {
                        return Err(RuntimeError::NotImplemented("Transpose not yet implemented".to_string()));
                    }
                    UnaryOp::Inverse => {
                        return Err(RuntimeError::NotImplemented("Inverse not yet implemented".to_string()));
                    }
                    UnaryOp::Determinant => {
                        return Err(RuntimeError::NotImplemented("Determinant not yet implemented".to_string()));
                    }
                    UnaryOp::Not => {
                        return Err(RuntimeError::TypeError("Not operation requires boolean".to_string()));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;

                Ok(Value::Tensor(result))
            }
            Value::Float(f) => {
                let result = match op {
                    UnaryOp::Neg => -f,
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for floats",
                            op
                        )));
                    }
                };
                Ok(Value::Float(result))
            }
            Value::Boolean(b) => {
                let result = match op {
                    UnaryOp::Not => !b,
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for booleans",
                            op
                        )));
                    }
                };
                Ok(Value::Boolean(result))
            }
            _ => Err(RuntimeError::TypeError(format!(
                "Unary operation {:?} not supported for this type",
                op
            ))),
        }
    }

    /// Evaluate embedding lookup: embed[entity]
    pub(super) fn eval_embedding_lookup(&mut self, embedding: &Identifier, entity: &EntityRef) -> RuntimeResult<Value> {
        use crate::ast::EntityRef;

        // Get embedding name
        let embed_name = embedding.as_str();

        // Lookup embedding
        let (entity_map, embedding_matrix) = self.embeddings
            .get(embed_name)
            .ok_or_else(|| RuntimeError::UndefinedVariable(embed_name.to_string()))?;

        // Resolve entity to index
        let entity_idx = match entity {
            EntityRef::Literal(entity_name) => {
                // Look up entity in mapping
                entity_map
                    .get(entity_name)
                    .copied()
                    .ok_or_else(|| RuntimeError::InvalidOperation(
                        format!("Unknown entity '{}' in embedding '{}'", entity_name, embed_name)
                    ))?
            }
            EntityRef::Variable(var_name) => {
                // Variable entity: try to resolve from environment, or use as symbol
                match self.env.get_variable(var_name.as_str()) {
                    Ok(var_value) => {
                        // Variable exists - use its value
                        match var_value {
                            Value::String(s) => {
                                entity_map
                                    .get(s)
                                    .copied()
                                    .ok_or_else(|| RuntimeError::InvalidOperation(
                                        format!("Unknown entity '{}' in embedding '{}'", s, embed_name)
                                    ))?
                            }
                            Value::Integer(idx) => *idx as usize,
                            _ => return Err(RuntimeError::TypeError(
                                format!("Expected String or Integer for entity, found {:?}", var_value)
                            )),
                        }
                    }
                    Err(RuntimeError::UndefinedVariable(_)) => {
                        // Variable doesn't exist - treat identifier as entity symbol
                        entity_map
                            .get(var_name.as_str())
                            .copied()
                            .ok_or_else(|| RuntimeError::InvalidOperation(
                                format!("Unknown entity '{}' in embedding '{}'", var_name.as_str(), embed_name)
                            ))?
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        // Extract row from embedding matrix
        let embedding_vec = embedding_matrix.to_vec();
        let dimension = embedding_matrix.shape().dims()[1];
        
        // Check bounds
        let num_entities = embedding_matrix.shape().dims()[0];
        if entity_idx >= num_entities {
            return Err(RuntimeError::InvalidOperation(
                format!("Entity index {} out of bounds (0..{})", entity_idx, num_entities)
            ));
        }

        // Extract embedding vector
        let start_idx = entity_idx * dimension;
        let end_idx = start_idx + dimension;
        let entity_embedding = embedding_vec[start_idx..end_idx].to_vec();

        // Create tensor from embedding vector on Metal GPU
        let embedding_tensor = Tensor::from_vec_metal(
            self.env.metal_device(),
            entity_embedding,
            vec![dimension]
        )?;

        Ok(Value::Tensor(embedding_tensor))
    }

    /// Evaluate tensor indexing: tensor[i, j, ...]
    pub(super) fn eval_tensor_index(&mut self, tensor_expr: &TensorExpr, indices: &[IndexExpr]) -> RuntimeResult<Value> {
        use crate::ast::IndexExpr;

        // Evaluate the tensor expression
        let tensor_value = self.eval_expr(tensor_expr)?;
        let tensor = tensor_value.as_tensor()?;

        // Convert indices to usizes
        let mut idx_values = Vec::new();
        for idx_expr in indices {
            match idx_expr {
                IndexExpr::Int(i) => {
                    if *i < 0 {
                        return Err(RuntimeError::InvalidOperation(
                            "Negative indices not supported".to_string()
                        ));
                    }
                    idx_values.push(*i as usize);
                }
                IndexExpr::Var(var) => {
                    let val = self.env.get_variable(var.as_str())?;
                    let i = val.as_integer()?;
                    if i < 0 {
                        return Err(RuntimeError::InvalidOperation(
                            "Negative indices not supported".to_string()
                        ));
                    }
                    idx_values.push(i as usize);
                }
                IndexExpr::Slice => {
                    return Err(RuntimeError::NotImplemented(
                        "Slice indexing not yet implemented".to_string()
                    ));
                }
            }
        }

        // Calculate linear index
        let dims = tensor.dims();
        if idx_values.len() != dims.len() {
            return Err(RuntimeError::InvalidOperation(format!(
                "Index dimension mismatch: tensor has {} dimensions, got {} indices",
                dims.len(),
                idx_values.len()
            )));
        }

        // Compute linear index
        let mut linear_idx = 0;
        let mut stride = 1;
        for i in (0..dims.len()).rev() {
            if idx_values[i] >= dims[i] {
                return Err(RuntimeError::InvalidOperation(format!(
                    "Index out of bounds: index {} = {}, dimension size = {}",
                    i, idx_values[i], dims[i]
                )));
            }
            linear_idx += idx_values[i] * stride;
            stride *= dims[i];
        }

        // Get the value at the index
        let data = tensor.to_vec();
        let value = data[linear_idx];

        // Return as a scalar float
        Ok(Value::Float(value.to_f32() as f64))
    }

    /// Evaluate Einstein summation: einsum("ij,jk->ik", A, B)
    pub(super) fn eval_einsum(&mut self, spec: &str, tensor_exprs: &[TensorExpr]) -> RuntimeResult<Value> {
        // Evaluate all tensor expressions
        let mut tensors = Vec::new();
        for expr in tensor_exprs {
            let value = self.eval_expr(expr)?;
            let tensor = value.as_tensor()?;
            tensors.push(tensor.clone());
        }

        // Create references for einsum call
        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();

        // Call einsum operation
        let result = Tensor::einsum(spec, &tensor_refs)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }
}
