//! Expression and statement evaluation logic for TensorLogic interpreter
//!
//! This module contains the core evaluation methods that execute TensorLogic code.

use super::*;
use crate::ast::*;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO};
use crate::tensor::Tensor;
use crate::device::EncoderProvider;
use std::collections::HashSet;

impl Interpreter {
    pub(super) fn execute_statement(&mut self, stmt: &Statement) -> RuntimeResult<()> {
        match stmt {
            Statement::TensorDecl(decl) => {
                // Handle tensor declaration (same logic as Let statement)
                if let Some(init_expr) = &decl.init_expr {
                    let value = self.eval_expr(init_expr)?;

                    // Check if we're inside a function call
                    if let Some(frame) = self.call_stack.last_mut() {
                        // Inside function: declare in local scope
                        frame.local_vars.insert(decl.name.as_str().to_string(), value);
                        Ok(())
                    } else {
                        // Global scope: declare in environment
                        self.env
                            .declare_variable(decl.name.as_str().to_string(), value)?;
                        Ok(())
                    }
                } else {
                    // No initializer - create uninitialized tensor (would need default value)
                    return Err(RuntimeError::TypeError(
                        "Tensor declarations must have initializers".to_string(),
                    ));
                }
            }
            Statement::Let { target, value } => {
                eprintln!("[DEBUG] Statement::Let: target={}", target.as_str());
                eprintln!("[DEBUG] Statement::Let: About to eval_expr...");
                let evaluated_value = self.eval_expr(value)?;
                eprintln!("[DEBUG] Statement::Let: eval_expr completed");

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
                // Equation types (~) - just execute both sides
                let _left = self.eval_expr(&eq.left)?;
                let _right = self.eval_expr(&eq.right)?;
                Ok(())
            }
            Statement::FunctionCall { name, args, resolved, span } => {
                // Save span for error reporting
                self.current_span = Some(span.clone());
                // Handle function calls as statements (e.g., print)
                if name.as_str() == "print" {
                    // Special handling for print with format string support
                    if args.is_empty() {
                        println!();
                        return Ok(());
                    }

                    // Check if first argument is a string literal (format string mode)
                    let first_val = self.eval_expr(&args[0])?;

                    if let Value::String(ref format_str) = first_val {
                        // Check if this is format string mode (contains {}) or simple mode
                        if format_str.contains("{}") {
                            // Format string mode: print("Hello {}", name)
                            if args.len() > 1 {
                                // Evaluate remaining arguments
                                let mut format_args = Vec::new();
                                for arg in &args[1..] {
                                    format_args.push(self.eval_expr(arg)?);
                                }

                                // Use format_string helper
                                let formatted = self.format_string(&format_str, &format_args)?;
                                println!("{}", formatted);
                            } else {
                                // Just a string, print it
                                println!("{}", format_str);
                            }
                        } else if args.len() == 1 {
                            // Just a single string, print it
                            println!("{}", format_str);
                        } else {
                            // Simple mode with multiple arguments: print("A", "B", "C")
                            print!("{}", self.value_to_display(&first_val));
                            for arg in &args[1..] {
                                print!(" ");
                                let val = self.eval_expr(arg)?;
                                print!("{}", self.value_to_display(&val));
                            }
                            println!();
                        }
                    } else {
                        // Simple mode: print(value1, value2, ...)
                        print!("{}", self.value_to_display(&first_val));
                        for arg in &args[1..] {
                            print!(" ");
                            let val = self.eval_expr(arg)?;
                            print!("{}", self.value_to_display(&val));
                        }
                        println!();
                    }
                    Ok(())
                } else {
                    // Other function calls - evaluate and discard result
                    self.eval_function_call(None, name, args, resolved.as_ref())?;
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

                    // Execute appropriate block using centralized function
                    if condition_result {
                        self.execute_block(then_block, false)?;
                    } else if let Some(else_stmts) = else_block {
                        self.execute_block(else_stmts, false)?;
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
                            match tensor_val {
                                Value::TensorF16(tensor) => {
                                    let data = tensor.to_vec();
                                    data.iter().map(|&v| Value::Float(v.to_f32() as f64)).collect()
                                }
                                Value::TensorF32(tensor) => {
                                    let data = tensor.to_vec();
                                    data.iter().map(|&v| Value::Float(v as f64)).collect()
                                }
                                _ => {
                                    return Err(RuntimeError::TypeError(
                                        "Expected tensor (f16 or f32) for iteration".to_string()
                                    ));
                                }
                            }
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
                    let loop_var_name = variable.as_str().to_string();

                    // Save original value if loop variable shadows an outer variable
                    let original_loop_var = self.env.variables.get(&loop_var_name).cloned();

                    for item in items {
                        // Set loop variable
                        self.env.variables.insert(loop_var_name.clone(), item);

                        // Track variables before body (including loop variable)
                        let vars_before: HashSet<String> = self.env.variables.keys().cloned().collect();

                        // Execute body using centralized function (allows break)
                        let result = self.execute_block(body, true);

                        // Clean up iteration variables (excluding loop variable)
                        let vars_after: HashSet<String> = self.env.variables.keys().cloned().collect();
                        for var in vars_after.difference(&vars_before) {
                            self.env.variables.remove(var);
                        }

                        match result {
                            Err(RuntimeError::BreakOutsideLoop) => break,
                            Err(e) => return Err(e),
                            Ok(_) => {}
                        }
                    }

                    // Restore original loop variable value or remove it
                    match original_loop_var {
                        Some(val) => {
                            // Loop variable shadowed an outer variable - restore it
                            self.env.variables.insert(loop_var_name, val);
                        }
                        None => {
                            // Loop variable didn't exist before - remove it
                            self.env.variables.remove(&loop_var_name);
                        }
                    }

                    Ok(())
                }
                ControlFlow::While {
                    condition,
                    body,
                } => {
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

                        // Execute body using centralized function (allows break)
                        match self.execute_block(body, true) {
                            Err(RuntimeError::BreakOutsideLoop) => break,
                            Err(e) => return Err(e),
                            Ok(_) => {}
                        }
                    }

                    Ok(())
                }
                ControlFlow::Loop { body } => {
                    loop {
                        // Execute body using centralized function (allows break)
                        match self.execute_block(body, true) {
                            Err(RuntimeError::BreakOutsideLoop) => break,
                            Err(e) => return Err(e),
                            Ok(_) => {}
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
                    // Handle as print function with format string support
                    if atom.terms.is_empty() {
                        println!();
                        return Ok(());
                    }

                    // Convert first term to expression and evaluate
                    let first_term = &atom.terms[0];
                    let first_expr = self.term_to_expr(first_term);
                    let first_val = self.eval_expr(&first_expr)?;

                    if let Value::String(ref format_str) = first_val {
                        // Check if this is format string mode (contains {}) or simple mode
                        if format_str.contains("{}") {
                            // Format string mode
                            if atom.terms.len() > 1 {
                                let mut format_args = Vec::new();
                                for term in &atom.terms[1..] {
                                    let expr = self.term_to_expr(term);
                                    format_args.push(self.eval_expr(&expr)?);
                                }
                                let formatted = self.format_string(&format_str, &format_args)?;
                                println!("{}", formatted);
                            } else {
                                println!("{}", format_str);
                            }
                        } else if atom.terms.len() == 1 {
                            // Just a single string, print it
                            println!("{}", format_str);
                        } else {
                            // Simple mode with multiple arguments: print("A", "B", "C")
                            print!("{}", self.value_to_display(&first_val));
                            for term in &atom.terms[1..] {
                                print!(" ");
                                let expr = self.term_to_expr(term);
                                let val = self.eval_expr(&expr)?;
                                print!("{}", self.value_to_display(&val));
                            }
                            println!();
                        }
                    } else {
                        // Simple mode
                        print!("{}", self.value_to_display(&first_val));
                        for term in &atom.terms[1..] {
                            print!(" ");
                            let expr = self.term_to_expr(term);
                            let val = self.eval_expr(&expr)?;
                            print!("{}", self.value_to_display(&val));
                        }
                        println!();
                    }
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
            Statement::Block { statements } => {
                self.execute_block(statements, false)
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
            Statement::Panic { format, args } => {
                // Evaluate arguments
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }

                // Format the panic message
                let msg = self.format_string(format, &arg_values)?;

                // Panic with the formatted message
                panic!("{}", msg);
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

    /// Read a single element from f16 tensor at linear index
    /// SIMPLIFIED: Use CPU transfer instead of GPU kernel to avoid command buffer complexity
    pub(super) fn read_element_f16(&self, tensor: &crate::tensor::Tensor<half::f16>, linear_idx: usize) -> RuntimeResult<f32> {
        use crate::device::{MetalBuffer, KernelExecutor};
        use half::f16;

        if linear_idx >= tensor.numel() {
            return Err(RuntimeError::InvalidOperation(
                format!("Index {} out of bounds for tensor with {} elements", linear_idx, tensor.numel())
            ));
        }

        // Fast path for CPU tensors: direct access without copying
        use crate::tensor::BufferHandle;
        if let BufferHandle::CPU(ref vec) = tensor.buffer() {
            return Ok(vec[linear_idx].to_f32());
        }

        let device = match tensor.device() {
            crate::device::Device::Metal(dev) => dev.clone(),
            _ => return Err(RuntimeError::InvalidOperation("read_element_f16 requires Metal device".to_string())),
        };

        let mut device_mut = device.clone();
        if device_mut.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device_mut.load_library(shader_source)
                .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to load shader: {}", e)))?;
        }

        let input_buf = tensor.buffer().as_metal()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to get Metal buffer: {}", e)))?;
        let output_buf = MetalBuffer::<f16>::new_uninit(device.metal_device(), 1)
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to create output buffer: {}", e)))?;

        let index_data = [linear_idx as u32];
        let index_buf = device.metal_device().new_buffer_with_data(
            index_data.as_ptr() as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut executor = KernelExecutor::new(device_mut);

        let pipeline = executor.get_or_compile_pipeline("read_element_f16")
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to compile kernel: {}", e)))?;

        // Use Commands manager for command buffer (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to get command buffer: {}", e)))?;
        let encoder = command_buffer.encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&output_buf.buffer), 0);
        encoder.set_buffer(2, Some(&index_buf), 0);

        // FIXED: Use dispatchThreadgroups for single-thread execution (more reliable than dispatch_threads)
        let threadgroups = metal::MTLSize::new(1, 1, 1);
        let threads_per_threadgroup = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        // Commands manager will flush and commit when needed
        // Since we need the result immediately, wait for completion
        device.wait_until_completed()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to wait for GPU: {}", e)))?;

        // Check for errors
        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::InvalidOperation(
                format!("GPU command buffer failed with status: {:?}", command_buffer.status())
            ));
        }

        // Read result from GPU
        let result_slice = unsafe {
            std::slice::from_raw_parts(output_buf.buffer.contents() as *const f16, 1)
        };
        Ok(result_slice[0].to_f32())
    }

    /// Read a single element from f32 tensor at linear index using GPU
    pub(super) fn read_element_f32(&self, tensor: &crate::tensor::Tensor<f32>, linear_idx: usize) -> RuntimeResult<f32> {
        use crate::device::{MetalBuffer, KernelExecutor};

        if linear_idx >= tensor.numel() {
            return Err(RuntimeError::InvalidOperation(
                format!("Index {} out of bounds for tensor with {} elements", linear_idx, tensor.numel())
            ));
        }

        // Fast path for CPU tensors: direct access without copying
        use crate::tensor::BufferHandle;
        if let BufferHandle::CPU(ref vec) = tensor.buffer() {
            return Ok(vec[linear_idx]);
        }
        let device = match tensor.device() {
            crate::device::Device::Metal(dev) => dev.clone(),
            _ => return Err(RuntimeError::InvalidOperation("read_element_f32 requires Metal device".to_string())),
        };
        let mut device_mut = device.clone();
        if device_mut.library().is_none() {
            let shader_source = include_str!("../../shaders/unified.metal");
            device_mut.load_library(shader_source)
                .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to load shader: {}", e)))?;
        }

        let input_buf = tensor.buffer().as_metal()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to get Metal buffer: {}", e)))?;
        let output_buf = MetalBuffer::<f32>::new_uninit(device.metal_device(), 1)
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to create output buffer: {}", e)))?;

        let index_data = [linear_idx as u32];
        let index_buf = device.metal_device().new_buffer_with_data(
            index_data.as_ptr() as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut executor = KernelExecutor::new(device_mut);
        let pipeline = executor.get_or_compile_pipeline("read_element_f32")
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to compile kernel: {}", e)))?;

        // Use Commands manager for command buffer (Candle pattern)
        let (_flushed, command_buffer) = device.command_buffer()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to get command buffer: {}", e)))?;
        let encoder = command_buffer.encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buf.buffer), 0);
        encoder.set_buffer(1, Some(&output_buf.buffer), 0);
        encoder.set_buffer(2, Some(&index_buf), 0);

        // FIXED: Use dispatchThreadgroups for single-thread execution (more reliable than dispatch_threads)
        let threadgroups = metal::MTLSize::new(1, 1, 1);
        let threads_per_threadgroup = metal::MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        // Commands manager will flush and commit when needed
        // Since we need the result immediately, wait for completion
        device.wait_until_completed()
            .map_err(|e| RuntimeError::InvalidOperation(format!("Failed to wait for GPU: {}", e)))?;

        // Check for errors
        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::InvalidOperation(
                format!("GPU command buffer failed with status: {:?}", command_buffer.status())
            ));
        }

        // Read result from GPU
        let result_slice = unsafe {
            std::slice::from_raw_parts(output_buf.buffer.contents() as *const f32, 1)
        };
        Ok(result_slice[0])
    }



    /// Evaluate an expression
    /// Execute a block of statements with automatic variable cleanup
    ///
    /// This is the common function used by all control structures (IF, WHILE, LOOP, FOR, Block)
    /// to ensure consistent scoping behavior.
    ///
    /// Parameters:
    /// - statements: The statements to execute
    /// - allow_break: Whether break statements are allowed (true for loops)
    fn execute_block(&mut self, statements: &[Statement], allow_break: bool) -> RuntimeResult<()> {
        // Save current variable state (for shadowing support)
        // Map: variable name -> Option<Value> (None if didn't exist before)
        let mut shadowed_vars: HashMap<String, Option<Value>> = HashMap::new();

        // Track which variables existed before the block
        let vars_before: HashSet<String> = self.env.variables.keys().cloned().collect();

        // Execute all statements in the block
        for stmt in statements {
            // If this is a let statement, track shadowing
            if let Statement::Let { target, .. } = stmt {
                let var_name = target.as_str().to_string();
                if !shadowed_vars.contains_key(&var_name) {
                    // Save the old value if it exists (for restoration)
                    shadowed_vars.insert(
                        var_name.clone(),
                        self.env.variables.get(&var_name).cloned()
                    );
                }
            }

            let result = self.execute_statement(stmt);
            match result {
                Err(RuntimeError::BreakOutsideLoop) if allow_break => {
                    // Restore shadowed variables before breaking
                    self.restore_shadowed_variables(&shadowed_vars, &vars_before);
                    return Err(RuntimeError::BreakOutsideLoop);
                }
                Err(RuntimeError::ReturnValue(_)) => {
                    // Propagate return upward (caller will handle cleanup)
                    return result;
                }
                Err(e) => return Err(e),
                Ok(_) => {}
            }
        }

        // Restore shadowed variables and clean up block-local variables
        self.restore_shadowed_variables(&shadowed_vars, &vars_before);

        Ok(())
    }

    /// Restore shadowed variables and clean up block-local variables
    fn restore_shadowed_variables(
        &mut self,
        shadowed_vars: &HashMap<String, Option<Value>>,
        vars_before: &HashSet<String>
    ) {
        // Get current variables after block execution
        let vars_after: HashSet<String> = self.env.variables.keys().cloned().collect();

        // Restore shadowed variables
        for (var_name, old_value) in shadowed_vars {
            match old_value {
                Some(val) => {
                    // Variable existed before - restore old value
                    self.env.variables.insert(var_name.clone(), val.clone());
                }
                None => {
                    // Variable didn't exist before - remove it
                    self.env.variables.remove(var_name);
                }
            }
        }

        // Clean up any other block-local variables (not shadowed)
        for var in vars_after.difference(vars_before) {
            if !shadowed_vars.contains_key(var) {
                self.env.variables.remove(var);
            }
        }
    }

    pub(super) fn eval_expr(&mut self, expr: &TensorExpr) -> RuntimeResult<Value> {
        // Debug: print expression type
        let expr_type = match expr {
            TensorExpr::Variable(_) => "Variable",
            TensorExpr::Literal(_) => "Literal",
            TensorExpr::BinaryOp { .. } => "BinaryOp",
            TensorExpr::UnaryOp { .. } => "UnaryOp",
            TensorExpr::FunctionCall { .. } => "FunctionCall",
            TensorExpr::TensorIndex { .. } => "TensorIndex",
            TensorExpr::EinSum { .. } => "EinSum",
            TensorExpr::EmbeddingLookup { .. } => "EmbeddingLookup",
            TensorExpr::PythonCall { .. } => "PythonCall",
            TensorExpr::PropertyAccess { .. } => "PropertyAccess",
            TensorExpr::MethodCall { .. } => "MethodCall",
        };
        eprintln!("[DEBUG] eval_expr: type={}", expr_type);

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

            TensorExpr::FunctionCall { type_namespace, name, args, resolved } => {
                self.eval_function_call(type_namespace.as_deref(), name, args, resolved.as_ref())
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
                    use crate::interpreter::value::ToValue;

                    // Ensure Python environment is initialized
                    if self.python_env.is_none() {
                        return Err(RuntimeError::InvalidOperation(
                            "Python environment not initialized. Import a module first with 'python import'".to_string()
                        ));
                    }

                    // Evaluate all arguments
                    let values: Vec<Value> = args.iter()
                        .map(|arg| self.eval_expr(arg))
                        .collect::<Result<_, _>>()?;

                    // Check if all tensors are f16 or all f32
                    let all_f16 = values.iter().all(|v| matches!(v, Value::TensorF16(_)));
                    let all_f32 = values.iter().all(|v| matches!(v, Value::TensorF32(_)));

                    if !all_f16 && !all_f32 {
                        return Err(RuntimeError::TypeError(
                            "Python function call requires all tensors to be the same type (all f16 or all f32)".to_string()
                        ));
                    }

                    if all_f16 {
                        // Extract f16 tensors
                        let tensor_args: Vec<_> = values.iter()
                            .filter_map(|v| match v {
                                Value::TensorF16(t) => Some(t.clone()),
                                _ => None
                            })
                            .collect();

                        // Create references for the call
                        let tensor_refs: Vec<&Tensor<f16>> = tensor_args.iter().collect();

                        // Call Python function
                        let result = self.python_env.as_ref().unwrap()
                            .call_function(function, tensor_refs)
                            .map_err(|e| RuntimeError::InvalidOperation(e))?;

                        println!("✓ Python call: {}({} args)", function, args.len());
                        Ok(result.to_value())
                    } else {
                        // Extract f32 tensors
                        let tensor_args: Vec<_> = values.iter()
                            .filter_map(|v| match v {
                                Value::TensorF32(t) => Some(t.clone()),
                                _ => None
                            })
                            .collect();

                        // Create references for the call
                        let tensor_refs: Vec<&Tensor<f32>> = tensor_args.iter().collect();

                        // Call Python function
                        let result = self.python_env.as_ref().unwrap()
                            .call_function(function, tensor_refs)
                            .map_err(|e| RuntimeError::InvalidOperation(e))?;

                        println!("✓ Python call: {}({} args)", function, args.len());
                        Ok(result.to_value())
                    }
                }
                #[cfg(not(any(feature = "python", feature = "python-extension")))]
                {
                    Err(RuntimeError::NotImplemented(
                        "Python integration not enabled (compile with --features python)".to_string()
                    ))
                }
            }

            TensorExpr::PropertyAccess { object, property } => {
                // NEW: Evaluate object first, then access property on the resulting value
                let obj_value = self.eval_expr(object)?;
                let prop_name = property.as_str();
                
                // Handle property access based on object type
                match obj_value {
                    // Model.property -> returns ModelLayerCollection or ModelFeature
                    Value::ModelF16(ref model) => {
                        // Try as layer collection first (e.g., "blk")
                        if let Some(collection) = model.build_layer_collection(prop_name) {
                            Ok(Value::ModelLayerCollectionF16(collection))
                        } else if let Some(feature) = model.get_property(prop_name) {
                            Ok(Value::ModelFeatureF16(feature))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Model does not have property '{}'", prop_name)
                            ))
                        }
                    }
                    Value::ModelF32(ref model) => {
                        // Try as layer collection first (e.g., "blk")
                        if let Some(collection) = model.build_layer_collection(prop_name) {
                            Ok(Value::ModelLayerCollectionF32(collection))
                        } else if let Some(feature) = model.get_property(prop_name) {
                            Ok(Value::ModelFeatureF32(feature))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Model does not have property '{}'", prop_name)
                            ))
                        }
                    }
                    
                    // ModelLayer.property -> returns ModelFeature
                    Value::ModelLayerF16(ref layer) => {
                        if let Some(feature) = layer.get_feature(prop_name) {
                            Ok(Value::ModelFeatureF16(feature.clone()))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Layer {} does not have feature '{}'", layer.index, prop_name)
                            ))
                        }
                    }
                    Value::ModelLayerF32(ref layer) => {
                        if let Some(feature) = layer.get_feature(prop_name) {
                            Ok(Value::ModelFeatureF32(feature.clone()))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Layer {} does not have feature '{}'", layer.index, prop_name)
                            ))
                        }
                    }
                    
                    // ModelFeature.property -> returns Tensor
                    Value::ModelFeatureF16(ref feature) => {
                        if let Some(tensor) = feature.get_property(prop_name) {
                            Ok(Value::TensorF16(tensor.clone()))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Feature '{}' does not have property '{}'", feature.name, prop_name)
                            ))
                        }
                    }
                    Value::ModelFeatureF32(ref feature) => {
                        if let Some(tensor) = feature.get_property(prop_name) {
                            Ok(Value::TensorF32(tensor.clone()))
                        } else {
                            Err(RuntimeError::InvalidOperation(
                                format!("Feature '{}' does not have property '{}'", feature.name, prop_name)
                            ))
                        }
                    }
                    
                    _ => {
                        Err(RuntimeError::TypeError(
                            format!("Property access not supported on {:?}", std::mem::discriminant(&obj_value))
                        ))
                    }
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
                            Value::TensorF16(ref t) => {
                                // Create shape tensor from dimensions
                                use half::f16;
                                let shape_data: Vec<f16> = t.shape().dims().iter()
                                    .map(|&d| f16::from_f32(d as f32))
                                    .collect();
                                let device = t.device().clone();
                                let shape_tensor = match &device {
                                    crate::device::Device::Metal(metal_device) => {
                                        crate::tensor::Tensor::from_vec_gpu(metal_device, shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::CPU => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::NeuralEngine => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                }.map_err(|e| RuntimeError::TensorError(e))?;
                                Ok(Value::TensorF16(shape_tensor))
                            }
                            Value::TensorF32(ref t) => {
                                // Create shape tensor from dimensions (f32 version)
                                let shape_data: Vec<f32> = t.shape().dims().iter()
                                    .map(|&d| d as f32)
                                    .collect();
                                let device = t.device().clone();
                                let shape_tensor = match &device {
                                    crate::device::Device::Metal(metal_device) => {
                                        crate::tensor::Tensor::from_vec_gpu(metal_device, shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::CPU => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                    crate::device::Device::NeuralEngine => {
                                        crate::tensor::Tensor::from_vec(shape_data, vec![t.shape().dims().len()])
                                    }
                                }.map_err(|e| RuntimeError::TensorError(e))?;
                                Ok(Value::TensorF32(shape_tensor))
                            }
                            _ => Err(RuntimeError::TypeError(
                                format!("Cannot call shape() on {:?}", obj_value)
                            ))
                        }
                    }
                    _ => {
                        // Type-based method dispatch
                        match (&obj_value, method.as_str()) {
                            // Tokenizer methods
                            (Value::Tokenizer(_), "tokenize") => {
                                let mut method_args = vec![(**object).clone()];
                                method_args.extend_from_slice(args);
                                self.eval_tokenize(&method_args)
                            }
                            (Value::Tokenizer(_), "detokenize") => {
                                let mut method_args = vec![(**object).clone()];
                                method_args.extend_from_slice(args);
                                self.eval_detokenize(&method_args)
                            }

                            // TokenIds methods
                            (Value::TokenIds(_), "append_token") => {
                                let mut method_args = vec![(**object).clone()];
                                method_args.extend_from_slice(args);
                                self.eval_append_token(&method_args)
                            }

                            // Tensor methods
                            (Value::TensorF32(_), "append") | (Value::TensorF16(_), "append") => {
                                let mut method_args = vec![(**object).clone()];
                                method_args.extend_from_slice(args);
                                self.eval_append_cache(&method_args)
                            }

                            // KVCache methods
                            (Value::KVCache(_), "set") => {
                                // kv_cache.set(layer_idx: int, k: Tensor, v: Tensor) -> Void
                                if args.len() != 3 {
                                    return Err(RuntimeError::TypeError(
                                        format!("KVCache.set() expects 3 arguments, got {}", args.len())
                                    ));
                                }

                                let cache_arc = match obj_value {
                                    Value::KVCache(cache) => cache,
                                    _ => unreachable!()
                                };

                                let layer_idx = match self.eval_expr(&args[0])? {
                                    Value::Integer(n) => n as usize,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.set() expects Integer as first argument, got {}", v.type_name())
                                    )),
                                };

                                let k = match self.eval_expr(&args[1])? {
                                    Value::TensorF16(t) => t,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.set() expects TensorF16 as second argument, got {}", v.type_name())
                                    )),
                                };

                                let v = match self.eval_expr(&args[2])? {
                                    Value::TensorF16(t) => t,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.set() expects TensorF16 as third argument, got {}", v.type_name())
                                    )),
                                };

                                let mut cache = cache_arc.lock().map_err(|e|
                                    RuntimeError::InvalidOperation(format!("Failed to lock cache: {}", e))
                                )?;

                                if layer_idx >= cache.kvs.len() {
                                    return Err(RuntimeError::InvalidOperation(
                                        format!("Layer index {} out of bounds (cache has {} layers)", layer_idx, cache.kvs.len())
                                    ));
                                }

                                cache.kvs[layer_idx] = Some((k, v));
                                Ok(Value::Void)
                            }

                            (Value::KVCache(_), "append") => {
                                // kv_cache.append(layer_idx: int, k: Tensor, v: Tensor) -> Void
                                if args.len() != 3 {
                                    return Err(RuntimeError::TypeError(
                                        format!("KVCache.append() expects 3 arguments, got {}", args.len())
                                    ));
                                }

                                let cache_arc = match obj_value {
                                    Value::KVCache(cache) => cache,
                                    _ => unreachable!()
                                };

                                let layer_idx = match self.eval_expr(&args[0])? {
                                    Value::Integer(n) => n as usize,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.append() expects Integer as first argument, got {}", v.type_name())
                                    )),
                                };

                                let k = match self.eval_expr(&args[1])? {
                                    Value::TensorF16(t) => t,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.append() expects TensorF16 as second argument, got {}", v.type_name())
                                    )),
                                };

                                let v = match self.eval_expr(&args[2])? {
                                    Value::TensorF16(t) => t,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.append() expects TensorF16 as third argument, got {}", v.type_name())
                                    )),
                                };

                                let mut cache = cache_arc.lock().map_err(|e|
                                    RuntimeError::InvalidOperation(format!("Failed to lock cache: {}", e))
                                )?;

                                let device = crate::device::MetalDevice::new()
                                    .map_err(|e| RuntimeError::TensorError(e))?;

                                cache.update(layer_idx, k, v, &device)
                                    .map_err(|e| RuntimeError::TensorError(e))?;

                                Ok(Value::Void)
                            }

                            (Value::KVCache(_), "get_k") => {
                                // kv_cache.get_k(layer_idx: int) -> Tensor
                                if args.len() != 1 {
                                    return Err(RuntimeError::TypeError(
                                        format!("KVCache.get_k() expects 1 argument, got {}", args.len())
                                    ));
                                }

                                let cache_arc = match obj_value {
                                    Value::KVCache(cache) => cache,
                                    _ => unreachable!()
                                };

                                let layer_idx = match self.eval_expr(&args[0])? {
                                    Value::Integer(n) => n as usize,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.get_k() expects Integer, got {}", v.type_name())
                                    )),
                                };

                                let cache = cache_arc.lock().map_err(|e|
                                    RuntimeError::InvalidOperation(format!("Failed to lock cache: {}", e))
                                )?;

                                let (k, _v) = cache.get(layer_idx).ok_or_else(||
                                    RuntimeError::InvalidOperation(format!("No cache entry for layer {}", layer_idx))
                                )?;

                                Ok(Value::TensorF16(k.clone()))
                            }

                            (Value::KVCache(_), "get_v") => {
                                // kv_cache.get_v(layer_idx: int) -> Tensor
                                if args.len() != 1 {
                                    return Err(RuntimeError::TypeError(
                                        format!("KVCache.get_v() expects 1 argument, got {}", args.len())
                                    ));
                                }

                                let cache_arc = match obj_value {
                                    Value::KVCache(cache) => cache,
                                    _ => unreachable!()
                                };

                                let layer_idx = match self.eval_expr(&args[0])? {
                                    Value::Integer(n) => n as usize,
                                    v => return Err(RuntimeError::TypeError(
                                        format!("KVCache.get_v() expects Integer, got {}", v.type_name())
                                    )),
                                };

                                let cache = cache_arc.lock().map_err(|e|
                                    RuntimeError::InvalidOperation(format!("Failed to lock cache: {}", e))
                                )?;

                                let (_k, v) = cache.get(layer_idx).ok_or_else(||
                                    RuntimeError::InvalidOperation(format!("No cache entry for layer {}", layer_idx))
                                )?;

                                Ok(Value::TensorF16(v.clone()))
                            }

                            _ => Err(RuntimeError::TypeError(
                                format!("Type {:?} has no method '{}'", obj_value.type_name(), method)
                            ))
                        }
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
        let _fn_start = std::time::Instant::now();
        eprintln!("[DEBUG] eval_array_literal: Entry, elements.len={}", elements.len());

        // Support empty arrays - return empty Tensor
        if elements.is_empty() {
            eprintln!("[DEBUG] eval_array_literal: Empty array, creating empty tensor");
            let tensor = Tensor::from_vec_gpu(self.env.metal_device(), vec![], vec![0])
                .map_err(|e| RuntimeError::TensorError(e))?;
            return Ok(Value::TensorF16(tensor));
        }

        // Recursively collect all scalar values
        eprintln!("[DEBUG] eval_array_literal: Calling collect_scalars...");
        let collect_start = std::time::Instant::now();
        let values = self.collect_scalars(elements)?;
        eprintln!("[DEBUG] eval_array_literal: collect_scalars completed in {:.3}ms, values.len={}",
                  collect_start.elapsed().as_secs_f64() * 1000.0, values.len());

        // Determine shape
        eprintln!("[DEBUG] eval_array_literal: Calling infer_shape...");
        let shape_start = std::time::Instant::now();
        let shape = self.infer_shape(elements)?;
        eprintln!("[DEBUG] eval_array_literal: infer_shape completed in {:.3}ms, shape={:?}",
                  shape_start.elapsed().as_secs_f64() * 1000.0, shape);

        // Determine if array contains float literals (f32) or only integers (f16)
        eprintln!("[DEBUG] eval_array_literal: Calling has_float_literal...");
        let float_check_start = std::time::Instant::now();
        let has_float_literal = self.has_float_literal(elements);
        eprintln!("[DEBUG] eval_array_literal: has_float_literal={} ({:.3}ms)",
                  has_float_literal, float_check_start.elapsed().as_secs_f64() * 1000.0);

        // OPTIMIZATION: Create array literals on CPU to avoid GPU sync overhead
        // Builtin functions (ones, zeros, reshape) now support CPU tensors via to_cpu_vec()
        // This eliminates per-element GPU sync when using arrays as shape parameters
        let numel = values.len();
        let use_cpu = true;

        if has_float_literal {
            eprintln!("[DEBUG] eval_array_literal: Creating f32 tensor (numel={}, cpu={})...", numel, use_cpu);
            let create_start = std::time::Instant::now();
            // Array contains float literals -> create f32 tensor
            let tensor = if use_cpu {
                Tensor::from_vec(values, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?
            } else {
                Tensor::from_vec_gpu(self.env.metal_device(), values, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?
            };
            eprintln!("[DEBUG] eval_array_literal: f32 tensor created in {:.3}ms", create_start.elapsed().as_secs_f64() * 1000.0);
            eprintln!("[DEBUG] eval_array_literal: TOTAL time: {:.3}ms", _fn_start.elapsed().as_secs_f64() * 1000.0);
            Ok(Value::TensorF32(tensor))
        } else {
            eprintln!("[DEBUG] eval_array_literal: Creating f16 tensor (numel={}, cpu={})...", numel, use_cpu);
            let convert_start = std::time::Instant::now();
            // Array contains only integers -> create f16 tensor (backward compatibility)
            let f16_values: Vec<f16> = values.into_iter().map(f16::from_f32).collect();
            eprintln!("[DEBUG] eval_array_literal: f16 conversion complete in {:.3}ms, creating tensor...", convert_start.elapsed().as_secs_f64() * 1000.0);
            let create_start = std::time::Instant::now();
            let tensor = if use_cpu {
                Tensor::from_vec(f16_values, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?
            } else {
                Tensor::from_vec_gpu(self.env.metal_device(), f16_values, shape)
                    .map_err(|e| RuntimeError::TensorError(e))?
            };
            eprintln!("[DEBUG] eval_array_literal: f16 tensor created in {:.3}ms", create_start.elapsed().as_secs_f64() * 1000.0);
            eprintln!("[DEBUG] eval_array_literal: TOTAL time: {:.3}ms", _fn_start.elapsed().as_secs_f64() * 1000.0);
            Ok(Value::TensorF16(tensor))
        }
    }

    /// Check if array elements contain float literals (for f32 vs f16 detection)
    fn has_float_literal(&self, elements: &[ArrayElement]) -> bool {
        use crate::ast::{ArrayElement, TensorLiteral, ScalarLiteral};

        for elem in elements {
            match elem {
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(_))) => {
                    return true;
                }
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    if self.has_float_literal(nested) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Collect all scalar values from nested arrays
    pub(super) fn collect_scalars(&mut self, elements: &[ArrayElement]) -> RuntimeResult<Vec<f32>> {
        eprintln!("[DEBUG] collect_scalars: Entry, elements.len={}", elements.len());
        let mut values = Vec::new();

        for (i, elem) in elements.iter().enumerate() {
            eprintln!("[DEBUG] collect_scalars: Processing element[{}]", i);
            match elem {
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Float(f))) => {
                    eprintln!("[DEBUG] collect_scalars: element[{}] is Float={}", i, f);
                    values.push(*f as f32);
                }
                ArrayElement::Literal(TensorLiteral::Scalar(ScalarLiteral::Integer(i_val))) => {
                    eprintln!("[DEBUG] collect_scalars: element[{}] is Integer={}", i, i_val);
                    values.push(*i_val as f32);
                }
                ArrayElement::Literal(TensorLiteral::Array(nested)) => {
                    eprintln!("[DEBUG] collect_scalars: element[{}] is nested array", i);
                    values.extend(self.collect_scalars(nested)?);
                }
                ArrayElement::Expression(expr) => {
                    eprintln!("[DEBUG] collect_scalars: element[{}] is Expression, evaluating...", i);
                    // Evaluate the expression (e.g., variable reference like seq_len, d_model)
                    let value = self.eval_expr(expr)?;
                    eprintln!("[DEBUG] collect_scalars: element[{}] eval_expr returned, matching value...", i);
                    match value {
                        Value::Float(f) => {
                            eprintln!("[DEBUG] collect_scalars: element[{}] evaluated to Float={}", i, f);
                            values.push(f as f32);
                        }
                        Value::Integer(i_val) => {
                            eprintln!("[DEBUG] collect_scalars: element[{}] evaluated to Integer={}", i, i_val);
                            values.push(i_val as f32);
                        }
                        _ => {
                            eprintln!("[DEBUG] collect_scalars: element[{}] evaluated to non-scalar type", i);
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
            (Value::TensorF16(l), Value::TensorF16(r)) => {
                let result = match op {
                    BinaryOp::Add => l.add(&r),
                    BinaryOp::Sub => l.sub(&r),
                    BinaryOp::Mul => l.mul(&r),
                    BinaryOp::Div => l.div(&r),
                    BinaryOp::Mod => {
                        return Err(RuntimeError::NotImplemented("Modulo not yet implemented for tensors".to_string()));
                    }
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

                Ok(Value::TensorF16(result))
            }
            (Value::TensorF32(l), Value::TensorF32(r)) => {
                let result = match op {
                    BinaryOp::Add => l.add(&r),
                    BinaryOp::Sub => l.sub(&r),
                    BinaryOp::Mul => l.mul(&r),
                    BinaryOp::Div => l.div(&r),
                    BinaryOp::Mod => {
                        return Err(RuntimeError::NotImplemented("Modulo not yet implemented for tensors".to_string()));
                    }
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

                Ok(Value::TensorF32(result))
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
                    BinaryOp::Mod => {
                        if r == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Float(l % r))
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
                    BinaryOp::Mod => {
                        if r == 0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        Ok(Value::Integer(l % r))
                    }
                    BinaryOp::Power => Ok(Value::Integer(l.pow(r as u32))),
                    BinaryOp::Eq => Ok(Value::Boolean(l == r)),
                    BinaryOp::Ne => Ok(Value::Boolean(l != r)),
                    BinaryOp::Lt => Ok(Value::Boolean(l < r)),
                    BinaryOp::Le => Ok(Value::Boolean(l <= r)),
                    BinaryOp::Gt => Ok(Value::Boolean(l > r)),
                    BinaryOp::Ge => Ok(Value::Boolean(l >= r)),
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for integers",
                            op
                        )));
                    }
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
            (Value::TensorF16(t), Value::Float(s)) => {
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
                Ok(Value::TensorF16(result))
            }
            // Float-Tensor operations (e.g., 0.5 * tensor)
            (Value::Float(s), Value::TensorF16(t)) => {
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
                Ok(Value::TensorF16(result))
            }
            // TensorF32-Float operations (e.g., tensor * 0.5)
            (Value::TensorF32(t), Value::Float(s)) => {
                let scalar_f32 = s as f32;
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f32),
                    BinaryOp::Sub => t.sub_scalar(scalar_f32),
                    BinaryOp::Mul => t.mul_scalar(scalar_f32),
                    BinaryOp::Div => {
                        if s == 0.0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        t.div_scalar(scalar_f32)
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation(format!(
                            "Operation {:?} not supported for TensorF32-Float",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            // Float-TensorF32 operations (e.g., 0.5 * tensor)
            (Value::Float(s), Value::TensorF32(t)) => {
                let scalar_f32 = s as f32;
                let result = match op {
                    BinaryOp::Add => t.add_scalar(scalar_f32),
                    BinaryOp::Mul => t.mul_scalar(scalar_f32),
                    BinaryOp::Sub => {
                        // s - tensor = -(tensor - s)
                        let temp = t.sub_scalar(scalar_f32)
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
                            "Operation {:?} not supported for Float-TensorF32",
                            op
                        )));
                    }
                }
                .map_err(|e| RuntimeError::TensorError(e))?;
                Ok(Value::TensorF32(result))
            }
            // Integer-Float mixed operations (convert integer to float)
            (Value::Integer(l), Value::Float(r)) => {
                let left_f = l as f64;
                self.eval_binary_op(op, Value::Float(left_f), Value::Float(r))
            }
            (Value::Float(l), Value::Integer(r)) => {
                let right_f = r as f64;
                self.eval_binary_op(op, Value::Float(l), Value::Float(right_f))
            }
            // TensorF16-Integer operations (convert integer to float)
            (Value::TensorF16(t), Value::Integer(i)) => {
                self.eval_binary_op(op, Value::TensorF16(t), Value::Float(i as f64))
            }
            (Value::Integer(i), Value::TensorF16(t)) => {
                self.eval_binary_op(op, Value::Float(i as f64), Value::TensorF16(t))
            }
            // TensorF32-Integer operations (convert integer to float)
            (Value::TensorF32(t), Value::Integer(i)) => {
                self.eval_binary_op(op, Value::TensorF32(t), Value::Float(i as f64))
            }
            (Value::Integer(i), Value::TensorF32(t)) => {
                self.eval_binary_op(op, Value::Float(i as f64), Value::TensorF32(t))
            }
            _ => Err(RuntimeError::TypeError(
                "Binary operation requires compatible types".to_string(),
            )),
        }
    }

    /// Evaluate a unary operation
    pub(super) fn eval_unary_op(&self, op: &UnaryOp, operand: Value) -> RuntimeResult<Value> {
        match operand {
            Value::TensorF16(t) => {
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

                Ok(Value::TensorF16(result))
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
        let embedding_tensor = Tensor::from_vec_gpu(
            self.env.metal_device(),
            entity_embedding,
            vec![dimension]
        )?;

        Ok(Value::TensorF16(embedding_tensor))
    }

    /// Evaluate tensor indexing: tensor[i, j, ...] OR collection[i]
    pub(super) fn eval_tensor_index(&mut self, tensor_expr: &TensorExpr, indices: &[IndexExpr]) -> RuntimeResult<Value> {
        use crate::ast::IndexExpr;

        // Evaluate the tensor/collection expression
        let value = self.eval_expr(tensor_expr)?;
        
        // NEW: Handle ModelLayerCollection[index] -> returns ModelLayer
        match value {
            Value::ModelLayerCollectionF16(ref collection) => {
                if indices.len() != 1 {
                    return Err(RuntimeError::InvalidOperation(
                        format!("ModelLayerCollection indexing requires exactly 1 index, got {}", indices.len())
                    ));
                }
                
                let layer_idx = match &indices[0] {
                    IndexExpr::Int(i) => {
                        if *i < 0 {
                            return Err(RuntimeError::InvalidOperation(
                                "Negative indices not supported".to_string()
                            ));
                        }
                        *i as usize
                    }
                    IndexExpr::Var(var) => {
                        let val = self.env.get_variable(var.as_str())?;
                        let i = val.as_integer()?;
                        if i < 0 {
                            return Err(RuntimeError::InvalidOperation(
                                "Negative indices not supported".to_string()
                            ));
                        }
                        i as usize
                    }
                    IndexExpr::Slice => {
                        return Err(RuntimeError::NotImplemented(
                            "Slice indexing not supported on ModelLayerCollection".to_string()
                        ));
                    }
                };
                
                if let Some(layer) = collection.get_layer(layer_idx) {
                    return Ok(Value::ModelLayerF16(layer));
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Layer index {} not found in collection", layer_idx)
                    ));
                }
            }
            Value::ModelLayerCollectionF32(ref collection) => {
                if indices.len() != 1 {
                    return Err(RuntimeError::InvalidOperation(
                        format!("ModelLayerCollection indexing requires exactly 1 index, got {}", indices.len())
                    ));
                }
                
                let layer_idx = match &indices[0] {
                    IndexExpr::Int(i) => {
                        if *i < 0 {
                            return Err(RuntimeError::InvalidOperation(
                                "Negative indices not supported".to_string()
                            ));
                        }
                        *i as usize
                    }
                    IndexExpr::Var(var) => {
                        let val = self.env.get_variable(var.as_str())?;
                        let i = val.as_integer()?;
                        if i < 0 {
                            return Err(RuntimeError::InvalidOperation(
                                "Negative indices not supported".to_string()
                            ));
                        }
                        i as usize
                    }
                    IndexExpr::Slice => {
                        return Err(RuntimeError::NotImplemented(
                            "Slice indexing not supported on ModelLayerCollection".to_string()
                        ));
                    }
                };
                
                if let Some(layer) = collection.get_layer(layer_idx) {
                    return Ok(Value::ModelLayerF32(layer));
                } else {
                    return Err(RuntimeError::InvalidOperation(
                        format!("Layer index {} not found in collection", layer_idx)
                    ));
                }
            }
            _ => {}
        }
        
        // Continue with original tensor indexing logic
        let tensor_value = value;

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

        // Handle both f16 and f32 tensors
        match tensor_value {
            Value::TensorF16(tensor) => {
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

                // OPTIMIZATION: Fast path for CPU tensors (no clone needed)
                let value = if tensor.buffer().is_cpu() {
                    // Direct access to CPU buffer (zero-copy)
                    let cpu_vec = tensor.buffer().as_cpu()
                        .map_err(|e| RuntimeError::InvalidOperation(e.to_string()))?;
                    cpu_vec[linear_idx]
                } else {
                    // GPU tensor: transfer to CPU
                    let cpu_data = tensor.buffer().to_cpu_vec();
                    cpu_data[linear_idx]
                };

                // Return as a scalar float
                Ok(Value::Float(value.to_f32() as f64))
            }
            Value::TensorF32(tensor) => {
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

                // OPTIMIZATION: Fast path for CPU tensors (no clone needed)
                let value = if tensor.buffer().is_cpu() {
                    // Direct access to CPU buffer (zero-copy)
                    let cpu_vec = tensor.buffer().as_cpu()
                        .map_err(|e| RuntimeError::InvalidOperation(e.to_string()))?;
                    cpu_vec[linear_idx]
                } else {
                    // GPU tensor: transfer to CPU
                    let cpu_data = tensor.buffer().to_cpu_vec();
                    cpu_data[linear_idx]
                };

                // Return as a scalar float
                Ok(Value::Float(value as f64))
            }
            _ => Err(RuntimeError::TypeError("Expected tensor (f16 or f32) for indexing".to_string()))
        }
    }

    /// Evaluate Einstein summation: einsum("ij,jk->ik", A, B)
    pub(super) fn eval_einsum(&mut self, spec: &str, tensor_exprs: &[TensorExpr]) -> RuntimeResult<Value> {
        use crate::interpreter::value::ToValue;

        if tensor_exprs.is_empty() {
            return Err(RuntimeError::TypeError("einsum requires at least one tensor".to_string()));
        }

        // Evaluate all tensor expressions
        let values: Vec<Value> = tensor_exprs.iter()
            .map(|expr| self.eval_expr(expr))
            .collect::<Result<_, _>>()?;

        // Check if all tensors are f16
        let all_f16 = values.iter().all(|v| matches!(v, Value::TensorF16(_)));
        let all_f32 = values.iter().all(|v| matches!(v, Value::TensorF32(_)));

        if !all_f16 && !all_f32 {
            return Err(RuntimeError::TypeError(
                "einsum requires all tensors to be the same type (all f16 or all f32)".to_string()
            ));
        }

        if all_f16 {
            // Extract f16 tensors
            let tensors: Vec<_> = values.iter()
                .filter_map(|v| match v {
                    Value::TensorF16(t) => Some(t.clone()),
                    _ => None
                })
                .collect();

            // Create references for einsum call
            let tensor_refs: Vec<&Tensor<f16>> = tensors.iter().collect();

            // Call einsum operation
            let result = Tensor::einsum(spec, &tensor_refs)
                .map_err(|e| RuntimeError::TensorError(e))?;

            Ok(result.to_value())
        } else {
            // Extract f32 tensors
            let tensors: Vec<_> = values.iter()
                .filter_map(|v| match v {
                    Value::TensorF32(t) => Some(t.clone()),
                    _ => None
                })
                .collect();

            // Create references for einsum call
            let tensor_refs: Vec<&Tensor<f32>> = tensors.iter().collect();

            // Call einsum operation
            let result = Tensor::einsum(spec, &tensor_refs)
                .map_err(|e| RuntimeError::TensorError(e))?;

            Ok(result.to_value())
        }
    }

    /// Format a string with arguments (like Rust's println!)
    pub(crate) fn format_string(&self, format: &str, args: &[Value]) -> RuntimeResult<String> {
        let mut result = String::new();
        let mut arg_idx = 0;
        let mut chars = format.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek() == Some(&'}') {
                    chars.next(); // consume '}'
                    if arg_idx < args.len() {
                        result.push_str(&Self::value_to_string(&args[arg_idx])?);
                        arg_idx += 1;
                    } else {
                        return Err(RuntimeError::TypeError(
                            "Not enough arguments for format string".to_string()
                        ));
                    }
                } else {
                    result.push(ch);
                }
            } else if ch == '\\' {
                // Handle escape sequences
                if let Some(&next_ch) = chars.peek() {
                    chars.next();
                    match next_ch {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        'r' => result.push('\r'),
                        '\\' => result.push('\\'),
                        _ => {
                            result.push('\\');
                            result.push(next_ch);
                        }
                    }
                } else {
                    result.push(ch);
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }

    /// Convert Value to string for display
    fn value_to_string(value: &Value) -> RuntimeResult<String> {
        Ok(match value {
            Value::TensorF16(t) => {
                format!("Tensor<f16>(shape={:?})", t.dims())
            }
            Value::TensorF32(t) => {
                format!("Tensor<f32>(shape={:?})", t.dims())
            }
            Value::Boolean(b) => b.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Void => "void".to_string(),
            Value::ModelF16(_) => "<ModelF16>".to_string(),
            Value::ModelF32(_) => "<ModelF32>".to_string(),
            Value::ModelLayerCollectionF16(_) => "<ModelLayerCollectionF16>".to_string(),
            Value::ModelLayerCollectionF32(_) => "<ModelLayerCollectionF32>".to_string(),
            Value::ModelLayerF16(_) => "<ModelLayerF16>".to_string(),
            Value::ModelLayerF32(_) => "<ModelLayerF32>".to_string(),
            Value::ModelFeatureF16(_) => "<ModelFeatureF16>".to_string(),
            Value::ModelFeatureF32(_) => "<ModelFeatureF32>".to_string(),
            Value::Tokenizer(_) => "<Tokenizer>".to_string(),
            Value::TokenIds(ids) => format!("{:?}", ids),
            Value::TokenIdArray(arr) => format!("{:?}", arr.data()),
            Value::Type(t) => format!("<Type: {}>", t),
            Value::KVCache(cache) => {
                let c = cache.lock().unwrap();
                format!("<KVCache: {} layers>", c.kvs.len())
            }
        })
    }

    /// Convert Value to display string for simple print mode
    pub(crate) fn value_to_display(&self, value: &Value) -> String {
        match value {
            Value::TensorF16(t) => {
                format!("Tensor<f16>(shape={:?})", t.dims())
            }
            Value::TensorF32(t) => {
                format!("Tensor<f32>(shape={:?})", t.dims())
            }
            Value::Boolean(b) => b.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Void => "void".to_string(),
            Value::ModelF16(_) => "<ModelF16>".to_string(),
            Value::ModelF32(_) => "<ModelF32>".to_string(),
            Value::ModelLayerCollectionF16(_) => "<ModelLayerCollectionF16>".to_string(),
            Value::ModelLayerCollectionF32(_) => "<ModelLayerCollectionF32>".to_string(),
            Value::ModelLayerF16(_) => "<ModelLayerF16>".to_string(),
            Value::ModelLayerF32(_) => "<ModelLayerF32>".to_string(),
            Value::ModelFeatureF16(_) => "<ModelFeatureF16>".to_string(),
            Value::ModelFeatureF32(_) => "<ModelFeatureF32>".to_string(),
            Value::Tokenizer(_) => "<Tokenizer>".to_string(),
            Value::TokenIds(ids) => format!("{:?}", ids),
            Value::TokenIdArray(arr) => format!("{:?}", arr.data()),
            Value::Type(t) => format!("<Type: {}>", t),
            Value::KVCache(cache) => {
                let c = cache.lock().unwrap();
                format!("<KVCache: {} layers>", c.kvs.len())
            }
        }
    }
}
