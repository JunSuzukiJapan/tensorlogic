//! TensorLogic CLI
//!
//! Command-line interface for running TensorLogic programs.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use tensorlogic::error_reporting::{helpers, ErrorReporter, FrameType, StackFrame, StackTrace};
use tensorlogic::interpreter::Interpreter;
use tensorlogic::parser::TensorLogicParser;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let command = &args[1];

    // Check for debug flag
    let debug_mode = args.contains(&"--debug".to_string()) || args.contains(&"-d".to_string());

    // Check for test/bench flags
    let test_mode = args.contains(&"--test".to_string());
    let bench_mode = args.contains(&"--bench".to_string());

    match command.as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path");
                eprintln!(
                    "Usage: {} run <file.tl> [--debug] [--test] [--bench]",
                    args[0]
                );
                std::process::exit(1);
            }
            let file_path = &args[2];

            #[cfg(feature = "llvm")]
            {
                // Parse LLVM-related options
                let use_jit = args.contains(&"--jit".to_string());
                let emit_llvm = parse_option_value(&args, "--emit-llvm");
                let emit_asm = parse_option_value(&args, "--emit-asm");
                let opt_level = parse_option_value(&args, "--opt-level")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(2);

                if let Err(e) = run_file(
                    file_path,
                    debug_mode,
                    test_mode,
                    bench_mode,
                    use_jit,
                    emit_llvm,
                    emit_asm,
                    opt_level,
                ) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
            #[cfg(not(feature = "llvm"))]
            {
                if let Err(e) = run_file(file_path, debug_mode, test_mode, bench_mode) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "repl" => {
            if let Err(e) = run_repl(debug_mode) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        "help" | "--help" | "-h" => {
            print_usage(&args[0]);
        }
        "version" | "--version" | "-v" => {
            println!("TensorLogic v{}", env!("CARGO_PKG_VERSION"));
        }
        _ => {
            eprintln!("Error: Unknown command '{}'", command);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }
}

/// Parse option value from command line arguments
/// Example: parse_option_value(&["prog", "--emit-llvm", "out.ll"], "--emit-llvm") => Some("out.ll")
fn parse_option_value(args: &[String], option: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == option && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn print_usage(program_name: &str) {
    println!("TensorLogic v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("    {} <COMMAND> [OPTIONS]", program_name);
    println!();
    println!("COMMANDS:");
    println!("    run <file>    Run a TensorLogic program file");
    println!("    repl          Start an interactive REPL session");
    println!("    help          Print this help message");
    println!("    version       Print version information");
    println!();
    println!("OPTIONS:");
    println!("    --debug, -d          Enable debug mode with detailed error information");
    println!("    --test               Run test blocks instead of main block");
    println!("    --bench              Run benchmark blocks with timing");
    #[cfg(feature = "llvm")]
    println!("    --jit                Use JIT compilation for faster execution");
    #[cfg(feature = "llvm")]
    println!("    --emit-llvm <file>   Emit LLVM IR to the specified file");
    #[cfg(feature = "llvm")]
    println!("    --emit-asm <file>    Emit native assembly to the specified file");
    #[cfg(feature = "llvm")]
    println!("    --opt-level <0-3>    Set optimization level (default: 2)");
    println!();
    println!("EXAMPLES:");
    println!("    {} run examples/linear_regression.tl", program_name);
    println!("    {} run examples/test.tl --debug", program_name);
    println!("    {} run examples/test.tl --test", program_name);
    println!("    {} run examples/benchmark.tl --bench", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --jit", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-llvm output.ll", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-asm output.s", program_name);
    println!("    {} repl --debug", program_name);
}

#[cfg(feature = "llvm")]
fn run_file(
    file_path: &str,
    debug_mode: bool,
    test_mode: bool,
    bench_mode: bool,
    use_jit: bool,
    emit_llvm: Option<String>,
    emit_asm: Option<String>,
    opt_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use tensorlogic::compiler::{CompilationMode, CompilerOptions, JITCompiler, OutputFormat, OutputWriter};
    use inkwell::{context::Context, OptimizationLevel};

    run_file_impl(
        file_path,
        debug_mode,
        test_mode,
        bench_mode,
        Some((use_jit, emit_llvm, emit_asm, opt_level)),
    )
}

#[cfg(not(feature = "llvm"))]
fn run_file(
    file_path: &str,
    debug_mode: bool,
    test_mode: bool,
    bench_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    run_file_impl(file_path, debug_mode, test_mode, bench_mode, None)
}

fn run_file_impl(
    file_path: &str,
    debug_mode: bool,
    test_mode: bool,
    bench_mode: bool,
    #[allow(unused_variables)]
    llvm_options: Option<(bool, Option<String>, Option<String>, u8)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File not found: {}", file_path).into());
    }

    // Read file contents
    let source = fs::read_to_string(path)?;

    // Create error reporter with source
    let mut error_reporter = ErrorReporter::with_source(source.clone());

    // Parse program
    if debug_mode {
        println!("[DEBUG] Parsing {}...", file_path);
        println!("[DEBUG] Source length: {} bytes", source.len());
    } else {
        println!("Parsing {}...", file_path);
    }

    let program = match TensorLogicParser::parse_program(&source) {
        Ok(program) => program,
        Err(e) => {
            // Report parse error with enhanced formatting
            let diag = helpers::parse_error_diagnostic(e.to_string(), None);
            error_reporter.report(diag);
            eprintln!("{}", error_reporter.format_all());

            if debug_mode {
                eprintln!("\n[DEBUG] Parse error details:");
                eprintln!("[DEBUG] Error: {:?}", e);
            }
            std::process::exit(1);
        }
    };

    if debug_mode {
        println!("[DEBUG] Parsed {} declarations", program.declarations.len());
        if program.main_block.is_some() {
            println!("[DEBUG] Found main block");
        }
        println!("[DEBUG] Found {} test blocks", program.test_blocks.len());
        println!("[DEBUG] Found {} bench blocks", program.bench_blocks.len());
    } else {
        println!("Parsed {} declarations", program.declarations.len());
        if program.main_block.is_some() {
            println!("Found main block");
        }
    }

    // Handle LLVM compilation if requested
    #[cfg(feature = "llvm")]
    if let Some((use_jit, emit_llvm, emit_asm, opt_level)) = llvm_options {
        use tensorlogic::compiler::{JITCompiler, OutputFormat, OutputWriter};
        use inkwell::{context::Context, OptimizationLevel};

        let opt = match opt_level {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Less,
            2 => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive,
        };

        // Handle --emit-llvm
        if let Some(output_path) = emit_llvm {
            println!("\n=== Compiling to LLVM IR ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            writer.write(&program, &output_path, OutputFormat::LLVMAssembly, opt)?;
            return Ok(());
        }

        // Handle --emit-asm
        if let Some(output_path) = emit_asm {
            println!("\n=== Compiling to Native Assembly ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            match writer.write(&program, &output_path, OutputFormat::NativeAssembly, opt) {
                Ok(_) => return Ok(()),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        // Handle --jit
        if use_jit {
            println!("\n=== Running with JIT Compilation ===\n");
            let context = Context::create();
            let mut jit = JITCompiler::new(&context);

            if let Err(e) = jit.compile(&program, opt) {
                eprintln!("JIT compilation failed: {}", e);
                eprintln!("Falling back to interpreter...");
            } else {
                // Try to execute with JIT
                unsafe {
                    match jit.execute_main() {
                        Ok(exit_code) => {
                            println!("\n✅ Program executed successfully (JIT)!");
                            if exit_code != 0 {
                                println!("Exit code: {}", exit_code);
                            }
                            return Ok(());
                        }
                        Err(e) => {
                            eprintln!("JIT execution failed: {}", e);
                            eprintln!("Falling back to interpreter...");
                        }
                    }
                }
            }
        }
    }

    // Execute program with interpreter
    let mut interpreter = Interpreter::new();

    // Set current file path for import resolution
    interpreter.set_current_file(path.canonicalize()?);

    // Determine which blocks to execute
    let result = if test_mode {
        // Run test blocks
        println!("\n=== Running Tests ===\n");
        interpreter.execute_tests(&program)
    } else if bench_mode {
        // Run benchmark blocks
        println!("\n=== Running Benchmarks ===\n");
        interpreter.execute_benchmarks(&program)
    } else {
        // Run main block (default)
        println!("\nExecuting...\n");
        interpreter.execute(&program)
    };

    if let Err(e) = result {
        // Build stack trace from error context
        let mut stack_trace = StackTrace::new();

        // Add main execution frame
        stack_trace.push(StackFrame::with_location(
            "main".to_string(),
            FrameType::MainBlock,
            file_path.to_string(),
            0,
        ));

        // Add error chain as stack frames
        let mut source = e.source();
        let mut level = 1;
        while let Some(err) = source {
            stack_trace.push(StackFrame::new(
                format!("error level {}", level),
                FrameType::Expression,
            ));
            source = err.source();
            level += 1;
        }

        // Report runtime error with stack trace
        let diag = helpers::runtime_error_with_trace(e.to_string(), None, stack_trace);
        error_reporter.report(diag);
        eprintln!("{}", error_reporter.format_all());

        if debug_mode {
            eprintln!("\n[DEBUG] Runtime error details:");
            eprintln!("[DEBUG] Error: {:?}", e);
        }
        std::process::exit(1);
    }

    println!("\n✅ Program executed successfully!");

    // Print final state
    // print_final_state(&interpreter);

    Ok(())
}

fn run_repl(debug_mode: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("TensorLogic REPL v{}", env!("CARGO_PKG_VERSION"));
    if debug_mode {
        println!("[DEBUG MODE ENABLED]");
    }
    println!("Type 'exit' or 'quit' to exit, 'help' for help");
    println!();

    let mut interpreter = Interpreter::new();
    let mut line_num = 1;

    loop {
        // Print prompt
        print!("tl[{}]> ", line_num);
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        // Handle special commands
        match input {
            "" => continue,
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                print_repl_help();
                continue;
            }
            "vars" => {
                print_variables(&interpreter);
                continue;
            }
            "clear" => {
                interpreter = Interpreter::new();
                println!("Environment cleared");
                continue;
            }
            _ => {}
        }

        // Try to parse and execute
        match execute_repl_input(&mut interpreter, input, debug_mode) {
            Ok(result) => {
                if let Some(msg) = result {
                    println!("{}", msg);
                }
            }
            Err(e) => {
                // Create error reporter for REPL input
                let mut error_reporter = ErrorReporter::with_source(input.to_string());
                let diag = helpers::parse_error_diagnostic(e.to_string(), None);
                error_reporter.report(diag);
                eprintln!("{}", error_reporter.format_all());

                if debug_mode {
                    eprintln!("[DEBUG] Error: {:?}", e);
                }
            }
        }

        line_num += 1;
    }

    Ok(())
}

fn execute_repl_input(
    interpreter: &mut Interpreter,
    input: &str,
    debug_mode: bool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    // Try to parse as a complete program
    let wrapped = format!("main {{ {} }}", input);

    if debug_mode {
        eprintln!("[DEBUG] Trying to parse as statement: {}", input);
    }

    match TensorLogicParser::parse_program(&wrapped) {
        Ok(program) => {
            if debug_mode {
                eprintln!("[DEBUG] Successfully parsed as statement");
            }
            interpreter.execute(&program)?;
            Ok(Some("✓".to_string()))
        }
        Err(_) => {
            if debug_mode {
                eprintln!("[DEBUG] Failed as statement, trying as declaration");
            }
            // Try as a declaration
            match TensorLogicParser::parse_program(input) {
                Ok(program) => {
                    if debug_mode {
                        eprintln!("[DEBUG] Successfully parsed as declaration");
                    }
                    interpreter.execute(&program)?;
                    Ok(Some("✓".to_string()))
                }
                Err(e) => Err(format!("Parse error: {}", e).into()),
            }
        }
    }
}

fn print_repl_help() {
    println!("REPL Commands:");
    println!("  exit, quit    Exit the REPL");
    println!("  help          Show this help message");
    println!("  vars          Show all variables");
    println!("  clear         Clear the environment");
    println!();
    println!("TensorLogic Syntax:");
    println!("  tensor w: float32[10, 20]           # Declare tensor");
    println!("  x := 10                              # Assignment");
    println!("  y := x + 5                           # Expression");
    println!("  if x > 0 {{ y := 1 }}                 # Control flow");
    println!("  for i in range(5) {{ x := x + i }}   # Loop");
    println!();
}

fn print_varialbes_sub(interpreter: &Interpreter) {
    let vars = interpreter.get_all_variables();
    if let Some(vars) = vars {
        for (name, value) in vars {
            match value {
                tensorlogic::interpreter::Value::Float(f) => {
                    println!("  {} = {}", name, f);
                }
                tensorlogic::interpreter::Value::Integer(i) => {
                    println!("  {} = {}", name, i);
                }
                tensorlogic::interpreter::Value::Boolean(b) => {
                    println!("  {} = {}", name, b);
                }
                tensorlogic::interpreter::Value::TensorF16(t) => {
                    println!("  {} = TensorF16{:?}", name, t.shape());
                },
                tensorlogic::interpreter::Value::TensorF32(t) => {
                    println!("  {} = TensorF32{:?}", name, t.shape());
                },
                _ => {
                    println!("  {} = {:?}", name, value);
                }
            }
        }
    }
}

fn print_variables(interpreter: &Interpreter) {
    println!("Variables:");
    print_varialbes_sub(interpreter);
}

#[allow(dead_code)]
fn print_final_state(interpreter: &Interpreter) {
    println!("\nFinal state:");
    print_varialbes_sub(interpreter);
}
