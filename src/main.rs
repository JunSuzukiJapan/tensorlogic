//! TensorLogic CLI
//!
//! Command-line interface for running TensorLogic programs.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process;

use sysinfo::{ProcessRefreshKind, System};
use tensorlogic::error_reporting::{helpers, ErrorReporter, FrameType, StackFrame, StackTrace};
use tensorlogic::interpreter::Interpreter;
use tensorlogic::parser::TensorLogicParser;

/// Check if another TensorLogic process is already running
/// If found, print error message and exit to prevent GPU conflicts
fn check_concurrent_execution() {
    let current_pid = process::id();
    let mut system = System::new();
    system.refresh_processes_specifics(ProcessRefreshKind::everything());

    let tl_processes: Vec<_> = system
        .processes()
        .iter()
        .filter(|(pid, process)| {
            // Check if it's a different process
            if pid.as_u32() == current_pid {
                return false;
            }

            // Check if the process name is exactly "tl" (our binary name)
            let name = process.name();
            name == "tl" || name.ends_with("/tl") || name.ends_with("\\tl")
        })
        .collect();

    if !tl_processes.is_empty() {
        eprintln!("╔════════════════════════════════════════════════════════════════╗");
        eprintln!("║          ⚠️  TensorLogic Concurrent Execution Error  ⚠️         ║");
        eprintln!("╠════════════════════════════════════════════════════════════════╣");
        eprintln!("║                                                                ║");
        eprintln!("║  Another TensorLogic process is already running!               ║");
        eprintln!("║                                                                ║");
        eprintln!("║  Running multiple TL scripts simultaneously causes:            ║");
        eprintln!("║  • GPU resource conflicts                                      ║");
        eprintln!("║  • System hangs and crashes                                    ║");
        eprintln!("║  • Potential data loss                                         ║");
        eprintln!("║                                                                ║");
        eprintln!("║  Detected processes:                                           ║");
        for (pid, process) in tl_processes.iter().take(5) {
            eprintln!("║  • PID {}: {}                                        ",
                     pid,
                     process.name());
        }
        eprintln!("║                                                                ║");
        eprintln!("║  Please wait for the other process to complete, or run:       ║");
        eprintln!("║  $ pkill tl                                                    ║");
        eprintln!("║                                                                ║");
        eprintln!("╚════════════════════════════════════════════════════════════════╝");

        process::exit(1);
    }
}

fn main() {
    // CRITICAL: Check for concurrent execution before doing anything
    // This prevents GPU conflicts and system hangs
    check_concurrent_execution();

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
                let emit_obj = parse_option_value(&args, "--emit-obj");
                let emit_lib = parse_option_value(&args, "--emit-lib");
                let emit_shared = parse_option_value(&args, "--emit-shared");
                let emit_bin = parse_option_value(&args, "--emit-bin");
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
                    emit_obj,
                    emit_lib,
                    emit_shared,
                    emit_bin,
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
    println!("    --emit-obj <file>    Emit object file (.o/.obj)");
    #[cfg(feature = "llvm")]
    println!("    --emit-lib <file>    Emit static library (.a/.lib)");
    #[cfg(feature = "llvm")]
    println!("    --emit-shared <file> Emit shared library (.so/.dll/.dylib)");
    #[cfg(feature = "llvm")]
    println!("    --emit-bin <file>    Emit executable binary");
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
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-obj output.o", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-lib libcompute.a", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-shared libcompute.so", program_name);
    #[cfg(feature = "llvm")]
    println!("    {} run examples/compute.tl --emit-bin compute", program_name);
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
    emit_obj: Option<String>,
    emit_lib: Option<String>,
    emit_shared: Option<String>,
    emit_bin: Option<String>,
    opt_level: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    use tensorlogic::compiler::{CompilationMode, CompilerOptions, JITCompiler, OutputFormat, OutputWriter};
    use inkwell::{context::Context, OptimizationLevel};

    run_file_impl(
        file_path,
        debug_mode,
        test_mode,
        bench_mode,
        Some((use_jit, emit_llvm, emit_asm, emit_obj, emit_lib, emit_shared, emit_bin, opt_level)),
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
    llvm_options: Option<(bool, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, u8)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // GPU memory before execution (captured outside the block)
    let gpu_memory_before = {
        #[cfg(target_os = "macos")]
        {
            use tensorlogic::device::MetalDevice;
            match MetalDevice::new() {
                Ok(device) => {
                    let allocated = device.current_allocated_size();
                    // Only show detailed log if TL_MEMORY_CHECK is set
                    if std::env::var("TL_MEMORY_CHECK").is_ok() {
                        eprintln!("\n=== GPU Memory Check: Before Execution ===");
                        eprintln!("GPU memory allocated: {:.2} MB", allocated as f64 / 1_048_576.0);
                        eprintln!("==========================================\n");
                    }
                    Some(allocated)
                }
                Err(_) => None
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    };

    // Execute in isolated block to ensure all variables are dropped before memory check
    let result = {
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
        if let Some((use_jit, emit_llvm, emit_asm, emit_obj, emit_lib, emit_shared, emit_bin, opt_level)) = llvm_options {
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

        // Handle --emit-obj
        if let Some(output_path) = emit_obj {
            println!("\n=== Compiling to Object File ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            writer.write(&program, &output_path, OutputFormat::ObjectFile, opt)?;
            return Ok(());
        }

        // Handle --emit-lib
        if let Some(output_path) = emit_lib {
            println!("\n=== Compiling to Static Library ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            writer.write(&program, &output_path, OutputFormat::StaticLibrary, opt)?;
            return Ok(());
        }

        // Handle --emit-shared
        if let Some(output_path) = emit_shared {
            println!("\n=== Compiling to Shared Library ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            writer.write(&program, &output_path, OutputFormat::SharedLibrary, opt)?;
            return Ok(());
        }

        // Handle --emit-bin
        if let Some(output_path) = emit_bin {
            println!("\n=== Compiling to Executable ===\n");
            let context = Context::create();
            let writer = OutputWriter::new(&context);
            writer.write(&program, &output_path, OutputFormat::Executable, opt)?;
            return Ok(());
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

        // Handle result and print success/error messages
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
        Ok(())
    }; // End of execution block - all variables including interpreter are now dropped

    // Check GPU memory after all variables are dropped
    if let Some(memory_before) = gpu_memory_before {
        #[cfg(target_os = "macos")]
        {
            use tensorlogic::device::MetalDevice;
            match MetalDevice::new() {
                Ok(device) => {
                    let memory_after = device.current_allocated_size();
                    let memory_diff = memory_after as i64 - memory_before as i64;

                    // Only show detailed stats if TL_MEMORY_CHECK is set
                    let show_details = std::env::var("TL_MEMORY_CHECK").is_ok();

                    if show_details {
                        eprintln!("\n=== GPU Memory Check: After Execution ===");
                        eprintln!("GPU memory allocated: {:.2} MB", memory_after as f64 / 1_048_576.0);
                        eprintln!("Memory change: {:+.2} MB", memory_diff as f64 / 1_048_576.0);

                        // Also show buffer pool stats for context
                        let pool_stats = device.buffer_pool().stats();
                        eprintln!("\nBuffer Pool Stats:");
                        eprintln!("  Pooled buffers: {}", pool_stats.total_pooled);
                        eprintln!("  Pool memory: {:.2} MB", pool_stats.total_memory as f64 / 1_048_576.0);

                        if pool_stats.allocation_count > 0 {
                            let reuse_rate = (pool_stats.reuse_count as f64 / (pool_stats.allocation_count + pool_stats.reuse_count) as f64) * 100.0;
                            eprintln!("  Reuse rate: {:.1}%", reuse_rate);
                        }
                    }

                    // Always detect and fix memory leaks
                    // Since this is after all variables are dropped, any remaining memory is a leak
                    if memory_diff > 0 {
                        eprintln!("\n⚠️  WARNING: GPU memory leak detected!");
                        eprintln!("   {:.2} MB of GPU memory was not freed after execution.", memory_diff as f64 / 1_048_576.0);

                        // Force purge all buffers to release GPU memory
                        eprintln!("🔧 Attempting to force-release GPU memory...");
                        device.purge_all_buffers();

                        // Check memory after purge
                        let memory_after_purge = device.current_allocated_size();
                        let purged_amount = memory_after as i64 - memory_after_purge as i64;
                        eprintln!("   After purge: {:.2} MB ({:+.2} MB freed)\n",
                                 memory_after_purge as f64 / 1_048_576.0,
                                 purged_amount as f64 / 1_048_576.0);
                    } else if show_details {
                        eprintln!("\n✅ No memory leaks detected");
                        eprintln!("   All GPU memory properly released.");
                        eprintln!("=========================================\n");
                    }
                }
                Err(_) => {}
            }
        }
    }

    result
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
