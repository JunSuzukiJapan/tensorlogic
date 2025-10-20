//! TensorLogic CLI
//!
//! Command-line interface for running TensorLogic programs.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use tensorlogic::parser::TensorLogicParser;
use tensorlogic::interpreter::Interpreter;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: Missing file path");
                eprintln!("Usage: {} run <file.tl>", args[0]);
                std::process::exit(1);
            }
            let file_path = &args[2];
            if let Err(e) = run_file(file_path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        "repl" => {
            if let Err(e) = run_repl() {
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
    println!("    {} <COMMAND>", program_name);
    println!();
    println!("COMMANDS:");
    println!("    run <file>    Run a TensorLogic program file");
    println!("    repl          Start an interactive REPL session");
    println!("    help          Print this help message");
    println!("    version       Print version information");
    println!();
    println!("EXAMPLES:");
    println!("    {} run examples/linear_regression.tl", program_name);
    println!("    {} repl", program_name);
}

fn run_file(file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Check if file exists
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File not found: {}", file_path).into());
    }

    // Read file contents
    let source = fs::read_to_string(path)?;

    // Parse program
    println!("Parsing {}...", file_path);
    let program = TensorLogicParser::parse_program(&source)
        .map_err(|e| format!("Parse error: {}", e))?;

    println!("Parsed {} declarations", program.declarations.len());
    if program.main_block.is_some() {
        println!("Found main block");
    }

    // Execute program
    println!("\nExecuting...\n");
    let mut interpreter = Interpreter::new();
    interpreter.execute(&program)
        .map_err(|e| format!("Runtime error: {}", e))?;

    println!("\n✅ Program executed successfully!");

    // Print final state
    print_final_state(&interpreter);

    Ok(())
}

fn run_repl() -> Result<(), Box<dyn std::error::Error>> {
    println!("TensorLogic REPL v{}", env!("CARGO_PKG_VERSION"));
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
        match execute_repl_input(&mut interpreter, input) {
            Ok(result) => {
                if let Some(msg) = result {
                    println!("{}", msg);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        line_num += 1;
    }

    Ok(())
}

fn execute_repl_input(
    interpreter: &mut Interpreter,
    input: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    // Try to parse as a complete program
    let wrapped = format!("main {{ {} }}", input);

    match TensorLogicParser::parse_program(&wrapped) {
        Ok(program) => {
            interpreter.execute(&program)?;
            Ok(Some("✓".to_string()))
        }
        Err(_) => {
            // Try as a declaration
            match TensorLogicParser::parse_program(input) {
                Ok(program) => {
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

fn print_variables(interpreter: &Interpreter) {
    println!("Variables:");
    let vars = interpreter.get_all_variables();
    if vars.is_empty() {
        println!("  (none)");
    } else {
        for (name, value) in vars {
            println!("  {} = {:?}", name, value);
        }
    }
}

fn print_final_state(interpreter: &Interpreter) {
    let vars = interpreter.get_all_variables();
    if !vars.is_empty() {
        println!("\nFinal state:");
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
                tensorlogic::interpreter::Value::Tensor(t) => {
                    println!("  {} = Tensor{:?}", name, t.shape());
                }
                _ => {
                    println!("  {} = {:?}", name, value);
                }
            }
        }
    }
}
