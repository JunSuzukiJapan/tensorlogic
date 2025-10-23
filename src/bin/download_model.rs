use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;

/// モデル情報
struct ModelInfo {
    name: &'static str,
    filename: &'static str,
    url: &'static str,
    size_mb: u32,
    description: &'static str,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "tinyllama",
        filename: "tinyllama-1.1b-chat-q4_0.gguf",
        url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        size_mb: 600,
        description: "TinyLlama 1.1B Chat (Q4_0) - Fast, lightweight model for simple conversations",
    },
    ModelInfo {
        name: "phi2",
        filename: "phi-2-q4_0.gguf",
        url: "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf",
        size_mb: 1600,
        description: "Phi-2 (Q4_0) - High-quality coding and reasoning",
    },
    ModelInfo {
        name: "mistral",
        filename: "mistral-7b-instruct-q4_0.gguf",
        url: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
        size_mb: 3800,
        description: "Mistral 7B Instruct (Q4_0) - Professional conversations and complex tasks",
    },
];

fn get_models_dir() -> PathBuf {
    let home = env::var("HOME").expect("HOME environment variable not set");
    let mut path = PathBuf::from(home);
    path.push(".llm");
    path.push("models");
    path
}

fn download_model(model: &ModelInfo) -> io::Result<()> {
    let models_dir = get_models_dir();

    // ディレクトリ作成
    fs::create_dir_all(&models_dir)?;

    let target_path = models_dir.join(model.filename);

    // 既に存在するかチェック
    if target_path.exists() {
        println!("✓ Model already exists: {}", target_path.display());
        println!("  To re-download, delete the file first");
        return Ok(());
    }

    println!("📥 Downloading: {}", model.name);
    println!("   Description: {}", model.description);
    println!("   Size: ~{}MB", model.size_mb);
    println!("   URL: {}", model.url);
    println!("   Target: {}", target_path.display());
    println!();

    // curlを使ってダウンロード（進捗表示付き）
    let status = Command::new("curl")
        .arg("-L") // リダイレクトをフォロー
        .arg("-o")
        .arg(&target_path)
        .arg("--progress-bar")
        .arg(model.url)
        .status()?;

    if !status.success() {
        eprintln!("❌ Download failed");
        if target_path.exists() {
            fs::remove_file(&target_path)?;
        }
        return Err(io::Error::new(io::ErrorKind::Other, "Download failed"));
    }

    println!();
    println!("✅ Successfully downloaded: {}", model.filename);
    println!("   Location: {}", target_path.display());
    println!();
    println!("To use this model in TensorLogic:");
    println!("   tl run examples/local_llm_chat.tl");
    println!();

    Ok(())
}

fn list_models() {
    println!("Available models:");
    println!();

    for model in MODELS {
        println!("  {} - {}", model.name, model.description);
        println!("    Size: ~{}MB", model.size_mb);
        println!("    Download: cargo run --bin download_model -- --model {}", model.name);
        println!();
    }
}

fn list_downloaded() {
    let models_dir = get_models_dir();

    println!("Downloaded models in {}:", models_dir.display());
    println!();

    if !models_dir.exists() {
        println!("  (no models directory yet)");
        return;
    }

    match fs::read_dir(&models_dir) {
        Ok(entries) => {
            let mut found_any = false;
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        let size_mb = metadata.len() / 1_000_000;
                        println!("  ✓ {} ({}MB)", entry.file_name().to_string_lossy(), size_mb);
                        found_any = true;
                    }
                }
            }
            if !found_any {
                println!("  (no models downloaded yet)");
            }
        }
        Err(e) => {
            eprintln!("  Error reading directory: {}", e);
        }
    }
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    println!("TensorLogic Model Downloader");
    println!("============================");
    println!();

    // 引数パース
    if args.len() < 2 {
        println!("Usage:");
        println!("  cargo run --bin download_model -- --list          # List available models");
        println!("  cargo run --bin download_model -- --downloaded    # List downloaded models");
        println!("  cargo run --bin download_model -- --model <name>  # Download a model");
        println!();
        list_models();
        list_downloaded();
        return;
    }

    match args[1].as_str() {
        "--list" => {
            list_models();
        }
        "--downloaded" => {
            list_downloaded();
        }
        "--model" => {
            if args.len() < 3 {
                eprintln!("❌ Error: --model requires a model name");
                println!();
                list_models();
                return;
            }

            let model_name = &args[2];

            if let Some(model) = MODELS.iter().find(|m| m.name == model_name) {
                if let Err(e) = download_model(model) {
                    eprintln!("❌ Download error: {}", e);
                    std::process::exit(1);
                }
            } else {
                eprintln!("❌ Error: Unknown model '{}'", model_name);
                println!();
                list_models();
                std::process::exit(1);
            }
        }
        "--help" | "-h" => {
            println!("TensorLogic Model Downloader");
            println!();
            println!("USAGE:");
            println!("    cargo run --bin download_model -- [OPTIONS]");
            println!();
            println!("OPTIONS:");
            println!("    --list              List available models");
            println!("    --downloaded        List already downloaded models");
            println!("    --model <name>      Download a specific model");
            println!("    --help, -h          Print help information");
            println!();
            println!("EXAMPLES:");
            println!("    # Download TinyLlama");
            println!("    cargo run --bin download_model -- --model tinyllama");
            println!();
            println!("    # List available models");
            println!("    cargo run --bin download_model -- --list");
            println!();
        }
        other => {
            eprintln!("❌ Error: Unknown option '{}'", other);
            eprintln!("Use --help for usage information");
            std::process::exit(1);
        }
    }
}
