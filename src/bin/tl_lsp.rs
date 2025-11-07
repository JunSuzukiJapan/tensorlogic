use tower_lsp::{LspService, Server};
use tensorlogic::lsp::TensorLogicBackend;

#[tokio::main]
async fn main() {
    // Set up logging
    env_logger::init();

    // Create stdin/stdout for LSP communication
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    // Create the language server
    let (service, socket) = LspService::new(|client| TensorLogicBackend::new(client));

    // Run the server
    Server::new(stdin, stdout, socket).serve(service).await;
}
