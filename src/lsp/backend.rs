use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::parser::TensorLogicParser;
use super::diagnostics::generate_diagnostics;
use super::completion::get_completions;
use super::hover::get_hover_info;
use super::goto_definition::get_definition;

/// TensorLogic Language Server Backend
pub struct TensorLogicBackend {
    client: Client,
    /// Document storage: URI -> source code
    documents: DashMap<String, String>,
}

impl TensorLogicBackend {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
        }
    }

    /// Validate document and publish diagnostics
    async fn validate_document(&self, uri: Url) {
        let text = match self.documents.get(&uri.to_string()) {
            Some(text) => text.clone(),
            None => return,
        };

        let diagnostics = generate_diagnostics(&text);
        self.client.publish_diagnostics(uri, diagnostics, None).await;
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for TensorLogicBackend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "TensorLogic Language Server".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![
                        ".".to_string(),
                        ":".to_string(),
                    ]),
                    ..Default::default()
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace: Some(WorkspaceServerCapabilities {
                    workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Left(true)),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "TensorLogic Language Server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        self.documents.insert(uri.clone(), params.text_document.text);
        self.validate_document(params.text_document.uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        if let Some(change) = params.content_changes.first() {
            self.documents.insert(uri.clone(), change.text.clone());
            self.validate_document(params.text_document.uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        self.documents.remove(&uri);
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri.to_string();
        let position = params.text_document_position.position;

        if let Some(text) = self.documents.get(&uri) {
            let completions = get_completions(&text, position);
            Ok(Some(CompletionResponse::Array(completions)))
        } else {
            Ok(None)
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri.to_string();
        let position = params.text_document_position_params.position;

        if let Some(text) = self.documents.get(&uri) {
            Ok(get_hover_info(&text, position))
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri.to_string();
        let position = params.text_document_position_params.position;

        if let Some(text) = self.documents.get(&uri) {
            if let Some(location) = get_definition(&text, position, &uri) {
                Ok(Some(GotoDefinitionResponse::Scalar(location)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri.to_string();

        if let Some(text) = self.documents.get(&uri) {
            // Parse and extract symbols
            let symbols = extract_symbols(&text);
            Ok(Some(DocumentSymbolResponse::Flat(symbols)))
        } else {
            Ok(None)
        }
    }
}

/// Extract document symbols (functions, tensors, etc.)
fn extract_symbols(text: &str) -> Vec<SymbolInformation> {
    let mut symbols = Vec::new();

    // Simple regex-based symbol extraction
    // In production, this should use the AST from the parser
    for (line_num, line) in text.lines().enumerate() {
        let line_num = line_num as u32;

        // Function declarations: fn name(...)
        if let Some(fn_pos) = line.find("fn ") {
            if let Some(name_start) = line[fn_pos + 3..].find(|c: char| c.is_alphanumeric()) {
                let name_start = fn_pos + 3 + name_start;
                if let Some(name_end) = line[name_start..].find(|c: char| !c.is_alphanumeric() && c != '_') {
                    let name = &line[name_start..name_start + name_end];
                    symbols.push(SymbolInformation {
                        name: name.to_string(),
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: Url::parse("file:///dummy").unwrap(),
                            range: Range {
                                start: Position { line: line_num, character: name_start as u32 },
                                end: Position { line: line_num, character: (name_start + name_end) as u32 },
                            },
                        },
                        container_name: None,
                    });
                }
            }
        }

        // Tensor declarations: tensor name: type
        if let Some(tensor_pos) = line.find("tensor ") {
            if let Some(name_start) = line[tensor_pos + 7..].find(|c: char| c.is_alphanumeric()) {
                let name_start = tensor_pos + 7 + name_start;
                if let Some(name_end) = line[name_start..].find(|c: char| c == ':' || c.is_whitespace()) {
                    let name = &line[name_start..name_start + name_end];
                    symbols.push(SymbolInformation {
                        name: name.to_string(),
                        kind: SymbolKind::VARIABLE,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: Url::parse("file:///dummy").unwrap(),
                            range: Range {
                                start: Position { line: line_num, character: name_start as u32 },
                                end: Position { line: line_num, character: (name_start + name_end) as u32 },
                            },
                        },
                        container_name: None,
                    });
                }
            }
        }
    }

    symbols
}
