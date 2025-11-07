# TensorLogic Language Server

TensorLogic Language Server provides IDE support for the TensorLogic programming language through the Language Server Protocol (LSP).

## Features

### ✅ Implemented

- **Diagnostics**: Real-time syntax error detection and reporting
- **Code Completion**:
  - Keyword completion (tensor, fn, main, if, for, while, learn, etc.)
  - Type completion (float16, float32, int32, bool, etc.)
  - Built-in function completion (zeros, ones, reshape, softmax, linear, etc.)
  - Code snippets (transformer blocks, LLM inference loops)
- **Hover Information**: Documentation for keywords, types, and built-in functions
- **Go to Definition**: Jump to function, tensor, and variable declarations
- **Document Symbols**: Outline view showing functions and tensors

### 🚧 Future Enhancements

- Semantic analysis (type checking, undefined variable detection)
- Find references
- Rename symbol
- Code formatting
- Signature help for function parameters
- Workspace symbols

## Installation

### Building from Source

```bash
cargo build --release --bin tl-lsp
```

The binary will be located at `target/release/tl-lsp`.

### Configuration

#### Visual Studio Code

1. Install a generic LSP client extension (e.g., "vscode-language-server-protocol")
2. Add to your `settings.json`:

```json
{
  "languageServerExample.trace.server": "verbose",
  "tensorlogic.languageServer": {
    "command": "/path/to/tl-lsp",
    "args": []
  }
}
```

3. Create a `.vscode/settings.json` in your TensorLogic project:

```json
{
  "files.associations": {
    "*.tl": "tensorlogic"
  }
}
```

#### Neovim (with nvim-lspconfig)

Add to your `init.lua`:

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define tensorlogic LSP
if not configs.tensorlogic then
  configs.tensorlogic = {
    default_config = {
      cmd = {'/path/to/tl-lsp'},
      filetypes = {'tensorlogic', 'tl'},
      root_dir = lspconfig.util.root_pattern('.git', 'Cargo.toml'),
      settings = {},
    },
  }
end

-- Enable tensorlogic LSP
lspconfig.tensorlogic.setup{}
```

#### Emacs (with lsp-mode)

Add to your `.emacs` or `init.el`:

```elisp
(require 'lsp-mode)

(add-to-list 'lsp-language-id-configuration '(tensorlogic-mode . "tensorlogic"))

(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection "/path/to/tl-lsp")
  :major-modes '(tensorlogic-mode)
  :server-id 'tensorlogic-ls))
```

## Usage

Once configured, open any `.tl` file and the language server will automatically:

1. **Highlight errors** as you type
2. **Provide completions** when you press Ctrl+Space (or your editor's completion trigger)
3. **Show documentation** when you hover over keywords or functions
4. **Jump to definitions** with Ctrl+Click or F12
5. **Show outline** of functions and tensors in the sidebar

## Examples

### Code Completion

Type `tensor` and press Tab to expand to:
```tensorlogic
tensor name: float16[dims] = value
```

Type `transf` and select `transformer_block` to insert:
```tensorlogic
// Transformer block
tensor attn_norm = rms_norm(input, norm_weight)
tensor q = linear(attn_norm, q_weight)
tensor k = linear(attn_norm, k_weight)
tensor v = linear(attn_norm, v_weight)
tensor attn_out = attention_with_cache(q, k, v, cache)
tensor ffn_norm = rms_norm(attn_out + input, ffn_norm_weight)
tensor output = linear(ffn_norm, ffn_weight)
```

### Hover Documentation

Hover over `softmax` to see:

```
softmax(input: tensor) -> tensor

Applies softmax activation function.

Formula: exp(x) / sum(exp(x))

Use case: Convert logits to probabilities
```

### Go to Definition

Click on a variable name while holding Ctrl (or Cmd) to jump to where it was defined:

```tensorlogic
main {
    tensor x: float16[3] = [1.0, 2.0, 3.0]  // Definition
    tensor y = x + x  // Click on 'x' here to jump to definition
}
```

## Architecture

The language server is built using:

- **tower-lsp**: LSP protocol implementation
- **tokio**: Async runtime
- **pest**: Parser (reusing TensorLogic's existing parser)
- **dashmap**: Thread-safe document storage

### Components

```
src/lsp/
├── mod.rs              # Module exports
├── backend.rs          # LSP backend implementation
├── diagnostics.rs      # Error and warning detection
├── completion.rs       # Code completion logic
├── hover.rs            # Hover information provider
└── goto_definition.rs  # Definition finder
```

### Protocol Flow

```
Editor ──(LSP JSON-RPC)──> Language Server
                              │
                              ├─> Parse document (diagnostics)
                              ├─> Store in memory (DashMap)
                              ├─> Compute completions
                              ├─> Find definitions
                              └─> Generate hover info
                              │
Editor <───(Responses)────────┘
```

## Development

### Running Tests

```bash
cargo test --lib lsp
```

### Debugging

Enable logging:

```bash
RUST_LOG=debug tl-lsp
```

The language server communicates over stdin/stdout, so logs are written to stderr.

### Adding New Features

1. Add handler method to `TensorLogicBackend` in `backend.rs`
2. Implement logic in appropriate module (e.g., `completion.rs`)
3. Update `ServerCapabilities` in `initialize()` to advertise the feature
4. Add tests

## Performance

- **Startup time**: < 100ms
- **Completion latency**: < 10ms
- **Diagnostics**: Incremental (only re-parse changed documents)
- **Memory**: ~5MB + document cache

## Troubleshooting

### Language server not starting

Check that:
1. `tl-lsp` binary exists and is executable
2. Path in editor config is correct
3. File has `.tl` extension

### No completions appearing

1. Check file is recognized as TensorLogic (bottom right in VS Code)
2. Verify LSP client is active (check status bar)
3. Enable verbose logging to see requests/responses

### Errors not showing

1. Ensure document is saved (some editors only send changes on save)
2. Check LSP connection is active
3. Try closing and reopening the file

## Contributing

Contributions are welcome! Priority areas:

1. **Semantic analysis**: Type checking, scope analysis
2. **Better completion**: Context-aware suggestions
3. **Refactoring**: Extract function, rename symbol
4. **Testing**: More integration tests with editors

## License

Same as TensorLogic project.
