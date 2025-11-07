use tower_lsp::lsp_types::*;

/// Get completion suggestions at a given position
pub fn get_completions(text: &str, position: Position) -> Vec<CompletionItem> {
    let mut completions = Vec::new();

    // Get the current line
    let lines: Vec<&str> = text.lines().collect();
    if position.line as usize >= lines.len() {
        return completions;
    }

    let current_line = lines[position.line as usize];
    let before_cursor = &current_line[..position.character.min(current_line.len() as u32) as usize];

    // Check context to determine what to complete
    if before_cursor.trim_end().ends_with(':') {
        // Type completion
        completions.extend(get_type_completions());
    } else if before_cursor.contains("tensor ") && !before_cursor.contains(':') {
        // After tensor keyword, suggest nothing specific (user types name)
    } else {
        // Default: keywords, builtins, and snippets
        completions.extend(get_keyword_completions());
        completions.extend(get_builtin_function_completions());
        completions.extend(get_snippet_completions());
    }

    completions
}

/// Get keyword completions
fn get_keyword_completions() -> Vec<CompletionItem> {
    let keywords = vec![
        ("tensor", "Tensor declaration", "tensor ${1:name}: ${2:float16}[${3:dims}] = ${4:value}"),
        ("fn", "Function declaration", "fn ${1:name}(${2:params}) -> ${3:type} {\n    ${4:body}\n}"),
        ("main", "Main block", "main {\n    ${1:body}\n}"),
        ("if", "If statement", "if ${1:condition} {\n    ${2:body}\n}"),
        ("else", "Else clause", "else {\n    ${1:body}\n}"),
        ("for", "For loop", "for ${1:var} in ${2:range} {\n    ${3:body}\n}"),
        ("while", "While loop", "while ${1:condition} {\n    ${2:body}\n}"),
        ("return", "Return statement", "return ${1:value}"),
        ("learn", "Learn block", "learn {\n    objective: ${1:loss}\n    optimizer: ${2:Adam}\n    epochs: ${3:100}\n}"),
        ("relation", "Relation declaration", "relation ${1:name}(${2:args})"),
        ("rule", "Logic rule", "rule ${1:name} :- ${2:conditions}"),
        ("embedding", "Embedding declaration", "embedding ${1:name}: ${2:dims}"),
        ("entity", "Entity declaration", "entity ${1:name}"),
        ("concept", "Concept declaration", "concept ${1:name}"),
        ("let", "Variable binding", "let ${1:name} = ${2:value}"),
    ];

    keywords
        .into_iter()
        .map(|(label, detail, snippet)| CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some(detail.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        })
        .collect()
}

/// Get type completions
fn get_type_completions() -> Vec<CompletionItem> {
    let types = vec![
        ("float16", "16-bit floating point"),
        ("float32", "32-bit floating point"),
        ("int16", "16-bit integer"),
        ("int32", "32-bit integer"),
        ("int64", "64-bit integer"),
        ("bool", "Boolean type"),
        ("complex16", "16-bit complex number"),
    ];

    types
        .into_iter()
        .map(|(label, detail)| CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::TYPE_PARAMETER),
            detail: Some(detail.to_string()),
            ..Default::default()
        })
        .collect()
}

/// Get builtin function completions
fn get_builtin_function_completions() -> Vec<CompletionItem> {
    vec![
        // Tensor operations
        completion_func("zeros", "Create tensor of zeros", "zeros(${1:shape})"),
        completion_func("ones", "Create tensor of ones", "ones(${1:shape})"),
        completion_func("reshape", "Reshape tensor", "reshape(${1:tensor}, ${2:shape})"),
        completion_func("transpose", "Transpose tensor", "transpose(${1:tensor})"),
        completion_func("concat", "Concatenate tensors", "concat(${1:tensors}, ${2:axis})"),
        completion_func("split", "Split tensor", "split(${1:tensor}, ${2:sizes}, ${3:axis})"),
        completion_func("permute", "Permute tensor dimensions", "permute(${1:tensor}, ${2:dims})"),

        // Neural network operations
        completion_func("linear", "Linear layer", "linear(${1:input}, ${2:weight}, ${3:bias})"),
        completion_func("rms_norm", "RMS normalization", "rms_norm(${1:input}, ${2:weight})"),
        completion_func("softmax", "Softmax activation", "softmax(${1:input})"),
        completion_func("rope", "Rotary position embedding", "rope(${1:input}, ${2:position})"),
        completion_func("attention_with_cache", "Attention with cache", "attention_with_cache(${1:query}, ${2:key}, ${3:value}, ${4:cache})"),
        completion_func("sigmoid", "Sigmoid activation", "sigmoid(${1:input})"),
        completion_func("relu", "ReLU activation", "relu(${1:input})"),
        completion_func("gelu", "GELU activation", "gelu(${1:input})"),
        completion_func("silu", "SiLU activation", "silu(${1:input})"),
        completion_func("tanh", "Tanh activation", "tanh(${1:input})"),

        // Math operations
        completion_func("sin", "Sine function", "sin(${1:input})"),
        completion_func("cos", "Cosine function", "cos(${1:input})"),
        completion_func("exp", "Exponential function", "exp(${1:input})"),
        completion_func("log", "Natural logarithm", "log(${1:input})"),
        completion_func("sqrt", "Square root", "sqrt(${1:input})"),
        completion_func("abs", "Absolute value", "abs(${1:input})"),
        completion_func("pow", "Power function", "pow(${1:base}, ${2:exponent})"),

        // Model loading
        completion_func("load_f16", "Load f16 model", "load_f16(${1:path})"),
        completion_func("load_f32", "Load f32 model", "load_f32(${1:path})"),
        completion_func("load_tokenizer", "Load tokenizer", "load_tokenizer(${1:path})"),

        // Sampling
        completion_func("sample_temperature", "Temperature sampling", "sample_temperature(${1:logits}, ${2:temperature})"),
        completion_func("sample_top_k", "Top-k sampling", "sample_top_k(${1:logits}, ${2:k})"),
        completion_func("sample_greedy", "Greedy sampling", "sample_greedy(${1:logits})"),

        // Utilities
        completion_func("shape", "Get tensor shape", "shape(${1:tensor})"),
        completion_func("print", "Print value", "print(${1:value})"),
        completion_func("env", "Get environment variable", "env(${1:name})"),
    ]
}

/// Helper to create function completion item
fn completion_func(label: &str, detail: &str, snippet: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(CompletionItemKind::FUNCTION),
        detail: Some(detail.to_string()),
        insert_text: Some(snippet.to_string()),
        insert_text_format: Some(InsertTextFormat::SNIPPET),
        ..Default::default()
    }
}

/// Get snippet completions
fn get_snippet_completions() -> Vec<CompletionItem> {
    vec![
        CompletionItem {
            label: "transformer_block".to_string(),
            kind: Some(CompletionItemKind::SNIPPET),
            detail: Some("Complete transformer block".to_string()),
            insert_text: Some(
                r#"// Transformer block
tensor attn_norm = rms_norm(${1:input}, ${2:norm_weight})
tensor q = linear(attn_norm, ${3:q_weight})
tensor k = linear(attn_norm, ${4:k_weight})
tensor v = linear(attn_norm, ${5:v_weight})
tensor attn_out = attention_with_cache(q, k, v, ${6:cache})
tensor ffn_norm = rms_norm(attn_out + ${1:input}, ${7:ffn_norm_weight})
tensor output = linear(ffn_norm, ${8:ffn_weight})
"#.to_string()
            ),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        },
        CompletionItem {
            label: "llm_inference".to_string(),
            kind: Some(CompletionItemKind::SNIPPET),
            detail: Some("LLM inference loop".to_string()),
            insert_text: Some(
                r#"// LLM inference loop
for i in range(${1:max_tokens}) {
    tensor logits = ${2:model_forward}(${3:input})
    tensor token = sample_temperature(logits, ${4:temperature})
    print(token)
    ${3:input} = token
}
"#.to_string()
            ),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_completions() {
        let completions = get_completions("", Position::new(0, 0));
        assert!(completions.len() > 0);
        assert!(completions.iter().any(|c| c.label == "tensor"));
    }

    #[test]
    fn test_type_completions() {
        let text = "tensor x: ";
        let completions = get_completions(text, Position::new(0, 10));
        assert!(completions.iter().any(|c| c.label == "float16"));
    }
}
