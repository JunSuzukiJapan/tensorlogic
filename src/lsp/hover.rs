use tower_lsp::lsp_types::*;

/// Get hover information at a given position
pub fn get_hover_info(text: &str, position: Position) -> Option<Hover> {
    let lines: Vec<&str> = text.lines().collect();
    if position.line as usize >= lines.len() {
        return None;
    }

    let line = lines[position.line as usize];
    let word = extract_word_at_position(line, position.character as usize)?;

    // Check if it's a keyword, builtin function, or type
    if let Some(info) = get_keyword_info(&word) {
        return Some(create_hover(info));
    }

    if let Some(info) = get_builtin_function_info(&word) {
        return Some(create_hover(info));
    }

    if let Some(info) = get_type_info(&word) {
        return Some(create_hover(info));
    }

    None
}

/// Extract word at cursor position
fn extract_word_at_position(line: &str, column: usize) -> Option<String> {
    if column > line.len() {
        return None;
    }

    // Find word boundaries
    let start = line[..column]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);

    let end = line[column..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| column + i)
        .unwrap_or(line.len());

    if start >= end {
        return None;
    }

    Some(line[start..end].to_string())
}

/// Create hover object from markdown content
fn create_hover(content: String) -> Hover {
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: content,
        }),
        range: None,
    }
}

/// Get information about keywords
fn get_keyword_info(word: &str) -> Option<String> {
    match word {
        "tensor" => Some(format!(
            "```tensorlogic\ntensor\n```\n\n\
            Declares a tensor variable.\n\n\
            **Syntax:** `tensor name: type[dims] = value`\n\n\
            **Example:**\n```tensorlogic\ntensor x: float16[3] = [1.0, 2.0, 3.0]\n```"
        )),
        "fn" => Some(format!(
            "```tensorlogic\nfn\n```\n\n\
            Declares a function.\n\n\
            **Syntax:** `fn name(params) -> return_type {{ body }}`\n\n\
            **Example:**\n```tensorlogic\nfn add(a: float16, b: float16) -> float16 {{\n    return a + b\n}}\n```"
        )),
        "main" => Some(format!(
            "```tensorlogic\nmain\n```\n\n\
            Entry point of the program.\n\n\
            **Syntax:** `main {{ statements }}`"
        )),
        "learn" => Some(format!(
            "```tensorlogic\nlearn\n```\n\n\
            Defines a learning/training block with automatic differentiation.\n\n\
            **Syntax:**\n```tensorlogic\nlearn {{\n    objective: loss_function\n    optimizer: Adam\n    epochs: 100\n}}\n```"
        )),
        "if" => Some(format!(
            "```tensorlogic\nif\n```\n\n\
            Conditional statement.\n\n\
            **Syntax:** `if condition {{ body }} else {{ alternative }}`"
        )),
        "for" => Some(format!(
            "```tensorlogic\nfor\n```\n\n\
            Loop over a range or collection.\n\n\
            **Syntax:** `for var in range(start, end) {{ body }}`"
        )),
        "while" => Some(format!(
            "```tensorlogic\nwhile\n```\n\n\
            Loop while condition is true.\n\n\
            **Syntax:** `while condition {{ body }}`"
        )),
        "return" => Some(format!(
            "```tensorlogic\nreturn\n```\n\n\
            Returns a value from a function.\n\n\
            **Syntax:** `return value`"
        )),
        _ => None,
    }
}

/// Get information about types
fn get_type_info(word: &str) -> Option<String> {
    match word {
        "float16" => Some(format!(
            "```tensorlogic\nfloat16\n```\n\n\
            16-bit floating point type (half precision).\n\n\
            **Optimized for:** Apple Silicon Neural Engine and Metal GPU\n\n\
            **Range:** ±65,504\n\
            **Precision:** ~3 decimal digits"
        )),
        "float32" => Some(format!(
            "```tensorlogic\nfloat32\n```\n\n\
            32-bit floating point type (single precision).\n\n\
            **Range:** ±3.4 × 10³⁸\n\
            **Precision:** ~7 decimal digits"
        )),
        "int16" => Some("16-bit signed integer type.\n\n**Range:** -32,768 to 32,767".to_string()),
        "int32" => Some("32-bit signed integer type.\n\n**Range:** -2,147,483,648 to 2,147,483,647".to_string()),
        "int64" => Some("64-bit signed integer type.\n\n**Range:** -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807".to_string()),
        "bool" => Some("Boolean type (true or false).".to_string()),
        "complex16" => Some("16-bit complex number type.".to_string()),
        _ => None,
    }
}

/// Get information about builtin functions
fn get_builtin_function_info(word: &str) -> Option<String> {
    match word {
        "zeros" => Some(format!(
            "```tensorlogic\nzeros(shape: int[]) -> tensor\n```\n\n\
            Creates a tensor filled with zeros.\n\n\
            **Example:**\n```tensorlogic\ntensor z = zeros([3, 4])  // 3x4 matrix of zeros\n```"
        )),
        "ones" => Some(format!(
            "```tensorlogic\nones(shape: int[]) -> tensor\n```\n\n\
            Creates a tensor filled with ones.\n\n\
            **Example:**\n```tensorlogic\ntensor o = ones([2, 3])  // 2x3 matrix of ones\n```"
        )),
        "reshape" => Some(format!(
            "```tensorlogic\nreshape(tensor: tensor, shape: int[]) -> tensor\n```\n\n\
            Reshapes a tensor to the specified shape.\n\n\
            **Example:**\n```tensorlogic\ntensor reshaped = reshape(x, [10, 20])\n```"
        )),
        "transpose" => Some(format!(
            "```tensorlogic\ntranspose(tensor: tensor) -> tensor\n```\n\n\
            Transposes a 2D tensor (swaps rows and columns).\n\n\
            **Example:**\n```tensorlogic\ntensor t = transpose(matrix)\n```"
        )),
        "concat" => Some(format!(
            "```tensorlogic\nconcat(tensors: tensor[], axis: int) -> tensor\n```\n\n\
            Concatenates tensors along the specified axis.\n\n\
            **Example:**\n```tensorlogic\ntensor c = concat([a, b, c], 0)\n```"
        )),
        "linear" => Some(format!(
            "```tensorlogic\nlinear(input: tensor, weight: tensor, bias: tensor) -> tensor\n```\n\n\
            Applies a linear transformation: `output = input @ weight + bias`\n\n\
            **Example:**\n```tensorlogic\ntensor out = linear(x, w, b)\n```"
        )),
        "rms_norm" => Some(format!(
            "```tensorlogic\nrms_norm(input: tensor, weight: tensor) -> tensor\n```\n\n\
            Root Mean Square Layer Normalization (used in Llama models).\n\n\
            **Formula:** `x * weight / sqrt(mean(x²) + ε)`"
        )),
        "softmax" => Some(format!(
            "```tensorlogic\nsoftmax(input: tensor) -> tensor\n```\n\n\
            Applies softmax activation function.\n\n\
            **Formula:** `exp(x) / sum(exp(x))`\n\n\
            **Use case:** Convert logits to probabilities"
        )),
        "rope" => Some(format!(
            "```tensorlogic\nrope(input: tensor, position: int) -> tensor\n```\n\n\
            Rotary Position Embedding (RoPE).\n\n\
            Encodes position information in transformer models."
        )),
        "attention_with_cache" => Some(format!(
            "```tensorlogic\nattention_with_cache(query: tensor, key: tensor, value: tensor, cache: tensor) -> tensor\n```\n\n\
            Scaled dot-product attention with KV cache for autoregressive generation.\n\n\
            **Formula:** `softmax(Q @ K^T / sqrt(d)) @ V`"
        )),
        "sigmoid" => Some("Sigmoid activation: `1 / (1 + exp(-x))`\n\nRange: (0, 1)".to_string()),
        "relu" => Some("ReLU activation: `max(0, x)`\n\nRectified Linear Unit".to_string()),
        "gelu" => Some("GELU activation: Gaussian Error Linear Unit\n\nSmooth approximation of ReLU".to_string()),
        "silu" => Some("SiLU activation: `x * sigmoid(x)`\n\nAlso known as Swish".to_string()),
        "tanh" => Some("Hyperbolic tangent activation\n\nRange: (-1, 1)".to_string()),
        "sin" => Some("Sine function".to_string()),
        "cos" => Some("Cosine function".to_string()),
        "exp" => Some("Exponential function: `e^x`".to_string()),
        "log" => Some("Natural logarithm: `ln(x)`".to_string()),
        "sqrt" => Some("Square root: `√x`".to_string()),
        "abs" => Some("Absolute value: `|x|`".to_string()),
        "pow" => Some("Power function: `x^y`".to_string()),
        "load_f16" => Some(format!(
            "```tensorlogic\nload_f16(path: string) -> tensor\n```\n\n\
            Loads a model in f16 precision from GGUF or SafeTensors format.\n\n\
            **Optimized for:** Apple Silicon Neural Engine"
        )),
        "load_f32" => Some("Loads a model in f32 precision from GGUF or SafeTensors format.".to_string()),
        "load_tokenizer" => Some("Loads a tokenizer from HuggingFace format.".to_string()),
        "sample_temperature" => Some(format!(
            "```tensorlogic\nsample_temperature(logits: tensor, temperature: float) -> int\n```\n\n\
            Samples a token using temperature scaling.\n\n\
            **Higher temperature:** More random\n\
            **Lower temperature:** More deterministic"
        )),
        "sample_top_k" => Some("Samples from top-k most likely tokens.".to_string()),
        "sample_greedy" => Some("Selects the most likely token (argmax).".to_string()),
        "shape" => Some("Returns the shape of a tensor as an array of integers.".to_string()),
        "print" => Some("Prints a value to the console.".to_string()),
        "env" => Some("Gets an environment variable value.".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_word() {
        assert_eq!(extract_word_at_position("tensor x", 3), Some("tensor".to_string()));
        assert_eq!(extract_word_at_position("  float16", 5), Some("float16".to_string()));
    }

    #[test]
    fn test_keyword_hover() {
        let info = get_keyword_info("tensor");
        assert!(info.is_some());
        assert!(info.unwrap().contains("tensor"));
    }

    #[test]
    fn test_builtin_hover() {
        let info = get_builtin_function_info("softmax");
        assert!(info.is_some());
        assert!(info.unwrap().contains("softmax"));
    }
}
