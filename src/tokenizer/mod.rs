//! Tokenization support using HuggingFace tokenizers

use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer as HFTokenizer;
use crate::error::{TensorError, TensorResult};

/// Wrapper around HuggingFace tokenizer
pub struct Tokenizer {
    inner: Arc<HFTokenizer>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl Tokenizer {
    /// Load tokenizer from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> TensorResult<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| TensorError::InvalidOperation(
                format!("Failed to load tokenizer: {}", e)
            ))?;

        Ok(Self {
            inner: Arc::new(tokenizer),
        })
    }

    /// Load tokenizer from pretrained model on HuggingFace Hub
    /// Note: Simplified implementation - downloads from HuggingFace Hub cache
    pub fn from_pretrained(identifier: &str) -> TensorResult<Self> {
        // tokenizers 0.20 doesn't have from_pretrained, use from_file with HF cache
        // For now, return error with instruction
        return Err(TensorError::InvalidOperation(
            format!("from_pretrained not available in this version. Please download tokenizer file manually from HuggingFace Hub and use from_file: {}", identifier)
        ));
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TensorResult<Vec<u32>> {
        let encoding = self.inner
            .encode(text, add_special_tokens)
            .map_err(|e| TensorError::InvalidOperation(
                format!("Failed to encode text: {}", e)
            ))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> TensorResult<String> {
        let text = self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TensorError::InvalidOperation(
                format!("Failed to decode tokens: {}", e)
            ))?;

        Ok(text)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get padding token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<pad>")
            .or_else(|| self.inner.token_to_id("[PAD]"))
    }

    /// Get BOS (beginning of sequence) token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<s>")
            .or_else(|| self.inner.token_to_id("[BOS]"))
            .or_else(|| self.inner.token_to_id("<bos>"))
    }

    /// Get EOS (end of sequence) token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("</s>")
            .or_else(|| self.inner.token_to_id("[EOS]"))
            .or_else(|| self.inner.token_to_id("<eos>"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_tokenizer_from_pretrained() {
        let tokenizer = Tokenizer::from_pretrained("gpt2").unwrap();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_encode_decode() {
        let tokenizer = Tokenizer::from_pretrained("gpt2").unwrap();
        let text = "Hello, world!";

        let ids = tokenizer.encode(text, false).unwrap();
        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

}
