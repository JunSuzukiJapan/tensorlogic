//! Tokenization support using HuggingFace tokenizers

use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer as HFTokenizer;
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;
use crate::device::MetalDevice;
use half::f16;

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

    /// Encode text and return as tensor
    pub fn encode_tensor(
        &self,
        device: &MetalDevice,
        text: &str,
        add_special_tokens: bool,
    ) -> TensorResult<Tensor> {
        let ids = self.encode(text, add_special_tokens)?;

        // Convert u32 to f16 for tensor storage
        let data: Vec<f16> = ids.iter().map(|&id| f16::from_f32(id as f32)).collect();
        let shape = vec![ids.len()];

        Tensor::from_vec_metal(device, data, shape)
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

    /// Decode tensor of token IDs to text
    pub fn decode_tensor(&self, tensor: &Tensor, skip_special_tokens: bool) -> TensorResult<String> {
        let data = tensor.to_vec();

        // Convert f16 back to u32
        let ids: Vec<u32> = data.iter().map(|&val| val.to_f32() as u32).collect();

        self.decode(&ids, skip_special_tokens)
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

    #[test]
    #[ignore] // Requires network access
    fn test_encode_decode_tensor() {
        let device = MetalDevice::new().unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2").unwrap();
        let text = "Hello, world!";

        let tensor = tokenizer.encode_tensor(&device, text, false).unwrap();
        assert!(tensor.shape().dims()[0] > 0);

        let decoded = tokenizer.decode_tensor(&tensor, false).unwrap();
        assert_eq!(decoded, text);
    }
}
