//! Model metadata structures

/// Model format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format (HuggingFace standard)
    SafeTensors,
    /// GGUF format (LLM quantized models)
    GGUF,
    /// CoreML format (Apple Neural Engine)
    CoreML,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFormat::SafeTensors => write!(f, "SafeTensors"),
            ModelFormat::GGUF => write!(f, "GGUF"),
            ModelFormat::CoreML => write!(f, "CoreML"),
        }
    }
}

/// Quantization type for models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// No quantization (full precision)
    None,
    /// 4-bit quantization
    Q4,
    /// 6-bit quantization
    Q6,
    /// 8-bit quantization
    Q8,
    /// 16-bit float (native TensorLogic format)
    F16,
    /// 32-bit float
    F32,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationType::None => write!(f, "None"),
            QuantizationType::Q4 => write!(f, "4-bit"),
            QuantizationType::Q6 => write!(f, "6-bit"),
            QuantizationType::Q8 => write!(f, "8-bit"),
            QuantizationType::F16 => write!(f, "FP16"),
            QuantizationType::F32 => write!(f, "FP32"),
        }
    }
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Original format
    pub format: ModelFormat,
    /// Quantization type (if any)
    pub quantization: Option<QuantizationType>,
}

impl ModelMetadata {
    /// Create new metadata
    pub fn new(name: String, format: ModelFormat) -> Self {
        Self {
            name,
            format,
            quantization: None,
        }
    }

    /// Create new metadata with quantization
    pub fn with_quantization(name: String, format: ModelFormat, quantization: QuantizationType) -> Self {
        Self {
            name,
            format,
            quantization: Some(quantization),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_format_display() {
        assert_eq!(ModelFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(ModelFormat::GGUF.to_string(), "GGUF");
        assert_eq!(ModelFormat::CoreML.to_string(), "CoreML");
    }

    #[test]
    fn test_quantization_display() {
        assert_eq!(QuantizationType::None.to_string(), "None");
        assert_eq!(QuantizationType::Q4.to_string(), "4-bit");
        assert_eq!(QuantizationType::Q8.to_string(), "8-bit");
        assert_eq!(QuantizationType::F16.to_string(), "FP16");
        assert_eq!(QuantizationType::F32.to_string(), "FP32");
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = ModelMetadata::new("test".to_string(), ModelFormat::SafeTensors);
        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.format, ModelFormat::SafeTensors);
        assert_eq!(metadata.quantization, None);
    }

    #[test]
    fn test_metadata_with_quantization() {
        let metadata = ModelMetadata::with_quantization(
            "test".to_string(),
            ModelFormat::GGUF,
            QuantizationType::Q8
        );
        assert_eq!(metadata.quantization, Some(QuantizationType::Q8));
    }
}
