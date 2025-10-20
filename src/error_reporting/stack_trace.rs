//! Stack trace support for runtime errors
//!
//! Provides execution context tracking and stack trace formatting
//! for better debugging experience.

use std::fmt;

/// A single frame in the execution stack
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function or context name
    pub name: String,
    /// File path (if available)
    pub file: Option<String>,
    /// Line number (if available)
    pub line: Option<usize>,
    /// Frame type (function call, statement, etc.)
    pub frame_type: FrameType,
}

/// Type of stack frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Function call
    FunctionCall,
    /// Statement execution
    Statement,
    /// Expression evaluation
    Expression,
    /// Main block
    MainBlock,
    /// Declaration processing
    Declaration,
}

impl fmt::Display for FrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameType::FunctionCall => write!(f, "function call"),
            FrameType::Statement => write!(f, "statement"),
            FrameType::Expression => write!(f, "expression"),
            FrameType::MainBlock => write!(f, "main block"),
            FrameType::Declaration => write!(f, "declaration"),
        }
    }
}

impl StackFrame {
    /// Create a new stack frame
    pub fn new(name: String, frame_type: FrameType) -> Self {
        StackFrame {
            name,
            file: None,
            line: None,
            frame_type,
        }
    }

    /// Create a stack frame with file and line information
    pub fn with_location(name: String, frame_type: FrameType, file: String, line: usize) -> Self {
        StackFrame {
            name,
            file: Some(file),
            line: Some(line),
            frame_type,
        }
    }

    /// Format the stack frame for display
    pub fn format(&self) -> String {
        let location = match (&self.file, self.line) {
            (Some(file), Some(line)) => format!(" at {}:{}", file, line),
            (Some(file), None) => format!(" at {}", file),
            _ => String::new(),
        };

        format!("  in {} ({}){}", self.name, self.frame_type, location)
    }
}

/// Stack trace for runtime errors
#[derive(Debug, Clone)]
pub struct StackTrace {
    frames: Vec<StackFrame>,
}

impl StackTrace {
    /// Create a new empty stack trace
    pub fn new() -> Self {
        StackTrace { frames: Vec::new() }
    }

    /// Create a stack trace with a single frame
    pub fn single(frame: StackFrame) -> Self {
        StackTrace {
            frames: vec![frame],
        }
    }

    /// Add a frame to the stack trace
    pub fn push(&mut self, frame: StackFrame) {
        self.frames.push(frame);
    }

    /// Get all frames
    pub fn frames(&self) -> &[StackFrame] {
        &self.frames
    }

    /// Check if the stack trace is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Format the stack trace for display
    pub fn format(&self) -> String {
        if self.frames.is_empty() {
            return String::new();
        }

        let mut output = String::from("Stack trace:\n");
        for (i, frame) in self.frames.iter().enumerate() {
            output.push_str(&format!("{:2}. {}\n", i + 1, frame.format()));
        }
        output
    }

    /// Format the stack trace in compact form (single line per frame)
    pub fn format_compact(&self) -> String {
        if self.frames.is_empty() {
            return String::new();
        }

        let frames: Vec<String> = self
            .frames
            .iter()
            .map(|f| {
                let location = match (&f.file, f.line) {
                    (Some(file), Some(line)) => format!("{}:{}", file, line),
                    (Some(file), None) => file.clone(),
                    _ => f.name.clone(),
                };
                location
            })
            .collect();

        format!("Stack: {}", frames.join(" -> "))
    }
}

impl Default for StackTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StackTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_frame_creation() {
        let frame = StackFrame::new("test_function".to_string(), FrameType::FunctionCall);
        assert_eq!(frame.name, "test_function");
        assert_eq!(frame.frame_type, FrameType::FunctionCall);
        assert!(frame.file.is_none());
        assert!(frame.line.is_none());
    }

    #[test]
    fn test_stack_frame_with_location() {
        let frame = StackFrame::with_location(
            "main".to_string(),
            FrameType::MainBlock,
            "test.tl".to_string(),
            42,
        );
        assert_eq!(frame.name, "main");
        assert_eq!(frame.file, Some("test.tl".to_string()));
        assert_eq!(frame.line, Some(42));
    }

    #[test]
    fn test_stack_trace_push() {
        let mut trace = StackTrace::new();
        assert!(trace.is_empty());

        trace.push(StackFrame::new(
            "func1".to_string(),
            FrameType::FunctionCall,
        ));
        assert_eq!(trace.frames().len(), 1);

        trace.push(StackFrame::new(
            "func2".to_string(),
            FrameType::FunctionCall,
        ));
        assert_eq!(trace.frames().len(), 2);
    }

    #[test]
    fn test_stack_trace_format() {
        let mut trace = StackTrace::new();
        trace.push(StackFrame::with_location(
            "main".to_string(),
            FrameType::MainBlock,
            "test.tl".to_string(),
            10,
        ));
        trace.push(StackFrame::with_location(
            "calculate".to_string(),
            FrameType::FunctionCall,
            "test.tl".to_string(),
            5,
        ));

        let formatted = trace.format();
        assert!(formatted.contains("Stack trace:"));
        assert!(formatted.contains("main"));
        assert!(formatted.contains("calculate"));
        assert!(formatted.contains("test.tl:10"));
        assert!(formatted.contains("test.tl:5"));
    }

    #[test]
    fn test_stack_trace_compact() {
        let mut trace = StackTrace::new();
        trace.push(StackFrame::with_location(
            "main".to_string(),
            FrameType::MainBlock,
            "test.tl".to_string(),
            10,
        ));
        trace.push(StackFrame::new(
            "calculate".to_string(),
            FrameType::FunctionCall,
        ));

        let compact = trace.format_compact();
        assert!(compact.contains("Stack:"));
        assert!(compact.contains("test.tl:10"));
        assert!(compact.contains("calculate"));
        assert!(compact.contains("->"));
    }
}
