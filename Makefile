.PHONY: test build run clean help test-ops test-numerical

# Default target
help:
	@echo "TensorLogic Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make test            - Run all tests with single thread (required for GPU tests)"
	@echo "  make test-ops        - Run tensor operations numerical tests only"
	@echo "  make test-numerical  - Alias for test-ops"
	@echo "  make build           - Build the project"
	@echo "  make run             - Run the REPL"
	@echo "  make clean           - Clean build artifacts"
	@echo "  make help            - Show this help message"

# Run all tests including doctests with single thread (required for Metal GPU tests)
test:
	@echo "Running unit and integration tests with --test-threads=1 for GPU compatibility..."
	@cargo test --lib --bins --tests -- --test-threads=1
	@echo ""
	@echo "Running doctests..."
	@cargo test --doc

# Run tensor operations numerical tests only
test-ops:
	@echo "Running tensor operations numerical correctness tests..."
	@cargo test --lib ops::tests::tensor_ops_tests -- --test-threads=1 --nocapture

# Alias for test-ops
test-numerical: test-ops

# Build the project
build:
	cargo build

# Build release version
release:
	cargo build --release

# Run the REPL
run:
	cargo run -- repl

# Clean build artifacts
clean:
	cargo clean

# Run specific example
run-example:
	@echo "Usage: make run-example FILE=examples/test.tl"
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified"; \
		exit 1; \
	fi
	cargo run -- run $(FILE)
