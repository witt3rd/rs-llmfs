# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust translation of Sebastian Raschka's "Build a Large Language Model (From Scratch)" book, originally written for Python/PyTorch. The project serves dual educational purposes: teaching LLM fundamentals while demonstrating Rust programming concepts. It's organized as a Cargo workspace with chapter-based modules corresponding to the book's structure (currently only ch02 exists).

## Architecture

The codebase follows a workspace structure where each chapter is a separate Rust crate:
- Root workspace configuration in `/Cargo.toml`
- Individual chapters as workspace members (e.g., `ch02/`)
- Shared `Cargo.lock` and `target/` directory at root level

## Common Commands

### Running a specific chapter
```bash
cargo run -p ch02
```

### Building and checking
```bash
# Build all workspace members
cargo build --workspace

# Check specific chapter
cargo check -p ch02

# Run tests for all chapters
cargo test --workspace
```

### Linting and formatting
```bash
# Format code
cargo fmt --all

# Run clippy linter
cargo clippy --workspace -- -D warnings
```

## Chapter 2: Working with Text Data

The ch02 module corresponds to Chapter 2 of the LLM book and demonstrates:
- Async file downloading for fetching text datasets
- Tokio runtime for async execution
- reqwest for HTTP client operations
- Error handling with Result<T, Box<dyn Error>>
- File I/O with proper error propagation

This chapter sets up the foundation for downloading and managing text corpora that will be used for training LLMs in later chapters.

Key dependencies:
- `tokio` v1.47.1 (full features) - async runtime
- `reqwest` v0.12.22 (with stream feature) - HTTP client for downloading datasets

## Development Notes

- This is a translation project: Python/PyTorch concepts from the book are translated to idiomatic Rust
- The project uses educational inline documentation to explain both LLM concepts and Rust patterns
- Each chapter corresponds to a chapter in the original book and is a self-contained Rust crate
- The workspace structure allows for easy addition of new chapters as translation progresses
- Build artifacts are shared across all workspace members in the root `target/` directory
- When implementing new chapters, maintain consistency with the book's educational approach while using Rust best practices

## Documentation References

- Rust CLI information can be found at: @docs\refs\rust\rust-cli-guide.md
- Rust Cargo information can be found at: @docs\refs\rust\cargo.md