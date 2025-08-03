# Build a Large Language Model (From Scratch) - Rust Edition

This repository contains Rust implementations of the code examples from Sebastian Raschka's book **"Build a Large Language Model (From Scratch)"**, originally written for Python and PyTorch. This translation serves as a dual learning resource for understanding both Large Language Models (LLMs) and the Rust programming language.

## About This Project

This project translates the educational code examples from the book into idiomatic Rust, providing:

- **LLM Education**: Following the book's curriculum to understand how GPT-like models work from first principles
- **Rust Learning**: Demonstrating advanced Rust concepts including async programming, error handling, and systems programming
- **Practical Examples**: Working implementations that mirror the book's Python examples in Rust

## Project Structure

The project is organized as a Cargo workspace with chapter-based modules:

```
llmfs/
├── Cargo.toml          # Workspace configuration
├── ch02/               # Chapter 2: Working with Text Data
│   ├── Cargo.toml
│   └── src/
│       └── main.rs     # Async file downloader example
└── target/             # Shared build artifacts
```

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)

### Building the Project

```bash
# Build all chapters
cargo build --workspace

# Build a specific chapter
cargo build -p ch02
```

### Running Examples

```bash
# Run a specific chapter
cargo run -p ch02

# Run with release optimizations
cargo run -p ch02 --release
```

## Chapter Overview

### Chapter 2: Working with Text Data
- **Focus**: Async programming and file I/O
- **Key Concepts**: 
  - Tokio async runtime
  - HTTP requests with reqwest
  - Error handling with Result types
  - File system operations

*More chapters will be added as the translation progresses*

## Learning Objectives

This Rust translation helps readers:

1. **Understand LLM Fundamentals** through hands-on implementation
2. **Learn Rust** by translating Python/PyTorch concepts to Rust equivalents
3. **Explore Systems Programming** approaches to ML infrastructure
4. **Practice Modern Rust Patterns** including async/await, error handling, and type safety

## About the Original Book

"Build a Large Language Model (From Scratch)" by Sebastian Raschka is a comprehensive guide that teaches readers how to build GPT-like language models from first principles. The book covers:

- Data preprocessing and tokenization
- Transformer architecture implementation
- Training and fine-tuning strategies
- Practical applications

For more information about the original book:
- [Manning Publications](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Official GitHub Repository](https://github.com/rasbt/LLMs-from-scratch)

## Contributing

This is an educational project. Contributions that improve code clarity, add explanatory comments, or implement additional chapters are welcome.

## License

This project follows the licensing terms of the original book's code examples. Please refer to the original repository for detailed licensing information.

## Acknowledgments

- Sebastian Raschka for writing the original book and providing clear, educational examples
- The Rust community for excellent documentation and libraries
- Manning Publications for publishing this valuable educational resource