# Build a Large Language Model (From Scratch) - Rust Edition

## 🎯 A Dual Learning Journey

This repository is a unique educational project that serves **two parallel learning paths**:

1. **🧠 Learn to Build LLMs from Scratch** - Following Sebastian Raschka's book to understand how GPT-like models work from first principles
2. **🦀 Learn Rust Through Real Projects** - Not through abstract tutorials, but by building a complete, practical LLM implementation

## 📚 Why This Project?

Most Rust tutorials teach syntax through toy examples. Most ML tutorials assume Python/PyTorch. This project bridges both gaps:

- **For ML Learners**: See how LLM concepts translate to a systems programming language
- **For Rust Learners**: Learn by building something substantial and meaningful
- **For Both**: Experience how educational theory meets practical implementation

## 🏗️ Project Structure

Each chapter contains dual lesson files reflecting both learning paths:

```
llmfs/
├── Cargo.toml              # Workspace configuration
├── README.md               # This file
├── CLAUDE.md               # AI assistant guidance
├── ch02/                   # Chapter 2: Working with Text Data
│   ├── Cargo.toml
│   ├── LESSONS-LLM.md      # What we learn about LLMs
│   ├── LESSONS-RS.md       # What we learn about Rust
│   └── src/
│       └── main.rs         # Implementation
└── target/                 # Shared build artifacts
```

## 🚀 Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)
- Curiosity about both LLMs and systems programming

### Building and Running

```bash
# Build all chapters
cargo build --workspace

# Run a specific chapter (e.g., Chapter 2)
cargo run -p ch02 -- --help

# Run tests
cargo test --workspace

# Format and lint
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## 📖 Chapter Progress

### Chapter 2: Working with Text Data

**LLM Concepts**: Data acquisition, corpus preparation, text preprocessing fundamentals

**Rust Concepts**: Async programming, CLI development, error handling, streaming I/O

```bash
# Download a file to the default "data" directory
cargo run -p ch02 -- https://example.com/file.pdf

# Specify a custom download directory  
cargo run -p ch02 -- https://example.com/file.zip -d /custom/path
```

*More chapters coming as the translation progresses...*

## 🎓 The Dual Pedagogical Approach

### Learning LLMs (LESSONS-LLM.md files)

Each chapter includes a `LESSONS-LLM.md` file that covers:
- Key concepts from the original book chapter
- How these concepts prepare us for building transformers
- Connections to broader ML/AI principles
- Prerequisites for understanding future chapters

### Learning Rust (LESSONS-RS.md files)

Each chapter includes a `LESSONS-RS.md` file that covers:
- Rust patterns and idioms demonstrated in the code
- Systems programming concepts
- Performance considerations
- Common pitfalls and best practices
- How Rust's features benefit ML implementations

## 🤝 Contributing

This is an educational project. We welcome contributions that:
- Improve code clarity and educational value
- Add detailed explanations to LESSONS files
- Implement additional chapters from the book
- Fix bugs or improve performance
- Add tests and documentation

## 📚 Resources

### About the Original Book

"Build a Large Language Model (From Scratch)" by Sebastian Raschka teaches:
- Data preprocessing and tokenization
- Transformer architecture implementation  
- Training and fine-tuning strategies
- Practical applications

- [Manning Publications](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Official GitHub Repository](https://github.com/rasbt/LLMs-from-scratch)

### Rust Learning Resources

- [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Async Programming in Rust](https://rust-lang.github.io/async-book/)

## 📄 License

This project follows the licensing terms of the original book's code examples. Please refer to the original repository for detailed licensing information.

## 🙏 Acknowledgments

- **Sebastian Raschka** for writing the original book and providing clear, educational examples
- **The Rust Community** for excellent documentation and libraries
- **Manning Publications** for publishing this valuable educational resource
- **All Contributors** who help make this dual learning journey possible