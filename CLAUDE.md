# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **dual-pedagogical project** that teaches two things simultaneously:

1. **Building LLMs from Scratch** - Following Sebastian Raschka's book to implement GPT-like models
2. **Learning Rust Through Real Projects** - Not abstract tutorials, but learning by building a substantial, practical system

The project translates Python/PyTorch examples from the book into idiomatic Rust, serving as both an LLM implementation guide and a Rust learning resource.

## Architecture

The codebase follows a workspace structure with dual lesson documentation:

```
llmfs/
├── Cargo.toml              # Workspace configuration
├── README.md               # User-facing documentation
├── CLAUDE.md               # This file - AI assistant guidance
├── ch02/                   # Each chapter follows this structure
│   ├── Cargo.toml
│   ├── LESSONS-LLM.md      # LLM concepts and theory
│   ├── LESSONS-RS.md       # Rust concepts and patterns
│   └── src/
│       └── main.rs         # Implementation
└── target/                 # Shared build artifacts
```

## Development Philosophy

### Dual Learning Approach

When implementing or modifying code:
1. **Consider both pedagogical goals** - How does this teach LLM concepts AND Rust patterns?
2. **Document learnings** - Update relevant LESSONS files when adding new concepts
3. **Prefer clarity over cleverness** - Code should be educational first, optimized second
4. **Explain the "why"** - Comments should explain both the LLM reasoning and Rust idioms

### Code Style Guidelines

- Use extensive inline documentation explaining both LLM concepts and Rust patterns
- Prefer explicit types over inference when it aids understanding
- Break complex operations into clear steps with explanatory comments
- Include examples in doc comments where helpful
- Avoid advanced Rust features unless they directly benefit the learning experience

## Common Commands

### Running a specific chapter
```bash
cargo run -p ch02 -- --help
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

**LLM Focus**: Understanding data acquisition and preparation for language models
**Rust Focus**: Async programming, CLI tools, error handling, streaming I/O

Current implementation includes:
- CLI argument parsing with clap
- Async file downloading with progress reporting
- Streaming downloads for memory efficiency
- Error handling with Result types
- Visual progress bars with indicatif

Key dependencies:
- `clap` v4.5 - CLI argument parsing
- `indicatif` v0.17 - Progress bars
- `tokio` v1.47.1 - Async runtime
- `reqwest` v0.12.22 - HTTP client
- `futures-util` v0.3 - Stream utilities

## Working with LESSONS Files

### LESSONS-LLM.md Structure

Each LESSONS-LLM.md should contain:
1. **Chapter Overview** - What LLM concepts are covered
2. **Key Concepts** - Core ideas with explanations
3. **Why This Matters** - How it connects to building complete LLMs
4. **Prerequisites** - What readers should understand first
5. **Looking Ahead** - How this prepares for future chapters

### LESSONS-RS.md Structure

Each LESSONS-RS.md should contain:
1. **Rust Concepts Introduced** - New language features used
2. **Patterns and Idioms** - Best practices demonstrated
3. **Common Pitfalls** - What to watch out for
4. **Performance Notes** - Why certain approaches were chosen
5. **Exercises** - Ways to extend or modify the code for learning

## Development Notes

- This is a translation project with dual teaching goals
- Maintain consistency with the book's educational approach
- Balance idiomatic Rust with clear, understandable code
- Each chapter should be self-contained but build on previous knowledge
- Test all code examples and ensure they compile without warnings
- Keep dependencies minimal and well-justified

## When Adding New Chapters

1. Create the chapter directory and Cargo.toml
2. Add to workspace members in root Cargo.toml
3. Create both LESSONS-LLM.md and LESSONS-RS.md
4. Implement with extensive educational comments
5. Update README.md with chapter description
6. Test thoroughly with various inputs

## Documentation References

- Rust CLI information can be found at: @docs\refs\rust\rust-cli-guide.md
- Rust Cargo information can be found at: @docs\refs\rust\cargo.md