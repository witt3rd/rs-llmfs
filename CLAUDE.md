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

- Rust CLI information can be found at: @llm-docs\rust\rust-cli-guide.md
- Rust Cargo information can be found at: @llm-docs\rust\cargo.md

**CRITICAL**: These files serve completely different audiences and purposes. They must maintain strict separation of concerns.

### Core Distinction

- **LESSONS-LLM.md**: Teaches **WHAT** we're building (LLM/ML concepts) - for people learning about language models
- **LESSONS-RS.md**: Teaches **HOW** to build it in Rust (language patterns) - for people learning Rust programming

### LESSONS-LLM.md - Machine Learning Concepts Only

#### Purpose

Documents the machine learning and LLM theory being implemented. This file teaches ML concepts, NOT programming.

#### Content Requirements

1. **Chapter Overview** - What LLM/ML concepts are covered
2. **Key Concepts** - Core ML theory (tokenization, attention, training, etc.)
3. **Mathematical Foundations** - Equations, algorithms, theoretical background
4. **Why This Matters** - How it connects to building complete LLMs
5. **Prerequisites** - What ML/LLM concepts readers should understand first
6. **Looking Ahead** - How this prepares for future LLM chapters
7. **Implementation Notes** - Brief mention that concepts are implemented in Rust (1-2 sentences max)

#### Code Examples

- **Use Rust syntax** since that's our implementation language
- **Focus on WHAT the code does** (ML perspective), not HOW Rust works
- **No Rust idiom explanations** - don't explain ownership, borrowing, traits, etc.

#### Strictly FORBIDDEN in LESSONS-LLM.md

- ❌ Rust language tutorials or explanations
- ❌ Command-line usage instructions (belongs in README)
- ❌ Rust error handling patterns
- ❌ Cargo/crate management
- ❌ Performance optimizations specific to Rust
- ❌ Comparing Rust to other languages

#### Good Example

```markdown
## Tokenization

Tokenization converts text into numerical representations. Byte-Pair Encoding (BPE)
builds a vocabulary through iterative merging of frequent character pairs.

\```rust
// BPE merges the most frequent pair
let most_frequent = find_most_frequent_pair(&tokens);
tokens = merge_pair(tokens, most_frequent);
\```

This process continues until reaching the target vocabulary size, balancing
between character-level and word-level representations.
```

### LESSONS-RS.md - Rust Programming Patterns Only

#### Purpose

Documents Rust programming patterns, idioms, and techniques. This file teaches Rust, using the LLM project as a vehicle for learning.

#### Content Requirements

1. **Rust Concepts Introduced** - New language features demonstrated
2. **Patterns and Idioms** - Best practices and idiomatic Rust
3. **Common Pitfalls** - Rust-specific gotchas and how to avoid them
4. **Performance Notes** - Why certain Rust approaches were chosen
5. **Error Handling** - How to properly handle errors in Rust
6. **Memory Management** - Ownership, borrowing, lifetimes as needed
7. **Exercises** - Rust-focused coding challenges

#### Code Examples

- **Focus on HOW Rust works**, not what the ML algorithm does
- **Explain Rust-specific patterns** - ownership, traits, generics, etc.
- **Show alternative implementations** to teach different Rust approaches

#### Strictly FORBIDDEN in LESSONS-RS.md

- ❌ ML/LLM theory or algorithms
- ❌ Tokenization explanations
- ❌ Neural network architecture
- ❌ Training concepts
- ❌ Mathematical foundations
- ❌ References to "GPT", "transformer", "attention" etc. except in passing

#### Good Example

```markdown
## Working with External Crates

When integrating external crates, handle API differences:

\```rust
// External crate requires owned data
fn process(data: &[u32]) -> Result<String, Error> {
    let owned = data.to_vec();  // Clone when necessary
    external_api.decode(owned)
}
\```

This pattern shows when to convert borrowed slices to owned vectors,
a common requirement when adapting external APIs.
```

### Common Mistakes to Avoid

1. **Mixing Concerns**: Don't explain Rust ownership in LESSONS-LLM.md
2. **Wrong Audience**: LESSONS-LLM readers care about ML, not Rust syntax
3. **Command Instructions**: CLI usage belongs in README, not LESSONS files
4. **Theory in Wrong Place**: Don't explain BPE algorithm in LESSONS-RS.md
5. **Language Comparisons**: Don't compare Python/Rust in either file

### Quick Test for Correct Placement

Ask yourself:

- Is this about **ML/AI theory**? → LESSONS-LLM.md
- Is this about **Rust programming**? → LESSONS-RS.md
- Is this about **how to run the code**? → README.md
- Is this about **both ML and Rust**? → Split it into both files appropriately

### When Updating LESSONS Files

1. **Read the existing file first** to understand its focus
2. **Maintain the separation** - don't let concerns bleed across files
3. **Use appropriate terminology** - ML terms in LLM file, Rust terms in RS file
4. **Keep examples focused** - one teaching goal per example
5. **Review after writing** - would a pure ML student or pure Rust learner understand?

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

## Available Chapters

### Chapter 2: Working with Text Data (`ch02`)

Implements data acquisition, text preprocessing, tokenization, and dataset preparation.

**Key features:**

- Multiple tokenization approaches (word-level V1/V2, BPE with tiktoken)
- Sliding window dataset creation for training data
- Token visualization with colored output
- Side-by-side tokenizer comparison
- Text corpus downloading with streaming

### BPE Chapter: Byte-Pair Encoding (`bpe`)

Focuses on understanding and implementing the BPE algorithm from scratch.

**Status:** In development

- remember to ALWAYS use the 'cargo' command to modify .toml files