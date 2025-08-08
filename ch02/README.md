# Chapter 2: Working with Text Data

This chapter implements text data acquisition and preparation for language models, following Chapter 2 of "Build a Large Language Model from Scratch" by Sebastian Raschka.

## Overview

This implementation provides tools for:
- Downloading and managing text corpora
- Analyzing text characteristics
- Multiple tokenization approaches
- Creating training data with sliding windows
- Organizing data into datasets for model training

## Installation

This chapter is part of the llmfs workspace. From the workspace root:

```bash
# Build this chapter
cargo build -p ch02

# Run with help
cargo run -p ch02 -- --help
```

## Usage

### Quick Start: Demo Mode

Run the complete book demonstration that shows all concepts:

```bash
cargo run -p ch02 -- demo
```

This runs through all 12 steps from the book:
1. Downloads "the-verdict.txt" 
2. Analyzes character, line, and word counts
3. Demonstrates text splitting methods
4. Builds vocabulary from unique tokens
5. Implements SimpleTokenizerV1 (errors on unknown tokens)
6. Implements SimpleTokenizerV2 (handles unknown with `<|unk|>`)
7. Shows special token usage (`<|endoftext|>`)
8. Demonstrates tiktoken (GPT-2 BPE tokenizer)
9. Compares tokenization efficiency
10. Visualizes text as token IDs
11. Shows sliding windows for context-target pairs
12. Creates GPTDatasetV1 for batch training

### Individual Commands

#### Download Text Files

```bash
# Download to default data directory
cargo run -p ch02 -- download <URL>

# Download to specific location
cargo run -p ch02 -- download <URL> <output-path>

# Example
cargo run -p ch02 -- download https://example.com/text.txt data/my-text.txt
```

#### Analyze Text Files

```bash
# Basic analysis
cargo run -p ch02 -- analyze <file-path>

# With custom preview length
cargo run -p ch02 -- analyze data/the-verdict.txt --preview-length 200
```

Shows:
- Total character count
- Line count
- Word count (approximate)
- Text preview

#### Text Splitting Demonstration

Visualize how text is split into tokens:

```bash
# Whitespace splitting
cargo run -p ch02 -- split --method ws

# Basic punctuation splitting
cargo run -p ch02 -- split --method punct

# Comprehensive punctuation (book method)
cargo run -p ch02 -- split --method all --max-display 50
```

Options:
- `--file-path`: Custom text file (default: the-verdict.txt)
- `--method`: Splitting method (ws/punct/all)
- `--max-display`: Number of tokens to show

#### Tokenization

Compare different tokenization approaches:

```bash
# GPT-2 tokenizer (tiktoken)
cargo run -p ch02 -- tokenize --tokenizer tiktoken --text "Hello, world!"

# With detailed token breakdown
cargo run -p ch02 -- tokenize --tokenizer tiktoken --detailed

# SimpleTokenizerV1 (requires vocabulary, will error on unknown)
cargo run -p ch02 -- tokenize --tokenizer v1 --file-path data/the-verdict.txt

# SimpleTokenizerV2 (handles unknown tokens)
cargo run -p ch02 -- tokenize --tokenizer v2 --text "Unknown word: xyz123"
```

Options:
- `--tokenizer`: v1/v2/tiktoken
- `--text`: Direct text input
- `--file-path`: Tokenize file contents
- `--detailed`: Show token-by-token analysis

#### Sliding Windows

Visualize training data creation with sliding windows:

```bash
# Basic sliding window
cargo run -p ch02 -- sliding-window

# Custom parameters with decoded text
cargo run -p ch02 -- sliding-window \
  --context-size 8 \
  --start-pos 100 \
  --max-windows 5 \
  --show-decoded
```

Options:
- `--context-size`: Window size (default: 4)
- `--start-pos`: Starting position (default: 50)
- `--max-windows`: Windows to display (default: 10)
- `--show-decoded`: Show text alongside token IDs
- `--file-path`: Custom text file

#### Dataset Creation

Create training datasets with configurable parameters:

```bash
# Basic dataset
cargo run -p ch02 -- dataset

# Detailed configuration
cargo run -p ch02 -- dataset \
  --max-length 8 \
  --stride 4 \
  --num-samples 10 \
  --show-decoded \
  --verbose
```

Options:
- `--max-length`: Sequence length per sample (default: 4)
- `--stride`: Tokens between samples (default: 1)
- `--num-samples`: Samples to display (0 for all)
- `--show-decoded`: Show decoded text
- `--verbose`: Show statistics (coverage, overlap)
- `--file-path`: Custom text file

## Examples

### Example 1: Process Custom Text

```bash
# Download a book
cargo run -p ch02 -- download \
  https://www.gutenberg.org/files/1342/1342-0.txt \
  data/pride-prejudice.txt

# Analyze it
cargo run -p ch02 -- analyze data/pride-prejudice.txt

# Tokenize with GPT-2
cargo run -p ch02 -- tokenize \
  --tokenizer tiktoken \
  --file-path data/pride-prejudice.txt \
  --detailed

# Create dataset
cargo run -p ch02 -- dataset \
  --file-path data/pride-prejudice.txt \
  --max-length 256 \
  --stride 128 \
  --verbose
```

### Example 2: Compare Tokenizers

```bash
# Same text, different tokenizers
echo "The quick brown fox jumps over the lazy dog." > test.txt

# SimpleTokenizerV2
cargo run -p ch02 -- tokenize --tokenizer v2 --file-path test.txt

# Tiktoken (GPT-2)
cargo run -p ch02 -- tokenize --tokenizer tiktoken --file-path test.txt --detailed
```

### Example 3: Training Data Preparation

```bash
# Small context for testing
cargo run -p ch02 -- dataset \
  --max-length 4 \
  --stride 1 \
  --num-samples 5 \
  --show-decoded

# Larger context for actual training
cargo run -p ch02 -- dataset \
  --max-length 256 \
  --stride 256 \
  --verbose
```

## Output Format

### Tokenization Output

```
Encoded text: [40, 367, 2885, 1464, ...]
Token count: 5145

Token Analysis:
[0] 40 → "I"
[1] 367 → " HAD"
[2] 2885 → " always"
...
```

### Dataset Output

```
Dataset Statistics:
  Total samples: 5141
  Input shape per sample: [4]
  Target shape per sample: [4]
  Coverage: 99.9%

Sample 0
  Input IDs: [40, 367, 2885, 1464]
  Target IDs: [367, 2885, 1464, 1807]
  Input Text: "I HAD always"
  Target Text: " HAD always thought"
```

## File Structure

```
ch02/
├── Cargo.toml           # Package configuration
├── README.md            # This file
├── LESSONS-LLM.md       # ML/LLM concepts explained
├── LESSONS-RS.md        # Rust patterns demonstrated
├── src/
│   └── main.rs          # Complete implementation
└── data/                # Downloaded text files (git-ignored)
    └── the-verdict.txt  # Default text corpus
```

## Dependencies

- `clap` - Command-line argument parsing
- `tokio` - Async runtime for I/O operations
- `reqwest` - HTTP client for downloads
- `indicatif` - Progress bars
- `colored` - Terminal colors
- `tiktoken-rs` - GPT-2 tokenizer
- `regex` - Text pattern matching
- `futures-util` - Stream processing

## Learning Resources

- **LESSONS-LLM.md** - Understand the ML concepts (tokenization, BPE, training data)
- **LESSONS-RS.md** - Learn Rust patterns (async/await, error handling, CLI design)

## Testing

```bash
# Run all tests
cargo test -p ch02

# Run with output
cargo test -p ch02 -- --nocapture
```

## Common Issues

### File Not Found
The demo command will automatically download the-verdict.txt if not present. For other commands, ensure the file exists or specify `--file-path`.

### Unknown Token Error
SimpleTokenizerV1 will error on unknown words. Use V2 or tiktoken for handling arbitrary text.

### Memory Usage
For very large files, the dataset command with small stride values can use significant memory. Increase stride to reduce memory usage.

## Next Steps

After completing this chapter, you understand:
- How to prepare text data for language models
- Different tokenization strategies and trade-offs
- How training data is structured as context-target pairs
- The foundation for building data loaders in future chapters

Continue to Chapter 3 for deeper tokenization concepts or Chapter 4 for data loading and batching strategies.