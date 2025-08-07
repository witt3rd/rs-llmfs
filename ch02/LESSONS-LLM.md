# Chapter 2: LLM Lessons - Working with Text Data

## Chapter Overview

This chapter introduces the fundamental first step in building any language model: acquiring and preparing text data. Before we can train a model to understand and generate text, we need high-quality training data. This chapter focuses on the practical aspects of downloading and managing text corpora that will form the foundation of our LLM.

## Key Concepts

### 1. **Text Corpora as Training Data**

- Language models learn patterns from large collections of text
- Quality and diversity of training data directly impacts model performance
- Common sources: books, articles, web pages, code repositories
- File formats: plain text, JSON, CSV, compressed archives

### 2. **Data Acquisition Strategies**

- **Batch downloading**: Fetching complete datasets at once
- **Streaming**: Processing data as it arrives (memory-efficient)
- **Incremental updates**: Adding new data to existing corpora
- **Error handling**: Dealing with network issues, partial downloads

### 3. **Initial Text Analysis**

Understanding your data before processing:

```python
# Book example:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

Key metrics to examine:

- **Character count**: Total size of the text
- **Line count**: Structure of the document
- **Word count**: Vocabulary size estimation
- **Preview**: Quick quality check of content

### 4. **Storage Considerations for ML**

- Organizing data for efficient access during training
- Directory structures that support train/validation/test splits
- Metadata tracking (source, download date, version)
- Compression vs. quick access trade-offs

### 5. **Preprocessing Pipeline Preparation**

- Setting up infrastructure for future text processing steps
- Considering tokenization requirements early
- Planning for data cleaning and filtering
- Establishing reproducible workflows

### 6. **Introduction to Tokenization**

Tokenization is the process of converting text into numerical representations that models can process:

#### Simple Tokenization Approaches

```python
# Word-level tokenization (our SimpleTokenizerV1)
text = "Hello, world!"
tokens = text.split()  # ["Hello,", "world!"]
vocab = {word: idx for idx, word in enumerate(unique_words)}
ids = [vocab[word] for word in tokens]
```

Limitations:

- Large vocabulary size (one entry per unique word)
- Cannot handle out-of-vocabulary (OOV) words
- Inefficient for morphologically rich languages

#### Subword Tokenization (BPE)

Modern LLMs use Byte Pair Encoding (BPE) as implemented in tiktoken:

```python
import tiktoken
encoding = tiktoken.get_encoding("r50k_base")  # GPT-2 tokenizer
tokens = encoding.encode("Hello, world!")
# Produces: [15496, 11, 995, 0]  # Fewer tokens, handles any text
```

Advantages:

- Smaller vocabulary (~50k tokens vs millions of words)
- Handles any input text (no OOV problem)
- Captures subword patterns ("running" → "run" + "ning")
- Better cross-lingual transfer

#### Special Tokens

Production tokenizers include special tokens for model control:

- `<|endoftext|>`: Marks document boundaries
- `<|unk|>`: Unknown token placeholder (rare with BPE)
- Task-specific tokens: `<|translate|>`, `<|summarize|>`, etc.

These tokens allow models to understand document structure and perform different tasks.

## Why This Matters

### Foundation for Everything

- **No data, no model**: LLMs are fundamentally pattern recognition systems trained on text
- **Data quality ceiling**: Your model can only be as good as its training data
- **Scale matters**: Modern LLMs train on billions of tokens

### Practical Skills

- Real-world ML projects spend significant time on data engineering
- Understanding data pipelines is crucial for production systems
- These patterns apply beyond LLMs to any data-intensive application

### Connection to Transformers

- Transformers process sequences of tokens derived from text
- Attention mechanisms learn relationships within this data
- The better your data preparation, the more effective training becomes

## Prerequisites

To fully understand this chapter, you should be familiar with:

- Basic file I/O concepts
- HTTP/HTTPS protocols
- Command-line interfaces
- File system organization

## Practical Implementation

Our Rust implementation provides comprehensive text processing capabilities:

### 1. **Guided Demo Mode**

Replicates the exact book examples with all 10 steps:

```bash
cargo run -p ch02 -- demo
```

- Downloads "the-verdict.txt"
- Analyzes character, line, and word counts
- Demonstrates text splitting (whitespace, punctuation, comprehensive)
- Builds vocabulary from unique tokens
- Implements SimpleTokenizerV1 (with error on unknown tokens)
- Implements SimpleTokenizerV2 (with <|unk|> handling)
- Shows special token usage (<|endoftext|>)
- **NEW: Demonstrates tiktoken (GPT-2 BPE tokenizer)**
- Compares tokenization efficiency across methods

### 2. **Generic Download Tool**

For experimenting with different texts:

```bash
cargo run -p ch02 -- download <URL> <output-path>
```

### 3. **Text Analysis Tool**

For examining any text file:

```bash
cargo run -p ch02 -- analyze <file-path> --preview-length 200
```

### 4. **Text Splitting Tool**

Visual demonstration of tokenization patterns:

```bash
cargo run -p ch02 -- split --method all --max-display 50
```

Methods available:

- `ws`: Whitespace only
- `punct`: Whitespace + basic punctuation
- `all`: Comprehensive punctuation (as in book)

### 5. **Tokenizer Comparison Tool**

Compare different tokenization approaches:

```bash
# Use tiktoken (GPT-2)
cargo run -p ch02 -- tokenize --tokenizer tiktoken --detailed

# Use simple tokenizer V1 (requires vocabulary)
cargo run -p ch02 -- tokenize --tokenizer v1

# Use simple tokenizer V2 (with <|unk|> handling)
cargo run -p ch02 -- tokenize --tokenizer v2
```

Options:

- `--text "custom text"`: Provide custom text
- `--file-path path/to/file`: Tokenize a file
- `--detailed`: Show token-by-token breakdown

This modular approach allows both guided learning and experimentation with real tokenization strategies used in production LLMs.

## Looking Ahead

This chapter prepares us for:

### Chapter 3: Tokenization

- Converting raw text into numerical representations
- Building vocabulary from our downloaded corpora
- Understanding different tokenization strategies

### Chapter 4: Data Loading

- Creating efficient data pipelines for training
- Batching and shuffling strategies
- Memory management for large datasets

### Future Training Chapters

- Using our prepared data to train transformer models
- Evaluating model performance on held-out data
- Fine-tuning on specific text domains

## Tokenization Insights

### Comparing Tokenization Approaches

Through our implementation, we can observe key differences:

| Approach | Vocabulary Size | OOV Handling | Token Efficiency | Use Case |
|----------|----------------|--------------|------------------|----------|
| Word-level (V1) | ~10k-100k+ | Fails on unknown | Poor (1 token/word) | Educational |
| Word-level + UNK (V2) | ~10k-100k+ | Maps to <\|unk\|> | Poor | Simple systems |
| BPE (tiktoken) | ~50k | Subword decomposition | Excellent | Production LLMs |

### Example Comparison

For the text: "The quick brown fox jumps over the lazy dog"

- **SimpleTokenizerV1/V2**: 9 tokens (one per word)
- **tiktoken (GPT-2)**: 9 tokens (efficient on common words)

But for: "supercalifragilisticexpialidocious"

- **SimpleTokenizerV1**: ERROR (not in vocabulary)
- **SimpleTokenizerV2**: 1 token (<\|unk\|>) - loses all information
- **tiktoken**: ~8 tokens (preserves subword meaning)

### Why BPE Dominates Modern LLMs

1. **Efficiency**: Fewer tokens = faster processing, less memory
2. **Flexibility**: Handles any text without vocabulary explosions
3. **Multilingual**: Same tokenizer works across languages
4. **Morphology**: Captures word roots and affixes naturally
5. **Rare words**: Gracefully decomposes instead of failing

### Tokenization's Impact on Model Behavior

- **Context length**: With 2048 token limit, BPE fits ~1500 words vs ~500 with word-level
- **Learning efficiency**: Subword patterns transfer across related words
- **Generation quality**: Can create novel words through subword combination
- **Cross-lingual transfer**: Shared subwords enable multilingual models

## Key Takeaways

1. **Data is the foundation** - Time invested in proper data acquisition pays dividends
2. **Think ahead** - Consider future processing needs when organizing data
3. **Automate wisely** - Build reusable tools for common data tasks
4. **Document everything** - Track data sources, versions, and processing steps
5. **Tokenization matters** - Choice of tokenizer fundamentally affects model capabilities
6. **BPE is production-ready** - Simple tokenizers help understanding, BPE powers real systems

## Reflection Questions

- What makes a good training corpus for language models?
- How might data requirements differ for specialized vs. general-purpose LLMs?
- What ethical considerations arise when collecting training data?
- How do modern LLMs handle multilingual data?
- Why is subword tokenization superior to word-level for neural language models?
- How does vocabulary size affect model training and inference speed?
- What are the trade-offs between different tokenization strategies?
- How might you design a tokenizer for a specialized domain (e.g., code, chemistry)?
- What role do special tokens play in controlling model behavior?
