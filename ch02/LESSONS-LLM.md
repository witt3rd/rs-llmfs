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

```rust
// Book example implemented in Rust:
let raw_text = fs::read_to_string("the-verdict.txt").await?;
println!("Total number of characters: {}", raw_text.len());
println!("{}", &raw_text[..99.min(raw_text.len())]);
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

```rust
// Word-level tokenization (our SimpleTokenizerV1)
let text = "Hello, world!";
let tokens: Vec<&str> = text.split_whitespace().collect();  // ["Hello,", "world!"]
let vocab: HashMap<String, usize> = unique_words.into_iter()
    .enumerate()
    .map(|(idx, word)| (word, idx))
    .collect();
let ids: Vec<usize> = tokens.iter()
    .map(|word| vocab[word])
    .collect();
```

Limitations:

- Large vocabulary size (one entry per unique word)
- Cannot handle out-of-vocabulary (OOV) words
- Inefficient for morphologically rich languages

#### Subword Tokenization (BPE)

Modern LLMs use Byte Pair Encoding (BPE) as implemented in tiktoken:

```rust
use tiktoken_rs::r50k_base;
let encoding = r50k_base()?;  // GPT-2 tokenizer
let tokens = encoding.encode_with_special_tokens("Hello, world!");
// Produces: [15496, 11, 995, 0]  // Fewer tokens, handles any text
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

## Implementation Notes

Our Rust implementation demonstrates all the concepts from this chapter:

- Text corpus acquisition and management
- Multiple tokenization approaches (word-level, subword/BPE)
- Vocabulary building and token encoding/decoding
- Sliding window creation for training data
- GPTDatasetV1 for organizing context-target pairs
- Comparative analysis of tokenization efficiency

The implementation follows the book's progression while adapting Python/PyTorch patterns to idiomatic Rust.

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

## Training Data Preparation

### Sliding Windows for Context-Target Pairs

Language models learn to predict the next token given a context. We create training data using a sliding window approach:

```rust
// For each position in the text, create a training example
let context_size = 4;
for i in 0..tokens.len() - context_size {
    let input_chunk = &tokens[i..i + context_size];      // Context: [t0, t1, t2, t3]
    let target_chunk = &tokens[i + 1..i + context_size + 1];  // Target: [t1, t2, t3, t4]
}
```

This creates overlapping sequences where:
- **Input**: A sequence of tokens representing the context
- **Target**: The same sequence shifted by one token (what to predict)

#### Growing Context Windows

During training, models learn from progressively larger contexts:

```
Position 0: []           → predict token 0
Position 1: [t0]         → predict token 1  
Position 2: [t0, t1]     → predict token 2
Position 3: [t0, t1, t2] → predict token 3
...
```

This teaches the model to generate text with varying amounts of context, from starting fresh to continuing long passages.

### GPTDatasetV1: Batch Training Data

For efficient training, we organize sliding windows into a dataset:

```rust
struct GPTDatasetV1 {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

impl GPTDatasetV1 {
    fn new(text: &str, max_length: usize, stride: usize) -> Self {
        let tokenizer = r50k_base().unwrap();
        let token_ids = tokenizer.encode_with_special_tokens(text);
        
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        
        // Create all training pairs upfront
        let mut i = 0;
        while i + max_length < token_ids.len() {
            let input_chunk = token_ids[i..i + max_length].to_vec();
            let target_chunk = token_ids[i + 1..i + max_length + 1].to_vec();
            input_ids.push(input_chunk);
            target_ids.push(target_chunk);
            i += stride;
        }
        
        GPTDatasetV1 { input_ids, target_ids }
    }
}
```

Key parameters:
- **max_length**: Size of each context window (e.g., 256, 1024 tokens)
- **stride**: How many tokens to skip between samples
  - stride=1: Maximum overlap, most training examples
  - stride=max_length: No overlap, fewer examples

#### Trade-offs in Data Preparation

| Parameter | Small Value | Large Value |
|-----------|------------|-------------|
| **max_length** | Faster training, less context | Slower training, more context |
| **stride** | More examples, high overlap | Fewer examples, no redundancy |

Example with max_length=4, stride=1:
```
Sample 0: [I, had, always, thought] → [had, always, thought, Jack]
Sample 1: [had, always, thought, Jack] → [always, thought, Jack, was]
Sample 2: [always, thought, Jack, was] → [thought, Jack, was, clever]
```

### Why This Approach Works

1. **Dense supervision**: Every token position provides a learning signal
2. **Context variety**: Model sees tokens in many different contexts
3. **Efficient batching**: Fixed-size tensors enable GPU parallelization
4. **Natural curriculum**: Short contexts are easier, preparing for longer ones

### Connection to Transformer Training

This data preparation directly feeds into the transformer training loop:

1. **Forward pass**: Input tokens → transformer → predicted tokens
2. **Loss calculation**: Compare predictions with target tokens
3. **Backpropagation**: Update weights to improve predictions
4. **Repeat**: Process millions/billions of such examples

The quality and diversity of these context-target pairs directly determine model capabilities.

## Key Takeaways

1. **Data is the foundation** - Time invested in proper data acquisition pays dividends
2. **Think ahead** - Consider future processing needs when organizing data
3. **Automate wisely** - Build reusable tools for common data tasks
4. **Document everything** - Track data sources, versions, and processing steps
5. **Tokenization matters** - Choice of tokenizer fundamentally affects model capabilities
6. **BPE is production-ready** - Simple tokenizers help understanding, BPE powers real systems
7. **Sliding windows create training data** - Every position in text becomes a learning opportunity
8. **Batch preparation enables efficient training** - GPTDataset organizes data for parallel processing

## DataLoaders: Efficient Batch Processing

### From Dataset to DataLoader

While a Dataset organizes our training data, a DataLoader handles the logistics of feeding this data to the model during training:

```python
# Dataset provides individual samples
dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=128)
single_sample = dataset[0]  # One input-target pair

# DataLoader provides batches for training
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
batch = next(iter(dataloader))  # 8 input-target pairs as tensors
```

### Why We Need DataLoaders

#### 1. **Batching for GPU Efficiency**

Modern GPUs process multiple examples in parallel:
- Single example: Wastes GPU compute capacity
- Batch of 8-64: Maximizes throughput
- Too large: May exceed memory

```
Without batching (sequential):
Sample 1: [████____] 25% GPU usage
Sample 2: [████____] 25% GPU usage
Sample 3: [████____] 25% GPU usage

With batching (parallel):
Batch of 8: [████████] 100% GPU usage
```

#### 2. **Shuffling for Better Learning**

Training on sequential data creates problems:
- Model memorizes order instead of patterns
- Gradient updates become correlated
- Poor generalization

DataLoader shuffling ensures:
- Random sampling across the dataset
- Different batch compositions each epoch
- Better gradient estimates

#### 3. **Memory Management**

DataLoaders load data on-demand:
```
Dataset: 1GB of tokenized text stored
DataLoader: Loads only current batch (e.g., 1MB) into GPU memory
```

This enables training on datasets larger than available RAM.

### Key DataLoader Parameters

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Examples per batch
    shuffle=True,       # Randomize order
    drop_last=True,     # Drop incomplete final batch
    num_workers=4       # Parallel data loading threads
)
```

#### batch_size
- **Small (1-8)**: More gradient updates, slower convergence, less memory
- **Medium (32-64)**: Good balance for most tasks
- **Large (128+)**: Faster training, needs more memory, may hurt generalization

#### shuffle
- **True for training**: Prevents memorization of order
- **False for validation**: Reproducible evaluation
- **Seed for reproducibility**: Same random order across runs

#### drop_last
- **True**: Ensures all batches have same size (important for some models)
- **False**: Uses all data, but last batch may be smaller

### The Training Loop Pattern

DataLoaders integrate seamlessly into the training loop:

```python
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()}")
```

### Batch Tensor Shapes

Understanding batch dimensions is crucial:

```python
# Single sample from dataset
input_ids = [101, 2054, 2003, 102]  # Shape: [4]

# Batch from dataloader (batch_size=8)
batch_inputs = [
    [101, 2054, 2003, 102],
    [101, 5328, 1045, 102],
    ...  # 6 more samples
]  # Shape: [8, 4] = [batch_size, sequence_length]
```

The model processes all 8 examples simultaneously, computing:
- 8 sets of embeddings
- 8 attention patterns
- 8 loss values (averaged for backprop)

### DataLoader vs Dataset

| Component | Purpose | When Called |
|-----------|---------|-------------|
| **Dataset** | Stores and organizes data | Once during initialization |
| **DataLoader** | Serves batches during training | Every training step |

```python
# Dataset: "Here's all the data, organized"
dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
print(f"Total samples: {len(dataset)}")  # e.g., 10,000

# DataLoader: "Here's how to serve it for training"
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Batches per epoch: {len(dataloader)}")  # e.g., 313 (10,000/32)
```

### Iteration Patterns

#### Training (Multiple Epochs)
```python
for epoch in range(10):  # Train for 10 epochs
    for batch in dataloader:  # Each epoch sees all data
        train_step(batch)
    # Shuffle happens automatically between epochs
```

#### Quick Testing (First Few Batches)
```python
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
# Useful for debugging or sanity checks
```

### Impact on Model Training

DataLoader design choices directly affect training dynamics:

1. **Larger batches → Smoother gradients**
   - More stable training
   - Can use higher learning rates
   - May converge to worse minima

2. **Smaller batches → Noisier gradients**
   - Acts as regularization
   - Needs smaller learning rates
   - Often better final performance

3. **Shuffling → Better generalization**
   - Breaks spurious correlations
   - Ensures diverse mini-batches
   - Critical for small datasets

### Connection to Transformer Training

For transformer models specifically:

- **Attention computation**: Scales with sequence length squared
- **Batch processing**: Parallelizes across batch dimension
- **Memory requirements**: `batch_size × sequence_length × model_dim`
- **Gradient accumulation**: Simulate larger batches on limited hardware

Example memory calculation:
```
batch_size = 8
sequence_length = 512
model_dim = 768 (BERT-base)
Memory ≈ 8 × 512 × 768 × 4 bytes = ~12 MB per batch
```

### Best Practices

1. **Start with smaller batches** during development (faster iteration)
2. **Monitor GPU memory** usage to find optimal batch size
3. **Use gradient accumulation** if optimal batch size exceeds memory
4. **Set different configs** for train/val/test dataloaders
5. **Save DataLoader state** for resuming interrupted training

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
- Why is batching essential for efficient GPU utilization?
- How does batch size affect the bias-variance tradeoff in training?
- What's the relationship between batch size and learning rate?
- When would you disable shuffling in a DataLoader?
- How do DataLoaders enable training on datasets larger than RAM?
