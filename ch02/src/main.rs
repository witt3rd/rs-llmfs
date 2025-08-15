//! # Chapter 2: Asynchronous File Downloader with CLI
//!
//! This module demonstrates downloading files from URLs using async Rust with:
//! - Command-line interface using clap
//! - Visual progress reporting using indicatif
//!
//! ## Key Concepts Covered
//! - Async/await programming with streaming downloads
//! - Error handling with `Result` and the `?` operator
//! - Working with trait objects (`Box<dyn Error>`)
//! - File I/O operations
//! - CLI argument parsing
//! - Progress bars and visual feedback

// Standard library imports (alphabetically sorted)
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::error::Error; // Import trait for Box<dyn Error>
use std::fmt; // Module import - we'll use fmt::Display
use std::path::{Path, PathBuf}; // Types imported directly
use std::sync::Arc; // For efficient immutable data sharing

// External crate imports (alphabetically sorted)
use burn::data::dataset::Dataset;
use burn::tensor::{Tensor, Int, TensorData};
use burn_ndarray::NdArray;
use clap::{Parser, Subcommand}; // Derive macro traits
use colored::*; // Colors for terminal output
use futures_util::StreamExt; // Trait needed for .next() on streams
use indicatif::{ProgressBar, ProgressStyle}; // Types imported directly
use regex::Regex; // Type imported directly
use reqwest; // Module import - we'll use reqwest::get() (function convention)
use tiktoken_rs::r50k_base; // GPT-2 tokenizer
use tokio::fs; // Module import - we'll use fs::read_to_string() (function convention)
use tokio::io::AsyncWriteExt; // Trait needed for .write_all() method

// Note: No internal imports as this is a binary crate (main.rs)

// Constants for demo
const VERDICT_FILENAME: &str = "data/the-verdict.txt";
const VERDICT_URL: &str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";

/// Color scheme for token display
struct ColorScheme {
    text_bg: Color,
    text_fg: Color,
    delimiter_bg: Color,
    delimiter_fg: Color,
}

impl ColorScheme {
    /// Default color scheme
    fn default() -> Self {
        ColorScheme {
            text_bg: Color::Blue,
            text_fg: Color::White,
            delimiter_bg: Color::BrightWhite,
            delimiter_fg: Color::Black,
        }
    }

    /// Apply text token colors
    fn style_text(&self, text: &str) -> ColoredString {
        text.color(self.text_fg).on_color(self.text_bg)
    }

    /// Apply delimiter token colors
    fn style_delimiter(&self, text: &str) -> ColoredString {
        text.color(self.delimiter_fg).on_color(self.delimiter_bg)
    }
}

/// Splitting method for text tokenization
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SplitMethod {
    /// Split on whitespace only
    #[value(name = "ws")]
    Whitespace,
    /// Split on whitespace, commas, and periods
    #[value(name = "punct")]
    Punctuation,
    /// Split on all punctuation and whitespace
    #[value(name = "all")]
    All,
}

/// Choice of tokenizer for the Tokenize subcommand
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TokenizerChoice {
    /// Our custom SimpleTokenizerV1 (requires vocabulary)
    #[value(name = "v1")]
    V1,
    /// Our custom SimpleTokenizerV2 with <|unk|> handling
    #[value(name = "v2")]
    V2,
    /// OpenAI's tiktoken (GPT-2 tokenizer)
    #[value(name = "tiktoken")]
    Tiktoken,
}

impl fmt::Display for SplitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitMethod::Whitespace => write!(f, "whitespace"),
            SplitMethod::Punctuation => write!(f, "punctuation"),
            SplitMethod::All => write!(f, "all"),
        }
    }
}

impl fmt::Display for TokenizerChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerChoice::V1 => write!(f, "SimpleTokenizerV1"),
            TokenizerChoice::V2 => write!(f, "SimpleTokenizerV2"),
            TokenizerChoice::Tiktoken => write!(f, "tiktoken (GPT-2)"),
        }
    }
}

/// A token from splitting (either text or delimiter)
#[derive(Debug, Clone)]
struct Token {
    content: String,
    is_delimiter: bool,
}

/// Simple tokenizer that converts between text and integer IDs
struct SimpleTokenizerV1 {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

/// Tokenizer V2 that handles unknown tokens with <|unk|>
struct SimpleTokenizerV2 {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

impl Token {
    /// Create a new token
    fn new(content: String, is_delimiter: bool) -> Self {
        Token { content, is_delimiter }
    }

    /// Display the token inline with background highlighting
    fn display_inline(&self, colors: &ColorScheme) -> ColoredString {
        if self.is_delimiter {
            // Delimiters get highlighted to show what was split on
            let display = match self.content.as_str() {
                "\n" => "↵\n",  // Show newline symbol then actual newline
                "\t" => "→",    // Tab arrow
                "\r" => "↵",    // Carriage return
                _ => &self.content,
            };
            colors.style_delimiter(display)
        } else {
            // Non-delimiter text tokens
            colors.style_text(&self.content)
        }
    }
}

impl SimpleTokenizerV1 {
    /// Create a new tokenizer from a vocabulary mapping
    fn new(vocab: HashMap<String, usize>) -> Self {
        // Create the reverse mapping (int to string)
        let int_to_str: HashMap<usize, String> = vocab
            .iter()
            .map(|(s, &i)| (i, s.clone()))
            .collect();

        SimpleTokenizerV1 {
            str_to_int: vocab,
            int_to_str,
        }
    }

    /// Encode text into a sequence of token IDs
    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        // Split text using comprehensive punctuation pattern
        let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

        // Split and filter empty strings
        let mut preprocessed = Vec::new();
        let mut last_end = 0;

        for mat in re.find_iter(text) {
            // Add text before match
            if mat.start() > last_end {
                let content = text[last_end..mat.start()].trim();
                if !content.is_empty() {
                    preprocessed.push(content);
                }
            }
            // Add the delimiter
            let content = mat.as_str().trim();
            if !content.is_empty() {
                preprocessed.push(content);
            }
            last_end = mat.end();
        }

        // Add remaining text
        if last_end < text.len() {
            let content = text[last_end..].trim();
            if !content.is_empty() {
                preprocessed.push(content);
            }
        }

        // Convert to IDs
        let mut ids = Vec::new();
        for token in preprocessed {
            match self.str_to_int.get(token) {
                Some(&id) => ids.push(id),
                None => return Err(format!("Token '{}' not in vocabulary", token)),
            }
        }

        Ok(ids)
    }

    /// Decode a sequence of token IDs back into text
    fn decode(&self, ids: &[usize]) -> Result<String, String> {
        // Convert IDs to strings
        let mut tokens = Vec::new();
        for &id in ids {
            match self.int_to_str.get(&id) {
                Some(token) => tokens.push(token.as_str()),
                None => return Err(format!("ID {} not in vocabulary", id)),
            }
        }

        // Join with spaces
        let mut text = tokens.join(" ");

        // Fix spacing around punctuation
        let re = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();
        text = re.replace_all(&text, "$1").to_string();

        Ok(text)
    }
}

impl SimpleTokenizerV2 {
    /// Create a new tokenizer from a vocabulary mapping
    fn new(vocab: HashMap<String, usize>) -> Self {
        // Create the reverse mapping (int to string)
        let int_to_str: HashMap<usize, String> = vocab
            .iter()
            .map(|(s, &i)| (i, s.clone()))
            .collect();

        SimpleTokenizerV2 {
            str_to_int: vocab,
            int_to_str,
        }
    }

    /// Encode text into a sequence of token IDs, using <|unk|> for unknown tokens
    fn encode(&self, text: &str) -> Vec<usize> {
        // Split text using comprehensive punctuation pattern
        let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

        // Split and filter empty strings
        let mut preprocessed = Vec::new();
        let mut last_end = 0;

        for mat in re.find_iter(text) {
            // Add text before match
            if mat.start() > last_end {
                let content = text[last_end..mat.start()].trim();
                if !content.is_empty() {
                    preprocessed.push(content);
                }
            }
            // Add the delimiter
            let content = mat.as_str().trim();
            if !content.is_empty() {
                preprocessed.push(content);
            }
            last_end = mat.end();
        }

        // Add remaining text
        if last_end < text.len() {
            let content = text[last_end..].trim();
            if !content.is_empty() {
                preprocessed.push(content);
            }
        }

        // Replace unknown tokens with <|unk|> and convert to IDs
        let mut ids = Vec::new();
        for token in preprocessed {
            let token_or_unk = if self.str_to_int.contains_key(token) {
                token
            } else {
                "<|unk|>"
            };

            // This should always succeed since we ensure <|unk|> is in vocabulary
            if let Some(&id) = self.str_to_int.get(token_or_unk) {
                ids.push(id);
            }
        }

        ids
    }

    /// Decode a sequence of token IDs back into text
    fn decode(&self, ids: &[usize]) -> Result<String, String> {
        // Convert IDs to strings
        let mut tokens = Vec::new();
        for &id in ids {
            match self.int_to_str.get(&id) {
                Some(token) => tokens.push(token.as_str()),
                None => return Err(format!("ID {} not in vocabulary", id)),
            }
        }

        // Join with spaces
        let mut text = tokens.join(" ");

        // Fix spacing around punctuation
        let re = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();
        text = re.replace_all(&text, "$1").to_string();

        Ok(text)
    }
}

/// A file downloader with progress reporting for LLM text data acquisition
#[derive(Parser, Debug)]
#[command(author, version, about = "Chapter 2: Working with Text Data", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download a file from a URL to a specific location
    Download {
        /// URL to download from
        url: String,

        /// Output path (required - can be file or directory)
        output: String,
    },
    /// Run the complete book demo (download + analyze the-verdict.txt)
    Demo,
    /// Analyze any text file (character count, preview, etc.)
    Analyze {
        /// Path to the text file to analyze
        file_path: String,

        /// Number of characters to preview
        #[arg(short, long, default_value = "99")]
        preview_length: usize,
    },
    /// Split text using various methods
    Split {
        /// Path to the text file to split (or use test string if not provided)
        file_path: Option<String>,

        /// Method to use for splitting
        #[arg(short, long, value_enum, default_value = "ws")]
        method: SplitMethod,

        /// Maximum number of tokens to display
        #[arg(long, default_value = "30")]
        max_display: usize,
    },
    /// Demonstrate various tokenizers (SimpleTokenizerV1, V2, and tiktoken)
    Tokenize {
        /// Text to tokenize (or use default examples if not provided)
        #[arg(short, long)]
        text: Option<String>,

        /// Path to file to tokenize (overrides text if provided)
        #[arg(short, long)]
        file_path: Option<String>,

        /// Which tokenizer to use
        #[arg(short = 'z', long, value_enum, default_value = "tiktoken")]
        tokenizer: TokenizerChoice,

        /// Show detailed token analysis
        #[arg(short, long)]
        detailed: bool,
    },
    /// Demonstrate sliding windows over encoded text for training data preparation
    SlidingWindow {
        /// Path to text file (defaults to the-verdict.txt if not provided)
        #[arg(short, long)]
        file_path: Option<String>,

        /// Context size (window size) for sliding window
        #[arg(short, long, default_value = "4")]
        context_size: usize,

        /// Starting position in the encoded text
        #[arg(short, long, default_value = "50")]
        start_pos: usize,

        /// Maximum number of windows to display
        #[arg(short, long, default_value = "10")]
        max_windows: usize,

        /// Show decoded text for each window
        #[arg(short = 'd', long)]
        show_decoded: bool,
    },
    /// Create a GPT dataset with input-target pairs for training
    Dataset {
        /// Path to text file (defaults to the-verdict.txt if not provided)
        #[arg(short, long)]
        file_path: Option<String>,

        /// Maximum sequence length for each sample
        #[arg(short, long, default_value = "4")]
        max_length: usize,

        /// Stride for sliding window (how many tokens to move forward)
        #[arg(short, long, default_value = "1")]
        stride: usize,

        /// Number of samples to display (0 for all)
        #[arg(short = 'n', long, default_value = "5")]
        num_samples: usize,

        /// Show decoded text for samples
        #[arg(short = 'd', long)]
        show_decoded: bool,

        /// Show dataset statistics
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    /// Demonstrate DataLoader for batch processing
    Dataloader {
        /// Path to text file (defaults to the-verdict.txt if not provided)
        #[arg(short, long)]
        file_path: Option<String>,

        /// Batch size for DataLoader
        #[arg(short, long, default_value = "4")]
        batch_size: usize,

        /// Maximum sequence length for each sample
        #[arg(short, long, default_value = "8")]
        max_length: usize,

        /// Stride for sliding window
        #[arg(short, long, default_value = "4")]
        stride: usize,

        /// Random seed for shuffling (omit for no shuffle)
        #[arg(long)]
        shuffle_seed: Option<u64>,

        /// Number of batches to display
        #[arg(short = 'n', long, default_value = "3")]
        num_batches: usize,

        /// Show decoded text for batches
        #[arg(short = 'd', long)]
        show_decoded: bool,
    },
}

/// Backend type for our tensors
type Backend = NdArray;

/// Dataset item containing input-target pairs
/// 
/// This struct is used when implementing Burn's Dataset trait.
/// In a real training scenario, a Batcher would consume these items
/// and convert them to tensors for batch processing.
#[allow(dead_code)]  // Used by Dataset trait implementation
#[derive(Clone, Debug)]
struct GPTDatasetItem {
    pub input_ids: Vec<i64>,  // Changed to i64 for Burn tensor compatibility
    pub target_ids: Vec<i64>,
}

/// GPT Dataset for creating input-target pairs for training
///
/// This dataset follows the pattern from the book where we create overlapping
/// sequences of tokens for language model training. Each sample consists of:
/// - Input: A sequence of tokens [t0, t1, ..., tn-1]
/// - Target: The next tokens [t1, t2, ..., tn]
///
/// ## PyTorch vs Rust/Burn Implementation
///
/// The PyTorch version is deceptively simple:
/// ```python
/// class GPTDatasetV1(Dataset):
///     def __init__(self, txt, tokenizer, max_length, stride):
///         self.input_ids = []
///         self.target_ids = []
///         token_ids = tokenizer.encode(txt)
///         for i in range(0, len(token_ids) - max_length, stride):
///             input_chunk = token_ids[i:i + max_length]
///             target_chunk = token_ids[i + 1: i + max_length + 1]
///             self.input_ids.append(torch.tensor(input_chunk))
///             self.target_ids.append(torch.tensor(target_chunk))
/// ```
///
/// The Rust/Burn version requires more explicit type handling due to:
///
/// ### 1. Library API Mismatch
/// - **tiktoken-rs** returns `Vec<u32>` for token IDs (inherited from C++)
/// - **Burn** expects `i64` for Int tensors (their design choice)
/// - **No bridge exists** between these two choices
///
/// ### 2. Rust's Type System
/// - **No implicit conversions**: Can't automatically convert u32 → i64
/// - **Ownership rules**: Slices are borrowed, tensors need owned data
/// - **Explicit everything**: Every conversion must be written out
///
/// ### 3. What Python Hides
/// In Python, `torch.tensor(token_ids[i:i+max_length])` does secretly:
/// 1. Slice the list (Python list, not numpy array)
/// 2. Infer the dtype (probably int64)
/// 3. Copy data to tensor storage
/// 4. Handle device placement
///
/// In Rust, we must do each step explicitly:
/// 1. Slice the Vec (`&token_ids[i..j]`) → gives `&[u32]`
/// 2. Convert each u32 to i64 (`.map(|&x| x as i64)`)
/// 3. Collect into owned Vec (`.collect::<Vec<_>>()`)
/// 4. Wrap in TensorData (`TensorData::from(...)`)
/// 5. Create tensor with device (`Tensor::from_data(..., &device)`)
///
/// This verbosity is the price of:
/// - Memory safety without GC
/// - Zero-cost abstractions
/// - Compile-time correctness
/// - Explicit resource management
struct GPTDatasetV1 {
    /// Input token sequences stored as tensors
    input_ids: Vec<Tensor<Backend, 1, Int>>,
    /// Target token sequences stored as tensors
    target_ids: Vec<Tensor<Backend, 1, Int>>,
}

impl GPTDatasetV1 {
    /// Helper function to create a tensor from a slice of u32 tokens
    /// 
    /// ## Why This Function Exists (The Type Conversion Problem)
    /// 
    /// We're caught between two incompatible library APIs:
    /// 1. **tiktoken returns `Vec<u32>`** - Their design choice for token IDs
    /// 2. **Burn expects `i64` for Int tensors** - Their design choice for integers
    /// 
    /// Additionally, Rust's ownership rules create friction:
    /// - Slicing gives us `&[u32]` (borrowed data)
    /// - TensorData::from needs owned data or specific types
    /// - No automatic u32 → i64 conversion in Rust (unlike Python)
    /// 
    /// ## What This Function Does
    /// 
    /// Takes a slice like `&token_ids[5..10]` and:
    /// 1. Iterates over each u32 token
    /// 2. Converts each to i64 (required by Burn)
    /// 3. Collects into a Vec<i64> (owned data)
    /// 4. Creates TensorData from the Vec
    /// 5. Wraps in a Tensor with device info
    /// 
    /// ## Why We Can't Simplify Further
    /// 
    /// In Python/PyTorch this would just be:
    /// ```python
    /// torch.tensor(token_ids[i:i+max_length])
    /// ```
    /// 
    /// But Rust makes all conversions explicit. We can't avoid this because:
    /// - Burn doesn't have `impl From<&[u32]> for TensorData<Int>`
    /// - tiktoken can't return i64 (it's a C++ library wrapper)
    /// - Rust won't implicitly convert numeric types
    /// 
    /// This is the trade-off: explicit and safe vs implicit and convenient.
    #[inline]
    fn slice_to_tensor(tokens: &[u32], device: &<Backend as burn::tensor::backend::Backend>::Device) -> Tensor<Backend, 1, Int> {
        // This chain of operations is necessary because:
        // 1. tokens is &[u32] (borrowed slice from tiktoken)
        // 2. We need Vec<i64> for TensorData (Burn requirement)
        // 3. The .map() converts each u32 to i64
        // 4. .collect::<Vec<_>>() creates owned Vec<i64>
        // 5. .as_slice() gives &[i64] for TensorData::from
        Tensor::<Backend, 1, Int>::from_data(
            TensorData::from(
                tokens.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice()
            ),
            device
        )
    }

    /// Create a new GPT dataset from text
    ///
    /// # Arguments
    /// * `text` - The input text to tokenize
    /// * `max_length` - Maximum sequence length for each sample
    /// * `stride` - How many tokens to move forward between samples
    ///
    /// # Returns
    /// A new GPTDatasetV1 instance with input-target pairs as Burn tensors
    fn new(text: &str, max_length: usize, stride: usize) -> Self {
        // Initialize device for tensor creation
        let device = Default::default();
        
        // Use tiktoken (GPT-2 tokenizer) for encoding
        let tokenizer = r50k_base().unwrap();
        let token_ids = tokenizer.encode_with_special_tokens(text);
        
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        
        // Create overlapping sequences with the specified stride
        // Following PyTorch: for i in range(0, len(token_ids) - max_length, stride)
        let mut i = 0;
        while i + max_length < token_ids.len() {
            // In PyTorch, this is simply:
            //   self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            //   self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))
            //
            // But in Rust/Burn we need explicit type conversions because:
            // - token_ids is Vec<u32> from tiktoken (C++ library choice)
            // - Burn tensors need i64 for Int type (Burn's design choice)
            // - Rust has no implicit numeric conversions (safety feature)
            //
            // The slice_to_tensor helper handles all the necessary conversions:
            // &[u32] → Iterator → map to i64 → Vec<i64> → TensorData → Tensor
            input_ids.push(Self::slice_to_tensor(&token_ids[i..i + max_length], &device));
            target_ids.push(Self::slice_to_tensor(&token_ids[i + 1..i + max_length + 1], &device));
            
            i += stride;
        }
        
        GPTDatasetV1 {
            input_ids,
            target_ids,
        }
    }
    
    /// Get the number of samples in the dataset
    /// 
    /// Equivalent to PyTorch's __len__ method
    fn len(&self) -> usize {
        self.input_ids.len()
    }
    
    /// Get a sample at the specified index
    ///
    /// Equivalent to PyTorch's __getitem__ method.
    /// Returns a tuple of (input_tensor, target_tensor) just like PyTorch.
    ///
    /// # Arguments
    /// * `idx` - The index of the sample to retrieve
    ///
    /// # Returns
    /// A tuple of (input_tensor, target_tensor) or None if index is out of bounds
    fn get_tensors(&self, idx: usize) -> Option<(Tensor<Backend, 1, Int>, Tensor<Backend, 1, Int>)> {
        if idx >= self.len() {
            return None;
        }
        Some((self.input_ids[idx].clone(), self.target_ids[idx].clone()))
    }
}

/// Implement the Burn Dataset trait for GPTDatasetV1
impl Dataset<GPTDatasetItem> for GPTDatasetV1 {
    fn get(&self, index: usize) -> Option<GPTDatasetItem> {
        if let Some((input_tensor, target_tensor)) = self.get_tensors(index) {
            // Convert tensors back to Vec<i64> for the dataset item
            // In a real training scenario, a Batcher would handle the tensor conversion
            let input_data = input_tensor.to_data();
            let target_data = target_tensor.to_data();
            
            let input_ids = input_data.to_vec::<i64>().unwrap();
            let target_ids = target_data.to_vec::<i64>().unwrap();
            
            Some(GPTDatasetItem {
                input_ids,
                target_ids,
            })
        } else {
            None
        }
    }
    
    fn len(&self) -> usize {
        self.input_ids.len()
    }
}

/// Batch structure for GPT training
/// 
/// This holds a batch of input and target tensors for training.
/// Similar to what PyTorch's DataLoader returns.
#[derive(Clone, Debug)]
struct GPTBatch<B: burn::tensor::backend::Backend> {
    pub inputs: Tensor<B, 2, Int>,   // Shape: [batch_size, seq_len]
    pub targets: Tensor<B, 2, Int>,  // Shape: [batch_size, seq_len]
}

/// Batcher for converting GPTDatasetItems into tensor batches
/// 
/// This is the Burn equivalent of PyTorch's collate_fn.
/// It takes individual dataset items and combines them into batches.
#[derive(Clone)]
struct GPTBatcher<B: burn::tensor::backend::Backend> {
    device: B::Device,
}

impl<B: burn::tensor::backend::Backend> GPTBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

use burn::data::dataloader::batcher::Batcher;
// DataLoader is built through DataLoaderBuilder

impl<B: burn::tensor::backend::Backend> Batcher<B, GPTDatasetItem, GPTBatch<B>> for GPTBatcher<B> {
    fn batch(&self, items: Vec<GPTDatasetItem>, _device: &B::Device) -> GPTBatch<B> {
        let batch_size = items.len();
        
        // Get sequence length from first item (assumes all are same length)
        let seq_len = items[0].input_ids.len();
        
        // Flatten all inputs and targets into single vectors
        let mut all_inputs = Vec::with_capacity(batch_size * seq_len);
        let mut all_targets = Vec::with_capacity(batch_size * seq_len);
        
        for item in items {
            all_inputs.extend(item.input_ids);
            all_targets.extend(item.target_ids);
        }
        
        // Create 2D tensors with shape [batch_size, seq_len]
        let inputs = Tensor::<B, 1, Int>::from_data(
            TensorData::from(all_inputs.as_slice()),
            &self.device
        ).reshape([batch_size, seq_len]);
        
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::from(all_targets.as_slice()),
            &self.device
        ).reshape([batch_size, seq_len]);
        
        GPTBatch { inputs, targets }
    }
}

/// Create a DataLoader for GPT training
/// 
/// This is the Rust/Burn equivalent of the Python function:
/// ```python
/// def create_dataloader_v1(txt, batch_size=4, max_length=256,
///                          stride=128, shuffle=True, drop_last=True,
///                          num_workers=0):
///     tokenizer = tiktoken.get_encoding("gpt2")
///     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
///     dataloader = DataLoader(
///         dataset,
///         batch_size=batch_size,
///         shuffle=shuffle,
///         drop_last=drop_last,
///         num_workers=num_workers
///     )
///     return dataloader
/// ```
/// 
/// ## Key Differences from PyTorch:
/// 
/// 1. **Tokenizer**: We use "r50k_base" (GPT-2) from tiktoken-rs
/// 2. **Shuffle**: Requires a seed in Burn (for reproducibility)
/// 3. **num_workers**: Handled differently in Burn (set via builder)
/// 4. **drop_last**: Available in Burn's DataLoaderBuilder
/// 5. **Return type**: Returns the built DataLoader directly
/// 
/// ## Usage Example:
/// ```rust
/// let dataloader = create_dataloader_v1(
///     text,
///     4,      // batch_size
///     256,    // max_length
///     128,    // stride
///     Some(42), // shuffle with seed
///     true,   // drop_last
/// )?;
/// 
/// for batch in dataloader.iter() {
///     // Training loop
///     let outputs = model.forward(batch.inputs);
///     let loss = loss_fn(outputs, batch.targets);
///     // ... backprop and optimization
/// }
/// ```
fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle_seed: Option<u64>,  // None for no shuffle, Some(seed) for reproducible shuffle
    drop_last: bool,
) -> Arc<dyn burn::data::dataloader::DataLoader<NdArray, GPTBatch<NdArray>>> {
    use burn::data::dataloader::DataLoaderBuilder;
    
    // Create the dataset (using r50k_base which is GPT-2 tokenizer)
    let dataset = GPTDatasetV1::new(txt, max_length, stride);
    
    // Create the batcher
    let device = Default::default();
    let batcher = GPTBatcher::<NdArray>::new(device);
    
    // Build the dataloader with explicit Backend type
    let mut builder = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size);
    
    // Add shuffle if requested (requires seed for reproducibility)
    if let Some(seed) = shuffle_seed {
        builder = builder.shuffle(seed);
    }
    
    // Note: drop_last is not directly available in Burn's DataLoaderBuilder
    // This would need to be handled differently or via a custom implementation
    // For now, we'll note this limitation
    if drop_last {
        // TODO: Implement drop_last functionality
        // Burn doesn't have built-in drop_last, would need custom implementation
    }
    
    // Note: num_workers is set differently in Burn (via builder.num_workers())
    // Default is usually appropriate for CPU-based datasets
    
    builder.build(dataset)
}

/// Downloads a file from a URL with progress reporting.
///
/// # Arguments
///
/// * `url` - The URL to download from
/// * `output_path` - The output path (can be a file or directory)
///
/// # Returns
///
/// * `Result<PathBuf, Box<dyn std::error::Error>>` - Path to downloaded file or error
///
/// # Behavior
///
/// - If output_path is a directory, the filename is extracted from the URL
/// - If output_path is a file path, the file is saved with that exact name
/// - Progress reporting shows download speed and estimated time remaining
async fn download_file(url: &str, output_path: Option<&str>) -> Result<PathBuf, Box<dyn Error>> {
    // Make an HTTP GET request to the URL
    let response = reqwest::get(url).await?;

    // Check if the HTTP response indicates success (status code 200-299)
    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    // Get the content length if available
    let total_size = response
        .content_length()
        .ok_or("Failed to get content length")?;

    // Determine the file path based on output_path
    let file_path = match output_path {
        Some(path) => {
            let path = Path::new(path);
            if path.extension().is_some() || path.to_string_lossy().contains('.') {
                // It's a file path
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).await?;
                }
                path.to_path_buf()
            } else {
                // It's a directory path
                fs::create_dir_all(path).await?;
                let file_name = url.split('/').last().unwrap_or("downloaded_file");
                path.join(file_name)
            }
        }
        None => {
            // Default to "data" directory
            fs::create_dir_all("data").await?;
            let file_name = url.split('/').last().unwrap_or("downloaded_file");
            Path::new("data").join(file_name)
        }
    };

    // Create a new file at the specified path
    let mut file = fs::File::create(&file_path).await?;

    // Create a progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")?
            .progress_chars("#>-")
    );
    pb.set_message(format!(
        "Downloading {}",
        file_path.file_name().unwrap_or_default().to_string_lossy()
    ));

    // Download the file in chunks with progress reporting
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk).await?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Downloaded to {}", file_path.display()));

    Ok(file_path)
}

/// Analyzes a text file by reading it and displaying statistics.
///
/// # Arguments
///
/// * `file_path` - Path to the text file to analyze
/// * `preview_length` - Number of characters to preview from the beginning
///
/// # Returns
///
/// * `Result<(), Box<dyn std::error::Error>>` - Success or error
///
/// # Behavior
///
/// This function replicates the book's Python example:
/// ```python
/// with open("the-verdict.txt", "r", encoding="utf-8") as f:
///     raw_text = f.read()
/// print("Total number of character:", len(raw_text))
/// print(raw_text[:99])
/// ```
async fn analyze_text(file_path: &str, preview_length: usize) -> Result<(), Box<dyn Error>> {
    // Read the entire file content with UTF-8 encoding
    let raw_text = fs::read_to_string(file_path).await?;

    // Display file information
    println!("File: {}", file_path);
    println!("Total number of characters: {}", raw_text.len());

    // Calculate line and word counts for additional insights
    let line_count = raw_text.lines().count();
    let word_count = raw_text.split_whitespace().count();

    println!("Total number of lines: {}", line_count);
    println!("Total number of words: {}", word_count);
    println!();

    // Display preview of the text
    println!("First {} characters:", preview_length);
    println!("---");

    // Use chars() to handle UTF-8 properly and take the requested number
    let preview: String = raw_text.chars().take(preview_length).collect();
    println!("{}", preview);

    if raw_text.len() > preview_length {
        println!("...");
    }

    Ok(())
}

/// Splits text on whitespace only.
///
/// Returns an `Arc<[Token]>` for efficient sharing of immutable token data.
/// Since tokens are created once and then only read, using Arc avoids
/// expensive cloning when sharing token lists between operations.
///
/// This replicates the book's Python example:
/// ```python
/// import re
/// text = "Hello, world. This, is a test."
/// result = re.split(r'(\s)', text)
/// print(result)
/// ```
///
/// # Arguments
///
/// * `text` - The text to split
///
/// # Returns
///
/// * `Arc<[Token]>` - Arc slice of tokens including whitespace
fn whitespace_split(text: &str) -> Arc<[Token]> {
    // Create regex that matches whitespace
    let re = Regex::new(r"\s").unwrap();

    // Split while keeping delimiters - this mimics Python's re.split with capturing group
    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(text) {
        // Add the text before the match (not a delimiter)
        if mat.start() > last_end {
            let content = text[last_end..mat.start()].to_string();
            result.push(Token::new(content, false));
        }
        // Add the whitespace match itself (is a delimiter)
        let content = mat.as_str().to_string();
        result.push(Token::new(content, true));
        last_end = mat.end();
    }

    // Add any remaining text after the last match (not a delimiter)
    if last_end < text.len() {
        let content = text[last_end..].to_string();
        result.push(Token::new(content, false));
    }

    result.into()
}

/// Splits text on whitespace, commas, and periods.
///
/// This replicates the book's Python example:
/// ```python
/// import re
/// text = "Hello, world. This, is a test."
/// result = re.split(r'([,.]|\s)', text)
/// print(result)
/// ```
///
/// # Arguments
///
/// * `text` - The text to split
///
/// # Returns
///
/// * `Arc<[Token]>` - Arc slice of tokens including delimiters
fn punctuation_split(text: &str) -> Arc<[Token]> {
    // Create regex that matches whitespace, commas, or periods
    let re = Regex::new(r"([,.]|\s)").unwrap();

    // Split while keeping delimiters - this mimics Python's re.split with capturing group
    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(text) {
        // Add the text before the match (not a delimiter)
        if mat.start() > last_end {
            let content = text[last_end..mat.start()].to_string();
            result.push(Token::new(content, false));
        }
        // Add the delimiter match itself (whitespace, comma, or period)
        let content = mat.as_str().to_string();
        result.push(Token::new(content, true));
        last_end = mat.end();
    }

    // Add any remaining text after the last match (not a delimiter)
    if last_end < text.len() {
        let content = text[last_end..].to_string();
        result.push(Token::new(content, false));
    }

    result.into()
}

/// Splits text on comprehensive punctuation and whitespace.
///
/// This replicates the book's Python example:
/// ```python
/// import re
/// raw_text = "..."
/// preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
/// preprocessed = [item.strip() for item in preprocessed if item.strip()]
/// print(len(preprocessed))
/// ```
///
/// # Arguments
///
/// * `text` - The text to split
///
/// # Returns
///
/// * `Arc<[Token]>` - Arc slice of tokens (empty tokens are filtered out)
fn all_split(text: &str) -> Arc<[Token]> {
    // Create regex that matches comprehensive punctuation, double dash, or whitespace
    let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

    // Split while keeping delimiters
    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(text) {
        // Add the text before the match (not a delimiter)
        if mat.start() > last_end {
            let content = text[last_end..mat.start()].trim().to_string();
            if !content.is_empty() {
                result.push(Token::new(content, false));
            }
        }
        // Add the delimiter match itself (but strip and check if not empty)
        let content = mat.as_str().trim().to_string();
        if !content.is_empty() {
            result.push(Token::new(content, true));
        }
        last_end = mat.end();
    }

    // Add any remaining text after the last match (not a delimiter)
    if last_end < text.len() {
        let content = text[last_end..].trim().to_string();
        if !content.is_empty() {
            result.push(Token::new(content, false));
        }
    }

    result.into()
}

/// Splits text using the specified method and returns tokens.
///
/// # Arguments
///
/// * `text` - The text to split
/// * `method` - The splitting method to use
///
/// # Returns
///
/// * `Arc<[Token]>` - Arc slice of tokens from the split
fn split_text(text: &str, method: SplitMethod) -> Arc<[Token]> {
    match method {
        SplitMethod::Whitespace => whitespace_split(text),
        SplitMethod::Punctuation => punctuation_split(text),
        SplitMethod::All => all_split(text),
    }
}

/// Handles the split subcommand, demonstrating various text splitting methods.
///
/// # Arguments
///
/// * `file_path` - Optional path to file, uses example text if None
/// * `method` - The splitting method to use
/// * `max_display` - Maximum number of tokens to display
async fn handle_split(
    file_path: Option<String>,
    method: SplitMethod,
    max_display: usize,
) -> Result<(), Box<dyn Error>> {
    // Get the text to split
    let text = match file_path {
        Some(path) => {
            println!("Reading text from: {}", path);
            fs::read_to_string(path).await?
        }
        None => {
            let example = "Hello, world. This, is a test.";
            println!("Using example text: \"{}\"", example);
            example.to_string()
        }
    };

    println!();
    println!("Splitting method: {}", method);
    println!("Python equivalent:");

    match method {
        SplitMethod::Whitespace => {
            println!("  import re");
            println!("  result = re.split(r'(\\s)', text)");
            println!();
        }
        SplitMethod::Punctuation => {
            println!("  import re");
            println!("  result = re.split(r'([,.]|\\s)', text)");
            println!();
        }
        SplitMethod::All => {
            println!("  import re");
            println!("  preprocessed = re.split(r'([,.:;?_!\"()\\']|--|\\s)', raw_text)");
            println!("  preprocessed = [item.strip() for item in preprocessed if item.strip()]");
            println!();
        }
    }

    let tokens = split_text(&text, method);

    // Display results
    println!("Total tokens: {}", tokens.len().to_string().bold());
    println!();

    // Create color scheme
    let colors = ColorScheme::default();

    // Display tokens inline with highlighting
    println!("Tokenized text ({} {} tokens):",
        colors.style_text("text"),
        colors.style_delimiter("delimiters")
    );
    println!();

    if tokens.len() <= max_display {
        // Show all tokens inline with visual separation
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                print!(" ");  // Space between tokens
            }
            print!("{}", token.display_inline(&colors));
        }
        println!();
    } else {
        // Show limited tokens inline with visual separation
        for (i, token) in tokens.iter().take(max_display).enumerate() {
            if i > 0 {
                print!(" ");  // Space between tokens
            }
            print!("{}", token.display_inline(&colors));
        }
        print!("{}", format!(" ... ({} more tokens)", tokens.len() - max_display).bright_black());
        println!();
    }

    Ok(())
}

/// Handles the sliding window subcommand, demonstrating training data preparation.
///
/// This replicates the book's Python example for creating input-output pairs:
/// ```python
/// enc_sample = enc_text[50:]
/// context_size = 4
/// x = enc_sample[:context_size]
/// y = enc_sample[1:context_size+1]
/// 
/// for i in range(1, context_size+1):
///     context = enc_sample[:i]
///     desired = enc_sample[i]
///     print(context, "---->", desired)
/// ```
async fn handle_sliding_window(
    file_path: Option<String>,
    context_size: usize,
    start_pos: usize,
    max_windows: usize,
    show_decoded: bool,
) -> Result<(), Box<dyn Error>> {
    // Load the text file
    let path = file_path.unwrap_or_else(|| VERDICT_FILENAME.to_string());
    println!("Loading text from: {}", path);
    
    if !Path::new(&path).exists() {
        println!("File not found. Run 'cargo run -p ch02 -- demo' first to download the sample text.");
        return Err("File not found".into());
    }
    
    let raw_text = fs::read_to_string(&path).await?;
    println!("Loaded {} characters", raw_text.len());
    println!();
    
    // Use tiktoken for encoding
    println!("Encoding text with tiktoken (GPT-2 tokenizer)...");
    let encoding = r50k_base().map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    let allowed_special = encoding.special_tokens();
    let (enc_text, _) = encoding.encode(&raw_text, &allowed_special);
    println!("Total tokens: {}", enc_text.len());
    println!();
    
    // Check if we have enough tokens
    if enc_text.len() <= start_pos {
        return Err(format!("Not enough tokens. Text has {} tokens but start position is {}", 
                          enc_text.len(), start_pos).into());
    }
    
    // Get the sample starting from the specified position
    let enc_sample = &enc_text[start_pos..];
    println!("Working with tokens starting at position {}", start_pos);
    println!("Sample has {} tokens available", enc_sample.len());
    println!();
    
    // Ensure we have enough tokens for the context window
    if enc_sample.len() <= context_size {
        return Err(format!("Not enough tokens in sample. Need at least {} but have {}", 
                          context_size + 1, enc_sample.len()).into());
    }
    
    // Show the initial context window
    println!("=== Initial Context Window ===");
    println!("Context size: {}", context_size);
    println!();
    
    let x = &enc_sample[..context_size];
    let y = &enc_sample[1..context_size + 1];
    
    println!("Input (x):  {:?}", x);
    println!("Target (y): {:?}", y);
    println!();
    
    if show_decoded {
        let x_decoded = encoding.decode(x.to_vec()).unwrap_or_else(|_| "<decode error>".to_string());
        let y_decoded = encoding.decode(y.to_vec()).unwrap_or_else(|_| "<decode error>".to_string());
        println!("Input decoded:  \"{}\"", x_decoded);
        println!("Target decoded: \"{}\"", y_decoded);
        println!();
    }
    
    // Show how the context grows from 1 to context_size
    println!("=== Growing Context Windows ===");
    println!("Showing how context grows and predicts the next token:");
    println!();
    
    for i in 1..=context_size.min(max_windows) {
        let context = &enc_sample[..i];
        let target = enc_sample[i];
        
        print!("{:?} ", context);
        print!("{}", "----->".yellow());
        println!(" {}", target);
        
        if show_decoded {
            let context_decoded = encoding.decode(context.to_vec())
                .unwrap_or_else(|_| "<decode error>".to_string());
            let target_decoded = encoding.decode(vec![target])
                .unwrap_or_else(|_| "<decode error>".to_string());
            
            println!("  \"{}\" {} \"{}\"", 
                    context_decoded.blue(), 
                    "----->".yellow(), 
                    target_decoded.green());
            println!();
        }
    }
    
    // Show sliding windows for batch training
    println!();
    println!("=== Sliding Windows for Batch Training ===");
    println!("Each window creates an input-target pair for training:");
    println!();
    
    let num_windows = ((enc_sample.len() - context_size - 1).min(max_windows)).min(10);
    
    for i in 0..num_windows {
        let input_start = i;
        let input_end = i + context_size;
        let target_start = i + 1;
        let target_end = i + context_size + 1;
        
        if target_end > enc_sample.len() {
            break;
        }
        
        let input = &enc_sample[input_start..input_end];
        let target = &enc_sample[target_start..target_end];
        
        println!("Window {}:", i + 1);
        println!("  Input:  {:?}", input);
        println!("  Target: {:?}", target);
        
        if show_decoded {
            let input_decoded = encoding.decode(input.to_vec())
                .unwrap_or_else(|_| "<decode error>".to_string());
            let target_decoded = encoding.decode(target.to_vec())
                .unwrap_or_else(|_| "<decode error>".to_string());
            
            println!("  Input (decoded):  \"{}\"", input_decoded.blue());
            println!("  Target (decoded): \"{}\"", target_decoded.green());
        }
        println!();
    }
    
    if num_windows < max_windows {
        println!("(Showing {} windows out of {} possible)", 
                num_windows, enc_sample.len() - context_size);
    }
    
    println!();
    println!("Python equivalent:");
    println!("```python");
    println!("with open(\"{}\", \"r\", encoding=\"utf-8\") as f:", path);
    println!("    raw_text = f.read()");
    println!();
    println!("# Encode with tiktoken");
    println!("import tiktoken");
    println!("encoding = tiktoken.get_encoding(\"r50k_base\")");
    println!("enc_text = encoding.encode(raw_text)");
    println!("print(len(enc_text))");
    println!();
    println!("# Create sliding windows");
    println!("enc_sample = enc_text[{}:]", start_pos);
    println!("context_size = {}", context_size);
    println!("x = enc_sample[:context_size]");
    println!("y = enc_sample[1:context_size+1]");
    println!("print(f\"x: {{x}}\")");
    println!("print(f\"y: {{y}}\")");
    println!();
    println!("# Show growing context");
    println!("for i in range(1, context_size+1):");
    println!("    context = enc_sample[:i]");
    println!("    desired = enc_sample[i]");
    println!("    print(context, \"---->\", desired)");
    if show_decoded {
        println!();
        println!("# With decoded text");
        println!("for i in range(1, context_size+1):");
        println!("    context = enc_sample[:i]");
        println!("    desired = enc_sample[i]");
        println!("    print(encoding.decode(context), \"---->\", encoding.decode([desired]))");
    }
    println!("```");
    
    Ok(())
}

/// Handles the dataset subcommand, demonstrating GPTDatasetV1 for training data preparation.
///
/// # Arguments
///
/// * `file_path` - Optional file path to read text from
/// * `max_length` - Maximum sequence length for each sample
/// * `stride` - How many tokens to move forward between samples
/// * `num_samples` - Number of samples to display
/// * `show_decoded` - Whether to show decoded text
/// * `verbose` - Whether to show dataset statistics
async fn handle_dataset(
    file_path: Option<String>,
    max_length: usize,
    stride: usize,
    num_samples: usize,
    show_decoded: bool,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    // Determine the file path to use
    let path = file_path.unwrap_or_else(|| VERDICT_FILENAME.to_string());
    
    // Check if file exists, download if necessary
    if !Path::new(&path).exists() {
        println!("File not found. Downloading from: {}", VERDICT_URL);
        download_file(VERDICT_URL, Some(&path)).await?;
    }
    
    // Read the text
    println!("Loading text from: {}", path);
    let text = fs::read_to_string(&path).await?;
    println!("Text length: {} characters", text.len());
    
    // Create the dataset
    println!("\n{}", "Creating GPTDatasetV1...".bold());
    println!("Parameters:");
    println!("  max_length: {}", max_length);
    println!("  stride: {}", stride);
    
    let dataset = GPTDatasetV1::new(&text, max_length, stride);
    
    // Show dataset statistics
    println!("\n{}", "Dataset Statistics:".bold());
    println!("  Total samples: {}", dataset.len());
    println!("  Input shape per sample: [{}]", max_length);
    println!("  Target shape per sample: [{}]", max_length);
    
    if verbose {
        // Calculate total tokens
        let tokenizer = r50k_base().unwrap();
        let total_tokens = tokenizer.encode_with_special_tokens(&text).len();
        println!("  Total tokens in text: {}", total_tokens);
        println!("  Coverage: {:.1}%", (dataset.len() * stride) as f64 / total_tokens as f64 * 100.0);
        println!("  Overlap between samples: {} tokens", max_length - stride);
    }
    
    // Display sample data
    println!("\n{}", "Sample Data:".bold());
    let samples_to_show = if num_samples == 0 {
        dataset.len()
    } else {
        min(num_samples, dataset.len())
    };
    
    for i in 0..samples_to_show {
        if let Some((input_tensor, target_tensor)) = dataset.get_tensors(i) {
            println!("\n{} {}", "Sample".cyan(), format!("{}", i).yellow());
            
            // Convert tensors to vectors for display
            let input_data = input_tensor.to_data();
            let target_data = target_tensor.to_data();
            let input_ids_vec = input_data.to_vec::<i64>().unwrap();
            let target_ids_vec = target_data.to_vec::<i64>().unwrap();
            
            // Show raw token IDs
            println!("  {}: {:?}", "Input IDs".green(), input_ids_vec);
            println!("  {}: {:?}", "Target IDs".red(), target_ids_vec);
            
            // Show decoded text if requested
            if show_decoded {
                let tokenizer = r50k_base().unwrap();
                
                // Convert back to u32 for tiktoken
                let input_ids_u32: Vec<u32> = input_ids_vec.iter().map(|&x| x as u32).collect();
                let target_ids_u32: Vec<u32> = target_ids_vec.iter().map(|&x| x as u32).collect();
                
                let input_text = tokenizer.decode(input_ids_u32).unwrap();
                let target_text = tokenizer.decode(target_ids_u32).unwrap();
                
                println!("  {} {:?}", "Input Text:".green(), input_text);
                println!("  {} {:?}", "Target Text:".red(), target_text);
            }
        }
    }
    
    // Show how to use with DataLoader (conceptual, as Burn's DataLoader is different from PyTorch)
    println!("\n{}", "Usage with Burn DataLoader:".bold());
    println!("{}", "```rust".dimmed());
    println!("// In a training context, you would use:");
    println!("// let dataloader = DataLoaderBuilder::new(batcher)");
    println!("//     .batch_size(8)");
    println!("//     .shuffle(42)");
    println!("//     .build(dataset);");
    println!("//");
    println!("// for batch in dataloader {{");
    println!("//     let (inputs, targets) = batch;");
    println!("//     // ... training step");
    println!("// }}");
    println!("{}", "```".dimmed());
    
    // Show Python equivalent
    println!("\n{}", "Python Equivalent:".bold());
    println!("{}", "```python".dimmed());
    println!("class GPTDatasetV1(Dataset):");
    println!("    def __init__(self, txt, tokenizer, max_length, stride):");
    println!("        self.input_ids = []");
    println!("        self.target_ids = []");
    println!("        token_ids = tokenizer.encode(txt)");
    println!("        ");
    println!("        for i in range(0, len(token_ids) - max_length, stride):");
    println!("            input_chunk = token_ids[i:i + max_length]");
    println!("            target_chunk = token_ids[i + 1: i + max_length + 1]");
    println!("            self.input_ids.append(torch.tensor(input_chunk))");
    println!("            self.target_ids.append(torch.tensor(target_chunk))");
    println!("{}", "```".dimmed());
    
    Ok(())
}

/// Handles the dataloader subcommand, demonstrating batch processing with DataLoader.
///
/// # Arguments
///
/// * `file_path` - Optional file path to read text from
/// * `batch_size` - Number of samples per batch
/// * `max_length` - Maximum sequence length for each sample
/// * `stride` - How many tokens to move forward between samples
/// * `shuffle_seed` - Optional seed for shuffling
/// * `num_batches` - Number of batches to display
/// * `show_decoded` - Whether to show decoded text
async fn handle_dataloader(
    file_path: Option<String>,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle_seed: Option<u64>,
    num_batches: usize,
    show_decoded: bool,
) -> Result<(), Box<dyn Error>> {
    // Determine the file path to use
    let path = file_path.unwrap_or_else(|| VERDICT_FILENAME.to_string());
    
    // Check if file exists, download if necessary
    if !Path::new(&path).exists() {
        println!("File not found. Downloading from: {}", VERDICT_URL);
        download_file(VERDICT_URL, Some(&path)).await?;
    }
    
    // Read the text
    println!("Loading text from: {}", path);
    let text = fs::read_to_string(&path).await?;
    println!("Text length: {} characters", text.len());
    
    // Create the dataloader
    println!("\n{}", "Creating DataLoader...".bold());
    println!("Parameters:");
    println!("  batch_size: {}", batch_size);
    println!("  max_length: {}", max_length);
    println!("  stride: {}", stride);
    println!("  shuffle: {}", if shuffle_seed.is_some() { "Yes" } else { "No" });
    
    let dataloader = create_dataloader_v1(
        &text,
        batch_size,
        max_length,
        stride,
        shuffle_seed,
        false,  // drop_last not fully implemented yet
    );
    
    // Get dataset info
    let dataset = GPTDatasetV1::new(&text, max_length, stride);
    let total_samples = dataset.len();
    let total_batches = (total_samples + batch_size - 1) / batch_size;  // Ceiling division
    
    println!("\n{}", "DataLoader Statistics:".bold());
    println!("  Total samples: {}", total_samples);
    println!("  Total batches: {}", total_batches);
    println!("  Samples per batch: {}", batch_size);
    println!("  Token shape per sample: [{}]", max_length);
    println!("  Batch tensor shape: [{}, {}]", batch_size, max_length);
    
    // Display sample batches
    println!("\n{}", "Sample Batches:".bold());
    let batches_to_show = min(num_batches, total_batches);
    
    // Note: Burn's DataLoader has an iter() method to get an iterator
    // We iterate over the batches
    let mut batch_count = 0;
    for batch_data in dataloader.iter() {
        if batch_count >= batches_to_show {
            break;
        }
        
        println!("\n{} {}", "Batch".cyan(), format!("{}", batch_count).yellow());
        
        // Show batch shapes and sample data
        let batch_inputs = &batch_data.inputs;
        let batch_targets = &batch_data.targets;
        
        // Get dimensions
        let dims = batch_inputs.dims();
        println!("  Batch shape: {:?}", dims);
        
        // Show first few sequences in the batch
        // Show full batch tensor (like PyTorch would)
        println!("  {}:", "Inputs (full batch tensor)".green());
        let input_data = batch_inputs.to_data();
        let input_values = input_data.to_vec::<i64>().unwrap();
        
        // Display as 2D array with proper formatting
        let actual_batch_size = dims[0];
        let seq_len = dims[1];
        for i in 0..actual_batch_size {
            let start = i * seq_len;
            let end = start + seq_len;
            let row = &input_values[start..end];
            println!("    [{}]: {:?}", i, row);
        }
        
        println!("\n  {}:", "Targets (full batch tensor)".red());
        let target_data = batch_targets.to_data();
        let target_values = target_data.to_vec::<i64>().unwrap();
        for i in 0..actual_batch_size {
            let start = i * seq_len;
            let end = start + seq_len;
            let row = &target_values[start..end];
            println!("    [{}]: {:?}", i, row);
        }
        
        if show_decoded {
            println!("\n  {}:", "Decoded text (first sequence)".cyan());
            let tokenizer = r50k_base().unwrap();
            
            // Show decoded text for first sequence only (for brevity)
            let first_input = &input_values[0..seq_len];
            let first_target = &target_values[0..seq_len];
            
            // Convert back to u32 for tiktoken
            let input_u32: Vec<u32> = first_input.iter().map(|&x| x as u32).collect();
            let target_u32: Vec<u32> = first_target.iter().map(|&x| x as u32).collect();
            
            let input_text = tokenizer.decode(input_u32).unwrap();
            let target_text = tokenizer.decode(target_u32).unwrap();
            
            println!("    {} {:?}", "Input:".green(), input_text);
            println!("    {} {:?}", "Target:".red(), target_text);
        }
        
        batch_count += 1;
    }
    
    if batches_to_show < total_batches {
        println!("\n(Showing {} batches out of {} total)", batches_to_show, total_batches);
    }
    
    // Show Python equivalent
    println!("\n{}", "Python Equivalent:".bold());
    println!("{}", "```python".dimmed());
    println!("from torch.utils.data import DataLoader");
    println!("import tiktoken");
    println!();
    println!("def create_dataloader_v1(txt, batch_size={}, max_length={},", batch_size, max_length);
    println!("                         stride={}, shuffle={}, drop_last=True,", stride, shuffle_seed.is_some());
    println!("                         num_workers=0):");
    println!("    tokenizer = tiktoken.get_encoding(\"gpt2\")");
    println!("    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)");
    println!("    dataloader = DataLoader(");
    println!("        dataset,");
    println!("        batch_size=batch_size,");
    println!("        shuffle=shuffle,");
    println!("        drop_last=drop_last,");
    println!("        num_workers=num_workers");
    println!("    )");
    println!("    return dataloader");
    println!();
    println!("dataloader = create_dataloader_v1(text)");
    println!("for batch_idx, (inputs, targets) in enumerate(dataloader):");
    println!("    print(f\"Batch {{batch_idx}}: inputs shape {{inputs.shape}}, targets shape {{targets.shape}}\")");
    println!("    if batch_idx >= {}:", num_batches - 1);
    println!("        break");
    println!("{}", "```".dimmed());
    
    Ok(())
}

/// Handles the tokenize subcommand, demonstrating various tokenizer implementations.
///
/// # Arguments
///
/// * `text` - Optional text to tokenize
/// * `file_path` - Optional file path (overrides text)
/// * `tokenizer` - Which tokenizer to use
/// * `detailed` - Whether to show detailed analysis
async fn handle_tokenize(
    text: Option<String>,
    file_path: Option<String>,
    tokenizer: TokenizerChoice,
    detailed: bool,
) -> Result<(), Box<dyn Error>> {
    // Get the text to tokenize
    let input_text = if let Some(path) = file_path {
        println!("Reading text from: {}", path);
        fs::read_to_string(path).await?
    } else if let Some(t) = text {
        t
    } else {
        // Default example text that works with all tokenizers
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.".to_string()
    };

    println!();
    println!("=== {} Tokenization Demo ===", tokenizer);
    println!();
    println!("Input text ({} characters):", input_text.len());
    println!("{}", input_text);
    println!();

    match tokenizer {
        TokenizerChoice::Tiktoken => {
            // Demonstrate tiktoken (GPT-2 tokenizer)
            println!("Loading GPT-2 tokenizer (r50k_base)...");
            let encoding = r50k_base().map_err(|e| format!("Failed to load tokenizer: {}", e))?;

            // Note: tiktoken-rs doesn't expose vocabulary size directly
            println!();

            // Get special tokens
            let allowed_special = encoding.special_tokens();

            // Encode text
            println!("Encoding text (allowing special tokens)...");
            let (tokens, _) = encoding.encode(&input_text, &allowed_special);

            println!("Token IDs: {:?}", tokens);
            println!("Number of tokens: {}", tokens.len());

            if detailed {
                println!();
                println!("Token details:");
                for (i, &token_id) in tokens.iter().enumerate() {
                    // Decode individual token
                    if let Ok(token_str) = encoding.decode(vec![token_id]) {
                        println!("  [{}] {} -> \"{}\"", i, token_id, token_str);
                    } else {
                        println!("  [{}] {} -> <special token>", i, token_id);
                    }
                }
            }

            // Decode back to text
            println!();
            println!("Decoding tokens back to text...");
            let decoded = encoding
                .decode(tokens.clone())
                .map_err(|e| format!("Decoding failed: {}", e))?;

            println!("Decoded text: {}", decoded);

            // Verify round-trip
            if decoded == input_text {
                println!("{}", "✓ Round-trip encoding/decoding successful!".green());
            } else {
                println!("{}", "⚠ Decoded text differs from original (expected for tiktoken with special handling)".yellow());
            }

            // Show Python equivalent
            println!();
            println!("Python equivalent:");
            println!("```python");
            println!("import tiktoken");
            println!();
            println!("# Load the GPT-2 tokenizer");
            println!("encoding = tiktoken.get_encoding(\"r50k_base\")");
            println!();
            println!("# Input text");
            println!("text = \"{}\"", input_text);
            println!();
            println!("# Encode with special tokens allowed");
            println!("tokens = encoding.encode(text, allowed_special=\"all\")");
            println!("print(f\"Tokens: {{tokens}}\")");
            println!("print(f\"Number of tokens: {{len(tokens)}}\")");
            println!();
            println!("# Decode back to text");
            println!("decoded = encoding.decode(tokens)");
            println!("print(f\"Decoded: {{decoded}}\")");
            println!("```");
        }
        TokenizerChoice::V1 | TokenizerChoice::V2 => {
            // For V1 and V2, we need to build a vocabulary first
            println!("Note: SimpleTokenizer{} requires a vocabulary built from training data.",
                if matches!(tokenizer, TokenizerChoice::V1) { "V1" } else { "V2" });
            println!("Loading sample text to build vocabulary...");

            // Use the verdict text to build vocabulary
            let sample_text = if Path::new(VERDICT_FILENAME).exists() {
                fs::read_to_string(VERDICT_FILENAME).await?
            } else {
                println!("Sample text file not found. Run 'cargo run -p ch02 -- demo' first to download it.");
                return Err("Sample text file not found".into());
            };

            // Build vocabulary from sample text
            let tokens = all_split(&sample_text);
            let unique_words: HashSet<&str> = tokens
                .iter()
                .map(|token| token.content.as_str())
                .collect();

            let mut sorted_words: Vec<&str> = unique_words.into_iter().collect();
            sorted_words.sort();

            // Add special tokens for V2
            if matches!(tokenizer, TokenizerChoice::V2) {
                sorted_words.push("<|endoftext|>");
                sorted_words.push("<|unk|>");
            }

            let vocab: HashMap<String, usize> = sorted_words
                .iter()
                .enumerate()
                .map(|(idx, &word)| (word.to_string(), idx))
                .collect();

            println!("Built vocabulary with {} entries", vocab.len());
            println!();

            match tokenizer {
                TokenizerChoice::V1 => {
                    let tokenizer = SimpleTokenizerV1::new(vocab);

                    match tokenizer.encode(&input_text) {
                        Ok(ids) => {
                            println!("Token IDs: {:?}", ids);
                            println!("Number of tokens: {}", ids.len());

                            match tokenizer.decode(&ids) {
                                Ok(decoded) => {
                                    println!();
                                    println!("Decoded text: {}", decoded);

                                    if decoded == input_text {
                                        println!("{}", "✓ Round-trip encoding/decoding successful!".green());
                                    } else {
                                        println!("{}", "⚠ Decoded text differs from original".yellow());
                                    }
                                }
                                Err(e) => println!("{}", format!("Error decoding: {}", e).red()),
                            }
                        }
                        Err(e) => println!("{}", format!("Error encoding: {}", e).red()),
                    }
                }
                TokenizerChoice::V2 => {
                    let tokenizer = SimpleTokenizerV2::new(vocab);

                    let ids = tokenizer.encode(&input_text);
                    println!("Token IDs: {:?}", ids);
                    println!("Number of tokens: {}", ids.len());

                    match tokenizer.decode(&ids) {
                        Ok(decoded) => {
                            println!();
                            println!("Decoded text: {}", decoded);

                            if decoded == input_text {
                                println!("{}", "✓ Round-trip encoding/decoding successful!".green());
                            } else {
                                println!("{}", "⚠ Decoded text may differ due to unknown token handling".yellow());
                            }
                        }
                        Err(e) => println!("{}", format!("Error decoding: {}", e).red()),
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(())
}

/// The main entry point of our application.
///
/// This function:
/// 1. Parses command-line arguments using clap
/// 2. Executes the appropriate subcommand
/// 3. Handles any errors that occur during execution
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments
    let args = Args::parse();

    match args.command {
        Commands::Download { url, output } => {
            println!("Starting download...");
            println!("URL: {}", url);
            println!("Output: {}", output);
            println!();

            let file_path = download_file(&url, Some(&output)).await?;

            println!("\nDownload complete!");
            println!("File saved to: {}", file_path.display());
        }
        Commands::Demo => {
            // Run the complete book example from Chapter 2
            println!("=== Chapter 2 Demo: Working with Text Data ===");
            println!();
            println!("This demo replicates the book's Python examples:");
            println!();

            // Step 1: Download the file
            println!("Step 1: Download the text file");
            println!("Python equivalent:");
            println!("  import urllib.request");
            println!("  url = (\"https://raw.githubusercontent.com/rasbt/\"");
            println!("         \"LLMs-from-scratch/main/ch02/01_main-chapter-code/\"");
            println!("         \"the-verdict.txt\")");
            println!("  file_path = \"the-verdict.txt\"");
            println!("  urllib.request.urlretrieve(url, file_path)");
            println!();

            let _file_path = download_file(VERDICT_URL, Some(VERDICT_FILENAME)).await?;

            println!("\n✓ Download complete!");
            println!();

            // Step 2: Analyze the file
            println!("Step 2: Read and analyze the text");
            println!("Python equivalent:");
            println!("  with open(\"the-verdict.txt\", \"r\", encoding=\"utf-8\") as f:");
            println!("      raw_text = f.read()");
            println!("  print(\"Total number of character:\", len(raw_text))");
            println!("  print(raw_text[:99])");
            println!();

            analyze_text(VERDICT_FILENAME, 99).await?;

            println!("\n✓ Analysis complete!");
            println!();

            // Step 3: Demonstrate text splitting
            println!("Step 3: Split text into tokens");
            println!("Python equivalent:");
            println!("  import re");
            println!("  text = \"Hello, world. This, is a test.\"");
            println!("  result = re.split(r'(\\s)', text)");
            println!("  print(result)");
            println!();

            // First show with example text
            println!("Example 1: Whitespace splitting");
            handle_split(None, SplitMethod::Whitespace, 30).await?;

            println!();
            println!("Example 2: Punctuation splitting");
            handle_split(None, SplitMethod::Punctuation, 30).await?;

            println!();
            println!("Example 3: All punctuation splitting (with trimming)");
            handle_split(None, SplitMethod::All, 30).await?;

            println!();
            println!("Step 4: Apply comprehensive splitting to the downloaded text");
            println!("Python equivalent:");
            println!("  preprocessed = re.split(r'([,.:;?_!\"()\\']|--|\\s)', raw_text)");
            println!("  preprocessed = [item.strip() for item in preprocessed if item.strip()]");
            println!("  print(len(preprocessed))");
            println!();

            // Read the entire file and apply comprehensive splitting
            let raw_text = fs::read_to_string(VERDICT_FILENAME).await?;
            let preprocessed = split_text(&raw_text, SplitMethod::All);

            println!("Total tokens after splitting: {}", preprocessed.len().to_string().bold());
            println!();

            // Create color scheme and display first 30 tokens
            let colors = ColorScheme::default();
            println!("First 30 tokens ({} {} tokens):",
                colors.style_text("text"),
                colors.style_delimiter("delimiters")
            );
            println!();

            for (i, token) in preprocessed.iter().take(30).enumerate() {
                if i > 0 {
                    print!(" ");  // Space between tokens
                }
                print!("{}", token.display_inline(&colors));
            }
            if preprocessed.len() > 30 {
                print!("{}", format!(" ... ({} more tokens)", preprocessed.len() - 30).bright_black());
            }
            println!();

            // Step 5: Create vocabulary from unique tokens
            println!();
            println!("Step 5: Create vocabulary from unique tokens");
            println!("Python equivalent:");
            println!("  all_words = sorted(set(preprocessed))");
            println!("  vocab_size = len(all_words)");
            println!("  print(vocab_size)");
            println!();

            // Create a set of unique words using HashSet
            let unique_words: HashSet<&str> = preprocessed
                .iter()
                .map(|token| token.content.as_str())
                .collect();

            let vocab_size = unique_words.len();

            // Convert to sorted vector for display
            let mut sorted_words: Vec<&str> = unique_words.into_iter().collect();
            sorted_words.sort();
            println!("Vocabulary size: {}", vocab_size.to_string().bold());
            println!();

            // Show first 50 words from vocabulary
            println!("First 50 words from vocabulary (alphabetically sorted):");
            println!();

            for (i, word) in sorted_words.iter().take(50).enumerate() {
                if i > 0 && i % 10 == 0 {
                    println!();  // New line every 10 words
                }
                print!("{:<12}", word);  // Left-aligned with 12 character width
            }

            if sorted_words.len() > 50 {
                println!();
                println!("{}", format!("... ({} more words)", sorted_words.len() - 50).bright_black());
            }
            println!();

            // Step 6: Create tokenizer and demonstrate encode/decode
            println!();
            println!("Step 6: Create SimpleTokenizerV1 and demonstrate encode/decode");
            println!("Python equivalent:");
            println!("  vocab = {{token: idx for idx, token in enumerate(sorted_words)}}");
            println!("  tokenizer = SimpleTokenizerV1(vocab)");
            println!("  text = \"\"\"\"It's the last he painted, you know,\" ");
            println!("         Mrs. Gisburn said with pardonable pride.\"\"\"");
            println!("  ids = tokenizer.encode(text)");
            println!("  decoded_text = tokenizer.decode(ids)");
            println!();

            // Create vocabulary mapping from sorted unique words
            let vocab: HashMap<String, usize> = sorted_words
                .iter()
                .enumerate()
                .map(|(idx, &word)| (word.to_string(), idx))
                .collect();

            let vocab_size = vocab.len();
            println!("Created vocabulary with {} entries", vocab_size);

            // Create tokenizer
            let tokenizer = SimpleTokenizerV1::new(vocab);

            // Example text from the book
            let example_text = "\"It's the last he painted, you know,\" \n       Mrs. Gisburn said with pardonable pride.";
            println!();
            println!("Example text:");
            println!("{}", example_text);

            // Encode the text
            match tokenizer.encode(example_text) {
                Ok(ids) => {
                    println!();
                    println!("Encoded IDs: {:?}", ids);
                    println!("Number of tokens: {}", ids.len());

                    // Decode back to text
                    match tokenizer.decode(&ids) {
                        Ok(decoded) => {
                            println!();
                            println!("Decoded text:");
                            println!("{}", decoded);

                            // Verify round-trip
                            if decoded == example_text {
                                println!("{}", "✓ Round-trip encoding/decoding successful!".green());
                            } else {
                                println!("{}", "⚠ Decoded text differs from original".yellow());
                                println!("  Original:  \"{}\"", example_text);
                                println!("  Decoded:   \"{}\"", decoded);
                            }
                        }
                        Err(e) => println!("{}", format!("Error decoding: {}", e).red()),
                    }
                }
                Err(e) => println!("{}", format!("Error encoding: {}", e).red()),
            }

            // Step 7: Extend vocabulary with special tokens
            println!();
            println!("Step 7: Extend vocabulary with special tokens");
            println!("Python equivalent:");
            println!("  all_tokens = sorted(list(set(preprocessed)))");
            println!("  all_tokens.extend([\"<|endoftext|>\", \"<|unk|>\"])");
            println!("  vocab = {{token:integer for integer,token in enumerate(all_tokens)}}");
            println!("  print(len(vocab.items()))");
            println!();

            // Create a new vocabulary with special tokens
            let mut all_tokens = sorted_words.clone();
            all_tokens.push("<|endoftext|>");
            all_tokens.push("<|unk|>");

            let vocab_with_special: HashMap<String, usize> = all_tokens
                .iter()
                .enumerate()
                .map(|(idx, &word)| (word.to_string(), idx))
                .collect();

            println!("Original vocabulary size: {}", vocab_size);
            println!("Extended vocabulary size: {}", vocab_with_special.len());

            // Show the last five entries of the vocabulary
            println!();
            println!("Last five entries of the updated vocabulary:");
            println!("Python equivalent:");
            println!("  for i, item in enumerate(list(vocab.items())[-5:]):");
            println!("      print(item)");
            println!();

            // Get sorted vocabulary items to show last 5
            let mut vocab_items: Vec<(&String, &usize)> = vocab_with_special.iter().collect();
            vocab_items.sort_by_key(|&(_, &id)| id);

            for &(token, id) in vocab_items.iter().rev().take(5).rev() {
                println!("('{}', {})", token, id);
            }

            // Step 8: Create SimpleTokenizerV2 and test with unknown tokens
            println!();
            println!("Step 8: Create SimpleTokenizerV2 to handle unknown tokens");
            println!("Python equivalent:");
            println!("  tokenizer = SimpleTokenizerV2(vocab)");
            println!("  text = \"Hello, do you like tea?\"");
            println!("  print(tokenizer.encode(text))");
            println!();

            // Create tokenizer V2 with extended vocabulary
            let tokenizer_v2 = SimpleTokenizerV2::new(vocab_with_special);

            let new_text = "Hello, do you like tea?";
            println!("New text: \"{}\"", new_text);

            let ids = tokenizer_v2.encode(new_text);
            println!("Encoded IDs: {:?}", ids);
            println!("Number of tokens: {}", ids.len());

            // Decode to show the result
            match tokenizer_v2.decode(&ids) {
                Ok(decoded) => {
                    println!("Decoded text: \"{}\"", decoded);

                    // Show which tokens were replaced with <|unk|>
                    println!();
                    println!("Token mapping:");
                    let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();
                    let mut tokens = Vec::new();
                    let mut last_end = 0;

                    for mat in re.find_iter(new_text) {
                        if mat.start() > last_end {
                            let content = new_text[last_end..mat.start()].trim();
                            if !content.is_empty() {
                                tokens.push(content);
                            }
                        }
                        let content = mat.as_str().trim();
                        if !content.is_empty() {
                            tokens.push(content);
                        }
                        last_end = mat.end();
                    }
                    if last_end < new_text.len() {
                        let content = new_text[last_end..].trim();
                        if !content.is_empty() {
                            tokens.push(content);
                        }
                    }

                    for (token, &id) in tokens.iter().zip(ids.iter()) {
                        let mapped = if tokenizer_v2.str_to_int.contains_key(*token) {
                            token
                        } else {
                            "<|unk|>"
                        };
                        println!("  {} → {} (id: {})", token, mapped, id);
                    }
                }
                Err(e) => println!("Error decoding: {}", e),
            }

            // Step 9: Demonstrate joining multiple texts with <|endoftext|>
            println!();
            println!("Step 9: Join multiple texts with <|endoftext|> token");
            println!("Python equivalent:");
            println!("  text1 = \"Hello, do you like tea?\"");
            println!("  text2 = \"In the sunlit terraces of the palace.\"");
            println!("  text = \" <|endoftext|> \".join((text1, text2))");
            println!("  print(text)");
            println!("  print(tokenizer.encode(text))");
            println!();

            let text1 = "Hello, do you like tea?";
            let text2 = "In the sunlit terraces of the palace.";
            let joined_text = format!("{} <|endoftext|> {}", text1, text2);

            println!("Text 1: \"{}\"", text1);
            println!("Text 2: \"{}\"", text2);
            println!("Joined text: \"{}\"", joined_text);
            println!();

            let ids = tokenizer_v2.encode(&joined_text);
            println!("Encoded IDs: {:?}", ids);
            println!("Number of tokens: {}", ids.len());

            // Decode to verify
            match tokenizer_v2.decode(&ids) {
                Ok(decoded) => {
                    println!("Decoded text: \"{}\"", decoded);
                }
                Err(e) => println!("Error decoding: {}", e),
            }

            // Step 10: Demonstrate tiktoken (GPT-2 tokenizer)
            println!();
            println!("Step 10: Demonstrate tiktoken (GPT-2 tokenizer)");
            println!("Python equivalent:");
            println!("  import tiktoken");
            println!("  encoding = tiktoken.get_encoding(\"r50k_base\")");
            println!("  text = \"Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.\"");
            println!("  tokens = encoding.encode(text, allowed_special=\"all\")");
            println!("  print(tokens)");
            println!("  decoded = encoding.decode(tokens)");
            println!("  print(decoded)");
            println!();

            println!("Loading GPT-2 tokenizer (r50k_base)...");
            let encoding = r50k_base().unwrap();

            let tiktoken_text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
            println!();
            println!("Input text: \"{}\"", tiktoken_text);

            // Get special tokens
            let allowed_special = encoding.special_tokens();

            // Encode text
            let (tokens, _) = encoding.encode(tiktoken_text, &allowed_special);
            println!("Token IDs: {:?}", tokens);
            println!("Number of tokens: {}", tokens.len());

            // Decode back to text
            let decoded = encoding.decode(tokens.clone()).unwrap();
            println!("Decoded text: \"{}\"", decoded);

            if decoded == tiktoken_text {
                println!("{}", "✓ Round-trip encoding/decoding successful!".green());
            }

            // Compare tokenization between our simple tokenizer and tiktoken
            println!();
            println!("Comparison of tokenization approaches:");
            println!("- SimpleTokenizerV2: {} tokens", ids.len());
            println!("- tiktoken (GPT-2): {} tokens", tokens.len());
            println!();
            println!("Note: tiktoken uses Byte Pair Encoding (BPE) which creates more");
            println!("efficient tokenization by learning common subword patterns.");

            // Step 11: Demonstrate sliding windows for training data preparation
            println!();
            println!("Step 11: Demonstrate sliding windows for training data preparation");
            println!("Python equivalent:");
            println!("  # Create training data with sliding windows");
            println!("  enc_sample = enc_text[50:]");
            println!("  context_size = 4");
            println!("  x = enc_sample[:context_size]");
            println!("  y = enc_sample[1:context_size+1]");
            println!("  print(f\"x: {{x}}\")");
            println!("  print(f\"y: {{y}}\")");
            println!();
            println!("  for i in range(1, context_size+1):");
            println!("      context = enc_sample[:i]");
            println!("      desired = enc_sample[i]");
            println!("      print(context, \"---->\", desired)");
            println!();

            // Re-encode the full text for sliding window demo
            let (full_enc_text, _) = encoding.encode(&raw_text, &allowed_special);
            
            // Create sliding windows starting at position 50
            let start_pos = 50;
            let context_size = 4;
            
            if full_enc_text.len() > start_pos + context_size {
                let enc_sample = &full_enc_text[start_pos..];
                
                println!("Working with {} total tokens, starting at position {}", full_enc_text.len(), start_pos);
                println!();
                
                // Show initial context window
                let x = &enc_sample[..context_size];
                let y = &enc_sample[1..context_size + 1];
                
                println!("Initial context window:");
                println!("  Input (x):  {:?}", x);
                println!("  Target (y): {:?}", y);
                println!();
                
                // Show growing context
                println!("Growing context windows (how model learns to predict next token):");
                for i in 1..=context_size {
                    let context = &enc_sample[..i];
                    let target = enc_sample[i];
                    
                    print!("  {:?} ", context);
                    print!("{}", "----->".yellow());
                    println!(" {}", target);
                    
                    // Decode to show the actual text
                    let context_decoded = encoding.decode(context.to_vec()).unwrap_or_else(|_| "<error>".to_string());
                    let target_decoded = encoding.decode(vec![target]).unwrap_or_else(|_| "<error>".to_string());
                    
                    println!("    \"{}\" {} \"{}\"", 
                            context_decoded.blue(), 
                            "----->".yellow(), 
                            target_decoded.green());
                }
                
                println!();
                println!("This sliding window approach creates input-target pairs for training:");
                println!("- Each position in the text provides a training example");
                println!("- The model learns to predict the next token given the context");
                println!("- This is how GPT models are trained to generate text");
            }

            // Step 12: Demonstrate GPTDatasetV1 for batch training data preparation
            println!();
            println!("Step 12: Create GPTDatasetV1 for batch training");
            println!("Python equivalent:");
            println!("  class GPTDatasetV1(Dataset):");
            println!("      def __init__(self, txt, tokenizer, max_length, stride):");
            println!("          self.input_ids = []");
            println!("          self.target_ids = []");
            println!("          token_ids = tokenizer.encode(txt)");
            println!("          for i in range(0, len(token_ids) - max_length, stride):");
            println!("              input_chunk = token_ids[i:i + max_length]");
            println!("              target_chunk = token_ids[i + 1: i + max_length + 1]");
            println!("              self.input_ids.append(torch.tensor(input_chunk))");
            println!("              self.target_ids.append(torch.tensor(target_chunk))");
            println!();
            println!("  dataset = GPTDatasetV1(raw_text, tokenizer, max_length=4, stride=1)");
            println!("  print(f\"Total samples: {{len(dataset)}}\")");
            println!();

            // Create the dataset with the same parameters
            let dataset = GPTDatasetV1::new(&raw_text, 4, 1);
            
            println!("Creating GPTDatasetV1 with max_length=4, stride=1...");
            println!("Total samples: {}", dataset.len());
            println!();
            
            // Show first few samples
            println!("First 3 samples:");
            for i in 0..3.min(dataset.len()) {
                if let Some((input_tensor, target_tensor)) = dataset.get_tensors(i) {
                    println!();
                    println!("Sample {}:", i);
                    
                    // Convert tensors to vectors for display
                    let input_data = input_tensor.to_data();
                    let target_data = target_tensor.to_data();
                    let input_ids_vec = input_data.to_vec::<i64>().unwrap();
                    let target_ids_vec = target_data.to_vec::<i64>().unwrap();
                    
                    println!("  Input IDs:  {:?}", input_ids_vec);
                    println!("  Target IDs: {:?}", target_ids_vec);
                    
                    // Convert back to u32 for tiktoken decoding
                    let input_ids_u32: Vec<u32> = input_ids_vec.iter().map(|&x| x as u32).collect();
                    let target_ids_u32: Vec<u32> = target_ids_vec.iter().map(|&x| x as u32).collect();
                    
                    // Decode to show text
                    let input_text = encoding.decode(input_ids_u32).unwrap();
                    let target_text = encoding.decode(target_ids_u32).unwrap();
                    
                    println!("  Input:  {:?}", input_text);
                    println!("  Target: {:?}", target_text);
                }
            }
            
            println!();
            println!("Key insights:");
            println!("- GPTDatasetV1 creates all input-target pairs upfront");
            println!("- Each sample is a vector of token IDs ready for batch processing");
            println!("- The stride parameter controls overlap between samples");
            println!("- This format is ideal for training with DataLoaders");
            println!("- In production with Burn, these would be converted to tensors");

            println!("\n=== Demo Complete ===");
            println!(
                "The file '{}' has been tokenized and prepared for training.",
                VERDICT_FILENAME
            );
        }
        Commands::Analyze {
            file_path,
            preview_length,
        } => {
            analyze_text(&file_path, preview_length).await?;
        }
        Commands::Split {
            file_path,
            method,
            max_display,
        } => {
            handle_split(file_path, method, max_display).await?;
        }
        Commands::Tokenize {
            text,
            file_path,
            tokenizer,
            detailed,
        } => {
            handle_tokenize(text, file_path, tokenizer, detailed).await?;
        }
        Commands::SlidingWindow {
            file_path,
            context_size,
            start_pos,
            max_windows,
            show_decoded,
        } => {
            handle_sliding_window(file_path, context_size, start_pos, max_windows, show_decoded).await?;
        }
        Commands::Dataset {
            file_path,
            max_length,
            stride,
            num_samples,
            show_decoded,
            verbose,
        } => {
            handle_dataset(file_path, max_length, stride, num_samples, show_decoded, verbose).await?;
        }
        Commands::Dataloader {
            file_path,
            batch_size,
            max_length,
            stride,
            shuffle_seed,
            num_batches,
            show_decoded,
        } => {
            handle_dataloader(file_path, batch_size, max_length, stride, shuffle_seed, num_batches, show_decoded).await?;
        }
    }

    Ok(())
}
