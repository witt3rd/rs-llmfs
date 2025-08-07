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

            println!("\n=== Demo Complete ===");
            println!(
                "The file '{}' has been tokenized and is ready for further processing.",
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
    }

    Ok(())
}
