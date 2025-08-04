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
use std::error::Error; // Import trait for Box<dyn Error>
use std::fmt; // Module import - we'll use fmt::Display
use std::path::{Path, PathBuf}; // Types imported directly

// External crate imports (alphabetically sorted)
use clap::{Parser, Subcommand}; // Derive macro traits
use colored::*; // Colors for terminal output
use futures_util::StreamExt; // Trait needed for .next() on streams
use indicatif::{ProgressBar, ProgressStyle}; // Types imported directly
use regex::Regex; // Type imported directly
use reqwest; // Module import - we'll use reqwest::get() (function convention)
use tokio::fs; // Module import - we'll use fs::read_to_string() (function convention)
use tokio::io::AsyncWriteExt; // Trait needed for .write_all() method

// Note: No internal imports as this is a binary crate (main.rs)

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
}

impl fmt::Display for SplitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitMethod::Whitespace => write!(f, "whitespace"),
            SplitMethod::Punctuation => write!(f, "punctuation"),
        }
    }
}

/// A token from splitting (either text or delimiter)
#[derive(Debug, Clone)]
struct Token {
    content: String,
    is_delimiter: bool,
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
        #[arg(long, default_value = "50")]
        max_display: usize,
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
/// * `Vec<Token>` - Vector of tokens including whitespace
fn whitespace_split(text: &str) -> Vec<Token> {
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

    result
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
/// * `Vec<Token>` - Vector of tokens including delimiters
fn punctuation_split(text: &str) -> Vec<Token> {
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

    result
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

    let tokens = match method {
        SplitMethod::Whitespace => {
            println!("  import re");
            println!("  result = re.split(r'(\\s)', text)");
            println!();
            whitespace_split(&text)
        }
        SplitMethod::Punctuation => {
            println!("  import re");
            println!("  result = re.split(r'([,.]|\\s)', text)");
            println!();
            punctuation_split(&text)
        }
    };

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
        // Show all tokens inline
        for token in &tokens {
            print!("{}", token.display_inline(&colors));
        }
        println!();
    } else {
        // Show limited tokens inline
        for token in tokens.iter().take(max_display) {
            print!("{}", token.display_inline(&colors));
        }
        print!("{}", format!(" ... ({} more tokens)", tokens.len() - max_display).bright_black());
        println!();
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

            let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
            let _file_path = download_file(url, Some("the-verdict.txt")).await?;

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

            analyze_text("the-verdict.txt", 99).await?;

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
            println!("Example with whitespace splitting:");
            handle_split(None, SplitMethod::Whitespace, 50).await?;

            println!();
            println!("Example with punctuation splitting:");
            println!("Python equivalent:");
            println!("  result = re.split(r'([,.]|\\s)', text)");
            println!();
            handle_split(None, SplitMethod::Punctuation, 50).await?;

            println!();
            println!("Now with our downloaded text (first 50 tokens):");
            handle_split(Some("the-verdict.txt".to_string()), SplitMethod::Punctuation, 50).await?;

            println!("\n=== Demo Complete ===");
            println!(
                "The file 'the-verdict.txt' is now ready for further text processing examples."
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
    }

    Ok(())
}
