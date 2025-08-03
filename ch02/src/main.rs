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

use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use std::cmp::min;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

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
async fn download_file(url: &str, output_path: Option<&str>) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
    pb.set_message(format!("Downloading {}", file_path.file_name().unwrap_or_default().to_string_lossy()));
    
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
async fn analyze_text(file_path: &str, preview_length: usize) -> Result<(), Box<dyn std::error::Error>> {
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

/// The main entry point of our application.
/// 
/// This function:
/// 1. Parses command-line arguments using clap
/// 2. Executes the appropriate subcommand
/// 3. Handles any errors that occur during execution
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
            
            println!("\n=== Demo Complete ===");
            println!("The file 'the-verdict.txt' is now ready for further text processing examples.");
        }
        Commands::Analyze { file_path, preview_length } => {
            analyze_text(&file_path, preview_length).await?;
        }
    }
    
    Ok(())
}