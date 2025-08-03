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

use clap::Parser;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use std::cmp::min;
use std::path::Path;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// A simple file downloader with progress reporting
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// URL to download from
    url: String,

    /// Directory to save the file (defaults to "data")
    #[arg(short, long, default_value = "data")]
    directory: String,
}

/// Downloads a file from a URL with progress reporting and saves it to a local directory.
///
/// # Arguments
/// 
/// * `url` - The URL to download from (borrowed string slice `&str`)
/// * `local_dir` - The local directory path to save the file
///
/// # Returns
/// 
/// * `Result<(), Box<dyn std::error::Error>>` - Success or any error type
///
/// # Progress Reporting
/// 
/// This function now includes visual progress reporting using the indicatif crate.
/// It shows:
/// - A progress bar with percentage completion
/// - Download speed in MB/s
/// - Estimated time remaining
/// - Total bytes downloaded
async fn download_file(url: &str, local_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
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
    
    // Extract the filename from the URL
    let file_name = url.split('/').last().unwrap_or("downloaded_file");
    
    // Create the full file path by joining the directory and filename
    let file_path = Path::new(local_dir).join(file_name);
    
    // Create the directory (and any parent directories) if they don't exist
    fs::create_dir_all(local_dir).await?;
    
    // Create a new file at the specified path
    let mut file = fs::File::create(&file_path).await?;
    
    // Create a progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")?
            .progress_chars("#>-")
    );
    pb.set_message(format!("Downloading {}", file_name));
    
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
    
    pb.finish_with_message(format!("Downloaded {} to {}", file_name, file_path.display()));
    
    Ok(())
}

/// The main entry point of our application.
/// 
/// This function:
/// 1. Parses command-line arguments using clap
/// 2. Calls the download function with progress reporting
/// 3. Handles any errors that occur during the download
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args = Args::parse();
    
    println!("Starting download...");
    println!("URL: {}", args.url);
    println!("Save to: {}", args.directory);
    println!();
    
    // Download the file with progress reporting
    download_file(&args.url, &args.directory).await?;
    
    println!("\nDownload complete!");
    
    Ok(())
}