//! # Chapter 2: Asynchronous File Downloader
//!
//! This module demonstrates downloading files from URLs using async Rust.
//! 
//! ## Key Concepts Covered
//! - Async/await programming
//! - Error handling with `Result` and the `?` operator
//! - Working with trait objects (`Box<dyn Error>`)
//! - File I/O operations

use reqwest;
use std::path::Path;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Downloads a file from a URL and saves it to a local directory.
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
/// # Understanding `&str` (String References)
/// 
/// The `&` means we're borrowing the string, not taking ownership.
/// `str` is the string slice type - a view into string data.
/// 
/// # Understanding `Box<dyn std::error::Error>`
/// 
/// This is Rust's way of saying "any type of error". Let's break it down:
/// 
/// * **`Box<...>`** - A smart pointer that stores data on the heap
/// * **`dyn`** - Short for "dynamic", means the exact type is determined at runtime
/// * **`std::error::Error`** - A trait that all error types implement
/// 
/// ## Why `dyn` is Required
/// 
/// In Rust, traits have two uses:
/// 1. As bounds on generics: `fn foo<T: Error>(err: T)`
/// 2. As types themselves: `fn foo(err: Box<dyn Error>)`
/// 
/// When using a trait as a type, you MUST use `dyn`. Without it:
/// ```compile_fail
/// // This won't compile!
/// fn example() -> Box<std::error::Error> { ... }
/// // Error: trait objects must include the `dyn` keyword
/// ```
/// 
/// ## When to Use `dyn` in Your Code
/// 
/// 1. **Heterogeneous collections**: `Vec<Box<dyn Animal>>` can hold Dogs, Cats, etc.
/// 2. **Multiple error types**: This function can return network errors, I/O errors, etc.
/// 3. **Runtime polymorphism**: When the exact type isn't known until runtime
/// 4. **Simplicity**: Avoids complex generic type parameters
/// 
/// ## Alternative Using Generics
/// 
/// ```rust
/// fn download_file<E: std::error::Error>(...) -> Result<(), E>
/// ```
/// But this is less flexible - all errors must be the same type `E`.
/// 
/// # Async Functions
/// 
/// The `async` keyword means:
/// - This function returns a `Future` that must be executed by a runtime (like tokio)
/// - Can use `.await` to pause and wait for other async operations
/// - Enables non-blocking I/O (the program can do other things while waiting)
/// 
/// # Examples
/// 
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// download_file("https://example.com/file.pdf", "./downloads").await?;
/// # Ok(())
/// # }
/// ```
async fn download_file(url: &str, local_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Make an HTTP GET request to the URL and wait for the response
    let response = reqwest::get(url).await?;
    // ## The `?` Operator Explained
    //
    // The `?` is Rust's error propagation operator. It's syntactic sugar for:
    // ```rust
    // let response = match reqwest::get(url).await {
    //     Ok(value) => value,
    //     Err(error) => return Err(error.into()),
    // };
    // ```
    // 
    // If the operation succeeds, it unwraps the `Ok` value and continues.
    // If it fails, it converts the error and returns early from the function.
    
    // Check if the HTTP response indicates success (status code 200-299)
    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
        // ## The `.into()` Method
        //
        // `.into()` converts one type into another using the `Into` trait.
        // Here it converts `String` → `Box<dyn Error>` automatically.
        // This works because `String` implements `Into<Box<dyn Error>>`.
    }
    
    // Extract the filename from the URL
    let file_name = url.split('/').last().unwrap_or("downloaded_file");
    // ## Understanding `unwrap_or()`
    //
    // Many Rust operations return `Option<T>`:
    // - `Some(value)` - contains a value
    // - `None` - no value
    //
    // `unwrap_or()` safely extracts the value or provides a default:
    // ```rust
    // match url.split('/').last() {
    //     Some(name) => name,
    //     None => "downloaded_file",
    // }
    // ```
    //
    // Other unwrap variants:
    // - `unwrap()` - Panics if None (avoid in production!)
    // - `unwrap_or_else(|| ...)` - Computes default with a closure
    // - `expect("msg")` - Panics with custom message
    
    // Create the full file path by joining the directory and filename
    let file_path = Path::new(local_dir).join(file_name);
    
    // Create the directory (and any parent directories) if they don't exist
    fs::create_dir_all(local_dir).await?;
    
    // Create a new file at the specified path
    let mut file = fs::File::create(&file_path).await?;
    // ## The `mut` Keyword
    //
    // In Rust, variables are immutable by default:
    // ```rust
    // let x = 5;
    // x = 6; // ERROR! Cannot assign twice to immutable variable
    // ```
    //
    // `mut` makes a variable mutable:
    // ```rust
    // let mut x = 5;
    // x = 6; // OK!
    // ```
    //
    // We need `mut` here because `write_all()` mutates the file's internal state.
    
    // Download the entire response body as bytes (raw data)
    let content = response.bytes().await?;
    
    // Write all the downloaded bytes to the file
    file.write_all(&content).await?;
    
    // Print a success message showing what was downloaded and where it was saved
    println!("Downloaded {} to {}", url, file_path.display());
    
    Ok(())
}

/// The main entry point of our application.
/// 
/// # The `#[tokio::main]` Attribute
/// 
/// Attributes in Rust (marked with `#[...]`) modify the item that follows them.
/// This attribute transforms our async `main()` into a synchronous entry point
/// that initializes and runs the tokio async runtime.
/// 
/// Without this attribute, you'd need to manually set up the runtime:
/// ```rust
/// fn main() {
///     let runtime = tokio::runtime::Runtime::new().unwrap();
///     runtime.block_on(async {
///         // your async code here
///     });
/// }
/// ```
/// 
/// # Why Async Main?
/// 
/// We use async because our program performs I/O operations (network requests,
/// file writes) that can block. Async allows the program to do other work
/// while waiting for these operations to complete.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    
    // Example usage (uncomment to test):
    // download_file("https://www.rust-lang.org/logos/rust-logo-512x512.png", "./downloads").await?;
    
    // Return Ok(()) to indicate the program completed successfully
    // This satisfies the Result return type - no error occurred
    Ok(())
}
