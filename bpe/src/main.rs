use clap::Parser;

/// Simple CLI for BPE text conversion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The text string to convert
    text: String,
}

/// Converts a string by shifting each byte value by 256 to create valid UTF-8 characters
fn convert_text(text: &str) -> String {
    text.bytes()
        .map(|byte| {
            let shifted = (byte as u32) + 256;
            std::char::from_u32(shifted).unwrap_or('�')
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    
    let converted = convert_text(&args.text);
    println!("{}", converted);
}
