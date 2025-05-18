use clap::{Parser, Subcommand};
use log::{error};
use std::process;

use vector_chat::cli::chat::run_chat;
use vector_chat::cli::embed::run_embed;

/// Vector Chat - Text embedding and chat with context
#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed text into vector database
    Embed {
        /// Path to text file to embed
        #[clap(short, long)]
        file: Option<String>,

        /// Text to embed directly
        #[clap(short, long)]
        text: Option<String>,

        /// List available text files
        #[clap(short = 'l', long)]
        list_files: bool,
    },

    /// Chat with OpenAI using vector context
    Chat {
        /// Disable context retrieval
        #[clap(long)]
        no_context: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::init_from_env(
        env_logger::Env::default().filter_or("RUST_LOG", "info,vector_chat=debug"),
    );

    // Load environment variables from .env file
    dotenv::dotenv().ok();

    // Parse command line arguments
    let cli = Cli::parse();

    // Run command
    match cli.command {
        Commands::Embed { file, text, list_files } => {
            match run_embed(file, text, list_files).await {
                Ok(_) => (),
                Err(e) => {
                    error!("Error running embed command: {}", e);
                    process::exit(1);
                }
            }
        }
        Commands::Chat { no_context } => {
            match run_chat(no_context).await {
                Ok(_) => (),
                Err(e) => {
                    error!("Error running chat command: {}", e);
                    process::exit(1);
                }
            }
        }
    }

    Ok(())
} 