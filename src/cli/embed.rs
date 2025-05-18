use anyhow::{anyhow, Result};
use log::{error, info};
use serde_json::Value;
use std::io::{self, BufRead};

use crate::clients::OpenAIClient;
use crate::config::{
    DEFAULT_EMBEDDING_MODEL, DEFAULT_MAX_SENTENCES_PER_CHUNK, QDRANT_COLLECTION,
    validate_environment,
};
use crate::services::chunker::{chunk_text, list_text_files, read_file_content};
use crate::services::qdrant_service::QdrantService;

/// Get input text from file or direct input.
async fn get_input_text(
    file_path: Option<String>,
    text_input: Option<String>,
    list_files: bool,
) -> Result<Option<(String, String)>> {
    // Check for file input
    if let Some(path) = file_path {
        match read_file_content(&path) {
            Ok(content) => return Ok(Some((content, path))),
            Err(e) => {
                error!("Could not read file: {}: {}", path, e);
                return Ok(None);
            }
        }
    }

    // Check for direct text input
    if let Some(text) = text_input {
        return Ok(Some((text, "command_line_input".to_string())));
    }

    // If no input provided, prompt user
    if !list_files {
        // List available files
        let files = list_text_files(".")?;

        if !files.is_empty() {
            info!("Available text files:");
            for (i, file) in files.iter().enumerate() {
                info!("{}. {}", i + 1, file);
            }

            print!("\nSelect a file number (or press Enter for manual input): ");
            let mut input = String::new();
            io::stdin().lock().read_line(&mut input)?;

            if let Ok(idx) = input.trim().parse::<usize>() {
                if idx > 0 && idx <= files.len() {
                    match read_file_content(&files[idx - 1]) {
                        Ok(content) => return Ok(Some((content, files[idx - 1].clone()))),
                        Err(e) => error!("Could not read file: {}", e),
                    }
                } else {
                    error!("Invalid selection");
                }
            }
        }

        // Manual input
        info!("Enter text to embed (press Ctrl+D or Ctrl+Z on Windows to finish):");
        let stdin = io::stdin();
        let mut text = String::new();
        for line in stdin.lock().lines() {
            text.push_str(&line?);
            text.push('\n');
        }

        if !text.trim().is_empty() {
            return Ok(Some((text, "manual_input".to_string())));
        }
    }

    Ok(None)
}

/// Embed text chunks and store in vector database.
async fn embed_text(
    text: &str,
    source_name: &str,
    model_name: &str,
    collection_name: &str,
    max_sentences: usize,
) -> Result<bool> {
    // Initialize OpenAI client
    let openai_client = OpenAIClient::new(None, None, Some(model_name.to_string()))?;

    // Process text into chunks
    let chunks_data = chunk_text(text, max_sentences, source_name);
    let chunks: Vec<String> = chunks_data
        .iter()
        .filter_map(|item| {
            if let Some(Value::String(chunk)) = item.get("chunk_text") {
                Some(chunk.clone())
            } else {
                None
            }
        })
        .collect();

    if chunks.is_empty() {
        error!("No chunks generated from text");
        return Ok(false);
    }

    info!("Text chunked into {} segments", chunks.len());
    for (i, chunk) in chunks.iter().enumerate().take(5) {
        let preview = if chunk.len() > 50 {
            format!("{}...", &chunk[..50])
        } else {
            chunk.clone()
        };
        info!("Chunk {}: {}", i + 1, preview);
    }

    // Generate embeddings
    info!("Generating embeddings using {}...", model_name);
    let vectors = openai_client.embed(&chunks).await?;

    // Prepare payloads with metadata
    let ids: Vec<u64> = (1..=chunks.len() as u64).collect();

    // Initialize Qdrant and store vectors
    let qdrant = QdrantService::new(
        Some(collection_name.to_string()),
        Some(openai_client.get_embedding_dimension()),
    )
    .await?;

    qdrant.upsert(ids, vectors, chunks_data).await?;

    info!(
        "Successfully embedded {} chunks into collection '{}'",
        chunks.len(),
        collection_name
    );
    Ok(true)
}

/// Main entry point for the embed command.
pub async fn run_embed(
    file: Option<String>,
    text: Option<String>,
    list_files: bool,
) -> Result<()> {
    // Validate environment
    if !validate_environment() {
        error!("Environment validation failed");
        return Err(anyhow!("Environment validation failed"));
    }

    // List files if requested
    if list_files {
        let files = list_text_files(".")?;
        if !files.is_empty() {
            info!("Available text files:");
            for file in files {
                info!("- {}", file);
            }
        } else {
            info!("No text files found in current directory");
        }
        return Ok(());
    }

    // Get input text
    let input_data = get_input_text(file, text, list_files).await?;
    if input_data.is_none() {
        error!("No input text provided");
        return Err(anyhow!("No input text provided"));
    }

    let (text, source) = input_data.unwrap();
    
    // Embed text
    match embed_text(
        &text,
        &source,
        &DEFAULT_EMBEDDING_MODEL,
        &QDRANT_COLLECTION,
        DEFAULT_MAX_SENTENCES_PER_CHUNK,
    )
    .await
    {
        Ok(true) => {
            info!("Text successfully embedded");
            Ok(())
        }
        Ok(false) => Err(anyhow!("Failed to embed text")),
        Err(e) => Err(anyhow!("Error embedding text: {}", e)),
    }
} 