use anyhow::Result;
use log::info;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::config::TEXT_FILE_EXTENSIONS;

/// Split text into chunks of sentences.
pub fn chunk_by_sentences(text: &str, max_sents: usize) -> Vec<String> {
    // Simple sentence splitting based on common punctuation
    let mut sentences = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Split on sentence-ending punctuation
        for c in line.chars() {
            current.push(c);
            if c == '.' || c == '?' || c == '!' {
                sentences.push(current.trim().to_string());
                current.clear();
            }
        }

        // Add any remaining text as a sentence
        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
            current.clear();
        }
    }

    // Group sentences into chunks
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();

    for sent in sentences {
        current_chunk.push(sent);
        if current_chunk.len() >= max_sents {
            chunks.push(current_chunk.join(" "));
            current_chunk.clear();
        }
    }

    // Add any remaining sentences
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }

    chunks
}

/// Process text into chunks with metadata.
pub fn chunk_text(
    text: &str,
    max_sents: usize,
    source_name: &str,
) -> Vec<HashMap<String, Value>> {
    let chunks = chunk_by_sentences(text, max_sents);
    
    chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut metadata = HashMap::new();
            metadata.insert("chunk_text".to_string(), Value::String(chunk.clone()));
            metadata.insert("source".to_string(), Value::String(source_name.to_string()));
            metadata.insert("chunk_index".to_string(), Value::Number(i.into()));
            metadata.insert("total_chunks".to_string(), Value::Number(chunks.len().into()));
            metadata
        })
        .collect()
}

/// List all text files in the directory.
pub fn list_text_files(directory: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();
    
    let entries = fs::read_dir(directory)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext = format!(".{}", extension.to_string_lossy().to_lowercase());
                if TEXT_FILE_EXTENSIONS.contains(&ext.as_str()) {
                    files.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    
    info!("Found {} text files in {}", files.len(), directory);
    Ok(files)
}

/// Read the content of a file.
pub fn read_file_content(file_path: &str) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    info!("Read {} characters from {}", content.len(), file_path);
    Ok(content)
}

/// Process a file into chunks with metadata.
pub fn process_file(
    file_path: &str,
    max_sents: usize,
) -> Result<Vec<HashMap<String, Value>>> {
    let content = read_file_content(file_path)?;
    
    let path = Path::new(file_path);
    let source_name = path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.to_string());
    
    Ok(chunk_text(&content, max_sents, &source_name))
} 