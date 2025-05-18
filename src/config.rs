use log::{error, info};
use std::env;
use std::collections::HashMap;
use once_cell::sync::Lazy;

// API Keys
pub static OPENAI_API_KEY: Lazy<Option<String>> = Lazy::new(|| env::var("OPENAI_API_KEY").ok());

// Qdrant settings
pub static QDRANT_URL: Lazy<String> = Lazy::new(|| env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string()));
pub static QDRANT_API_KEY: Lazy<Option<String>> = Lazy::new(|| env::var("QDRANT_API_KEY").ok());
pub static QDRANT_COLLECTION: Lazy<String> = Lazy::new(|| env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "openai_embeddings".to_string()));

// OpenAI models
pub static DEFAULT_CHAT_MODEL: Lazy<String> = Lazy::new(|| env::var("DEFAULT_CHAT_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()));
pub static DEFAULT_EMBEDDING_MODEL: Lazy<String> = Lazy::new(|| env::var("DEFAULT_EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string()));

// Available embedding models
pub static AVAILABLE_EMBEDDING_MODELS: Lazy<Vec<&'static str>> = Lazy::new(|| vec![
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]);

// Embedding dimensions by model
pub static EMBEDDING_DIMENSIONS: Lazy<HashMap<&'static str, usize>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("text-embedding-3-small", 1536);
    map.insert("text-embedding-3-large", 3072);
    map.insert("text-embedding-ada-002", 1536);
    map
});

// Emoji indicators for different information sources
pub const EMOJI_SEARCH: &str = "🔍";  // Searching
pub const EMOJI_CONTEXT: &str = "📚";  // Using context from Qdrant
pub const EMOJI_AI: &str = "🤖";  // AI general knowledge
pub const EMOJI_ERROR: &str = "⚠️";  // Error indicator

// Default chunking settings
pub const DEFAULT_MAX_SENTENCES_PER_CHUNK: usize = 3;

// Text file extensions for auto-detection
pub static TEXT_FILE_EXTENSIONS: Lazy<Vec<&'static str>> = Lazy::new(|| vec![
    ".txt",
    ".md",
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
]);

/// Validate that required environment variables are set.
pub fn validate_environment() -> bool {
    if OPENAI_API_KEY.is_none() {
        error!("OPENAI_API_KEY environment variable is not set");
        return false;
    }

    info!("Environment validation successful");
    true
} 