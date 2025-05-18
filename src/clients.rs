use anyhow::{anyhow, Result};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::{DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, OPENAI_API_KEY};

// OpenAI API types
#[derive(Debug, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatResponseChoice>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Client for interacting with OpenAI APIs for both chat completions and embeddings.
pub struct OpenAIClient {
    client: HttpClient,
    api_key: String,
    chat_model: String,
    embedding_model: String,
    conversation_history: Vec<ChatMessage>,
    embedding_dimension: usize,
}

impl OpenAIClient {
    /// Initialize OpenAI client for both chat completions and embeddings.
    pub fn new(
        api_key: Option<String>,
        chat_model: Option<String>,
        embedding_model: Option<String>,
    ) -> Result<Self> {
        let api_key = api_key.or_else(|| OPENAI_API_KEY.clone());
        
        if api_key.is_none() {
            return Err(anyhow!(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter."
            ));
        }

        let client = HttpClient::new();
        let chat_model = chat_model.unwrap_or_else(|| DEFAULT_CHAT_MODEL.clone());
        let embedding_model = embedding_model.unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.clone());
        
        // Get embedding dimension based on model
        let embedding_dimension = EMBEDDING_DIMENSIONS
            .get(embedding_model.as_str())
            .copied()
            .unwrap_or(1536);

        Ok(Self {
            client,
            api_key: api_key.unwrap(),
            chat_model,
            embedding_model,
            conversation_history: Vec::new(),
            embedding_dimension,
        })
    }

    /// Add a system message to the conversation history.
    pub fn add_system_message(&mut self, content: &str) {
        self.conversation_history.push(ChatMessage {
            role: "system".to_string(),
            content: content.to_string(),
        });
    }

    /// Add a user message to the conversation history.
    pub fn add_user_message(&mut self, content: &str) {
        self.conversation_history.push(ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        });
    }

    /// Add an assistant message to the conversation history.
    pub fn add_assistant_message(&mut self, content: &str) {
        self.conversation_history.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
        });
    }

    /// Get a response from the chat model based on conversation history.
    pub async fn get_response(&mut self, temperature: f32) -> Result<String> {
        let request = ChatRequest {
            model: self.chat_model.clone(),
            messages: self.conversation_history.clone(),
            temperature,
            response_format: None,
        };

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error: {}", error_text));
        }

        let chat_response: ChatResponse = response.json().await?;
        if let Some(choice) = chat_response.choices.first() {
            if let Some(content) = &choice.message.content {
                self.add_assistant_message(content);
                return Ok(content.to_string());
            }
        }
        
        Err(anyhow!("No content in response"))
    }

    /// Create embeddings using OpenAI's embedding model.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_vectors = Vec::new();
        let batch_size = 64;

        for chunk in texts.chunks(batch_size) {
            let request = EmbeddingRequest {
                model: self.embedding_model.clone(),
                input: chunk.to_vec(),
            };

            let response = self.client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(anyhow!("API error: {}", error_text));
            }

            let embedding_response: EmbeddingResponse = response.json().await?;
            let vectors: Vec<Vec<f32>> = embedding_response
                .data
                .into_iter()
                .map(|item| item.embedding)
                .collect();
            
            all_vectors.extend(vectors);
        }

        Ok(all_vectors)
    }

    /// Reset the conversation history, optionally keeping system messages.
    pub fn reset_conversation(&mut self, keep_system_messages: bool) {
        if keep_system_messages {
            self.conversation_history = self
                .conversation_history
                .iter()
                .filter(|msg| msg.role == "system")
                .cloned()
                .collect();
        } else {
            self.conversation_history.clear();
        }
    }

    /// Get the embedding dimension for the current model
    pub fn get_embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }
}
