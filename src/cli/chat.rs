use anyhow::{anyhow, Result};
use colored::Colorize;
use log::{error, info};
use rustyline::{error::ReadlineError, DefaultEditor};
use serde_json::Value;

use crate::clients::OpenAIClient;
use crate::config::{
    DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, EMOJI_AI, EMOJI_CONTEXT, EMOJI_ERROR,
    EMOJI_SEARCH, QDRANT_COLLECTION, validate_environment,
};
use crate::services::qdrant_service::QdrantService;

/// Get relevant context for a query.
async fn get_context(
    query: &str,
    openai_client: &OpenAIClient,
    qdrant_client: &QdrantService,
    top_k: u64,
    score_threshold: f32,
) -> Result<(bool, Option<String>)> {
    // Generate query embedding
    info!("{} Searching for relevant information...", EMOJI_SEARCH);
    let q_vec = openai_client.embed(&[query.to_string()]).await?[0].clone();

    // Search for relevant chunks
    let results = qdrant_client
        .search(q_vec, top_k, score_threshold)
        .await?;

    if results.is_empty() {
        info!("{} No relevant context found", EMOJI_SEARCH);
        return Ok((false, None));
    }

    // Prepare context from search results
    let mut context_parts = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let score = result.1;
        let payload = &result.2;

        let text = if let Some(Value::String(chunk_text)) = payload.get("chunk_text") {
            chunk_text
        } else {
            continue;
        };

        let source_info = if let Some(Value::String(source)) = payload.get("source") {
            format!(" (from {})", source)
        } else {
            String::from(" (from unknown source)")
        };

        let model_info = if let Some(Value::String(model_name)) = payload.get("model_name") {
            format!(" [model: {}]", model_name)
        } else {
            String::new()
        };

        context_parts.push(format!(
            "Context {} (Relevance: {:.2}){}{}: {}",
            i + 1,
            score,
            source_info,
            model_info,
            text
        ));
    }

    let context = context_parts.join("\n\n");
    info!("{} Found {} relevant context chunks", EMOJI_CONTEXT, results.len());

    Ok((true, Some(context)))
}

/// Run the interactive chat loop.
async fn chat_loop(
    openai_client: &mut OpenAIClient,
    qdrant_client: Option<&QdrantService>,
    top_k: u64,
    score_threshold: f32,
) -> Result<()> {
    println!("\nChat with OpenAI (type 'exit' to quit, 'reset' to clear conversation history):");

    if qdrant_client.is_some() {
        println!(
            "\n{} = Using saved context | {} = AI knowledge | {} = Searching",
            EMOJI_CONTEXT, EMOJI_AI, EMOJI_SEARCH
        );
    } else {
        println!("\n{} = AI knowledge (no context retrieval enabled)", EMOJI_AI);
    }

    let mut rl = DefaultEditor::new()?;

    loop {
        // Get user query
        let query = match rl.readline("\nYou: ") {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("\nExiting chat...");
                break;
            }
            Err(err) => {
                error!("Error reading input: {}", err);
                continue;
            }
        };

        // Check for special commands
        match query.to_lowercase().as_str() {
            "exit" | "quit" | "bye" => {
                println!("Goodbye!");
                break;
            }
            "reset" => {
                openai_client.reset_conversation(true);
                println!("\n{} Conversation history has been reset.", EMOJI_AI);
                continue;
            }
            _ => {}
        }

        // Add user query to conversation
        openai_client.add_user_message(&query);

        // Try to find relevant context if available
        if let Some(qdrant) = qdrant_client {
            match get_context(&query, openai_client, qdrant, top_k, score_threshold).await {
                Ok((context_found, Some(context))) if context_found => {
                    // Add context to chat as system message
                    openai_client.add_system_message(&format!(
                        "Here is some relevant context to help answer the question. \
                        Use this information if it's helpful for answering the question:\n{}",
                        context
                    ));
                    
                    // Get response with context
                    match openai_client.get_response(0.7).await {
                        Ok(response) => {
                            println!("\n{} {}", EMOJI_CONTEXT, response.bright_green());
                        }
                        Err(e) => {
                            error!("Error getting response: {}", e);
                            println!("\n{} Error getting response", EMOJI_ERROR);
                        }
                    }
                }
                _ => {
                    // Get response without context
                    match openai_client.get_response(0.7).await {
                        Ok(response) => {
                            println!("\n{} {}", EMOJI_AI, response.bright_cyan());
                        }
                        Err(e) => {
                            error!("Error getting response: {}", e);
                            println!("\n{} Error getting response", EMOJI_ERROR);
                        }
                    }
                }
            }
        } else {
            // No context retrieval, just get response
            match openai_client.get_response(0.7).await {
                Ok(response) => {
                    println!("\n{} {}", EMOJI_AI, response.bright_cyan());
                }
                Err(e) => {
                    error!("Error getting response: {}", e);
                    println!("\n{} Error getting response", EMOJI_ERROR);
                }
            }
        }
    }

    Ok(())
}

/// Main entry point for the chat command.
pub async fn run_chat(no_context: bool) -> Result<()> {
    // Validate environment
    if !validate_environment() {
        error!("Environment validation failed");
        return Err(anyhow!("Environment validation failed"));
    }

    // Initialize OpenAI client
    let mut openai_client = OpenAIClient::new(
        None,
        Some(DEFAULT_CHAT_MODEL.clone()),
        Some(DEFAULT_EMBEDDING_MODEL.clone()),
    )?;

    // Add system message
    openai_client.add_system_message(
        "You are a helpful assistant that can answer questions based on provided context or general knowledge. \
        If context is provided, prioritize that information in your answers. \
        If no context is provided or the question is outside the scope of the context, \
        use your general knowledge to provide a helpful response. \
        Always be honest about what you know and don't know."
    );

    // Initialize Qdrant client if context is enabled
    let qdrant_client = if !no_context {
        match QdrantService::new(Some(QDRANT_COLLECTION.clone()), None).await {
            Ok(client) => {
                info!("Connected to Qdrant collection: {}", *QDRANT_COLLECTION);
                Some(client)
            }
            Err(e) => {
                error!("Error connecting to Qdrant: {}", e);
                info!("Continuing without context retrieval");
                None
            }
        }
    } else {
        None
    };

    // Start chat loop
    chat_loop(
        &mut openai_client,
        qdrant_client.as_ref(),
        3,  // top_k
        0.3, // score_threshold
    )
    .await?;

    Ok(())
} 