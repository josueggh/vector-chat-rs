use anyhow::{anyhow, Result};
use log::{debug, info};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::config::{QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL};

// Qdrant API types
#[derive(Debug, Serialize)]
struct VectorParams {
    size: usize,
    distance: String,
}

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    vectors: VectorParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum PointId {
    Num(u64),
    Uuid(String),
}

#[derive(Debug, Serialize)]
struct Point {
    id: u64,
    vector: Vec<f32>,
    payload: Map<String, Value>,
}

#[derive(Debug, Serialize)]
struct UpsertRequest {
    points: Vec<Point>,
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    limit: u64,
    with_payload: bool,
    score_threshold: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CollectionDescription {
    name: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ListCollectionsResponse {
    result: Option<Vec<CollectionDescription>>,
    status: Option<String>,
    time: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SearchResponseHit {
    id: u64,
    score: f32,
    payload: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    result: Vec<SearchResponseHit>,
}

/// Service for interacting with Qdrant vector database.
pub struct QdrantService {
    client: HttpClient,
    base_url: String,
    collection_name: String,
    api_key: Option<String>,
}

impl QdrantService {
    /// Initialize Qdrant client and ensure collection exists.
    pub async fn new(
        collection_name: Option<String>,
        vector_size: Option<usize>,
    ) -> Result<Self> {
        let base_url = QDRANT_URL.clone();
        let api_key = QDRANT_API_KEY.clone();
        let client = HttpClient::new();
        let collection_name = collection_name.unwrap_or_else(|| QDRANT_COLLECTION.clone());

        let service = Self {
            client,
            base_url,
            collection_name,
            api_key,
        };

        // Check if collection exists, create if needed
        let collections = service.list_collections().await?;
        let collection_exists = collections
            .iter()
            .any(|c| c == &service.collection_name);

        if !collection_exists {
            if let Some(vector_size) = vector_size {
                info!("Creating collection '{}' with vector size {}", service.collection_name, vector_size);
                
                service.create_collection(vector_size).await?;
            } else {
                return Err(anyhow!(
                    "Collection '{}' does not exist. Provide vector_size to create it.",
                    service.collection_name
                ));
            }
        } else {
            info!("Using existing collection: {}", service.collection_name);
        }

        Ok(service)
    }

    // Helper to build a request with auth headers
    fn request_builder(&self, url: &str) -> reqwest::RequestBuilder {
        let builder = self.client.get(url);
        
        if let Some(api_key) = &self.api_key {
            builder.header("api-key", api_key)
        } else {
            builder
        }
    }

    /// List all collections
    async fn list_collections(&self) -> Result<Vec<String>> {
        let url = format!("{}/collections", self.base_url);
        let response = self.request_builder(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error: {}", error_text));
        }

        // For debugging purposes, print the raw response
        let response_text = response.text().await?;
        debug!("Raw Qdrant response: {}", response_text);
        
        // Try to parse as JSON Value first to inspect structure
        match serde_json::from_str::<Value>(&response_text) {
            Ok(json_value) => {
                // Check if the response has a result field with collections
                if let Some(result) = json_value.get("result") {
                    if let Some(collections) = result.get("collections").and_then(|c| c.as_array()) {
                        let mut collection_names = Vec::new();
                        
                        for collection in collections {
                            if let Some(name) = collection.get("name").and_then(|n| n.as_str()) {
                                collection_names.push(name.to_string());
                            }
                        }
                        
                        return Ok(collection_names);
                    }
                }
                
                // If we couldn't extract collections using the expected structure, return empty list
                info!("No collections found or unexpected response format");
                Ok(Vec::new())
            },
            Err(e) => {
                return Err(anyhow!("Failed to parse Qdrant response: {} - Response was: {}", e, response_text));
            }
        }
    }

    /// Create a new collection
    async fn create_collection(&self, vector_size: usize) -> Result<()> {
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: vector_size,
                distance: "Cosine".to_string(),
            },
        };

        let response = self.client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error creating collection: {}", error_text));
        }

        Ok(())
    }

    /// Insert or update vectors in the collection.
    pub async fn upsert(
        &self,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        payloads: Vec<HashMap<String, Value>>,
    ) -> Result<()> {
        if ids.len() != vectors.len() || ids.len() != payloads.len() {
            return Err(anyhow!("Ids, vectors, and payloads must have the same length"));
        }

        let mut points = Vec::new();
        for ((id, vector), payload) in ids.into_iter().zip(vectors).zip(payloads) {
            // Convert HashMap<String, Value> to Map<String, Value>
            let payload_map: Map<String, Value> = payload.into_iter().collect();
            
            points.push(Point {
                id,
                vector,
                payload: payload_map,
            });
        }

        let points_len = points.len();
        let url = format!("{}/collections/{}/points", self.base_url, self.collection_name);
        let request = UpsertRequest { points };

        let response = self.client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error upserting points: {}", error_text));
        }

        info!(
            "Upserted {} vectors into collection '{}'",
            points_len,
            self.collection_name
        );

        Ok(())
    }

    /// Search for similar vectors in the collection.
    pub async fn search(
        &self,
        vector: Vec<f32>,
        top_k: u64,
        score_threshold: f32,
    ) -> Result<Vec<(u64, f32, HashMap<String, Value>)>> {
        let url = format!("{}/collections/{}/points/search", self.base_url, self.collection_name);
        
        let request = SearchRequest {
            vector,
            limit: top_k,
            with_payload: true,
            score_threshold,
        };

        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error searching points: {}", error_text));
        }

        let search_response: SearchResponse = response.json().await?;
        
        let results: Vec<(u64, f32, HashMap<String, Value>)> = search_response.result
            .into_iter()
            .map(|hit| {
                let id = hit.id;
                let score = hit.score;
                
                // Convert Map<String, Value> to HashMap<String, Value>
                let payload: HashMap<String, Value> = hit.payload
                    .into_iter()
                    .collect();

                (id, score, payload)
            })
            .collect();

        debug!("Found {} results for search query", results.len());
        Ok(results)
    }

    /// Check if the collection exists.
    pub async fn check_collection_exists(&self) -> Result<bool> {
        let collections = self.list_collections().await?;
        let exists = collections.contains(&self.collection_name);
        Ok(exists)
    }
}
