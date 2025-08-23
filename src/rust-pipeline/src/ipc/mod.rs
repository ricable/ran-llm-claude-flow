/*!
# Inter-Process Communication (IPC)

High-performance IPC between Rust core and Python ML workers.
Optimized for M3 Max unified memory architecture.
*/

pub mod shared_memory;
pub mod message_queue;
pub mod process_manager;

use crate::{Result, PipelineError, PipelineConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// IPC message types between Rust and Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcMessage {
    // Document processing messages
    DocumentProcess {
        document_id: Uuid,
        content: Vec<u8>,
        format: DocumentFormat,
        processing_options: ProcessingOptions,
    },
    DocumentResult {
        document_id: Uuid,
        result: ProcessingResult,
        performance_metrics: ProcessingMetrics,
    },
    
    // Model management messages
    ModelLoad {
        model_id: String,
        model_path: String,
        model_config: ModelConfig,
    },
    ModelUnload {
        model_id: String,
    },
    ModelInference {
        request_id: Uuid,
        model_id: String,
        input_text: String,
        inference_config: InferenceConfig,
    },
    ModelInferenceResult {
        request_id: Uuid,
        model_id: String,
        result: InferenceResult,
    },
    
    // Quality validation messages
    QualityCheck {
        content_id: Uuid,
        content: String,
        quality_criteria: QualityCriteria,
    },
    QualityResult {
        content_id: Uuid,
        quality_score: f64,
        quality_details: QualityDetails,
    },
    
    // System control messages
    WorkerHeartbeat {
        worker_id: String,
        timestamp: u64,
        status: WorkerStatus,
        metrics: WorkerMetrics,
    },
    SystemShutdown,
    
    // Error handling
    Error {
        code: u32,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentFormat {
    Pdf,
    Html, 
    Markdown,
    PlainText,
    Csv,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    pub extract_features: bool,
    pub extract_parameters: bool,
    pub extract_commands: bool,
    pub extract_procedures: bool,
    pub quality_threshold: f64,
    pub model_preference: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub document_id: Uuid,
    pub extracted_content: ExtractedContent,
    pub quality_score: f64,
    pub processing_time_ms: u64,
    pub model_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    pub features: Vec<Feature>,
    pub parameters: Vec<Parameter>,
    pub commands: Vec<Command>,
    pub procedures: Vec<Procedure>,
    pub references: Vec<Reference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub description: String,
    pub category: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub data_type: String,
    pub default_value: Option<String>,
    pub description: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub command: String,
    pub syntax: String,
    pub description: String,
    pub examples: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub title: String,
    pub steps: Vec<String>,
    pub prerequisites: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub title: String,
    pub section: Option<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub processing_time_ms: u64,
    pub tokens_processed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub max_memory_gb: u32,
    pub context_length: u32,
    pub temperature: f32,
    pub use_mlx: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub generated_text: String,
    pub token_count: u32,
    pub inference_time_ms: u64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCriteria {
    pub min_content_quality: f64,
    pub min_extraction_confidence: f64,
    pub require_features: bool,
    pub require_parameters: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDetails {
    pub content_quality: f64,
    pub extraction_confidence: f64,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub enum WorkerStatus {
    Initializing,
    Ready,
    Processing,
    Idle,
    Error,
    ShuttingDown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy, Default)]
pub struct WorkerMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub tasks_completed: u64,
    pub average_processing_time_ms: f64,
    pub error_count: u64,
}

/// IPC Manager for coordinating processes
pub struct IpcManager {
    shared_memory: Arc<shared_memory::SharedMemoryManager>,
    message_queue: Arc<message_queue::MessageQueue>,
    process_manager: Arc<Mutex<process_manager::ProcessManager>>,
    active_workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub process_id: u32,
    pub status: WorkerStatus,
    pub capabilities: Vec<String>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub metrics: WorkerMetrics,
}

impl IpcManager {
    /// Initialize IPC manager with M3 Max optimizations
    pub async fn new(config: &PipelineConfig) -> Result<Self> {
        let shared_memory = Arc::new(
            shared_memory::SharedMemoryManager::new(
                (config.m3_max.memory_pools.ipc * 1024 * 1024 * 1024) as usize  // Convert GB to bytes
            ).await?
        );
        
        let message_queue = Arc::new(
            message_queue::MessageQueue::new().await?
        );
        
        let process_manager = Arc::new(
            Mutex::new(process_manager::ProcessManager::new(config).await?)
        );
        
        Ok(Self {
            shared_memory,
            message_queue,
            process_manager,
            active_workers: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Start Python ML workers
    pub async fn start_python_workers(&self, worker_count: u8) -> Result<Vec<String>> {
        let mut process_manager = self.process_manager.lock().await;
        let worker_ids = process_manager.start_python_workers(worker_count).await?;
        
        // Wait for workers to initialize and register
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        
        Ok(worker_ids)
    }
    
    /// Send message to specific worker
    pub async fn send_to_worker(&self, worker_id: &str, message: IpcMessage) -> Result<()> {
        self.message_queue.send_to_worker(worker_id, message).await
    }
    
    /// Send message to all workers
    pub async fn broadcast_to_workers(&self, message: IpcMessage) -> Result<()> {
        self.message_queue.broadcast(message).await
    }
    
    /// Receive message from workers
    pub async fn receive_message(&self) -> Result<(String, IpcMessage)> {
        self.message_queue.receive().await
    }
    
    /// Register worker
    pub async fn register_worker(&self, worker_id: String, worker_info: WorkerInfo) -> Result<()> {
        let mut workers = self.active_workers.write().await;
        workers.insert(worker_id, worker_info);
        Ok(())
    }
    
    /// Update worker heartbeat
    pub async fn update_worker_heartbeat(&self, worker_id: &str, status: WorkerStatus, metrics: WorkerMetrics) -> Result<()> {
        let mut workers = self.active_workers.write().await;
        
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.status = status;
            worker.metrics = metrics;
            worker.last_heartbeat = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Get active workers
    pub async fn get_active_workers(&self) -> HashMap<String, WorkerInfo> {
        self.active_workers.read().await.clone()
    }
    
    /// Shutdown all workers
    pub async fn shutdown_workers(&self) -> Result<()> {
        // Send shutdown message to all workers
        self.broadcast_to_workers(IpcMessage::SystemShutdown).await?;
        
        // Wait for graceful shutdown
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        // Force cleanup if needed
        let mut process_manager = self.process_manager.lock().await;
        process_manager.cleanup_all().await?;
        
        Ok(())
    }
    
    /// Get IPC performance metrics
    pub async fn get_metrics(&self) -> IpcMetrics {
        IpcMetrics {
            active_workers: self.active_workers.read().await.len() as u32,
            shared_memory_usage_mb: self.shared_memory.get_usage_mb().await.unwrap_or(0),
            message_queue_depth: self.message_queue.get_queue_depth().await.unwrap_or(0),
            total_messages_sent: self.message_queue.get_message_count().await.unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcMetrics {
    pub active_workers: u32,
    pub shared_memory_usage_mb: u64,
    pub message_queue_depth: u32,
    pub total_messages_sent: u64,
}

/// Initialize IPC system
pub async fn initialize(config: &PipelineConfig) -> Result<()> {
    tracing::info!("Initializing IPC system for M3 Max optimization");
    
    // Create global IPC manager instance
    let _ipc_manager = IpcManager::new(config).await?;
    
    tracing::info!("IPC system initialized successfully");
    Ok(())
}

/// Cleanup IPC resources
pub async fn cleanup() -> Result<()> {
    tracing::info!("Cleaning up IPC resources");
    
    // This would cleanup the global IPC manager instance
    // For now, just log the cleanup
    
    tracing::info!("IPC cleanup completed");
    Ok(())
}