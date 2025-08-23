/*!
# Model Context Protocol (MCP) Implementation

MCP server, client, and host for coordinating Rust-Python pipeline operations.
*/

pub mod server;
pub mod client; 
pub mod host;
pub mod protocol;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// MCP message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpMessage {
    // Pipeline control messages
    PipelineStart { config: PipelineStartConfig },
    PipelineStop { pipeline_id: Uuid },
    PipelineStatus { pipeline_id: Uuid },
    
    // Task orchestration messages  
    TaskSubmit { task: TaskDefinition },
    TaskStatus { task_id: Uuid },
    TaskResult { task_id: Uuid, result: TaskResult },
    TaskCancel { task_id: Uuid },
    
    // Model management messages
    ModelLoad { model_id: String, variant: ModelVariant },
    ModelUnload { model_id: String },
    ModelSwitch { from_model: String, to_model: String },
    ModelStatus { model_id: Option<String> },
    
    // Performance monitoring messages
    MetricsRequest { component: Option<String> },
    MetricsResponse { metrics: PerformanceMetrics },
    AlertNotification { alert: Alert },
    
    // IPC coordination messages
    IpcRegister { process_id: String, capabilities: Vec<String> },
    IpcMessage { from: String, to: String, payload: Vec<u8> },
    IpcBroadcast { sender: String, payload: Vec<u8> },
    
    // System control messages
    HealthCheck,
    SystemShutdown,
    
    // Response messages
    Success { request_id: Uuid, data: Option<serde_json::Value> },
    Error { request_id: Uuid, error: McpError },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStartConfig {
    pub pipeline_id: Uuid,
    pub input_sources: Vec<String>,
    pub output_destination: String,
    pub quality_threshold: f64,
    pub model_preferences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub task_id: Uuid,
    pub task_type: TaskType,
    pub input_data: serde_json::Value,
    pub model_requirements: ModelRequirements,
    pub priority: TaskPriority,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    DocumentProcessing,
    FeatureExtraction,
    QualityValidation,
    ConversationGeneration,
    ModelInference,
    PerformanceAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub min_model_size: ModelSize,
    pub preferred_models: Vec<String>,
    pub max_memory_gb: u32,
    pub require_local: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Small,    // qwen3-1.7b
    Medium,   // qwen3-7b  
    Large,    // qwen3-30b
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2, 
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: Uuid,
    pub status: TaskStatus,
    pub result_data: Option<serde_json::Value>,
    pub performance_metrics: TaskMetrics,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub execution_time_ms: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub quality_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelVariant {
    Qwen3_1_7B,
    Qwen3_7B,
    Qwen3_30B,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_metrics: SystemMetrics,
    pub pipeline_metrics: PipelineMetrics,
    pub model_metrics: HashMap<String, ModelMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub memory_usage_percent: f64,
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: f64,
    pub disk_io_mbps: f64,
    pub network_io_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub documents_processed: u64,
    pub average_processing_time_ms: f64,
    pub quality_score_average: f64,
    pub throughput_docs_per_hour: f64,
    pub error_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_id: String,
    pub inference_count: u64,
    pub average_inference_time_ms: f64,
    pub memory_usage_mb: u64,
    pub accuracy_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_id: Uuid,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: u32,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

// Standard MCP error codes
impl McpError {
    pub fn invalid_request(message: &str) -> Self {
        Self {
            code: 400,
            message: message.to_string(),
            details: None,
        }
    }
    
    pub fn not_found(message: &str) -> Self {
        Self {
            code: 404,
            message: message.to_string(),
            details: None,
        }
    }
    
    pub fn internal_error(message: &str) -> Self {
        Self {
            code: 500,
            message: message.to_string(),
            details: None,
        }
    }
    
    pub fn timeout_error(message: &str) -> Self {
        Self {
            code: 408,
            message: message.to_string(),
            details: None,
        }
    }
    
    pub fn resource_exhausted(message: &str) -> Self {
        Self {
            code: 503,
            message: message.to_string(),
            details: None,
        }
    }
}

/// MCP connection information
#[derive(Debug, Clone)]
pub struct McpConnection {
    pub connection_id: Uuid,
    pub client_type: ClientType,
    pub capabilities: Vec<String>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientType {
    RustCore,
    PythonWorker,
    MonitoringAgent,
    OrchestrationHost,
    ExternalTool,
}