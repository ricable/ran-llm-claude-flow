use serde::{Deserialize, Serialize};
// use std::collections::HashMap; // Unused
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Document formats supported by the pipeline
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentFormat {
    Markdown,
    Html,
    Pdf,
    Csv,
    Gpp3, // 3GPP specifications
    Text,
}

/// Processing complexity levels for model selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Fast,     // Use Qwen3-1.7B
    Balanced, // Use Qwen3-7B  
    Quality,  // Use Qwen3-30B
}

/// Document structure after preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub path: PathBuf,
    pub format: DocumentFormat,
    pub content: String,
    pub metadata: DocumentMetadata,
    pub size_bytes: usize,
    pub created_at: DateTime<Utc>,
}

/// Metadata extracted from documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub feature_name: Option<String>,
    pub product_info: Option<String>,
    pub feature_state: Option<String>,
    pub parameters: Vec<Parameter>,
    pub counters: Vec<Counter>,
    pub technical_terms: Vec<String>,
    pub complexity_hints: ComplexityHints,
}

/// RAN parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub mo_class: Option<String>,
    pub data_type: Option<String>,
    pub valid_values: Option<String>,
    pub default_value: Option<String>,
    pub description: Option<String>,
}

/// Performance counter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub name: String,
    pub description: Option<String>,
    pub mo_class: Option<String>,
    pub counter_type: Option<String>,
}

/// Hints for complexity assessment and model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityHints {
    pub parameter_count: usize,
    pub counter_count: usize,
    pub technical_term_density: f64,
    pub content_length: usize,
    pub estimated_complexity: ComplexityLevel,
}

/// Document after Rust preprocessing, ready for ML processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub document: Document,
    pub structural_quality: StructuralQuality,
    pub processing_hints: ProcessingHints,
    pub checksum: u32,
}

/// Structural quality assessment (fast Rust validation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralQuality {
    pub completeness_score: f64,
    pub parameter_extraction_quality: f64,
    pub counter_extraction_quality: f64,
    pub technical_density_score: f64,
    pub overall_score: f64,
}

/// Hints for Python ML processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingHints {
    pub recommended_model: ComplexityLevel,
    pub expected_qa_pairs: usize,
    pub processing_priority: ProcessingPriority,
    pub use_cache: bool,
    pub batch_with_similar: bool,
    pub batch_processing_eligible: bool,
    pub expected_processing_time: Duration,
    pub memory_optimization: MemoryOptimization,
}

/// Processing priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Memory optimization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryOptimization {
    Standard,
    M3MaxUnified,
    HighThroughput,
    LowLatency,
}

/// Document content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentContent {
    Text(String),
    Binary(Vec<u8>),
    Reference(String),
}

/// Processing metadata for documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub source_type: String,
    pub processing_timestamp: DateTime<Utc>,
    pub quality_score: f64,
    pub estimated_complexity: f64,
    pub document_type: String,
    pub tokens_processed: usize,
    pub memory_used_mb: f64,
}

/// Error types for the Rust core system
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IPC communication error: {0}")]
    IpcError(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("CPU affinity error: {0}")]
    CPUAffinityError(String),
    
    #[error("Memory pool not found: {0}")]
    MemoryPoolNotFound(u32),
    
    #[error("Memory segment not found: {0}")]
    SegmentNotFound(String),
    
    #[error("Insufficient memory: requested {requested}GB, available {available}GB in pool {pool_type}")]
    InsufficientMemory {
        requested: usize,
        available: usize,
        pool_type: String,
    },
    
    #[error("Document processing error: {0}")]
    ProcessingError(String),
    
    #[error("Quality validation failed: {0}")]
    QualityError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Generic error: {0}")]
    Other(String),
}

/// Performance metrics for the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_docs_per_hour: f64,
    pub average_processing_time_ms: f64,
    pub memory_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub error_rate_percent: f64,
    pub quality_score_average: f64,
}


/// ML processing request sent to Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLProcessingRequest {
    pub request_id: Uuid,
    pub document: ProcessedDocument,
    pub processing_options: MLProcessingOptions,
    pub created_at: DateTime<Utc>,
}

/// Options for ML processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLProcessingOptions {
    pub model_preference: Option<ComplexityLevel>,
    pub max_qa_pairs: Option<usize>,
    pub quality_threshold: f64,
    pub enable_diversity_enhancement: bool,
    pub batch_processing: bool,
}

/// Response from Python ML processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLProcessingResponse {
    pub request_id: Uuid,
    pub qa_pairs: Vec<QAPair>,
    pub semantic_quality: SemanticQuality,
    pub processing_metadata: MLProcessingMetadata,
    pub model_used: String,
    pub processing_time: Duration,
}

/// Generated question-answer pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAPair {
    pub id: Uuid,
    pub question: String,
    pub answer: String,
    pub context: Option<String>,
    pub confidence: f64,
    pub metadata: QAMetadata,
}

/// Metadata for QA pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAMetadata {
    pub question_type: QuestionType,
    pub technical_terms: Vec<String>,
    pub parameters_mentioned: Vec<String>,
    pub counters_mentioned: Vec<String>,
    pub complexity_level: ComplexityLevel,
}

/// Types of generated questions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    Factual,
    Conceptual,
    Procedural,
    Analytical,
    Comparative,
}

/// Semantic quality assessment from Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticQuality {
    pub coherence_score: f64,
    pub relevance_score: f64,
    pub technical_accuracy_score: f64,
    pub diversity_score: f64,
    pub overall_score: f64,
}

/// ML processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLProcessingMetadata {
    pub model_name: String,
    pub model_version: String,
    pub inference_time: Duration,
    pub tokens_processed: usize,
    pub memory_used_mb: usize,
    pub gpu_utilization: Option<f64>,
}

/// Final processed result combining Rust and Python processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub document_id: Uuid,
    pub original_document: Document,
    pub structural_quality: StructuralQuality,
    pub semantic_quality: SemanticQuality,
    pub qa_pairs: Vec<QAPair>,
    pub combined_quality_score: f64,
    pub processing_stats: ProcessingStats,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub rust_processing_time: Duration,
    pub ml_processing_time: Duration,
    pub ipc_overhead_time: Duration,
    pub total_processing_time: Duration,
    pub memory_peak_mb: usize,
    pub model_used: String,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub documents_processed: usize,
    pub total_qa_pairs_generated: usize,
    pub average_quality: f64,
    pub total_time: Duration,
    pub memory_peak_gb: f64,
    pub errors_encountered: usize,
}

impl PipelineStats {
    pub fn throughput_per_hour(&self) -> f64 {
        if self.total_time.as_secs() == 0 {
            return 0.0;
        }
        (self.documents_processed as f64) * 3600.0 / (self.total_time.as_secs() as f64)
    }
}

/// Error types for the pipeline
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("Document processing error: {0}")]
    DocumentProcessing(#[from] std::io::Error),
    
    #[error("IPC communication error: {0}")]
    IpcError(String),
    
    #[error("Quality validation failed: {0}")]
    QualityValidation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
}

/// Result type for pipeline operations
pub type PipelineResult<T> = Result<T, PipelineError>;

/// M3 Max processing statistics for optimization tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3ProcessingStats {
    pub successful_documents: usize,
    pub processing_time: Duration,
    pub peak_memory_gb: f64,
    pub average_memory_per_doc_mb: usize,
    pub max_memory_delta_mb: usize,
    pub concurrency_used: usize,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            documents_processed: 0,
            total_qa_pairs_generated: 0,
            average_quality: 0.0,
            total_time: Duration::from_secs(0),
            memory_peak_gb: 0.0,
            errors_encountered: 0,
        }
    }
}