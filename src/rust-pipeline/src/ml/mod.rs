/*!
# ML Module

Machine Learning integration module for Rust-Python hybrid pipeline.
Provides dynamic model selection and workload analysis for Qwen3 variants.
*/

pub mod memory_predictor;
pub mod model_selector;
pub mod performance_benchmark;
pub mod workload_analyzer;

pub use memory_predictor::{initialize as memory_predictor_initialize, get_statistics as memory_predictor_get_statistics, *};
pub use model_selector::{initialize as model_selector_initialize, get_statistics as model_selector_get_statistics, *};
pub use performance_benchmark::*;
pub use workload_analyzer::{initialize as workload_analyzer_initialize, get_statistics as workload_analyzer_get_statistics, *};

use crate::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::time::SystemTime;
use uuid::Uuid;

/// ML processing request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLRequest {
    pub request_id: Uuid,
    pub document_type: DocumentType,
    pub document_size_bytes: usize,
    pub complexity_score: f64,
    pub priority: Priority,
    pub quality_requirements: QualityRequirements,
    pub processing_deadline: Option<Duration>,
}

/// Document type enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DocumentType {
    /// Simple text documents
    PlainText,
    /// Markdown documents with formatting
    Markdown,
    /// PDF documents requiring OCR
    Pdf,
    /// CSV data files
    Csv,
    /// Technical specifications
    Technical,
    /// 3GPP standards documents
    Standards3Gpp,
    /// Ericsson documentation
    EricssonDoc,
}

/// Processing priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality requirements for ML processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub min_score: f64,
    pub consistency_target: f64,
    pub accuracy_threshold: f64,
    pub enable_validation: bool,
}

/// ML processing response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLResponse {
    pub request_id: Uuid,
    pub model_used: Qwen3Model,
    pub processing_time: Duration,
    pub quality_score: f64,
    pub memory_used_mb: u64,
    pub status: ProcessingStatus,
    pub error_message: Option<String>,
}

/// Processing status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessingStatus {
    Success,
    QualityFailure,
    TimeoutFailure,
    MemoryFailure,
    ModelFailure,
    SystemFailure,
}

/// Qwen3 model variants
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Qwen3Model {
    /// Qwen3 1.7B - Fast processing for simple tasks
    Qwen3_1_7B,
    /// Qwen3 7B - Balanced processing for general tasks
    Qwen3_7B,
    /// Qwen3 30B - High-quality processing for complex tasks
    Qwen3_30B,
}

impl Qwen3Model {
    /// Get model specifications
    pub fn specs(&self) -> ModelSpecs {
        match self {
            Qwen3Model::Qwen3_1_7B => ModelSpecs {
                parameters: 1_700_000_000,
                memory_gb: 12,
                tokens_per_second: 150,
                quality_score: 0.72,
                use_cases: vec![
                    "embedding_generation".to_string(),
                    "simple_extraction".to_string(),
                    "text_classification".to_string(),
                ],
            },
            Qwen3Model::Qwen3_7B => ModelSpecs {
                parameters: 7_000_000_000,
                memory_gb: 28,
                tokens_per_second: 80,
                quality_score: 0.82,
                use_cases: vec![
                    "conversation_generation".to_string(),
                    "balanced_processing".to_string(),
                    "moderate_analysis".to_string(),
                ],
            },
            Qwen3Model::Qwen3_30B => ModelSpecs {
                parameters: 30_000_000_000,
                memory_gb: 45,
                tokens_per_second: 25,
                quality_score: 0.92,
                use_cases: vec![
                    "complex_analysis".to_string(),
                    "quality_assessment".to_string(),
                    "research_tasks".to_string(),
                ],
            },
        }
    }

    /// Get model name string
    pub fn name(&self) -> &'static str {
        match self {
            Qwen3Model::Qwen3_1_7B => "qwen3-1.7b",
            Qwen3Model::Qwen3_7B => "qwen3-7b",
            Qwen3Model::Qwen3_30B => "qwen3-30b",
        }
    }

    /// Get all available models
    pub fn all_models() -> Vec<Qwen3Model> {
        vec![
            Qwen3Model::Qwen3_1_7B,
            Qwen3Model::Qwen3_7B,
            Qwen3Model::Qwen3_30B,
        ]
    }
}

/// Model specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpecs {
    pub parameters: u64,
    pub memory_gb: u32,
    pub tokens_per_second: u32,
    pub quality_score: f64,
    pub use_cases: Vec<String>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model: Qwen3Model,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub average_processing_time: Duration,
    pub average_quality_score: f64,
    pub memory_efficiency: f64,
    pub throughput_docs_per_hour: f64,
    pub error_rate: f64,
    pub last_used: SystemTime,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            model: Qwen3Model::Qwen3_7B,
            total_requests: 0,
            successful_requests: 0,
            average_processing_time: Duration::from_secs(0),
            average_quality_score: 0.0,
            memory_efficiency: 0.0,
            throughput_docs_per_hour: 0.0,
            error_rate: 0.0,
            last_used: SystemTime::now(),
        }
    }
}

/// Initialize ML module
pub async fn initialize_ml_module() -> Result<()> {
    tracing::info!("Initializing ML module with dynamic model selection");

    // Initialize model selector
    model_selector::initialize().await?;

    // Initialize workload analyzer
    workload_analyzer::initialize().await?;

    // Initialize performance benchmark
    performance_benchmark::initialize().await?;

    // Initialize memory predictor
    memory_predictor::initialize().await?;

    tracing::info!("ML module initialization complete");
    Ok(())
}

/// Process ML request with dynamic model selection
pub async fn process_request(request: MLRequest) -> Result<MLResponse> {
    let start_time = SystemTime::now();

    tracing::debug!("Processing ML request: {:?}", request.request_id);

    // Analyze workload to determine optimal model
    let workload = workload_analyzer::analyze(&request).await?;

    // Select optimal model based on workload analysis
    let selected_model = model_selector::select_model(&request, &workload).await?;

    tracing::info!(
        "Selected model {} for request {}",
        selected_model.name(),
        request.request_id
    );

    // Process request with selected model
    let response = execute_ml_processing(request, selected_model, start_time).await?;

    // Update model metrics
    model_selector::update_metrics(&response).await?;

    Ok(response)
}

/// Execute ML processing with specified model
async fn execute_ml_processing(
    request: MLRequest,
    model: Qwen3Model,
    start_time: SystemTime,
) -> Result<MLResponse> {
    // Simulate ML processing (in real implementation, this would call Python workers)
    let processing_time = Duration::from_millis(100);
    tokio::time::sleep(processing_time).await;

    let specs = model.specs();
    let quality_score = specs.quality_score + (rand::random::<f64>() - 0.5) * 0.1;
    let quality_score = quality_score.max(0.0).min(1.0);

    let status = if quality_score >= request.quality_requirements.min_score {
        ProcessingStatus::Success
    } else {
        ProcessingStatus::QualityFailure
    };

    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));

    Ok(MLResponse {
        request_id: request.request_id,
        model_used: model,
        processing_time: elapsed,
        quality_score,
        memory_used_mb: specs.memory_gb as u64 * 1024,
        status,
        error_message: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_model_specs() {
        let model = Qwen3Model::Qwen3_7B;
        let specs = model.specs();
        assert_eq!(specs.parameters, 7_000_000_000);
        assert_eq!(specs.memory_gb, 28);
        assert!(specs.quality_score > 0.8);
    }

    #[tokio::test]
    async fn test_ml_request_processing() {
        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::PlainText,
            document_size_bytes: 1024,
            complexity_score: 0.5,
            priority: Priority::Medium,
            quality_requirements: QualityRequirements {
                min_score: 0.7,
                consistency_target: 0.75,
                accuracy_threshold: 0.8,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(300)),
        };

        // Test that we can create a valid ML request
        assert_eq!(request.document_type, DocumentType::PlainText);
        assert_eq!(request.priority, Priority::Medium);
    }
}
