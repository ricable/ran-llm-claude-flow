//! Mock factories for testing ML integration components

use rust_core::types::*;
use uuid::Uuid;
use std::time::Duration;

/// Mock IPC Manager for testing
pub struct MockIpcManager {
    pub response_delay: Duration,
    pub should_fail: bool,
}

impl MockIpcManager {
    pub fn new(response_delay: Duration, should_fail: bool) -> Self {
        Self { response_delay, should_fail }
    }
}

/// Create mock ML response
pub fn create_mock_ml_response() -> MLProcessingResponse {
    MLProcessingResponse {
        request_id: Uuid::new_v4(),
        qa_pairs: vec![
            QAPair {
                id: Uuid::new_v4(),
                question: "What is carrier aggregation?".to_string(),
                answer: "Carrier aggregation combines multiple carriers for increased bandwidth.".to_string(),
                context: Some("LTE feature context".to_string()),
                confidence: 0.85,
                metadata: QAMetadata {
                    question_type: QuestionType::Factual,
                    technical_terms: vec!["carrier aggregation".to_string(), "bandwidth".to_string()],
                    parameters_mentioned: vec![],
                    counters_mentioned: vec![],
                    complexity_level: ComplexityLevel::Balanced,
                },
            },
        ],
        semantic_quality: SemanticQuality {
            coherence_score: 0.82,
            relevance_score: 0.85,
            technical_accuracy_score: 0.88,
            diversity_score: 0.75,
            overall_score: 0.825,
        },
        processing_metadata: MLProcessingMetadata {
            model_name: "qwen3-7b-mlx".to_string(),
            model_version: "v1.0.0".to_string(),
            inference_time: Duration::from_millis(2000),
            tokens_processed: 150,
            memory_used_mb: 4096,
            gpu_utilization: Some(0.75),
        },
        model_used: "qwen3-7b-mlx".to_string(),
        processing_time: Duration::from_millis(2500),
    }
}