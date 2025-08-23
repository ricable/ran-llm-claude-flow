//! Test helpers and utilities for ML integration testing

use rust_core::types::*;
use uuid::Uuid;
use std::time::Duration;
use chrono::Utc;

/// Create a test document with specified complexity
pub fn create_test_document(complexity: ComplexityLevel) -> ProcessedDocument {
    let parameter_count = match complexity {
        ComplexityLevel::Fast => 2,
        ComplexityLevel::Balanced => 8,
        ComplexityLevel::Quality => 15,
    };
    
    ProcessedDocument {
        document: Document {
            id: Uuid::new_v4(),
            path: std::path::PathBuf::from("test_document.md"),
            format: DocumentFormat::Markdown,
            content: "# Test Feature\n\nThis is a test document for ML processing.".to_string(),
            metadata: DocumentMetadata {
                title: Some("Test Feature".to_string()),
                feature_name: Some("TestFeature".to_string()),
                product_info: Some("Test Product".to_string()),
                feature_state: Some("Active".to_string()),
                parameters: vec![Parameter {
                    name: "testParameter".to_string(),
                    mo_class: Some("TestMO".to_string()),
                    data_type: Some("boolean".to_string()),
                    valid_values: Some("true, false".to_string()),
                    default_value: Some("false".to_string()),
                    description: Some("Test parameter description".to_string()),
                }],
                counters: vec![Counter {
                    name: "pmTestCounter".to_string(),
                    description: Some("Test counter description".to_string()),
                    mo_class: Some("TestMO".to_string()),
                    counter_type: Some("gauge".to_string()),
                }],
                technical_terms: vec!["LTE".to_string(), "QoS".to_string()],
                complexity_hints: ComplexityHints {
                    parameter_count,
                    counter_count: 2,
                    technical_term_density: 2.5,
                    content_length: 1000,
                    estimated_complexity: complexity.clone(),
                },
            },
            size_bytes: 1000,
            created_at: Utc::now(),
        },
        structural_quality: StructuralQuality {
            completeness_score: 0.85,
            parameter_extraction_quality: 0.90,
            counter_extraction_quality: 0.88,
            technical_density_score: 0.82,
            overall_score: 0.86,
        },
        processing_hints: ProcessingHints {
            recommended_model: complexity,
            expected_qa_pairs: 5,
            processing_priority: ProcessingPriority::Normal,
            use_cache: true,
            batch_with_similar: true,
            batch_processing_eligible: true,
            expected_processing_time: Duration::from_millis(2000),
            memory_optimization: MemoryOptimization::M3MaxUnified,
        },
        checksum: 12345678,
    }
}

/// Temporary placeholder for missing ML integration config types
#[derive(Debug, Clone)]
pub struct MLIntegrationConfig {
    pub enable_dynamic_model_selection: bool,
    pub batch_processing_enabled: bool,
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub quality_enhancement_enabled: bool,
    pub performance_tracking_enabled: bool,
    pub adaptive_complexity_enabled: bool,
    pub m3_max_optimization: M3MaxMLConfig,
}

#[derive(Debug, Clone)]
pub struct M3MaxMLConfig {
    pub use_unified_memory: bool,
    pub max_concurrent_models: usize,
    pub model_cache_size_gb: usize,
    pub enable_simd_acceleration: bool,
    pub optimize_for_throughput: bool,
}

/// Create test ML integration configuration
pub fn create_test_config() -> MLIntegrationConfig {
    MLIntegrationConfig {
        enable_dynamic_model_selection: true,
        batch_processing_enabled: true,
        max_batch_size: 4,
        batch_timeout_ms: 1000,
        quality_enhancement_enabled: true,
        performance_tracking_enabled: true,
        adaptive_complexity_enabled: true,
        m3_max_optimization: M3MaxMLConfig {
            use_unified_memory: true,
            max_concurrent_models: 2,
            model_cache_size_gb: 8,
            enable_simd_acceleration: true,
            optimize_for_throughput: true,
        },
    }
}