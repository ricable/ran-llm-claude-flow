/*!
# Core Pipeline Components

High-performance document processing core optimized for M3 Max hardware.
*/

pub mod document_processor;
pub mod performance_monitor;
pub mod pipeline_coordinator;
pub mod quality_validator;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Document processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub document_id: Uuid,
    pub processing_time_ms: u64,
    pub quality_score: f64,
    pub tokens_processed: u64,
    pub extraction_results: ExtractionResults,
    pub metadata: DocumentMetadata,
}

/// Document extraction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResults {
    pub features: Vec<Feature>,
    pub parameters: Vec<Parameter>,
    pub commands: Vec<Command>,
    pub procedures: Vec<Procedure>,
    pub troubleshooting: Vec<TroubleshootingStep>,
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
    pub value_type: String,
    pub default_value: Option<String>,
    pub range: Option<String>,
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
pub struct TroubleshootingStep {
    pub issue: String,
    pub solution: String,
    pub severity: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub title: String,
    pub section: Option<String>,
    pub page: Option<u32>,
    pub url: Option<String>,
    pub confidence: f64,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub file_path: PathBuf,
    pub file_size_bytes: u64,
    pub content_type: String,
    pub processing_stage: ProcessingStage,
    pub quality_metrics: QualityMetrics,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    Raw,
    Converted,
    Preprocessed,
    Extracted,
    Validated,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_score: f64,
    pub content_quality: f64,
    pub extraction_confidence: f64,
    pub consistency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub processing_time_ms: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub tokens_per_second: f64,
}
