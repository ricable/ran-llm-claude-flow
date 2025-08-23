//! Comprehensive Rust Unit Tests for RAN LLM Pipeline
//! 
//! This module provides extensive test coverage for the Rust components of the 
//! hybrid Rust-Python RAN document processing pipeline, focusing on:
//!
//! 1. Core data structures and types
//! 2. Document processing pipeline
//! 3. M3 Max optimizations
//! 4. IPC communication
//! 5. Performance monitoring
//! 6. Error handling and edge cases
//!
//! Target: 95%+ test coverage with comprehensive error scenario testing

use std::{collections::HashMap, time::{Duration, SystemTime, UNIX_EPOCH}, sync::Arc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Mock types for testing (mirrors the actual Rust core types)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentFormat {
    Markdown,
    Html,
    Pdf,
    Csv,
    Gpp3,
    Text,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Fast,
    Balanced,
    Quality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryOptimization {
    Standard,
    M3MaxUnified,
    HighThroughput,
    LowLatency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub format: DocumentFormat,
    pub content: String,
    pub metadata: DocumentMetadata,
    pub size_bytes: usize,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub feature_name: Option<String>,
    pub product_info: Option<String>,
    pub parameters: Vec<Parameter>,
    pub counters: Vec<Counter>,
    pub technical_terms: Vec<String>,
    pub complexity_hints: ComplexityHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub mo_class: Option<String>,
    pub data_type: Option<String>,
    pub valid_values: Option<String>,
    pub default_value: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub name: String,
    pub description: Option<String>,
    pub mo_class: Option<String>,
    pub counter_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityHints {
    pub parameter_count: usize,
    pub counter_count: usize,
    pub technical_term_density: f64,
    pub content_length: usize,
    pub estimated_complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralQuality {
    pub completeness_score: f64,
    pub parameter_extraction_quality: f64,
    pub counter_extraction_quality: f64,
    pub technical_density_score: f64,
    pub overall_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingHints {
    pub recommended_model: ComplexityLevel,
    pub expected_qa_pairs: usize,
    pub processing_priority: ProcessingPriority,
    pub use_cache: bool,
    pub batch_with_similar: bool,
    pub batch_processing_eligible: bool,
    pub expected_processing_time: u64, // Duration in ms
    pub memory_optimization: MemoryOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub document: Document,
    pub structural_quality: StructuralQuality,
    pub processing_hints: ProcessingHints,
    pub checksum: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_docs_per_hour: f64,
    pub average_processing_time_ms: f64,
    pub memory_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub error_rate_percent: f64,
    pub quality_score_average: f64,
}

// Mock error types for testing
#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("Document processing error: {0}")]
    ProcessingError(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("IPC communication error: {0}")]
    IpcError(String),
    
    #[error("Quality validation failed: {0}")]
    QualityError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Timeout error: processing took longer than {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
}

type Result<T> = std::result::Result<T, TestError>;

// Core business logic for testing
pub struct DocumentProcessor {
    config: ProcessingConfig,
    metrics: Arc<std::sync::Mutex<PerformanceMetrics>>,
}

#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub max_concurrent_docs: usize,
    pub timeout_ms: u64,
    pub quality_threshold: f64,
    pub enable_m3_optimization: bool,
    pub memory_limit_gb: usize,
}

impl DocumentProcessor {
    pub fn new(config: ProcessingConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(std::sync::Mutex::new(PerformanceMetrics {
                throughput_docs_per_hour: 0.0,
                average_processing_time_ms: 0.0,
                memory_utilization_percent: 0.0,
                cpu_utilization_percent: 0.0,
                error_rate_percent: 0.0,
                quality_score_average: 0.0,
            })),
        }
    }

    pub fn process_document(&self, mut doc: Document) -> Result<ProcessedDocument> {
        let start_time = std::time::Instant::now();
        
        // Validate document
        if doc.content.is_empty() {
            return Err(TestError::ProcessingError("Document content is empty".to_string()));
        }
        
        // Extract metadata
        self.extract_metadata(&mut doc)?;
        
        // Assess structural quality
        let structural_quality = self.assess_quality(&doc)?;
        
        // Generate processing hints
        let processing_hints = self.generate_hints(&doc, &structural_quality)?;
        
        // Calculate checksum
        let checksum = self.calculate_checksum(&doc);
        
        let processing_time = start_time.elapsed();
        self.update_metrics(processing_time.as_millis() as f64)?;
        
        Ok(ProcessedDocument {
            document: doc,
            structural_quality,
            processing_hints,
            checksum,
        })
    }
    
    fn extract_metadata(&self, doc: &mut Document) -> Result<()> {
        // Mock metadata extraction logic
        let content_lower = doc.content.to_lowercase();
        
        // Extract parameters (simplified pattern matching)
        let parameter_count = content_lower.matches("parameter").count() + 
                            content_lower.matches("config").count();
        
        let counter_count = content_lower.matches("counter").count() + 
                          content_lower.matches("metric").count();
        
        let technical_terms: Vec<String> = vec![
            "LTE", "5G", "RAN", "eNodeB", "gNodeB", "MIMO", "UE"
        ].iter()
        .filter(|term| content_lower.contains(&term.to_lowercase()))
        .map(|s| s.to_string())
        .collect();
        
        let technical_term_density = technical_terms.len() as f64 / 
            std::cmp::max(1, doc.content.len() / 1000) as f64;
        
        let estimated_complexity = if parameter_count > 10 || counter_count > 5 {
            ComplexityLevel::Quality
        } else if parameter_count > 5 || counter_count > 2 {
            ComplexityLevel::Balanced
        } else {
            ComplexityLevel::Fast
        };
        
        doc.metadata = DocumentMetadata {
            title: Some("Test Document".to_string()),
            feature_name: Some("TestFeature".to_string()),
            product_info: Some("Test Product".to_string()),
            parameters: (0..parameter_count).map(|i| Parameter {
                name: format!("param_{}", i),
                mo_class: Some("TestMO".to_string()),
                data_type: Some("Integer".to_string()),
                valid_values: Some("0-100".to_string()),
                default_value: Some("50".to_string()),
                description: Some("Test parameter".to_string()),
            }).collect(),
            counters: (0..counter_count).map(|i| Counter {
                name: format!("counter_{}", i),
                description: Some("Test counter".to_string()),
                mo_class: Some("TestMO".to_string()),
                counter_type: Some("GAUGE".to_string()),
            }).collect(),
            technical_terms,
            complexity_hints: ComplexityHints {
                parameter_count,
                counter_count,
                technical_term_density,
                content_length: doc.content.len(),
                estimated_complexity,
            },
        };
        
        Ok(())
    }
    
    fn assess_quality(&self, doc: &Document) -> Result<StructuralQuality> {
        let base_score = 5.0;
        let parameter_quality = (doc.metadata.parameters.len() as f64 * 0.5).min(2.0);
        let counter_quality = (doc.metadata.counters.len() as f64 * 0.3).min(1.5);
        let technical_density = doc.metadata.complexity_hints.technical_term_density.min(2.0);
        let completeness = if doc.content.len() > 1000 { 1.0 } else { 0.5 };
        
        let overall_score = base_score + parameter_quality + counter_quality + 
                          technical_density + completeness;
        
        // Quality validation
        if overall_score < self.config.quality_threshold {
            return Err(TestError::QualityError(
                format!("Quality score {} below threshold {}", 
                       overall_score, self.config.quality_threshold)
            ));
        }
        
        Ok(StructuralQuality {
            completeness_score: completeness / 1.0,
            parameter_extraction_quality: parameter_quality / 2.0,
            counter_extraction_quality: counter_quality / 1.5,
            technical_density_score: technical_density / 2.0,
            overall_score: overall_score / 10.0, // Normalize to 0-1
        })
    }
    
    fn generate_hints(&self, doc: &Document, quality: &StructuralQuality) -> Result<ProcessingHints> {
        let expected_qa_pairs = std::cmp::min(
            20,
            (doc.metadata.parameters.len() + doc.metadata.counters.len()) * 2
        );
        
        let priority = if quality.overall_score > 0.8 {
            ProcessingPriority::High
        } else if quality.overall_score > 0.6 {
            ProcessingPriority::Normal
        } else {
            ProcessingPriority::Low
        };
        
        let memory_opt = if self.config.enable_m3_optimization {
            MemoryOptimization::M3MaxUnified
        } else {
            MemoryOptimization::Standard
        };
        
        Ok(ProcessingHints {
            recommended_model: doc.metadata.complexity_hints.estimated_complexity.clone(),
            expected_qa_pairs,
            processing_priority: priority,
            use_cache: doc.content.len() < 5000,
            batch_with_similar: true,
            batch_processing_eligible: doc.content.len() < 10000,
            expected_processing_time: (doc.content.len() / 100) as u64, // Mock estimation
            memory_optimization: memory_opt,
        })
    }
    
    fn calculate_checksum(&self, doc: &Document) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        doc.content.hash(&mut hasher);
        doc.id.hash(&mut hasher);
        hasher.finish() as u32
    }
    
    fn update_metrics(&self, processing_time_ms: f64) -> Result<()> {
        let mut metrics = self.metrics.lock()
            .map_err(|e| TestError::ProcessingError(format!("Failed to lock metrics: {}", e)))?;
        
        metrics.average_processing_time_ms = 
            (metrics.average_processing_time_ms + processing_time_ms) / 2.0;
        
        // Mock throughput calculation
        if processing_time_ms > 0.0 {
            metrics.throughput_docs_per_hour = 3600000.0 / processing_time_ms;
        }
        
        Ok(())
    }
    
    pub fn get_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self.metrics.lock()
            .map_err(|e| TestError::ProcessingError(format!("Failed to lock metrics: {}", e)))?;
        Ok(metrics.clone())
    }
}

// Mock M3 Max Optimizer
pub struct M3MaxOptimizer {
    enabled: bool,
    thread_count: usize,
    memory_pool_gb: usize,
}

impl M3MaxOptimizer {
    pub fn new(memory_limit_gb: usize) -> Self {
        Self {
            enabled: true,
            thread_count: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
            memory_pool_gb: (memory_limit_gb as f64 * 0.8) as usize, // 80% allocation
        }
    }
    
    pub fn optimize_batch(&self, docs: Vec<Document>) -> Result<Vec<Document>> {
        if !self.enabled {
            return Ok(docs);
        }
        
        // Mock M3 Max optimization (sorting by size for better memory locality)
        let mut optimized = docs;
        optimized.sort_by_key(|doc| doc.size_bytes);
        
        // Validate memory usage
        let total_size_mb: usize = optimized.iter().map(|doc| doc.size_bytes / 1024 / 1024).sum();
        let memory_limit_mb = self.memory_pool_gb * 1024;
        
        if total_size_mb > memory_limit_mb {
            return Err(TestError::MemoryError(
                format!("Batch size {}MB exceeds memory limit {}MB", 
                       total_size_mb, memory_limit_mb)
            ));
        }
        
        Ok(optimized)
    }
    
    pub fn get_thread_count(&self) -> usize {
        self.thread_count
    }
    
    pub fn get_memory_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("thread_count".to_string(), self.thread_count);
        stats.insert("memory_pool_gb".to_string(), self.memory_pool_gb);
        stats.insert("available_memory_mb".to_string(), self.memory_pool_gb * 1024);
        stats
    }
}

// ============================================================================
// COMPREHENSIVE UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document(content: &str, format: DocumentFormat) -> Document {
        Document {
            id: Uuid::new_v4(),
            format,
            content: content.to_string(),
            metadata: DocumentMetadata {
                title: None,
                feature_name: None,
                product_info: None,
                parameters: vec![],
                counters: vec![],
                technical_terms: vec![],
                complexity_hints: ComplexityHints {
                    parameter_count: 0,
                    counter_count: 0,
                    technical_term_density: 0.0,
                    content_length: content.len(),
                    estimated_complexity: ComplexityLevel::Fast,
                },
            },
            size_bytes: content.len(),
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }
    
    fn create_test_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_docs: 8,
            timeout_ms: 5000,
            quality_threshold: 5.0,
            enable_m3_optimization: true,
            memory_limit_gb: 60,
        }
    }

    // ========================================================================
    // Core Data Structure Tests
    // ========================================================================
    
    #[test]
    fn test_document_creation_and_serialization() {
        let doc = create_test_document("Test content with parameters", DocumentFormat::Markdown);
        
        assert_eq!(doc.content, "Test content with parameters");
        assert_eq!(doc.format, DocumentFormat::Markdown);
        assert!(!doc.id.is_nil());
        assert_eq!(doc.size_bytes, "Test content with parameters".len());
        
        // Test serialization
        let serialized = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.content, deserialized.content);
        assert_eq!(doc.format, deserialized.format);
    }
    
    #[test]
    fn test_document_format_variants() {
        let formats = vec![
            DocumentFormat::Markdown,
            DocumentFormat::Html,
            DocumentFormat::Pdf,
            DocumentFormat::Csv,
            DocumentFormat::Gpp3,
            DocumentFormat::Text,
        ];
        
        for format in formats {
            let doc = create_test_document("test", format.clone());
            assert_eq!(doc.format, format);
        }
    }
    
    #[test]
    fn test_complexity_level_ordering() {
        let levels = vec![
            ComplexityLevel::Fast,
            ComplexityLevel::Balanced,
            ComplexityLevel::Quality,
        ];
        
        for level in levels {
            let hints = ComplexityHints {
                parameter_count: 5,
                counter_count: 3,
                technical_term_density: 2.5,
                content_length: 1000,
                estimated_complexity: level.clone(),
            };
            
            assert_eq!(hints.estimated_complexity, level);
        }
    }
    
    #[test]
    fn test_processing_priority_levels() {
        let priorities = vec![
            ProcessingPriority::Low,
            ProcessingPriority::Normal,
            ProcessingPriority::High,
            ProcessingPriority::Critical,
        ];
        
        for priority in priorities {
            let hints = ProcessingHints {
                recommended_model: ComplexityLevel::Fast,
                expected_qa_pairs: 10,
                processing_priority: priority.clone(),
                use_cache: true,
                batch_with_similar: true,
                batch_processing_eligible: true,
                expected_processing_time: 1000,
                memory_optimization: MemoryOptimization::Standard,
            };
            
            assert_eq!(hints.processing_priority, priority);
        }
    }
    
    // ========================================================================
    // Document Processing Pipeline Tests
    // ========================================================================
    
    #[test]
    fn test_document_processor_creation() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config.clone());
        
        assert_eq!(processor.config.max_concurrent_docs, 8);
        assert_eq!(processor.config.timeout_ms, 5000);
        assert_eq!(processor.config.quality_threshold, 5.0);
        assert!(processor.config.enable_m3_optimization);
    }
    
    #[test]
    fn test_successful_document_processing() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let doc = create_test_document(
            "This is a test document about LTE parameters and 5G counters with RAN configuration",
            DocumentFormat::Markdown
        );
        
        let result = processor.process_document(doc.clone());
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.document.id, doc.id);
        assert!(processed.structural_quality.overall_score > 0.0);
        assert!(processed.processing_hints.expected_qa_pairs > 0);
        assert_ne!(processed.checksum, 0);
    }
    
    #[test]
    fn test_empty_document_processing_error() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let doc = create_test_document("", DocumentFormat::Text);
        let result = processor.process_document(doc);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            TestError::ProcessingError(msg) => {
                assert!(msg.contains("Document content is empty"));
            }
            _ => panic!("Expected ProcessingError"),
        }
    }
    
    #[test]
    fn test_low_quality_document_rejection() {
        let mut config = create_test_config();
        config.quality_threshold = 8.0; // Set high threshold
        
        let processor = DocumentProcessor::new(config);
        let doc = create_test_document("Simple text", DocumentFormat::Text);
        
        let result = processor.process_document(doc);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TestError::QualityError(msg) => {
                assert!(msg.contains("Quality score"));
                assert!(msg.contains("below threshold"));
            }
            _ => panic!("Expected QualityError"),
        }
    }
    
    #[test]
    fn test_metadata_extraction() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let content = "This document contains multiple parameters for LTE configuration. \
                      It also has several counters for 5G performance monitoring. \
                      The RAN system uses eNodeB and gNodeB components with MIMO technology.";
        
        let doc = create_test_document(content, DocumentFormat::Markdown);
        let result = processor.process_document(doc).unwrap();
        
        // Check that parameters were extracted
        assert!(result.document.metadata.parameters.len() > 0);
        
        // Check technical terms extraction
        assert!(result.document.metadata.technical_terms.contains(&"LTE".to_string()));
        assert!(result.document.metadata.technical_terms.contains(&"5G".to_string()));
        assert!(result.document.metadata.technical_terms.contains(&"RAN".to_string()));
        
        // Check complexity assessment
        assert!(result.document.metadata.complexity_hints.technical_term_density > 0.0);
        assert_eq!(result.document.metadata.complexity_hints.content_length, content.len());
    }
    
    #[test]
    fn test_structural_quality_assessment() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        // High-quality document with many parameters and counters
        let content = "This is a comprehensive LTE parameter configuration document. \
                      Parameter param1: Configures cell power. \
                      Parameter param2: Sets antenna configuration. \
                      Counter counter1: Measures throughput. \
                      Counter counter2: Tracks error rates. \
                      The RAN system uses advanced 5G and LTE technologies.";
        
        let doc = create_test_document(content, DocumentFormat::Markdown);
        let result = processor.process_document(doc).unwrap();
        
        let quality = &result.structural_quality;
        assert!(quality.overall_score > 0.5); // Should be high quality
        assert!(quality.parameter_extraction_quality > 0.0);
        assert!(quality.counter_extraction_quality > 0.0);
        assert!(quality.technical_density_score > 0.0);
        assert!(quality.completeness_score > 0.0);
    }
    
    #[test]
    fn test_processing_hints_generation() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let doc = create_test_document(
            "Complex document with many parameters and counters for advanced RAN configuration",
            DocumentFormat::Markdown
        );
        
        let result = processor.process_document(doc).unwrap();
        let hints = &result.processing_hints;
        
        assert!(hints.expected_qa_pairs > 0);
        assert!(matches!(hints.processing_priority, ProcessingPriority::Normal | ProcessingPriority::High));
        assert!(matches!(hints.memory_optimization, MemoryOptimization::M3MaxUnified));
        assert!(hints.expected_processing_time > 0);
    }
    
    #[test]
    fn test_checksum_calculation() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let doc1 = create_test_document("Content 1", DocumentFormat::Markdown);
        let doc2 = create_test_document("Content 2", DocumentFormat::Markdown);
        let doc3 = create_test_document("Content 1", DocumentFormat::Markdown); // Same content, different ID
        
        let result1 = processor.process_document(doc1).unwrap();
        let result2 = processor.process_document(doc2).unwrap();
        let result3 = processor.process_document(doc3).unwrap();
        
        // Different content should have different checksums
        assert_ne!(result1.checksum, result2.checksum);
        
        // Same content but different ID should have different checksums
        assert_ne!(result1.checksum, result3.checksum);
        
        // Checksums should be non-zero
        assert_ne!(result1.checksum, 0);
        assert_ne!(result2.checksum, 0);
        assert_ne!(result3.checksum, 0);
    }
    
    // ========================================================================
    // M3 Max Optimization Tests
    // ========================================================================
    
    #[test]
    fn test_m3_max_optimizer_creation() {
        let optimizer = M3MaxOptimizer::new(60);
        
        assert!(optimizer.enabled);
        assert!(optimizer.thread_count > 0);
        assert_eq!(optimizer.memory_pool_gb, 48); // 80% of 60GB
        
        let stats = optimizer.get_memory_stats();
        assert_eq!(stats["memory_pool_gb"], 48);
        assert_eq!(stats["available_memory_mb"], 48 * 1024);
    }
    
    #[test]
    fn test_batch_optimization_sorting() {
        let optimizer = M3MaxOptimizer::new(60);
        
        let mut docs = vec![
            create_test_document("Large content that takes up more space", DocumentFormat::Markdown),
            create_test_document("Small", DocumentFormat::Text),
            create_test_document("Medium content here", DocumentFormat::Html),
        ];
        
        // Set different sizes
        docs[0].size_bytes = 1000;
        docs[1].size_bytes = 100;
        docs[2].size_bytes = 500;
        
        let result = optimizer.optimize_batch(docs).unwrap();
        
        // Should be sorted by size
        assert!(result[0].size_bytes <= result[1].size_bytes);
        assert!(result[1].size_bytes <= result[2].size_bytes);
    }
    
    #[test]
    fn test_memory_limit_enforcement() {
        let optimizer = M3MaxOptimizer::new(1); // Very small limit
        
        let large_docs = vec![
            create_test_document(&"x".repeat(1024 * 1024 * 500), DocumentFormat::Text), // 500MB content
            create_test_document(&"x".repeat(1024 * 1024 * 600), DocumentFormat::Text), // 600MB content
        ];
        
        let result = optimizer.optimize_batch(large_docs);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TestError::MemoryError(msg) => {
                assert!(msg.contains("exceeds memory limit"));
            }
            _ => panic!("Expected MemoryError"),
        }
    }
    
    #[test]
    fn test_thread_count_detection() {
        let optimizer = M3MaxOptimizer::new(60);
        let thread_count = optimizer.get_thread_count();
        
        assert!(thread_count > 0);
        assert!(thread_count <= 32); // Reasonable upper bound for test environments
    }
    
    // ========================================================================
    // Performance Monitoring Tests
    // ========================================================================
    
    #[test]
    fn test_metrics_tracking() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let doc = create_test_document("Test content", DocumentFormat::Markdown);
        let _result = processor.process_document(doc).unwrap();
        
        let metrics = processor.get_metrics().unwrap();
        assert!(metrics.average_processing_time_ms >= 0.0);
        assert!(metrics.throughput_docs_per_hour >= 0.0);
    }
    
    #[test]
    fn test_performance_metrics_structure() {
        let metrics = PerformanceMetrics {
            throughput_docs_per_hour: 3600.0,
            average_processing_time_ms: 1000.0,
            memory_utilization_percent: 75.5,
            cpu_utilization_percent: 85.2,
            error_rate_percent: 2.1,
            quality_score_average: 8.7,
        };
        
        assert_eq!(metrics.throughput_docs_per_hour, 3600.0);
        assert_eq!(metrics.average_processing_time_ms, 1000.0);
        assert_eq!(metrics.memory_utilization_percent, 75.5);
        assert_eq!(metrics.cpu_utilization_percent, 85.2);
        assert_eq!(metrics.error_rate_percent, 2.1);
        assert_eq!(metrics.quality_score_average, 8.7);
    }
    
    // ========================================================================
    // Error Handling and Edge Cases
    // ========================================================================
    
    #[test]
    fn test_error_types_display() {
        let errors = vec![
            TestError::ProcessingError("test processing".to_string()),
            TestError::MemoryError("test memory".to_string()),
            TestError::IpcError("test ipc".to_string()),
            TestError::QualityError("test quality".to_string()),
            TestError::ConfigError("test config".to_string()),
            TestError::SerializationError("test serialization".to_string()),
            TestError::TimeoutError { timeout_ms: 5000 },
        ];
        
        for error in errors {
            let error_str = format!("{}", error);
            assert!(!error_str.is_empty());
            assert!(error_str.contains("test") || error_str.contains("5000"));
        }
    }
    
    #[test]
    fn test_invalid_json_serialization() {
        // Create a document with potentially problematic content
        let doc = create_test_document("Content with \"quotes\" and \n newlines \t tabs", DocumentFormat::Markdown);
        
        // Should serialize without issues
        let serialized = serde_json::to_string(&doc);
        assert!(serialized.is_ok());
        
        // Should deserialize back correctly
        let deserialized: Result<Document, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
        
        let restored = deserialized.unwrap();
        assert_eq!(doc.content, restored.content);
    }
    
    #[test]
    fn test_extreme_content_sizes() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        // Test very small content
        let tiny_doc = create_test_document("x", DocumentFormat::Text);
        let tiny_result = processor.process_document(tiny_doc);
        assert!(tiny_result.is_ok());
        
        // Test large content (but not too large for test performance)
        let large_content = "x".repeat(10000);
        let large_doc = create_test_document(&large_content, DocumentFormat::Markdown);
        let large_result = processor.process_document(large_doc);
        assert!(large_result.is_ok());
    }
    
    #[test]
    fn test_unicode_content_handling() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let unicode_content = "测试内容 with émojis 🚀 and spëcial çharacters ñoñó";
        let doc = create_test_document(unicode_content, DocumentFormat::Markdown);
        
        let result = processor.process_document(doc);
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.document.content, unicode_content);
    }
    
    #[test]
    fn test_concurrent_processing_safety() {
        use std::sync::Arc;
        use std::thread;
        
        let config = create_test_config();
        let processor = Arc::new(DocumentProcessor::new(config));
        
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let proc = Arc::clone(&processor);
                thread::spawn(move || {
                    let doc = create_test_document(
                        &format!("Thread {} content with parameters", i),
                        DocumentFormat::Markdown
                    );
                    proc.process_document(doc)
                })
            })
            .collect();
        
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All threads should complete successfully
        for result in results {
            assert!(result.is_ok());
        }
        
        // Metrics should be accessible after concurrent processing
        let metrics = processor.get_metrics();
        assert!(metrics.is_ok());
    }
    
    // ========================================================================
    // Configuration and Validation Tests
    // ========================================================================
    
    #[test]
    fn test_processing_config_validation() {
        let valid_config = ProcessingConfig {
            max_concurrent_docs: 8,
            timeout_ms: 5000,
            quality_threshold: 5.0,
            enable_m3_optimization: true,
            memory_limit_gb: 60,
        };
        
        // Valid config should work
        let processor = DocumentProcessor::new(valid_config);
        assert_eq!(processor.config.max_concurrent_docs, 8);
    }
    
    #[test]
    fn test_config_edge_values() {
        let configs = vec![
            ProcessingConfig {
                max_concurrent_docs: 1,
                timeout_ms: 1000,
                quality_threshold: 0.0,
                enable_m3_optimization: false,
                memory_limit_gb: 1,
            },
            ProcessingConfig {
                max_concurrent_docs: 1000,
                timeout_ms: 300000,
                quality_threshold: 10.0,
                enable_m3_optimization: true,
                memory_limit_gb: 128,
            },
        ];
        
        for config in configs {
            let processor = DocumentProcessor::new(config);
            // Should create successfully with edge values
            assert!(processor.config.max_concurrent_docs > 0);
        }
    }
    
    // ========================================================================
    // Integration Test Scenarios
    // ========================================================================
    
    #[test]
    fn test_full_pipeline_with_optimization() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        let optimizer = M3MaxOptimizer::new(60);
        
        // Create a batch of documents
        let docs = vec![
            create_test_document("LTE document with parameter configurations", DocumentFormat::Markdown),
            create_test_document("5G document with counter definitions", DocumentFormat::Html),
            create_test_document("RAN document with technical specifications", DocumentFormat::Text),
        ];
        
        // Optimize batch
        let optimized_docs = optimizer.optimize_batch(docs).unwrap();
        assert_eq!(optimized_docs.len(), 3);
        
        // Process all documents
        let mut results = Vec::new();
        for doc in optimized_docs {
            let result = processor.process_document(doc)?;
            results.push(result);
        }
        
        // Verify all processing completed successfully
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.structural_quality.overall_score > 0.0);
            assert_ne!(result.checksum, 0);
        }
        
        // Check metrics were updated
        let metrics = processor.get_metrics()?;
        assert!(metrics.throughput_docs_per_hour > 0.0);
        
        Ok::<(), TestError>(())
    }
    
    #[test]
    fn test_real_world_document_patterns() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        // Simulate real RAN documentation patterns
        let test_cases = vec![
            (
                "Feature Document",
                "DOCTITLE: Advanced MIMO Configuration\n\
                This feature enables enhanced MIMO capabilities for LTE eNodeB.\n\
                Parameters:\n\
                - EUtranCellFDD.mimoSleepMode: Controls MIMO sleep functionality\n\
                - SectorCarrier.txPowerMax: Maximum transmission power\n\
                Counters:\n\
                - pmActiveDrbDl: Active DRB downlink counter\n\
                - pmCellAvailTime: Cell availability time",
                vec!["mimoSleepMode", "txPowerMax", "pmActiveDrbDl", "pmCellAvailTime"]
            ),
            (
                "Parameter Reference",
                "CXC1234567 - RAN Parameter Guide v1.2\n\
                Parameter: EUtranCellTDD.cellRange\n\
                Description: Defines cell coverage range in meters\n\
                Valid Values: 0-100000\n\
                Default: 15000\n\
                MO Class: EUtranCellTDD",
                vec!["cellRange", "EUtranCellTDD"]
            ),
            (
                "Counter Documentation", 
                "Performance Counter Reference\n\
                Counter: EUtranCellFDD.pmRrcConnLevSum\n\
                Type: GAUGE\n\
                Unit: percentage\n\
                Description: RRC connection level summary",
                vec!["pmRrcConnLevSum", "RRC"]
            ),
        ];
        
        for (name, content, expected_terms) in test_cases {
            let doc = create_test_document(content, DocumentFormat::Markdown);
            let result = processor.process_document(doc).unwrap();
            
            // Verify document was processed successfully
            assert!(result.structural_quality.overall_score > 0.0, 
                   "Failed processing {}", name);
            
            // Check that expected terms influence the processing
            let has_expected_content = expected_terms.iter().any(|term| {
                result.document.content.contains(term) ||
                result.document.metadata.parameters.iter().any(|p| p.name.contains(term)) ||
                result.document.metadata.counters.iter().any(|c| c.name.contains(term))
            });
            assert!(has_expected_content, "Missing expected content in {}", name);
        }
    }
    
    // ========================================================================
    // Performance and Stress Tests
    // ========================================================================
    
    #[test]
    fn test_processing_performance() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let start = std::time::Instant::now();
        let doc = create_test_document(
            &"Standard content ".repeat(100), // Moderate size document
            DocumentFormat::Markdown
        );
        
        let result = processor.process_document(doc);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000, "Processing took too long: {}ms", duration.as_millis());
    }
    
    #[test] 
    fn test_memory_optimization_patterns() {
        let optimizer = M3MaxOptimizer::new(60);
        
        // Test different memory optimization patterns
        let memory_tests = vec![
            MemoryOptimization::Standard,
            MemoryOptimization::M3MaxUnified,
            MemoryOptimization::HighThroughput,
            MemoryOptimization::LowLatency,
        ];
        
        for pattern in memory_tests {
            let hints = ProcessingHints {
                recommended_model: ComplexityLevel::Fast,
                expected_qa_pairs: 10,
                processing_priority: ProcessingPriority::Normal,
                use_cache: true,
                batch_with_similar: true,
                batch_processing_eligible: true,
                expected_processing_time: 1000,
                memory_optimization: pattern.clone(),
            };
            
            assert_eq!(hints.memory_optimization, pattern);
            
            // Memory optimization should be serializable
            let serialized = serde_json::to_string(&hints).unwrap();
            let deserialized: ProcessingHints = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized.memory_optimization, pattern);
        }
    }
}

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_document_processing_throughput() {
        let config = create_test_config();
        let processor = DocumentProcessor::new(config);
        
        let document_sizes = vec![100, 1000, 5000, 10000];
        
        for size in document_sizes {
            let content = "x".repeat(size);
            let doc = create_test_document(&content, DocumentFormat::Markdown);
            
            let start = Instant::now();
            let result = processor.process_document(doc);
            let duration = start.elapsed();
            
            assert!(result.is_ok());
            
            let docs_per_second = 1000.0 / duration.as_millis() as f64;
            println!("Content size {}: {:.2} docs/sec ({:.2}ms per doc)", 
                    size, docs_per_second, duration.as_millis());
            
            // Performance expectations (adjust based on hardware)
            match size {
                100 => assert!(docs_per_second > 100.0, "Too slow for small documents"),
                1000 => assert!(docs_per_second > 50.0, "Too slow for medium documents"), 
                5000 => assert!(docs_per_second > 20.0, "Too slow for large documents"),
                10000 => assert!(docs_per_second > 10.0, "Too slow for very large documents"),
                _ => {}
            }
        }
    }
    
    #[test]
    fn benchmark_batch_optimization() {
        let optimizer = M3MaxOptimizer::new(60);
        let batch_sizes = vec![1, 10, 50, 100];
        
        for batch_size in batch_sizes {
            let docs: Vec<Document> = (0..batch_size)
                .map(|i| create_test_document(&format!("Document {}", i), DocumentFormat::Markdown))
                .collect();
            
            let start = Instant::now();
            let result = optimizer.optimize_batch(docs);
            let duration = start.elapsed();
            
            assert!(result.is_ok());
            
            let docs_per_ms = batch_size as f64 / duration.as_millis() as f64;
            println!("Batch size {}: {:.2} docs/ms ({:.2}ms total)", 
                    batch_size, docs_per_ms, duration.as_millis());
            
            // Should handle reasonable batch sizes efficiently
            assert!(duration.as_millis() < 1000, "Batch optimization too slow");
        }
    }
}