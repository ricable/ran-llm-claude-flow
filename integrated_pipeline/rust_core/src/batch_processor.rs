use crate::types::*;
use crate::quality_validator::{QualityValidator, ValidationConfig};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore, RwLock};
use tracing::{info, warn, debug, error};
use rayon::prelude::*;
use dashmap::DashMap;
use uuid::Uuid;
use chrono::Utc;

/// High-performance batch processor for concurrent document processing
/// Optimized for M3 Max with intelligent workload distribution
#[derive(Debug)]
pub struct BatchProcessor {
    config: BatchProcessorConfig,
    quality_validator: Arc<QualityValidator>,
    processing_stats: Arc<RwLock<BatchProcessingStats>>,
    active_batches: Arc<DashMap<Uuid, BatchInfo>>,
    semaphore: Arc<Semaphore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessorConfig {
    pub max_concurrent_documents: usize,
    pub max_concurrent_batches: usize,
    pub batch_size_hint: usize,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub processing_timeout: Duration,
    pub enable_adaptive_batching: bool,
    pub memory_limit_gb: f64,
    pub optimization_strategy: OptimizationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Throughput,
    Quality,
    Memory,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInfo {
    pub batch_id: Uuid,
    pub document_count: usize,
    pub start_time: Instant,
    pub processing_hints: Vec<ProcessingHints>,
    pub status: BatchStatus,
    pub estimated_completion: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingStats {
    pub total_batches_processed: usize,
    pub total_documents_processed: usize,
    pub total_processing_time: Duration,
    pub average_batch_size: f64,
    pub throughput_docs_per_hour: f64,
    pub memory_peak_gb: f64,
    pub error_rate: f64,
    pub quality_average: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch_id: Uuid,
    pub processed_documents: Vec<ProcessedDocument>,
    pub failed_documents: Vec<(Uuid, String)>,
    pub batch_stats: BatchExecutionStats,
    pub quality_summary: QualitySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExecutionStats {
    pub documents_processed: usize,
    pub processing_time: Duration,
    pub memory_used_gb: f64,
    pub throughput_docs_per_hour: f64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    pub average_quality: f64,
    pub quality_distribution: HashMap<String, usize>,
    pub parameter_extraction_rate: f64,
    pub counter_extraction_rate: f64,
    pub technical_density_average: f64,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_documents: num_cpus::get() * 2,
            max_concurrent_batches: 4,
            batch_size_hint: 8,
            min_batch_size: 2,
            max_batch_size: 32,
            processing_timeout: Duration::from_secs(300),
            enable_adaptive_batching: true,
            memory_limit_gb: 60.0, // 60GB for Rust core
            optimization_strategy: OptimizationStrategy::Balanced,
        }
    }
}

impl BatchProcessor {
    pub fn new(config: BatchProcessorConfig) -> Self {
        let validation_config = ValidationConfig {
            min_parameter_count: 3,
            min_technical_density: 0.15,
            completeness_threshold: 0.6,
            enable_caching: true,
            parallel_processing: true,
        };

        let quality_validator = Arc::new(QualityValidator::new(validation_config));
        let processing_stats = Arc::new(RwLock::new(BatchProcessingStats::default()));
        let active_batches = Arc::new(DashMap::new());
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_documents));

        Self {
            config,
            quality_validator,
            processing_stats,
            active_batches,
            semaphore,
        }
    }

    /// Process multiple documents in optimized batches
    pub async fn process_documents_batch(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<BatchResult>> {
        let start_time = Instant::now();
        info!("Starting batch processing of {} documents", documents.len());

        // Create batches using intelligent grouping
        let batches = self.create_optimal_batches(documents)?;
        info!("Created {} optimal batches", batches.len());

        // Process batches concurrently
        let results = self.process_batches_concurrent(batches).await?;

        // Update processing statistics
        self.update_batch_stats(&results, start_time).await?;

        info!(
            "Batch processing complete: {} batches, {} total documents in {:?}",
            results.len(),
            results.iter().map(|r| r.processed_documents.len()).sum::<usize>(),
            start_time.elapsed()
        );

        Ok(results)
    }

    /// Create optimal batches based on document characteristics and system constraints
    fn create_optimal_batches(&self, mut documents: Vec<Document>) -> Result<Vec<Vec<Document>>> {
        if !self.config.enable_adaptive_batching {
            return Ok(self.create_fixed_size_batches(documents));
        }

        // Analyze documents for optimal grouping
        let analysis = self.analyze_documents_for_batching(&documents)?;
        
        // Sort documents by processing complexity for better load balancing
        documents.sort_by(|a, b| {
            let complexity_a = self.estimate_processing_complexity(a);
            let complexity_b = self.estimate_processing_complexity(b);
            complexity_a.partial_cmp(&complexity_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_complexity_score = 0.0;
        let mut current_memory_estimate = 0.0;

        let target_complexity = self.calculate_target_batch_complexity(&analysis);
        let memory_per_batch = self.config.memory_limit_gb / self.config.max_concurrent_batches as f64;

        for document in documents {
            let doc_complexity = self.estimate_processing_complexity(&document);
            let doc_memory = self.estimate_memory_requirement(&document);

            let would_exceed_complexity = current_complexity_score + doc_complexity > target_complexity * 1.2;
            let would_exceed_memory = current_memory_estimate + doc_memory > memory_per_batch * 0.9;
            let would_exceed_max_size = current_batch.len() >= self.config.max_batch_size;

            if (would_exceed_complexity || would_exceed_memory || would_exceed_max_size) 
                && current_batch.len() >= self.config.min_batch_size {
                
                batches.push(std::mem::take(&mut current_batch));
                current_complexity_score = 0.0;
                current_memory_estimate = 0.0;
            }

            current_batch.push(document);
            current_complexity_score += doc_complexity;
            current_memory_estimate += doc_memory;
        }

        // Add remaining documents as final batch
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        info!(
            "Created {} adaptive batches with sizes: {:?}",
            batches.len(),
            batches.iter().map(|b| b.len()).collect::<Vec<_>>()
        );

        Ok(batches)
    }

    fn create_fixed_size_batches(&self, documents: Vec<Document>) -> Vec<Vec<Document>> {
        documents
            .chunks(self.config.batch_size_hint)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn analyze_documents_for_batching(&self, documents: &[Document]) -> Result<BatchingAnalysis> {
        let complexities: Vec<f64> = documents
            .par_iter()
            .map(|doc| self.estimate_processing_complexity(doc))
            .collect();

        let memory_estimates: Vec<f64> = documents
            .par_iter()
            .map(|doc| self.estimate_memory_requirement(doc))
            .collect();

        let avg_complexity = complexities.iter().sum::<f64>() / complexities.len() as f64;
        let max_complexity = complexities.iter().copied().fold(0.0f64, f64::max);
        let avg_memory = memory_estimates.iter().sum::<f64>() / memory_estimates.len() as f64;
        let max_memory = memory_estimates.iter().copied().fold(0.0f64, f64::max);

        Ok(BatchingAnalysis {
            total_documents: documents.len(),
            avg_complexity,
            max_complexity,
            avg_memory_gb: avg_memory,
            max_memory_gb: max_memory,
            complexity_variance: self.calculate_variance(&complexities),
        })
    }

    fn estimate_processing_complexity(&self, document: &Document) -> f64 {
        let length_factor = (document.content.len() as f64 / 10000.0).min(1.0);
        let param_factor = (document.metadata.parameters.len() as f64 / 50.0).min(1.0);
        let counter_factor = (document.metadata.counters.len() as f64 / 20.0).min(1.0);
        let format_factor = match document.format {
            DocumentFormat::Pdf => 1.2,
            DocumentFormat::Html => 1.1,
            DocumentFormat::Gpp3 => 1.3,
            _ => 1.0,
        };

        (length_factor * 0.4 + param_factor * 0.3 + counter_factor * 0.3) * format_factor
    }

    fn estimate_memory_requirement(&self, document: &Document) -> f64 {
        // Estimate memory requirement in GB
        let base_memory = document.content.len() as f64 / (1024.0 * 1024.0 * 1024.0); // Content size in GB
        let processing_overhead = 0.05; // 50MB processing overhead
        let format_multiplier = match document.format {
            DocumentFormat::Pdf => 2.0,
            DocumentFormat::Html => 1.5,
            _ => 1.2,
        };

        (base_memory * format_multiplier + processing_overhead).max(0.01)
    }

    fn calculate_target_batch_complexity(&self, analysis: &BatchingAnalysis) -> f64 {
        match self.config.optimization_strategy {
            OptimizationStrategy::Throughput => analysis.avg_complexity * 0.8,
            OptimizationStrategy::Quality => analysis.avg_complexity * 1.2,
            OptimizationStrategy::Memory => analysis.avg_complexity * 0.6,
            OptimizationStrategy::Balanced => analysis.avg_complexity,
        }
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }

    /// Process multiple batches concurrently with intelligent scheduling
    async fn process_batches_concurrent(&self, batches: Vec<Vec<Document>>) -> Result<Vec<BatchResult>> {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_batches));
        let mut handles = Vec::new();

        for batch in batches {
            let processor = Arc::new(self.clone_for_batch());
            let sem = semaphore.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("Semaphore acquire failed");
                processor.process_single_batch(batch).await
            });
            
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => {
                    error!("Batch processing failed: {}", e);
                    return Err(e);
                }
                Err(e) => {
                    error!("Batch task panicked: {}", e);
                    return Err(anyhow::anyhow!("Batch task panicked: {}", e));
                }
            }
        }

        Ok(results)
    }

    /// Process a single batch of documents
    async fn process_single_batch(&self, documents: Vec<Document>) -> Result<BatchResult> {
        let batch_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        info!("Processing batch {} with {} documents", batch_id, documents.len());

        // Register batch
        let batch_info = BatchInfo {
            batch_id,
            document_count: documents.len(),
            start_time,
            processing_hints: documents.iter().map(|d| self.create_processing_hints(d)).collect(),
            status: BatchStatus::Processing,
            estimated_completion: Some(start_time + self.estimate_processing_time(&documents)),
        };
        
        self.active_batches.insert(batch_id, batch_info);

        // Process documents in parallel within batch
        let processed_results = self.process_documents_parallel(documents).await?;

        let processing_time = start_time.elapsed();
        
        // Separate successful and failed results
        let mut processed_documents = Vec::new();
        let mut failed_documents = Vec::new();
        
        for result in processed_results {
            match result {
                Ok(doc) => processed_documents.push(doc),
                Err((doc_id, error)) => failed_documents.push((doc_id, error)),
            }
        }

        // Calculate batch statistics
        let batch_stats = self.calculate_batch_stats(&processed_documents, processing_time);
        let quality_summary = self.calculate_quality_summary(&processed_documents);

        // Update batch status
        if let Some(mut batch) = self.active_batches.get_mut(&batch_id) {
            batch.status = if failed_documents.is_empty() {
                BatchStatus::Completed
            } else {
                BatchStatus::Failed(format!("{} documents failed", failed_documents.len()))
            };
        }

        info!(
            "Batch {} completed: {}/{} documents successful in {:?}",
            batch_id,
            processed_documents.len(),
            processed_documents.len() + failed_documents.len(),
            processing_time
        );

        Ok(BatchResult {
            batch_id,
            processed_documents,
            failed_documents,
            batch_stats,
            quality_summary,
        })
    }

    /// Process documents in parallel within a batch
    async fn process_documents_parallel(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<Result<ProcessedDocument, (Uuid, String)>>> {
        let semaphore = self.semaphore.clone();
        let mut handles = Vec::new();

        for document in documents {
            let validator = self.quality_validator.clone();
            let sem = semaphore.clone();
            let timeout = self.config.processing_timeout;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("Semaphore acquire failed");
                
                // Set processing timeout
                let processing_future = Self::process_single_document(validator, document);
                
                match tokio::time::timeout(timeout, processing_future).await {
                    Ok(result) => result,
                    Err(_) => {
                        let doc_id = Uuid::new_v4(); // Would normally get from document
                        Err((doc_id, "Processing timeout".to_string()))
                    }
                }
            });
            
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    let doc_id = Uuid::new_v4();
                    results.push(Err((doc_id, format!("Task panicked: {}", e))));
                }
            }
        }

        Ok(results)
    }

    /// Process a single document with comprehensive validation
    async fn process_single_document(
        validator: Arc<QualityValidator>,
        document: Document,
    ) -> Result<ProcessedDocument, (Uuid, String)> {
        let doc_id = document.id;
        
        debug!("Processing document: {}", doc_id);

        match validator.validate_document(&document) {
            Ok(structural_quality) => {
                let checksum = crc32fast::hash(document.content.as_bytes());
                let processing_hints = ProcessingHints {
                    recommended_model: Self::determine_model_from_complexity(&document),
                    expected_qa_pairs: Self::estimate_qa_pairs(&document),
                    processing_priority: Self::determine_priority(&structural_quality),
                    use_cache: true,
                    batch_with_similar: true,
                };

                Ok(ProcessedDocument {
                    document,
                    structural_quality,
                    processing_hints,
                    checksum,
                })
            }
            Err(e) => {
                error!("Failed to process document {}: {}", doc_id, e);
                Err((doc_id, e.to_string()))
            }
        }
    }

    // Helper methods
    fn create_processing_hints(&self, document: &Document) -> ProcessingHints {
        let complexity = self.estimate_processing_complexity(document);
        let recommended_model = if complexity < 0.3 {
            ComplexityLevel::Fast
        } else if complexity < 0.7 {
            ComplexityLevel::Balanced
        } else {
            ComplexityLevel::Quality
        };

        ProcessingHints {
            recommended_model,
            expected_qa_pairs: Self::estimate_qa_pairs(document),
            processing_priority: ProcessingPriority::Normal,
            use_cache: true,
            batch_with_similar: true,
        }
    }

    fn estimate_processing_time(&self, documents: &[Document]) -> Duration {
        let avg_doc_time = Duration::from_millis(500); // 500ms per document estimate
        let complexity_factor = documents.iter()
            .map(|d| self.estimate_processing_complexity(d))
            .fold(0.0, f64::max);
        
        let adjusted_time = avg_doc_time.mul_f64(complexity_factor * 1.5);
        adjusted_time * documents.len() as u32
    }

    fn calculate_batch_stats(&self, documents: &[ProcessedDocument], processing_time: Duration) -> BatchExecutionStats {
        let documents_processed = documents.len();
        let memory_used_gb = self.estimate_batch_memory_usage(documents);
        let throughput = if processing_time.as_secs() > 0 {
            (documents_processed as f64) * 3600.0 / processing_time.as_secs() as f64
        } else {
            0.0
        };

        BatchExecutionStats {
            documents_processed,
            processing_time,
            memory_used_gb,
            throughput_docs_per_hour: throughput,
            success_rate: 1.0, // All documents in this batch were successful
        }
    }

    fn calculate_quality_summary(&self, documents: &[ProcessedDocument]) -> QualitySummary {
        if documents.is_empty() {
            return QualitySummary::default();
        }

        let qualities: Vec<f64> = documents.iter()
            .map(|d| d.structural_quality.overall_score)
            .collect();

        let average_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;

        let mut quality_distribution = HashMap::new();
        for quality in &qualities {
            let bucket = if *quality < 0.5 {
                "low"
            } else if *quality < 0.7 {
                "medium"
            } else if *quality < 0.85 {
                "high"
            } else {
                "excellent"
            };
            *quality_distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }

        let param_extraction_rate = documents.iter()
            .map(|d| d.structural_quality.parameter_extraction_quality)
            .sum::<f64>() / documents.len() as f64;

        let counter_extraction_rate = documents.iter()
            .map(|d| d.structural_quality.counter_extraction_quality)
            .sum::<f64>() / documents.len() as f64;

        let technical_density_average = documents.iter()
            .map(|d| d.structural_quality.technical_density_score)
            .sum::<f64>() / documents.len() as f64;

        QualitySummary {
            average_quality,
            quality_distribution,
            parameter_extraction_rate,
            counter_extraction_rate,
            technical_density_average,
        }
    }

    fn estimate_batch_memory_usage(&self, documents: &[ProcessedDocument]) -> f64 {
        documents.iter()
            .map(|d| self.estimate_memory_requirement(&d.document))
            .sum()
    }

    async fn update_batch_stats(&self, results: &[BatchResult], start_time: Instant) -> Result<()> {
        let mut stats = self.processing_stats.write().await;
        
        stats.total_batches_processed += results.len();
        stats.total_documents_processed += results.iter()
            .map(|r| r.processed_documents.len())
            .sum::<usize>();
        
        let total_time = start_time.elapsed();
        stats.total_processing_time += total_time;
        
        if stats.total_batches_processed > 0 {
            stats.average_batch_size = stats.total_documents_processed as f64 / stats.total_batches_processed as f64;
        }
        
        if total_time.as_secs() > 0 {
            stats.throughput_docs_per_hour = (stats.total_documents_processed as f64) * 3600.0 / 
                stats.total_processing_time.as_secs() as f64;
        }
        
        // Calculate quality average
        let all_qualities: Vec<f64> = results.iter()
            .map(|r| r.quality_summary.average_quality)
            .collect();
        
        if !all_qualities.is_empty() {
            stats.quality_average = all_qualities.iter().sum::<f64>() / all_qualities.len() as f64;
        }
        
        // Update error rate
        let total_failed = results.iter()
            .map(|r| r.failed_documents.len())
            .sum::<usize>();
        let total_attempted = stats.total_documents_processed + total_failed;
        
        if total_attempted > 0 {
            stats.error_rate = total_failed as f64 / total_attempted as f64;
        }

        info!(
            "Updated batch stats: {} batches, {} docs, {:.2} docs/hour, {:.3} avg quality, {:.2}% error rate",
            stats.total_batches_processed,
            stats.total_documents_processed,
            stats.throughput_docs_per_hour,
            stats.quality_average,
            stats.error_rate * 100.0
        );

        Ok(())
    }

    // Static helper methods
    fn determine_model_from_complexity(document: &Document) -> ComplexityLevel {
        let param_count = document.metadata.parameters.len();
        let counter_count = document.metadata.counters.len();
        let content_length = document.content.len();

        if param_count > 20 || counter_count > 10 || content_length > 8000 {
            ComplexityLevel::Quality
        } else if param_count > 10 || counter_count > 5 || content_length > 3000 {
            ComplexityLevel::Balanced
        } else {
            ComplexityLevel::Fast
        }
    }

    fn estimate_qa_pairs(document: &Document) -> usize {
        let base_pairs = (document.content.len() / 1000).max(1);
        let param_bonus = document.metadata.parameters.len() / 2;
        let counter_bonus = document.metadata.counters.len() / 3;
        
        (base_pairs + param_bonus + counter_bonus).min(15)
    }

    fn determine_priority(quality: &StructuralQuality) -> ProcessingPriority {
        if quality.overall_score >= 0.8 {
            ProcessingPriority::High
        } else if quality.overall_score >= 0.6 {
            ProcessingPriority::Normal
        } else {
            ProcessingPriority::Low
        }
    }

    fn clone_for_batch(&self) -> Self {
        Self {
            config: self.config.clone(),
            quality_validator: self.quality_validator.clone(),
            processing_stats: self.processing_stats.clone(),
            active_batches: self.active_batches.clone(),
            semaphore: self.semaphore.clone(),
        }
    }

    /// Get current processing statistics
    pub async fn get_processing_stats(&self) -> BatchProcessingStats {
        self.processing_stats.read().await.clone()
    }

    /// Get information about active batches
    pub fn get_active_batches(&self) -> Vec<BatchInfo> {
        self.active_batches.iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Clear completed batches from tracking
    pub fn cleanup_completed_batches(&self) {
        self.active_batches.retain(|_, batch| {
            !matches!(batch.status, BatchStatus::Completed | BatchStatus::Failed(_))
        });
    }
}

#[derive(Debug, Clone)]
struct BatchingAnalysis {
    total_documents: usize,
    avg_complexity: f64,
    max_complexity: f64,
    avg_memory_gb: f64,
    max_memory_gb: f64,
    complexity_variance: f64,
}

impl Default for BatchProcessingStats {
    fn default() -> Self {
        Self {
            total_batches_processed: 0,
            total_documents_processed: 0,
            total_processing_time: Duration::from_secs(0),
            average_batch_size: 0.0,
            throughput_docs_per_hour: 0.0,
            memory_peak_gb: 0.0,
            error_rate: 0.0,
            quality_average: 0.0,
        }
    }
}

impl Default for QualitySummary {
    fn default() -> Self {
        Self {
            average_quality: 0.0,
            quality_distribution: HashMap::new(),
            parameter_extraction_rate: 0.0,
            counter_extraction_rate: 0.0,
            technical_density_average: 0.0,
        }
    }
}

impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            quality_validator: self.quality_validator.clone(),
            processing_stats: self.processing_stats.clone(),
            active_batches: self.active_batches.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}