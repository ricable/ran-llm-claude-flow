use crate::config::ProcessingConfig;
use crate::ipc_manager::IpcManager;
use crate::quality_validator::QualityValidator;
use crate::types::*;

use anyhow::Result;
use parking_lot::RwLock;
use rayon::prelude::*;
use regex::Regex;
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use tokio::task::JoinHandle;
use futures::future::try_join_all;
use std::sync::atomic::{AtomicUsize, Ordering};
use once_cell::sync::Lazy;
use sysinfo::{System, SystemExt};

/// High-performance document processor optimized for M3 Max
pub struct DocumentProcessor {
    config: ProcessingConfig,
    ipc_manager: Arc<IpcManager>,
    quality_validator: Arc<QualityValidator>,
    format_detectors: Arc<FormatDetectors>,
    processing_semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<ProcessingStats>>,
}

/// Format detection and parsing utilities
pub struct FormatDetectors {
    markdown_patterns: MarkdownPatterns,
    parameter_regex: Regex,
    counter_regex: Regex,
    technical_terms: Arc<Vec<String>>,
}

/// Markdown-specific pattern matchers
pub struct MarkdownPatterns {
    pub title_regex: Regex,
    pub feature_name_regex: Regex,
    pub product_info_regex: Regex,
    pub parameter_section_regex: Regex,
    pub counter_section_regex: Regex,
}

/// Aggregated processing statistics
#[derive(Debug, Default)]
pub struct ProcessingStatsData {
    pub documents_processed: usize,
    pub total_qa_pairs_generated: usize,
    pub total_processing_time: Duration,
    pub average_quality: f64,
    pub memory_peak_mb: usize,
    pub errors_count: usize,
}

impl DocumentProcessor {
    /// Create new document processor with M3 Max optimization
    pub async fn new(config: ProcessingConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        let memory_allocations = config.get_memory_allocations();
        info!("Memory allocation plan:");
        info!("  Document pool: {:.2}GB", memory_allocations.document_pool_gb());
        info!("  IPC pool: {:.2}GB", memory_allocations.ipc_pool_gb());
        info!("  System reserve: {:.2}GB", memory_allocations.system_reserve_gb());
        
        // Initialize IPC manager
        let ipc_manager = Arc::new(IpcManager::new(&config.ipc).await?);
        
        // Initialize quality validator
        let quality_validator = Arc::new(QualityValidator::new(&config.quality)?);
        
        // Initialize format detectors
        let format_detectors = Arc::new(FormatDetectors::new().await?);
        
        // Create semaphore for concurrency control
        let processing_semaphore = Arc::new(Semaphore::new(config.processing.max_concurrent_docs));
        
        // Initialize statistics
        let stats = Arc::new(RwLock::new(ProcessingStatsData::default()));
        
        // Configure Rayon thread pool for M3 Max optimization
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.get_optimal_thread_count())
            .build_global()?;
        
        info!("Document processor initialized");
        info!("Thread pool size: {}", config.get_optimal_thread_count());
        info!("Max concurrent documents: {}", config.processing.max_concurrent_docs);
        
        Ok(Self {
            config,
            ipc_manager,
            quality_validator,
            format_detectors,
            processing_semaphore,
            stats,
        })
    }
    
    /// Process entire directory of documents with M3 Max optimization
    pub async fn process_directory(&self, input_dir: &Path, output_dir: &Path) -> Result<PipelineStats> {
        let start_time = Instant::now();
        info!("Starting M3 Max optimized directory processing: {:?}", input_dir);
        
        // Log system information for optimization tracking
        self.log_system_information().await;
        
        // Ensure output directory exists
        fs::create_dir_all(output_dir).await?;
        
        // Discover documents with parallel file system scanning
        let document_paths = self.discover_documents_parallel(input_dir).await?;
        info!("Found {} documents to process", document_paths.len());
        
        if document_paths.is_empty() {
            warn!("No documents found in directory: {:?}", input_dir);
            return Ok(PipelineStats::default());
        }
        
        // Create high-capacity processing channels
        let channel_capacity = (document_paths.len() * 2).min(10000);
        let (result_tx, result_rx) = mpsc::channel::<ProcessingResult>(channel_capacity);
        
        // Start result collector with enhanced buffering
        let output_dir = output_dir.to_owned();
        let collector_handle = tokio::spawn(async move {
            Self::collect_results_optimized(result_rx, &output_dir).await
        });
        
        // Adaptive concurrency based on M3 Max capabilities
        let optimal_concurrency = self.calculate_m3_max_concurrency(document_paths.len());
        info!("Using M3 Max optimized concurrency: {} parallel streams", optimal_concurrency);
        
        // Process documents with intelligent work distribution
        let processing_stats = self.process_documents_m3_optimized(
            document_paths,
            result_tx,
            optimal_concurrency
        ).await?;
        
        // Wait for result collection to complete
        let collection_results = collector_handle.await??;
        
        let total_time = start_time.elapsed();
        let final_stats = self.build_enhanced_pipeline_stats(
            total_time, 
            &collection_results, 
            processing_stats
        );
        
        info!("M3 Max processing completed in {:?}", total_time);
        info!("Throughput: {:.2} docs/hour (Target: 20-30)", final_stats.throughput_per_hour());
        info!("Memory utilization: {:.2}GB peak", final_stats.memory_peak_gb);
        
        // Log performance metrics for optimization tracking
        self.log_performance_metrics(&final_stats).await;
        
        Ok(final_stats)
    }
    
    /// Process a batch of documents
    async fn process_batch(
        &self,
        batch_idx: usize,
        document_paths: Vec<PathBuf>,
        result_tx: mpsc::Sender<ProcessingResult>
    ) -> Result<()> {
        info!("Processing batch {} with {} documents", batch_idx, document_paths.len());
        
        // Process documents in parallel within the batch
        let handles: Vec<_> = document_paths
            .into_iter()
            .enumerate()
            .map(|(doc_idx, path)| {
                let processor = self.clone();
                let tx = result_tx.clone();
                let doc_id = format!("batch-{}-doc-{}", batch_idx, doc_idx);
                
                tokio::spawn(async move {
                    processor.process_single_document(doc_id, path, tx).await
                })
            })
            .collect();
        
        // Wait for all documents in batch to complete
        for handle in handles {
            if let Err(e) = handle.await? {
                error!("Document processing error: {}", e);
            }
        }
        
        debug!("Batch {} completed", batch_idx);
        Ok(())
    }
    
    /// Process a single document through the entire pipeline
    async fn process_single_document(
        &self,
        doc_id: String,
        path: PathBuf,
        result_tx: mpsc::Sender<ProcessingResult>
    ) -> Result<()> {
        // Acquire processing permit
        let _permit = self.processing_semaphore.acquire().await?;
        
        let start_time = Instant::now();
        debug!("Processing document: {:?}", path);
        
        // Load and preprocess document
        let document = match self.load_and_preprocess_document(&path).await {
            Ok(doc) => doc,
            Err(e) => {
                error!("Failed to load document {:?}: {}", path, e);
                self.increment_error_count();
                return Err(e);
            }
        };
        
        // Perform structural quality assessment
        let structural_quality = self.quality_validator.assess_structural_quality(&document)?;
        
        // Early quality filter
        if self.config.quality.enable_early_filtering && 
           structural_quality.overall_score < self.config.quality.quality_threshold {
            warn!("Document {:?} failed early quality filter (score: {:.2})", 
                  path, structural_quality.overall_score);
            return Ok(());
        }
        
        // Generate processing hints
        let processing_hints = self.generate_processing_hints(&document, &structural_quality)?;
        
        // Create processed document
        let processed_doc = ProcessedDocument {
            document: document.clone(),
            structural_quality,
            processing_hints,
            checksum: self.calculate_checksum(&document.content),
        };
        
        // Send to Python ML engine via IPC
        let ml_response = match self.ipc_manager.send_for_ml_processing(processed_doc).await {
            Ok(response) => response,
            Err(e) => {
                error!("ML processing failed for {:?}: {}", path, e);
                self.increment_error_count();
                return Err(e.into());
            }
        };
        
        // Combine results
        let processing_time = start_time.elapsed();
        let result = ProcessingResult {
            document_id: document.id,
            original_document: document,
            structural_quality: ml_response.structural_quality,
            semantic_quality: ml_response.semantic_quality,
            qa_pairs: ml_response.qa_pairs,
            combined_quality_score: self.calculate_combined_quality_score(
                &ml_response.structural_quality,
                &ml_response.semantic_quality
            ),
            processing_stats: ProcessingStats {
                rust_processing_time: processing_time - ml_response.processing_time,
                ml_processing_time: ml_response.processing_time,
                ipc_overhead_time: Duration::from_millis(10), // Estimated
                total_processing_time: processing_time,
                memory_peak_mb: self.estimate_memory_usage(&ml_response),
                model_used: ml_response.model_used,
            },
        };
        
        // Send result
        if let Err(e) = result_tx.send(result).await {
            error!("Failed to send processing result: {}", e);
        }
        
        self.update_stats(processing_time, &ml_response);
        
        debug!("Document processed in {:?}: {:?}", processing_time, path);
        Ok(())
    }
    
    /// Load and preprocess document with format detection
    async fn load_and_preprocess_document(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path).await?;
        let size_bytes = content.len();
        
        // Detect format
        let format = self.detect_document_format(path, &content)?;
        
        // Extract metadata based on format
        let metadata = self.extract_metadata(&content, &format)?;
        
        let document = Document {
            id: Uuid::new_v4(),
            path: path.to_owned(),
            format,
            content,
            metadata,
            size_bytes,
            created_at: chrono::Utc::now(),
        };
        
        Ok(document)
    }
    
    /// Detect document format from extension and content analysis
    fn detect_document_format(&self, path: &Path, content: &str) -> Result<DocumentFormat> {
        // Check extension first
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "md" | "markdown" => return Ok(DocumentFormat::Markdown),
                "html" | "htm" => return Ok(DocumentFormat::Html),
                "pdf" => return Ok(DocumentFormat::Pdf),
                "csv" => return Ok(DocumentFormat::Csv),
                "txt" => return Ok(DocumentFormat::Text),
                _ => {}
            }
        }
        
        // Content-based detection
        if content.contains("# ") || content.contains("## ") {
            Ok(DocumentFormat::Markdown)
        } else if content.contains("<html") || content.contains("<!DOCTYPE") {
            Ok(DocumentFormat::Html)
        } else if content.contains("3GPP") || content.contains("TS ") {
            Ok(DocumentFormat::Gpp3)
        } else {
            Ok(DocumentFormat::Text)
        }
    }
    
    /// Extract metadata from document content
    fn extract_metadata(&self, content: &str, format: &DocumentFormat) -> Result<DocumentMetadata> {
        match format {
            DocumentFormat::Markdown => self.extract_markdown_metadata(content),
            DocumentFormat::Html => self.extract_html_metadata(content),
            DocumentFormat::Pdf => self.extract_pdf_metadata(content),
            DocumentFormat::Csv => self.extract_csv_metadata(content),
            DocumentFormat::Gpp3 => self.extract_3gpp_metadata(content),
            DocumentFormat::Text => self.extract_text_metadata(content),
        }
    }
    
    /// Extract metadata from Markdown documents (Ericsson RAN features)
    fn extract_markdown_metadata(&self, content: &str) -> Result<DocumentMetadata> {
        let patterns = &self.format_detectors.markdown_patterns;
        
        // Extract title
        let title = patterns.title_regex
            .captures(content)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string());
        
        // Extract feature name
        let feature_name = patterns.feature_name_regex
            .captures(content)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string());
        
        // Extract product info
        let product_info = patterns.product_info_regex
            .captures(content)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string());
        
        // Extract parameters
        let parameters = self.extract_parameters(content)?;
        
        // Extract counters
        let counters = self.extract_counters(content)?;
        
        // Extract technical terms
        let technical_terms = self.extract_technical_terms(content);
        
        // Calculate complexity hints
        let complexity_hints = self.calculate_complexity_hints(
            &parameters,
            &counters,
            &technical_terms,
            content.len()
        );
        
        Ok(DocumentMetadata {
            title,
            feature_name,
            product_info,
            feature_state: None, // TODO: Extract from content
            parameters,
            counters,
            technical_terms,
            complexity_hints,
        })
    }
    
    /// Extract parameters from content
    fn extract_parameters(&self, content: &str) -> Result<Vec<Parameter>> {
        let mut parameters = Vec::new();
        
        // Look for parameter patterns in markdown
        for cap in self.format_detectors.parameter_regex.captures_iter(content) {
            if let Some(param_match) = cap.get(1) {
                let param_text = param_match.as_str();
                
                // Parse parameter details
                let parameter = self.parse_parameter_text(param_text)?;
                parameters.push(parameter);
            }
        }
        
        Ok(parameters)
    }
    
    /// Parse parameter text to extract structured information
    fn parse_parameter_text(&self, text: &str) -> Result<Parameter> {
        // Extract parameter name (first part before colon or description)
        let name_regex = Regex::new(r"^[\*\-\s]*([A-Za-z0-9_\.]+)")?;
        let name = name_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        
        // Extract MO class if present
        let mo_class_regex = Regex::new(r"MO Class:\s*([A-Za-z0-9_]+)")?;
        let mo_class = mo_class_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string());
        
        // Extract valid values if present
        let valid_values_regex = Regex::new(r"Valid Values?:\s*([^\n]+)")?;
        let valid_values = valid_values_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string());
        
        // Extract default value if present
        let default_regex = Regex::new(r"Default:\s*([^\n]+)")?;
        let default_value = default_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string());
        
        Ok(Parameter {
            name,
            mo_class,
            data_type: None, // TODO: Infer from valid values
            valid_values,
            default_value,
            description: Some(text.to_string()),
        })
    }
    
    /// Extract counters from content
    fn extract_counters(&self, content: &str) -> Result<Vec<Counter>> {
        let mut counters = Vec::new();
        
        for cap in self.format_detectors.counter_regex.captures_iter(content) {
            if let Some(counter_match) = cap.get(1) {
                let counter_text = counter_match.as_str();
                let counter = self.parse_counter_text(counter_text)?;
                counters.push(counter);
            }
        }
        
        Ok(counters)
    }
    
    /// Parse counter text to extract structured information
    fn parse_counter_text(&self, text: &str) -> Result<Counter> {
        let name_regex = Regex::new(r"^[\*\-\s]*([A-Za-z0-9_\.]+)")?;
        let name = name_regex
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        
        Ok(Counter {
            name,
            description: Some(text.to_string()),
            mo_class: None,
            counter_type: None,
        })
    }
    
    /// Extract technical terms using predefined vocabulary
    fn extract_technical_terms(&self, content: &str) -> Vec<String> {
        let content_lower = content.to_lowercase();
        let mut found_terms = Vec::new();
        
        for term in self.format_detectors.technical_terms.iter() {
            if content_lower.contains(&term.to_lowercase()) {
                found_terms.push(term.clone());
            }
        }
        
        found_terms
    }
    
    /// Calculate complexity hints for model selection
    fn calculate_complexity_hints(
        &self,
        parameters: &[Parameter],
        counters: &[Counter],
        technical_terms: &[String],
        content_length: usize
    ) -> ComplexityHints {
        let parameter_count = parameters.len();
        let counter_count = counters.len();
        let technical_density = technical_terms.len() as f64 / (content_length as f64 / 1000.0);
        
        // Estimate complexity based on multiple factors
        let complexity_score = self.calculate_complexity_score(
            parameter_count,
            counter_count,
            technical_density,
            content_length
        );
        
        let estimated_complexity = if complexity_score < 0.3 {
            ComplexityLevel::Fast
        } else if complexity_score < 0.7 {
            ComplexityLevel::Balanced
        } else {
            ComplexityLevel::Quality
        };
        
        ComplexityHints {
            parameter_count,
            counter_count,
            technical_term_density: technical_density,
            content_length,
            estimated_complexity,
        }
    }
    
    /// Calculate normalized complexity score (0.0 to 1.0)
    fn calculate_complexity_score(
        &self,
        parameter_count: usize,
        counter_count: usize,
        technical_density: f64,
        content_length: usize
    ) -> f64 {
        // Normalized factors (0.0 to 1.0 each)
        let param_factor = (parameter_count as f64 / 20.0).min(1.0);
        let counter_factor = (counter_count as f64 / 10.0).min(1.0);
        let density_factor = (technical_density / 10.0).min(1.0);
        let length_factor = (content_length as f64 / 10000.0).min(1.0);
        
        // Weighted average
        (param_factor * 0.3 + counter_factor * 0.2 + density_factor * 0.3 + length_factor * 0.2)
    }
    
    /// Generate processing hints for Python ML engine
    fn generate_processing_hints(
        &self,
        document: &Document,
        structural_quality: &StructuralQuality
    ) -> Result<ProcessingHints> {
        let complexity = &document.metadata.complexity_hints;
        
        // Determine processing priority
        let processing_priority = if structural_quality.overall_score > 0.9 {
            ProcessingPriority::High
        } else if structural_quality.overall_score > 0.7 {
            ProcessingPriority::Normal
        } else {
            ProcessingPriority::Low
        };
        
        // Estimate QA pairs based on content complexity
        let expected_qa_pairs = self.estimate_qa_pair_count(&document.metadata.complexity_hints);
        
        Ok(ProcessingHints {
            recommended_model: complexity.estimated_complexity.clone(),
            expected_qa_pairs,
            processing_priority,
            use_cache: document.size_bytes > 5000, // Cache for larger documents
            batch_with_similar: complexity.estimated_complexity == ComplexityLevel::Fast,
        })
    }
    
    /// Estimate number of QA pairs based on complexity
    fn estimate_qa_pair_count(&self, hints: &ComplexityHints) -> usize {
        let base_pairs = match hints.estimated_complexity {
            ComplexityLevel::Fast => 2,
            ComplexityLevel::Balanced => 4,
            ComplexityLevel::Quality => 8,
        };
        
        // Adjust based on content characteristics
        let param_bonus = (hints.parameter_count / 3).min(3);
        let counter_bonus = (hints.counter_count / 2).min(2);
        
        base_pairs + param_bonus + counter_bonus
    }
    
    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, content: &str) -> u32 {
        crc32fast::hash(content.as_bytes())
    }
    
    /// Calculate combined quality score from structural and semantic assessments
    fn calculate_combined_quality_score(
        &self,
        structural: &StructuralQuality,
        semantic: &SemanticQuality
    ) -> f64 {
        // Weighted average: structural quality (40%) + semantic quality (60%)
        structural.overall_score * 0.4 + semantic.overall_score * 0.6
    }
    
    /// Estimate memory usage for processing statistics
    fn estimate_memory_usage(&self, _response: &MLProcessingResponse) -> usize {
        // Simplified memory estimation - in production would use actual measurements
        1024 // 1GB placeholder
    }
    
    /// Update processing statistics
    fn update_stats(&self, processing_time: Duration, response: &MLProcessingResponse) {
        let mut stats = self.stats.write();
        stats.documents_processed += 1;
        stats.total_qa_pairs_generated += response.qa_pairs.len();
        stats.total_processing_time += processing_time;
        
        // Update average quality (running average)
        let new_quality = response.semantic_quality.overall_score;
        if stats.documents_processed == 1 {
            stats.average_quality = new_quality;
        } else {
            stats.average_quality = (stats.average_quality * (stats.documents_processed - 1) as f64 + new_quality) 
                                  / stats.documents_processed as f64;
        }
    }
    
    /// Increment error counter
    fn increment_error_count(&self) {
        let mut stats = self.stats.write();
        stats.errors_count += 1;
    }
    
    /// Discover all processable documents in directory with parallel scanning
    async fn discover_documents_parallel(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        use futures::stream::{self, StreamExt};
        
        let mut documents = Vec::new();
        let mut dir_queue = vec![dir.to_owned()];
        
        // Process directories in parallel batches
        while !dir_queue.is_empty() {
            let current_batch: Vec<_> = dir_queue.drain(..).collect();
            
            let batch_results: Vec<_> = stream::iter(current_batch)
                .map(|path| async move {
                    let mut batch_documents = Vec::new();
                    let mut batch_directories = Vec::new();
                    
                    if let Ok(mut entries) = fs::read_dir(&path).await {
                        while let Some(entry) = entries.next_entry().await.unwrap_or(None) {
                            let entry_path = entry.path();
                            
                            if entry_path.is_file() {
                                if let Some(extension) = entry_path.extension().and_then(|e| e.to_str()) {
                                    let ext_with_dot = format!(".{}", extension);
                                    if self.config.processing.supported_extensions.contains(&ext_with_dot) {
                                        batch_documents.push(entry_path);
                                    }
                                }
                            } else if entry_path.is_dir() {
                                batch_directories.push(entry_path);
                            }
                        }
                    }
                    
                    (batch_documents, batch_directories)
                })
                .buffer_unordered(16) // Process up to 16 directories concurrently
                .collect()
                .await;
            
            // Collect results from batch processing
            for (batch_docs, batch_dirs) in batch_results {
                documents.extend(batch_docs);
                dir_queue.extend(batch_dirs);
            }
        }
        
        info!("Parallel document discovery found {} files", documents.len());
        Ok(documents)
    }
    
    /// Legacy single-threaded document discovery for compatibility
    async fn discover_documents(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut documents = Vec::new();
        let mut entries = fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if self.config.processing.supported_extensions.contains(&format!(".{}", extension)) {
                        documents.push(path);
                    }
                }
            } else if path.is_dir() {
                // Recursively discover documents
                let mut sub_documents = self.discover_documents(&path).await?;
                documents.append(&mut sub_documents);
            }
        }
        
        Ok(documents)
    }
    
    /// Calculate optimal batch size based on available resources
    fn calculate_optimal_batch_size(&self, total_documents: usize) -> usize {
        let base_batch_size = self.config.processing.batch_size;
        let max_concurrent = self.config.processing.max_concurrent_docs;
        
        if total_documents <= max_concurrent {
            total_documents
        } else {
            std::cmp::min(base_batch_size, total_documents / 4).max(1)
        }
    }
    
    /// Build final pipeline statistics
    fn build_pipeline_stats(&self, total_time: Duration, _collection_results: &CollectionResults) -> PipelineStats {
        let stats = self.stats.read();
        PipelineStats {
            documents_processed: stats.documents_processed,
            total_qa_pairs_generated: stats.total_qa_pairs_generated,
            average_quality: stats.average_quality,
            total_time,
            memory_peak_gb: (stats.memory_peak_mb as f64) / 1024.0,
            errors_encountered: stats.errors_count,
        }
    }
    
    /// Build enhanced pipeline statistics with M3 Max metrics
    fn build_enhanced_pipeline_stats(
        &self, 
        total_time: Duration, 
        collection_results: &CollectionResults,
        processing_stats: M3ProcessingStats
    ) -> PipelineStats {
        let stats = self.stats.read();
        PipelineStats {
            documents_processed: stats.documents_processed,
            total_qa_pairs_generated: stats.total_qa_pairs_generated,
            average_quality: stats.average_quality,
            total_time,
            memory_peak_gb: processing_stats.peak_memory_gb,
            errors_encountered: stats.errors_count,
        }
    }
    
    /// Calculate M3 Max optimized concurrency
    fn calculate_m3_max_concurrency(&self, document_count: usize) -> usize {
        let base_concurrency = if self.config.m3_optimization.use_all_performance_cores {
            16 // M3 Max has 16 cores
        } else {
            8  // Use half the cores
        };
        
        // Adaptive scaling based on document count and memory constraints
        let memory_constrained_concurrency = 
            (self.config.memory.memory_limit_gb * 1024 / 256).min(base_concurrency); // ~256MB per document
        
        let optimal_concurrency = std::cmp::min(
            base_concurrency,
            std::cmp::max(1, document_count / 4) // Don't over-parallelize small batches
        ).min(memory_constrained_concurrency);
        
        info!("Calculated optimal concurrency: {} (base: {}, memory-constrained: {}, docs: {})",
              optimal_concurrency, base_concurrency, memory_constrained_concurrency, document_count);
        
        optimal_concurrency
    }
    
    /// Process documents with M3 Max optimization
    async fn process_documents_m3_optimized(
        &self,
        document_paths: Vec<PathBuf>,
        result_tx: mpsc::Sender<ProcessingResult>,
        concurrency: usize
    ) -> Result<M3ProcessingStats> {
        use futures::stream::{self, StreamExt};
        
        let start_time = Instant::now();
        let mut peak_memory_mb = 0;
        let memory_monitor = Arc::new(AtomicUsize::new(0));
        
        // Create semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(concurrency));
        
        // Process documents in parallel streams with memory monitoring
        let processing_results: Vec<_> = stream::iter(document_paths.into_iter().enumerate())
            .map(|(idx, path)| {
                let processor = self.clone();
                let tx = result_tx.clone();
                let sem = Arc::clone(&semaphore);
                let mem_monitor = Arc::clone(&memory_monitor);
                
                async move {
                    let _permit = sem.acquire().await.unwrap();
                    
                    // Monitor memory usage
                    let memory_before = Self::get_current_memory_usage();
                    mem_monitor.store(memory_before, Ordering::Relaxed);
                    
                    let result = processor.process_single_document(
                        format!("doc-{}", idx),
                        path.clone(),
                        tx
                    ).await;
                    
                    let memory_after = Self::get_current_memory_usage();
                    let memory_delta = memory_after.saturating_sub(memory_before);
                    
                    (result, memory_delta, path)
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;
        
        // Calculate statistics
        let mut successful_documents = 0;
        let mut total_memory_used = 0;
        let mut max_memory_delta = 0;
        
        for (result, memory_delta, path) in processing_results {
            match result {
                Ok(_) => {
                    successful_documents += 1;
                    total_memory_used += memory_delta;
                    max_memory_delta = max_memory_delta.max(memory_delta);
                }
                Err(e) => {
                    error!("Failed to process document {:?}: {}", path, e);
                }
            }
        }
        
        let processing_time = start_time.elapsed();
        let peak_memory_gb = (memory_monitor.load(Ordering::Relaxed) as f64) / 1024.0 / 1024.0 / 1024.0;
        
        Ok(M3ProcessingStats {
            successful_documents,
            processing_time,
            peak_memory_gb,
            average_memory_per_doc_mb: if successful_documents > 0 {
                total_memory_used / successful_documents
            } else {
                0
            },
            max_memory_delta_mb: max_memory_delta,
            concurrency_used: concurrency,
        })
    }
    
    /// Get current memory usage in MB
    fn get_current_memory_usage() -> usize {
        // Simplified memory usage tracking
        // In production, would use more sophisticated memory monitoring
        use std::sync::Mutex;
        static SYSTEM: Mutex<Option<System>> = Mutex::new(None);
        
        let mut system_guard = SYSTEM.lock().unwrap_or_else(|poison| poison.into_inner());
        if system_guard.is_none() {
            *system_guard = Some(System::new());
        }
        
        if let Some(ref mut sys) = *system_guard {
            sys.refresh_memory();
            (sys.used_memory() / 1024 / 1024) as usize // Convert to MB
        } else {
            0
        }
    }
    
    /// Log system information for optimization tracking
    async fn log_system_information(&self) {
        let mut sys = System::new();
        sys.refresh_all();
        
        info!("=== M3 Max System Information ===");
        info!("Total memory: {:.2}GB", sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
        info!("Available memory: {:.2}GB", sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
        info!("CPU cores: {}", sys.cpus().len());
        info!("Configured memory limit: {}GB", self.config.memory.memory_limit_gb);
        info!("Thread pool size: {}", self.config.get_optimal_thread_count());
        info!("====================================");
    }
    
    /// Log performance metrics for optimization tracking
    async fn log_performance_metrics(&self, stats: &PipelineStats) {
        info!("=== Performance Metrics ===");
        info!("Documents processed: {}", stats.documents_processed);
        info!("QA pairs generated: {}", stats.total_qa_pairs_generated);
        info!("Average quality: {:.3}", stats.average_quality);
        info!("Total processing time: {:?}", stats.total_time);
        info!("Throughput: {:.2} docs/hour", stats.throughput_per_hour());
        info!("Memory peak: {:.2}GB", stats.memory_peak_gb);
        info!("Errors: {}", stats.errors_encountered);
        info!("========================");
        
        // Performance target analysis
        let target_throughput = 25.0; // 20-30 docs/hour target
        let actual_throughput = stats.throughput_per_hour();
        
        if actual_throughput >= target_throughput {
            info!("✅ Performance target achieved: {:.2} >= {:.2} docs/hour", 
                  actual_throughput, target_throughput);
        } else {
            warn!("⚠️ Below performance target: {:.2} < {:.2} docs/hour", 
                  actual_throughput, target_throughput);
        }
        
        // Memory utilization analysis
        let target_memory_utilization = 0.90; // 90% target
        let actual_utilization = stats.memory_peak_gb / (self.config.memory.memory_limit_gb as f64);
        
        if actual_utilization >= target_memory_utilization {
            info!("✅ Memory utilization target achieved: {:.1}% >= {:.1}%", 
                  actual_utilization * 100.0, target_memory_utilization * 100.0);
        } else {
            info!("📊 Memory utilization: {:.1}% (target: {:.1}%)", 
                  actual_utilization * 100.0, target_memory_utilization * 100.0);
        }
    }
    
    /// Collect and aggregate processing results with optimization
    async fn collect_results_optimized(
        mut receiver: mpsc::Receiver<ProcessingResult>,
        output_dir: &Path
    ) -> Result<CollectionResults> {
        use tokio::io::AsyncWriteExt;
        
        let mut results = Vec::new();
        let training_data_path = output_dir.join("training_data.jsonl");
        let quality_report_path = output_dir.join("quality_report.json");
        let performance_log_path = output_dir.join("performance.log");
        
        // Create output files with buffering for better I/O performance
        let training_file = fs::File::create(&training_data_path).await?;
        let mut training_writer = tokio::io::BufWriter::new(training_file);
        
        let mut quality_data = Vec::new();
        let mut performance_metrics = Vec::new();
        let mut batch_buffer = Vec::new();
        const BATCH_SIZE: usize = 100;
        
        info!("Starting optimized result collection");
        let collection_start = Instant::now();
        
        // Collect all results with batched I/O
        while let Some(result) = receiver.recv().await {
            batch_buffer.push(result);
            
            // Process batch when it reaches optimal size
            if batch_buffer.len() >= BATCH_SIZE {
                Self::process_result_batch(
                    &mut batch_buffer,
                    &mut training_writer,
                    &mut quality_data,
                    &mut performance_metrics,
                    &mut results
                ).await?;
            }
        }
        
        // Process remaining results
        if !batch_buffer.is_empty() {
            Self::process_result_batch(
                &mut batch_buffer,
                &mut training_writer,
                &mut quality_data,
                &mut performance_metrics,
                &mut results
            ).await?;
        }
        
        // Flush and close training data file
        training_writer.flush().await?;
        drop(training_writer);
        
        // Write quality report
        let quality_report = serde_json::to_string_pretty(&quality_data)?;
        fs::write(&quality_report_path, quality_report).await?;
        
        // Write performance log
        let performance_report = serde_json::to_string_pretty(&performance_metrics)?;
        fs::write(&performance_log_path, performance_report).await?;
        
        let collection_time = collection_start.elapsed();
        info!("Optimized result collection completed in {:?}: {} documents processed", 
              collection_time, results.len());
        
        Ok(CollectionResults {
            total_results: results.len(),
            training_data_path,
            quality_report_path,
        })
    }
    
    /// Process a batch of results efficiently
    async fn process_result_batch(
        batch: &mut Vec<ProcessingResult>,
        training_writer: &mut tokio::io::BufWriter<fs::File>,
        quality_data: &mut Vec<serde_json::Value>,
        performance_metrics: &mut Vec<serde_json::Value>,
        results: &mut Vec<ProcessingResult>
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        
        for result in batch.drain(..) {
            // Write QA pairs to training data file in batch
            for qa_pair in &result.qa_pairs {
                let training_entry = serde_json::json!({
                    "conversation": [
                        {"role": "user", "content": qa_pair.question},
                        {"role": "assistant", "content": qa_pair.answer}
                    ],
                    "metadata": {
                        "document_id": result.document_id,
                        "quality_score": qa_pair.confidence,
                        "complexity": qa_pair.metadata.complexity_level,
                        "parameters": qa_pair.metadata.parameters_mentioned,
                        "technical_terms": qa_pair.metadata.technical_terms,
                        "feature_name": result.original_document.metadata.feature_name,
                        "source_file": result.original_document.path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                    }
                });
                
                let json_line = format!("{}\n", serde_json::to_string(&training_entry)?);
                training_writer.write_all(json_line.as_bytes()).await?;
            }
            
            // Collect quality data with enhanced metrics
            quality_data.push(serde_json::json!({
                "document_id": result.document_id,
                "document_path": result.original_document.path,
                "structural_quality": result.structural_quality,
                "semantic_quality": result.semantic_quality,
                "combined_quality": result.combined_quality_score,
                "qa_pairs_generated": result.qa_pairs.len(),
                "processing_stats": result.processing_stats,
                "feature_metadata": {
                    "feature_name": result.original_document.metadata.feature_name,
                    "parameters_count": result.original_document.metadata.parameters.len(),
                    "counters_count": result.original_document.metadata.counters.len(),
                    "technical_terms_count": result.original_document.metadata.technical_terms.len(),
                    "complexity_level": result.original_document.metadata.complexity_hints.estimated_complexity
                }
            }));
            
            // Collect performance metrics
            performance_metrics.push(serde_json::json!({
                "document_id": result.document_id,
                "processing_time_ms": result.processing_stats.total_processing_time.as_millis(),
                "rust_processing_time_ms": result.processing_stats.rust_processing_time.as_millis(),
                "ml_processing_time_ms": result.processing_stats.ml_processing_time.as_millis(),
                "memory_usage_mb": result.processing_stats.memory_peak_mb,
                "model_used": result.processing_stats.model_used,
                "document_size_bytes": result.original_document.size_bytes,
                "throughput_chars_per_sec": if result.processing_stats.total_processing_time.as_secs() > 0 {
                    result.original_document.size_bytes as f64 / result.processing_stats.total_processing_time.as_secs() as f64
                } else {
                    0.0
                }
            }));
            
            results.push(result);
        }
        
        Ok(())
    }
    
    /// Legacy result collection for compatibility
    async fn collect_results(
        mut receiver: mpsc::Receiver<ProcessingResult>,
        output_dir: &Path
    ) -> Result<CollectionResults> {
        let mut results = Vec::new();
        let training_data_path = output_dir.join("training_data.jsonl");
        let quality_report_path = output_dir.join("quality_report.json");
        
        // Create output files
        let mut training_file = fs::File::create(&training_data_path).await?;
        let mut quality_data = Vec::new();
        
        // Collect all results
        while let Some(result) = receiver.recv().await {
            // Write QA pairs to training data file
            for qa_pair in &result.qa_pairs {
                let training_entry = serde_json::json!({
                    "conversation": [
                        {"role": "user", "content": qa_pair.question},
                        {"role": "assistant", "content": qa_pair.answer}
                    ],
                    "metadata": {
                        "document_id": result.document_id,
                        "quality_score": qa_pair.confidence,
                        "complexity": qa_pair.metadata.complexity_level,
                        "parameters": qa_pair.metadata.parameters_mentioned,
                        "technical_terms": qa_pair.metadata.technical_terms
                    }
                });
                
                let json_line = format!("{}\n", serde_json::to_string(&training_entry)?);
                tokio::io::AsyncWriteExt::write_all(&mut training_file, json_line.as_bytes()).await?;
            }
            
            // Collect quality data
            quality_data.push(serde_json::json!({
                "document_id": result.document_id,
                "structural_quality": result.structural_quality,
                "semantic_quality": result.semantic_quality,
                "combined_quality": result.combined_quality_score,
                "processing_stats": result.processing_stats
            }));
            
            results.push(result);
        }
        
        // Write quality report
        let quality_report = serde_json::to_string_pretty(&quality_data)?;
        fs::write(quality_report_path, quality_report).await?;
        
        info!("Results collected: {} documents processed", results.len());
        
        Ok(CollectionResults {
            total_results: results.len(),
            training_data_path,
            quality_report_path,
        })
    }
    
    // Placeholder implementations for other format extractors
    fn extract_html_metadata(&self, _content: &str) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata::default())
    }
    
    fn extract_pdf_metadata(&self, _content: &str) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata::default())
    }
    
    fn extract_csv_metadata(&self, _content: &str) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata::default())
    }
    
    fn extract_3gpp_metadata(&self, _content: &str) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata::default())
    }
    
    fn extract_text_metadata(&self, _content: &str) -> Result<DocumentMetadata> {
        Ok(DocumentMetadata::default())
    }
}

/// Collection results summary
#[derive(Debug)]
pub struct CollectionResults {
    pub total_results: usize,
    pub training_data_path: PathBuf,
    pub quality_report_path: PathBuf,
}

impl Clone for DocumentProcessor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            ipc_manager: Arc::clone(&self.ipc_manager),
            quality_validator: Arc::clone(&self.quality_validator),
            format_detectors: Arc::clone(&self.format_detectors),
            processing_semaphore: Arc::clone(&self.processing_semaphore),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl FormatDetectors {
    async fn new() -> Result<Self> {
        // Initialize regex patterns for markdown processing
        let markdown_patterns = MarkdownPatterns {
            title_regex: Regex::new(r"(?i)DOCTITLE:\s*(.+)")?,
            feature_name_regex: Regex::new(r"(?i)(?:Feature|Title):\s*(.+)")?,
            product_info_regex: Regex::new(r"(?i)Product:\s*(.+)")?,
            parameter_section_regex: Regex::new(r"(?i)## Parameters\s*(.*?)(?=\n##|\n$)")?,
            counter_section_regex: Regex::new(r"(?i)## Counters\s*(.*?)(?=\n##|\n$)")?,
        };
        
        let parameter_regex = Regex::new(r"[-*]\s*\*\*([^*]+)\*\*[:\s]*([^\n]+)")?;
        let counter_regex = Regex::new(r"[-*]\s*([A-Za-z0-9_]+):\s*([^\n]+)")?;
        
        // Load technical terms from predefined vocabulary
        let technical_terms = Arc::new(Self::load_technical_terms().await?);
        
        Ok(Self {
            markdown_patterns,
            parameter_regex,
            counter_regex,
            technical_terms,
        })
    }
    
    async fn load_technical_terms() -> Result<Vec<String>> {
        // RAN and 3GPP technical terms
        Ok(vec![
            "LTE".to_string(), "5G".to_string(), "NR".to_string(),
            "eNodeB".to_string(), "gNodeB".to_string(), "UE".to_string(),
            "QoS".to_string(), "MIMO".to_string(), "CA".to_string(),
            "VoLTE".to_string(), "IMS".to_string(), "EPC".to_string(),
            "RRC".to_string(), "PDCP".to_string(), "RLC".to_string(),
            "MAC".to_string(), "PHY".to_string(), "TTI".to_string(),
            "HARQ".to_string(), "CQI".to_string(), "PMI".to_string(),
            "RI".to_string(), "SRS".to_string(), "PUCCH".to_string(),
            "PUSCH".to_string(), "PDCCH".to_string(), "PDSCH".to_string(),
            "PRACH".to_string(), "RACH".to_string(), "DRX".to_string(),
            "eDRX".to_string(), "PSM".to_string(), "TAU".to_string(),
        ])
    }
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            title: None,
            feature_name: None,
            product_info: None,
            feature_state: None,
            parameters: Vec::new(),
            counters: Vec::new(),
            technical_terms: Vec::new(),
            complexity_hints: ComplexityHints {
                parameter_count: 0,
                counter_count: 0,
                technical_term_density: 0.0,
                content_length: 0,
                estimated_complexity: ComplexityLevel::Fast,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_document_processor_creation() {
        let config = ProcessingConfig::default_m3_max(4, 8);
        let processor = DocumentProcessor::new(config).await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test] 
    async fn test_format_detection() {
        let config = ProcessingConfig::default_m3_max(4, 8);
        let processor = DocumentProcessor::new(config).await.unwrap();
        
        let md_content = "# Test Document\n## Parameters\n- **param1**: test";
        let format = processor.detect_document_format(Path::new("test.md"), md_content).unwrap();
        assert_eq!(format, DocumentFormat::Markdown);
    }
    
    #[tokio::test]
    async fn test_complexity_calculation() {
        let config = ProcessingConfig::default_m3_max(4, 8);
        let processor = DocumentProcessor::new(config).await.unwrap();
        
        let score = processor.calculate_complexity_score(5, 3, 2.5, 1000);
        assert!(score > 0.0 && score <= 1.0);
    }
}