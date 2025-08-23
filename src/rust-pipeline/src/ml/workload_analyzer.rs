/*!
# Workload Analyzer for Dynamic Model Selection

Analyzes incoming processing requests to determine complexity, resource requirements,
and optimal model selection criteria for Qwen3 variants in the M3 Max environment.
*/

use crate::ml::{DocumentType, MLRequest, Priority, Qwen3Model};
use crate::{PipelineError, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Workload analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    pub complexity_score: f64,
    pub estimated_processing_time: Duration,
    pub memory_requirements_gb: u32,
    pub cpu_intensity: f64,
    pub io_intensity: f64,
    pub recommended_model: Qwen3Model,
}

/// Document complexity characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub size_complexity: f64,
    pub content_complexity: f64,
    pub structure_complexity: f64,
    pub format_complexity: f64,
    pub processing_complexity: f64,
}

/// Resource utilization pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePattern {
    pub cpu_bound: bool,
    pub memory_bound: bool,
    pub io_bound: bool,
    pub gpu_accelerated: bool,
    pub neural_engine_suitable: bool,
}

/// Historical workload data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadHistory {
    pub document_type: DocumentType,
    pub average_complexity: f64,
    pub average_processing_time: Duration,
    pub average_memory_usage: u32,
    pub success_rate: f64,
    pub optimal_model: Qwen3Model,
    pub sample_count: u64,
}

/// Workload Analyzer
pub struct WorkloadAnalyzer {
    complexity_calculator: Arc<ComplexityCalculator>,
    resource_predictor: Arc<ResourcePredictor>,
    historical_data: Arc<RwLock<HashMap<DocumentType, WorkloadHistory>>>,
    analysis_cache: Arc<RwLock<HashMap<String, (WorkloadAnalysis, Instant)>>>,
    total_analyses: AtomicU64,
    cache_hits: AtomicU64,
}

#[derive(Debug)]
struct ComplexityCalculator {
    size_thresholds: SizeThresholds,
    content_patterns: Arc<Mutex<HashMap<String, f64>>>,
    format_weights: HashMap<DocumentType, f64>,
}

#[derive(Debug)]
struct SizeThresholds {
    small_bytes: usize,
    medium_bytes: usize,
    large_bytes: usize,
    huge_bytes: usize,
}

impl Default for SizeThresholds {
    fn default() -> Self {
        Self {
            small_bytes: 1024,            // 1KB
            medium_bytes: 1024 * 100,     // 100KB
            large_bytes: 1024 * 1024,     // 1MB
            huge_bytes: 1024 * 1024 * 10, // 10MB
        }
    }
}

#[derive(Debug)]
struct ResourcePredictor {
    model_profiles: HashMap<Qwen3Model, ModelProfile>,
    document_profiles: HashMap<DocumentType, DocumentProfile>,
    system_calibration: SystemCalibration,
}

#[derive(Debug, Clone)]
struct ModelProfile {
    base_memory_gb: u32,
    memory_per_token: f64,
    tokens_per_second: u32,
    quality_score: f64,
    cpu_efficiency: f64,
    gpu_utilization: f64,
}

#[derive(Debug, Clone)]
struct DocumentProfile {
    average_tokens_per_byte: f64,
    complexity_multiplier: f64,
    memory_overhead: f64,
    processing_difficulty: f64,
}

#[derive(Debug, Clone)]
struct SystemCalibration {
    m3_max_performance_multiplier: f64,
    neural_engine_speedup: f64,
    memory_bandwidth_gbps: f64,
    cache_efficiency: f64,
}

static WORKLOAD_ANALYZER: std::sync::OnceLock<Arc<WorkloadAnalyzer>> = std::sync::OnceLock::new();

impl WorkloadAnalyzer {
    /// Create new workload analyzer
    pub fn new() -> Self {
        let complexity_calculator = Arc::new(ComplexityCalculator::new());
        let resource_predictor = Arc::new(ResourcePredictor::new());

        Self {
            complexity_calculator,
            resource_predictor,
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            total_analyses: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// Analyze workload for the given ML request
    pub async fn analyze(&self, request: &MLRequest) -> Result<WorkloadAnalysis> {
        self.total_analyses.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();

        debug!(
            "Starting workload analysis for request {}",
            request.request_id
        );

        // Check cache first
        let cache_key = self.generate_cache_key(request);
        if let Some(cached) = self.check_cache(&cache_key).await {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            debug!("Cache hit for workload analysis: {}", cache_key);
            return Ok(cached);
        }

        // Calculate complexity metrics
        let complexity = self.calculate_complexity(request).await?;

        // Predict resource requirements
        let resources = self.predict_resources(request, &complexity).await?;

        // Determine resource utilization pattern
        let pattern = self
            .analyze_resource_pattern(request, &complexity, &resources)
            .await?;

        // Recommend optimal model
        let recommended_model = self
            .recommend_model(request, &complexity, &resources, &pattern)
            .await?;

        // Create workload analysis
        let analysis = WorkloadAnalysis {
            complexity_score: complexity.processing_complexity,
            estimated_processing_time: resources.estimated_time,
            memory_requirements_gb: resources.memory_gb,
            cpu_intensity: pattern.cpu_intensity,
            io_intensity: pattern.io_intensity,
            recommended_model,
        };

        // Cache the result
        self.cache_analysis(cache_key, analysis.clone()).await;

        // Update historical data
        self.update_historical_data(request, &analysis).await?;

        let analysis_time = start_time.elapsed();
        debug!(
            "Workload analysis completed in {}ms for request {}",
            analysis_time.as_millis(),
            request.request_id
        );

        Ok(analysis)
    }

    /// Calculate document complexity metrics
    async fn calculate_complexity(&self, request: &MLRequest) -> Result<ComplexityAnalysis> {
        let calculator = &self.complexity_calculator;

        // Size-based complexity
        let size_complexity = calculator.calculate_size_complexity(request.document_size_bytes);

        // Content-based complexity (estimated from document type and size)
        let content_complexity = calculator
            .calculate_content_complexity(&request.document_type, request.document_size_bytes);

        // Structure complexity based on document type
        let structure_complexity =
            calculator.calculate_structure_complexity(&request.document_type);

        // Format-specific complexity
        let format_complexity = calculator.get_format_complexity(&request.document_type);

        // Overall processing complexity
        let processing_complexity = self.combine_complexity_scores(
            size_complexity,
            content_complexity,
            structure_complexity,
            format_complexity,
            request.complexity_score,
        );

        Ok(ComplexityAnalysis {
            size_complexity,
            content_complexity,
            structure_complexity,
            format_complexity,
            processing_complexity,
        })
    }

    /// Predict resource requirements
    async fn predict_resources(
        &self,
        request: &MLRequest,
        complexity: &ComplexityAnalysis,
    ) -> Result<ResourceRequirements> {
        let predictor = &self.resource_predictor;

        // Estimate tokens based on document size and type
        let estimated_tokens = predictor.estimate_tokens(request);

        // Base memory requirements
        let base_memory = predictor.calculate_base_memory_requirements(
            &request.document_type,
            estimated_tokens,
            complexity.processing_complexity,
        );

        // Processing time estimation
        let estimated_time = predictor.estimate_processing_time(
            estimated_tokens,
            complexity.processing_complexity,
            &request.priority,
        );

        // Memory requirements with overhead
        let memory_gb = predictor.calculate_memory_with_overhead(
            base_memory,
            complexity.processing_complexity,
            &request.document_type,
        );

        Ok(ResourceRequirements {
            estimated_tokens,
            memory_gb,
            estimated_time,
            base_memory,
        })
    }

    /// Analyze resource utilization pattern
    async fn analyze_resource_pattern(
        &self,
        request: &MLRequest,
        complexity: &ComplexityAnalysis,
        resources: &ResourceRequirements,
    ) -> Result<ResourceUtilizationPattern> {
        // CPU intensity based on complexity and document type
        let cpu_intensity = match request.document_type {
            DocumentType::PlainText => 0.3 + complexity.content_complexity * 0.3,
            DocumentType::Markdown => 0.4 + complexity.structure_complexity * 0.3,
            DocumentType::Pdf => 0.7 + complexity.format_complexity * 0.2,
            DocumentType::Csv => 0.2 + complexity.size_complexity * 0.4,
            DocumentType::Technical => 0.6 + complexity.processing_complexity * 0.3,
            DocumentType::Standards3Gpp => 0.8 + complexity.processing_complexity * 0.2,
            DocumentType::EricssonDoc => 0.7 + complexity.content_complexity * 0.3,
        }
        .min(1.0);

        // I/O intensity based on size and format
        let io_intensity = if request.document_size_bytes > 1024 * 1024 {
            // > 1MB
            0.6 + complexity.size_complexity * 0.3
        } else {
            0.2 + complexity.size_complexity * 0.5
        }
        .min(1.0);

        // Memory intensity based on requirements
        let memory_intensity = (resources.memory_gb as f64 / 50.0).min(1.0); // Normalize to 50GB max

        // GPU suitability based on model requirements
        let gpu_suitable = resources.memory_gb > 20 && complexity.processing_complexity > 0.6;

        // Neural Engine suitability for specific tasks
        let neural_engine_suitable = matches!(
            request.document_type,
            DocumentType::PlainText | DocumentType::Markdown
        ) && complexity.processing_complexity < 0.7;

        Ok(ResourceUtilizationPattern {
            cpu_intensity,
            io_intensity,
            memory_intensity,
            gpu_suitable,
            neural_engine_suitable,
        })
    }

    /// Recommend optimal model based on analysis
    async fn recommend_model(
        &self,
        request: &MLRequest,
        complexity: &ComplexityAnalysis,
        resources: &ResourceRequirements,
        pattern: &ResourceUtilizationPattern,
    ) -> Result<Qwen3Model> {
        // Check historical data for this document type
        let historical_model = {
            let history = self.historical_data.read();
            history.get(&request.document_type).and_then(|h| {
                if h.success_rate > 0.8 {
                    Some(h.optimal_model.clone())
                } else {
                    None
                }
            })
        };

        // If we have successful historical data, consider it strongly
        if let Some(hist_model) = historical_model {
            let hist_specs = hist_model.specs();
            if resources.memory_gb <= hist_specs.memory_gb
                && complexity.processing_complexity >= 0.5
            {
                return Ok(hist_model);
            }
        }

        // Model selection logic based on multiple factors
        let model =
            if request.priority == Priority::Critical || complexity.processing_complexity > 0.8 {
                // High priority or complex tasks: use best model
                Qwen3Model::Qwen3_30B
            } else if complexity.processing_complexity < 0.3 && resources.memory_gb <= 12 {
                // Simple tasks: use fastest model
                Qwen3Model::Qwen3_1_7B
            } else if pattern.memory_intensity < 0.5 && pattern.cpu_intensity > 0.6 {
                // CPU-bound, memory-efficient tasks: balanced model
                Qwen3Model::Qwen3_7B
            } else if resources.memory_gb > 35 || request.quality_requirements.min_score > 0.85 {
                // High memory or quality requirements: use largest model
                Qwen3Model::Qwen3_30B
            } else {
                // Default to balanced model
                Qwen3Model::Qwen3_7B
            };

        // Validate the recommendation against constraints
        let specs = model.specs();
        if resources.memory_gb > specs.memory_gb {
            warn!("Selected model {} may not have sufficient memory ({} GB required, {} GB available)", 
                  model.name(), resources.memory_gb, specs.memory_gb);
        }

        debug!(
            "Recommended model {} for request {} (complexity: {:.3}, memory: {} GB)",
            model.name(),
            request.request_id,
            complexity.processing_complexity,
            resources.memory_gb
        );

        Ok(model)
    }

    /// Combine multiple complexity scores
    fn combine_complexity_scores(
        &self,
        size: f64,
        content: f64,
        structure: f64,
        format: f64,
        provided: f64,
    ) -> f64 {
        // Weighted combination of complexity scores
        let calculated = size * 0.2 + content * 0.3 + structure * 0.3 + format * 0.2;

        // If user provided a complexity score, blend it with our calculation
        if provided > 0.0 {
            (calculated * 0.7 + provided * 0.3).min(1.0)
        } else {
            calculated.min(1.0)
        }
    }

    /// Generate cache key for request
    fn generate_cache_key(&self, request: &MLRequest) -> String {
        format!(
            "{}:{}:{}:{:.2}",
            request.document_type as u8,
            request.document_size_bytes / 1024, // KB granularity
            request.priority as u8,
            request.complexity_score
        )
    }

    /// Check analysis cache
    async fn check_cache(&self, key: &str) -> Option<WorkloadAnalysis> {
        let cache = self.analysis_cache.read();
        if let Some((analysis, timestamp)) = cache.get(key) {
            // Cache valid for 5 minutes
            if timestamp.elapsed() < Duration::from_secs(300) {
                return Some(analysis.clone());
            }
        }
        None
    }

    /// Cache analysis result
    async fn cache_analysis(&self, key: String, analysis: WorkloadAnalysis) {
        let mut cache = self.analysis_cache.write();
        cache.insert(key, (analysis, Instant::now()));

        // Limit cache size
        if cache.len() > 1000 {
            // Remove oldest entries
            let cutoff = Instant::now() - Duration::from_secs(600); // 10 minutes
            cache.retain(|_, (_, timestamp)| *timestamp > cutoff);
        }
    }

    /// Update historical workload data
    async fn update_historical_data(
        &self,
        request: &MLRequest,
        analysis: &WorkloadAnalysis,
    ) -> Result<()> {
        let mut history = self.historical_data.write();

        let entry = history
            .entry(request.document_type.clone())
            .or_insert(WorkloadHistory {
                document_type: request.document_type.clone(),
                average_complexity: 0.0,
                average_processing_time: Duration::from_secs(0),
                average_memory_usage: 0,
                success_rate: 1.0, // Start optimistic
                optimal_model: analysis.recommended_model.clone(),
                sample_count: 0,
            });

        // Update rolling averages
        let old_count = entry.sample_count;
        let new_count = old_count + 1;
        let weight = 1.0 / new_count as f64;
        let old_weight = 1.0 - weight;

        entry.average_complexity =
            entry.average_complexity * old_weight + analysis.complexity_score * weight;
        entry.average_processing_time = Duration::from_nanos(
            (entry.average_processing_time.as_nanos() as f64 * old_weight
                + analysis.estimated_processing_time.as_nanos() as f64 * weight) as u64,
        );
        entry.average_memory_usage = (entry.average_memory_usage as f64 * old_weight
            + analysis.memory_requirements_gb as f64 * weight)
            as u32;
        entry.sample_count = new_count;

        Ok(())
    }

    /// Get workload statistics
    pub async fn get_statistics(&self) -> Result<WorkloadStats> {
        let total_analyses = self.total_analyses.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_hit_rate = if total_analyses > 0 {
            cache_hits as f64 / total_analyses as f64
        } else {
            0.0
        };

        let historical_data = self.historical_data.read().clone();
        let cache_size = self.analysis_cache.read().len();

        Ok(WorkloadStats {
            total_analyses,
            cache_hits,
            cache_hit_rate,
            cache_size,
            historical_data,
        })
    }
}

impl ComplexityCalculator {
    fn new() -> Self {
        let mut format_weights = HashMap::new();
        format_weights.insert(DocumentType::PlainText, 0.2);
        format_weights.insert(DocumentType::Markdown, 0.4);
        format_weights.insert(DocumentType::Pdf, 0.8);
        format_weights.insert(DocumentType::Csv, 0.3);
        format_weights.insert(DocumentType::Technical, 0.7);
        format_weights.insert(DocumentType::Standards3Gpp, 0.9);
        format_weights.insert(DocumentType::EricssonDoc, 0.8);

        Self {
            size_thresholds: SizeThresholds::default(),
            content_patterns: Arc::new(Mutex::new(HashMap::new())),
            format_weights,
        }
    }

    fn calculate_size_complexity(&self, size_bytes: usize) -> f64 {
        match size_bytes {
            s if s <= self.size_thresholds.small_bytes => 0.1,
            s if s <= self.size_thresholds.medium_bytes => 0.3,
            s if s <= self.size_thresholds.large_bytes => 0.6,
            s if s <= self.size_thresholds.huge_bytes => 0.8,
            _ => 1.0,
        }
    }

    fn calculate_content_complexity(&self, doc_type: &DocumentType, size_bytes: usize) -> f64 {
        let base_complexity = match doc_type {
            DocumentType::PlainText => 0.2,
            DocumentType::Markdown => 0.4,
            DocumentType::Pdf => 0.7,
            DocumentType::Csv => 0.3,
            DocumentType::Technical => 0.8,
            DocumentType::Standards3Gpp => 0.9,
            DocumentType::EricssonDoc => 0.8,
        };

        // Adjust for size - larger documents tend to be more complex
        let size_factor = (size_bytes as f64 / 1_000_000.0).min(1.0); // Normalize to 1MB
        (base_complexity + size_factor * 0.2).min(1.0)
    }

    fn calculate_structure_complexity(&self, doc_type: &DocumentType) -> f64 {
        match doc_type {
            DocumentType::PlainText => 0.1,
            DocumentType::Markdown => 0.5,
            DocumentType::Pdf => 0.8,
            DocumentType::Csv => 0.4,
            DocumentType::Technical => 0.7,
            DocumentType::Standards3Gpp => 0.9,
            DocumentType::EricssonDoc => 0.7,
        }
    }

    fn get_format_complexity(&self, doc_type: &DocumentType) -> f64 {
        *self.format_weights.get(doc_type).unwrap_or(&0.5)
    }
}

impl ResourcePredictor {
    fn new() -> Self {
        let mut model_profiles = HashMap::new();

        model_profiles.insert(
            Qwen3Model::Qwen3_1_7B,
            ModelProfile {
                base_memory_gb: 12,
                memory_per_token: 0.001,
                tokens_per_second: 150,
                quality_score: 0.72,
                cpu_efficiency: 0.9,
                gpu_utilization: 0.3,
            },
        );

        model_profiles.insert(
            Qwen3Model::Qwen3_7B,
            ModelProfile {
                base_memory_gb: 28,
                memory_per_token: 0.004,
                tokens_per_second: 80,
                quality_score: 0.82,
                cpu_efficiency: 0.8,
                gpu_utilization: 0.6,
            },
        );

        model_profiles.insert(
            Qwen3Model::Qwen3_30B,
            ModelProfile {
                base_memory_gb: 45,
                memory_per_token: 0.015,
                tokens_per_second: 25,
                quality_score: 0.92,
                cpu_efficiency: 0.7,
                gpu_utilization: 0.9,
            },
        );

        let mut document_profiles = HashMap::new();
        document_profiles.insert(
            DocumentType::PlainText,
            DocumentProfile {
                average_tokens_per_byte: 0.25,
                complexity_multiplier: 1.0,
                memory_overhead: 1.1,
                processing_difficulty: 1.0,
            },
        );

        document_profiles.insert(
            DocumentType::Pdf,
            DocumentProfile {
                average_tokens_per_byte: 0.15,
                complexity_multiplier: 2.0,
                memory_overhead: 1.5,
                processing_difficulty: 2.5,
            },
        );

        // Add other document profiles...

        Self {
            model_profiles,
            document_profiles,
            system_calibration: SystemCalibration {
                m3_max_performance_multiplier: 1.3,
                neural_engine_speedup: 2.0,
                memory_bandwidth_gbps: 400.0,
                cache_efficiency: 0.9,
            },
        }
    }

    fn estimate_tokens(&self, request: &MLRequest) -> u64 {
        let profile = self
            .document_profiles
            .get(&request.document_type)
            .cloned()
            .unwrap_or(DocumentProfile {
                average_tokens_per_byte: 0.2,
                complexity_multiplier: 1.0,
                memory_overhead: 1.2,
                processing_difficulty: 1.5,
            });

        ((request.document_size_bytes as f64 * profile.average_tokens_per_byte) as u64).max(100)
    }

    fn calculate_base_memory_requirements(
        &self,
        doc_type: &DocumentType,
        tokens: u64,
        complexity: f64,
    ) -> u32 {
        let profile = self
            .document_profiles
            .get(doc_type)
            .cloned()
            .unwrap_or_default();

        let base_memory =
            (tokens as f64 * 0.001 * profile.memory_overhead * (1.0 + complexity)) as u32;
        base_memory.max(4) // Minimum 4GB
    }

    fn estimate_processing_time(
        &self,
        tokens: u64,
        complexity: f64,
        priority: &Priority,
    ) -> Duration {
        // Base processing time estimate
        let base_seconds = (tokens as f64 / 100.0) * (1.0 + complexity);

        // Priority adjustment
        let priority_multiplier = match priority {
            Priority::Critical => 0.7, // Faster processing for critical
            Priority::High => 0.8,
            Priority::Medium => 1.0,
            Priority::Low => 1.2,
        };

        let adjusted_seconds = base_seconds * priority_multiplier;
        Duration::from_secs_f64(adjusted_seconds.max(1.0))
    }

    fn calculate_memory_with_overhead(
        &self,
        base_memory: u32,
        complexity: f64,
        doc_type: &DocumentType,
    ) -> u32 {
        let profile = self
            .document_profiles
            .get(doc_type)
            .cloned()
            .unwrap_or_default();

        let overhead_multiplier = profile.memory_overhead * (1.0 + complexity * 0.5);
        ((base_memory as f64 * overhead_multiplier) as u32).min(128) // M3 Max limit
    }
}

impl Default for DocumentProfile {
    fn default() -> Self {
        Self {
            average_tokens_per_byte: 0.2,
            complexity_multiplier: 1.0,
            memory_overhead: 1.2,
            processing_difficulty: 1.5,
        }
    }
}

// Helper structs
#[derive(Debug, Clone)]
struct ComplexityAnalysis {
    size_complexity: f64,
    content_complexity: f64,
    structure_complexity: f64,
    format_complexity: f64,
    processing_complexity: f64,
}

#[derive(Debug, Clone)]
struct ResourceRequirements {
    estimated_tokens: u64,
    memory_gb: u32,
    estimated_time: Duration,
    base_memory: u32,
}

#[derive(Debug, Clone)]
struct ResourceUtilizationPattern {
    cpu_intensity: f64,
    io_intensity: f64,
    memory_intensity: f64,
    gpu_suitable: bool,
    neural_engine_suitable: bool,
}

/// Workload analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadStats {
    pub total_analyses: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: f64,
    pub cache_size: usize,
    pub historical_data: HashMap<DocumentType, WorkloadHistory>,
}

/// Initialize workload analyzer
pub async fn initialize() -> Result<()> {
    info!("Initializing Workload Analyzer for dynamic model selection");

    let analyzer = WorkloadAnalyzer::new();

    WORKLOAD_ANALYZER.set(Arc::new(analyzer)).map_err(|_| {
        PipelineError::Optimization("Failed to initialize workload analyzer".to_string())
    })?;

    info!("Workload Analyzer initialized successfully");
    Ok(())
}

/// Analyze workload for ML request
pub async fn analyze(request: &MLRequest) -> Result<WorkloadAnalysis> {
    let analyzer = WORKLOAD_ANALYZER.get().ok_or_else(|| {
        PipelineError::Optimization("Workload analyzer not initialized".to_string())
    })?;

    analyzer.analyze(request).await
}

/// Get workload analysis statistics
pub async fn get_statistics() -> Result<WorkloadStats> {
    let analyzer = WORKLOAD_ANALYZER.get().ok_or_else(|| {
        PipelineError::Optimization("Workload analyzer not initialized".to_string())
    })?;

    analyzer.get_statistics().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_workload_analyzer_initialization() {
        initialize().await.unwrap();
        let stats = get_statistics().await.unwrap();
        assert_eq!(stats.total_analyses, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn test_complexity_calculation() {
        let calculator = ComplexityCalculator::new();

        // Test size complexity
        assert_eq!(calculator.calculate_size_complexity(500), 0.1);
        assert_eq!(calculator.calculate_size_complexity(50_000), 0.3);
        assert_eq!(calculator.calculate_size_complexity(500_000), 0.6);

        // Test content complexity
        let plain_text = calculator.calculate_content_complexity(&DocumentType::PlainText, 1000);
        let pdf = calculator.calculate_content_complexity(&DocumentType::Pdf, 1000);
        assert!(pdf > plain_text);
    }

    #[tokio::test]
    async fn test_workload_analysis() {
        initialize().await.unwrap();

        let request = crate::ml::MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::Technical,
            document_size_bytes: 50_000,
            complexity_score: 0.7,
            priority: Priority::High,
            quality_requirements: crate::ml::QualityRequirements {
                min_score: 0.8,
                consistency_target: 0.85,
                accuracy_threshold: 0.9,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(120)),
        };

        let analysis = analyze(&request).await.unwrap();

        assert!(analysis.complexity_score > 0.0);
        assert!(analysis.complexity_score <= 1.0);
        assert!(analysis.estimated_processing_time > Duration::from_secs(0));
        assert!(analysis.memory_requirements_gb > 0);
        assert!(analysis.cpu_intensity >= 0.0 && analysis.cpu_intensity <= 1.0);
        assert!(analysis.io_intensity >= 0.0 && analysis.io_intensity <= 1.0);
    }
}
