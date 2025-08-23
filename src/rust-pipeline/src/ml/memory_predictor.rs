/*!
# Memory Usage Predictor for Qwen3 Models

Advanced memory prediction system optimized for M3 Max unified memory architecture.
Predicts memory usage patterns, bottlenecks, and optimal allocation strategies.
*/

use crate::ml::{DocumentType, MLRequest, Priority, Qwen3Model};
use crate::optimization::m3_max::{M3MaxMemoryManager, MemoryStats};
use crate::{PipelineError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Memory prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPrediction {
    pub model: Qwen3Model,
    pub predicted_usage_gb: f64,
    pub peak_usage_gb: f64,
    pub allocation_pattern: AllocationPattern,
    pub fragmentation_risk: f64,
    pub gc_pressure: f64,
    pub bandwidth_utilization: f64,
    pub pool_recommendations: Vec<PoolRecommendation>,
    pub optimization_suggestions: Vec<String>,
}

/// Memory allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    pub initial_load_gb: f64,
    pub working_set_gb: f64,
    pub peak_temporary_gb: f64,
    pub gradient_memory_gb: f64,
    pub cache_memory_gb: f64,
    pub overhead_gb: f64,
}

/// Pool recommendation for M3 Max optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolRecommendation {
    pub pool_type: String, // Serializable version of PoolType
    pub recommended_size_gb: f64,
    pub priority: u8,
    pub optimization_flags: String, // Serializable version of flags
}

/// Memory efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEfficiencyMetrics {
    pub utilization_score: f64,
    pub fragmentation_score: f64,
    pub bandwidth_efficiency: f64,
    pub cache_hit_ratio: f64,
    pub allocation_speed: f64,
    pub deallocation_speed: f64,
}

/// Memory bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub bottleneck_type: BottleneckType,
    pub severity: f64,
    pub estimated_impact: Duration,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BottleneckType {
    InsufficientMemory,
    BandwidthLimited,
    FragmentationHigh,
    GarbageCollection,
    CacheMisses,
    AllocationOverhead,
    None,
}

/// Memory usage pattern for different document types
#[derive(Debug, Clone)]
struct DocumentMemoryProfile {
    base_memory_multiplier: f64,
    peak_memory_multiplier: f64,
    gradient_memory_ratio: f64,
    cache_efficiency: f64,
    fragmentation_tendency: f64,
}

/// Model-specific memory characteristics
#[derive(Debug, Clone)]
struct ModelMemoryProfile {
    base_model_size_gb: f64,
    context_window_memory_ratio: f64,
    activation_memory_ratio: f64,
    kv_cache_ratio: f64,
    gradient_memory_ratio: f64,
    optimizer_memory_ratio: f64,
}

/// Memory Predictor
pub struct MemoryPredictor {
    model_profiles: HashMap<Qwen3Model, ModelMemoryProfile>,
    document_profiles: HashMap<DocumentType, DocumentMemoryProfile>,
    historical_data: Arc<RwLock<HashMap<String, HistoricalMemoryData>>>,
    prediction_cache: Arc<RwLock<HashMap<String, (MemoryPrediction, Instant)>>>,
    memory_manager: Option<Arc<M3MaxMemoryManager>>,
    total_predictions: AtomicU64,
    accurate_predictions: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoricalMemoryData {
    model: Qwen3Model,
    document_type: DocumentType,
    average_usage_gb: f64,
    peak_usage_gb: f64,
    allocation_patterns: Vec<AllocationPattern>,
    efficiency_metrics: MemoryEfficiencyMetrics,
    sample_count: u64,
    last_updated: u64, // timestamp
}

static MEMORY_PREDICTOR: std::sync::OnceLock<Arc<MemoryPredictor>> = std::sync::OnceLock::new();

impl MemoryPredictor {
    /// Create new memory predictor
    pub fn new() -> Self {
        let mut model_profiles = HashMap::new();

        // Qwen3 1.7B profile
        model_profiles.insert(
            Qwen3Model::Qwen3_1_7B,
            ModelMemoryProfile {
                base_model_size_gb: 3.4, // FP16 weights
                context_window_memory_ratio: 0.1,
                activation_memory_ratio: 0.15,
                kv_cache_ratio: 0.05,
                gradient_memory_ratio: 0.3,  // During training/fine-tuning
                optimizer_memory_ratio: 0.2, // Adam states
            },
        );

        // Qwen3 7B profile
        model_profiles.insert(
            Qwen3Model::Qwen3_7B,
            ModelMemoryProfile {
                base_model_size_gb: 14.0,
                context_window_memory_ratio: 0.12,
                activation_memory_ratio: 0.18,
                kv_cache_ratio: 0.08,
                gradient_memory_ratio: 0.35,
                optimizer_memory_ratio: 0.25,
            },
        );

        // Qwen3 30B profile
        model_profiles.insert(
            Qwen3Model::Qwen3_30B,
            ModelMemoryProfile {
                base_model_size_gb: 60.0,
                context_window_memory_ratio: 0.15,
                activation_memory_ratio: 0.2,
                kv_cache_ratio: 0.12,
                gradient_memory_ratio: 0.4,
                optimizer_memory_ratio: 0.3,
            },
        );

        let mut document_profiles = HashMap::new();

        document_profiles.insert(
            DocumentType::PlainText,
            DocumentMemoryProfile {
                base_memory_multiplier: 1.0,
                peak_memory_multiplier: 1.2,
                gradient_memory_ratio: 0.8,
                cache_efficiency: 0.9,
                fragmentation_tendency: 0.1,
            },
        );

        document_profiles.insert(
            DocumentType::Pdf,
            DocumentMemoryProfile {
                base_memory_multiplier: 1.5,
                peak_memory_multiplier: 2.0,
                gradient_memory_ratio: 1.2,
                cache_efficiency: 0.7,
                fragmentation_tendency: 0.3,
            },
        );

        document_profiles.insert(
            DocumentType::Technical,
            DocumentMemoryProfile {
                base_memory_multiplier: 1.3,
                peak_memory_multiplier: 1.8,
                gradient_memory_ratio: 1.1,
                cache_efficiency: 0.8,
                fragmentation_tendency: 0.2,
            },
        );

        // Add profiles for other document types
        for doc_type in [
            DocumentType::Markdown,
            DocumentType::Csv,
            DocumentType::Standards3Gpp,
            DocumentType::EricssonDoc,
        ] {
            document_profiles.insert(
                doc_type,
                DocumentMemoryProfile {
                    base_memory_multiplier: 1.1,
                    peak_memory_multiplier: 1.5,
                    gradient_memory_ratio: 0.9,
                    cache_efficiency: 0.8,
                    fragmentation_tendency: 0.15,
                },
            );
        }

        Self {
            model_profiles,
            document_profiles,
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            memory_manager: None,
            total_predictions: AtomicU64::new(0),
            accurate_predictions: AtomicU64::new(0),
        }
    }

    /// Set M3 Max memory manager
    pub fn set_memory_manager(&mut self, memory_manager: Arc<M3MaxMemoryManager>) {
        self.memory_manager = Some(memory_manager);
    }

    /// Predict memory usage for model and request
    pub async fn predict_memory_usage(
        &self,
        model: &Qwen3Model,
        request: &MLRequest,
    ) -> Result<MemoryPrediction> {
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();

        debug!(
            "Predicting memory usage for model {} and request {}",
            model.name(),
            request.request_id
        );

        // Check prediction cache
        let cache_key = self.generate_cache_key(model, request);
        if let Some(cached) = self.check_prediction_cache(&cache_key).await {
            debug!("Cache hit for memory prediction: {}", cache_key);
            return Ok(cached);
        }

        // Get model and document profiles
        let model_profile = self.model_profiles.get(model).ok_or_else(|| {
            PipelineError::Optimization(format!("No memory profile for model {}", model.name()))
        })?;

        let doc_profile = self
            .document_profiles
            .get(&request.document_type)
            .ok_or_else(|| {
                PipelineError::Optimization(format!(
                    "No memory profile for document type {:?}",
                    request.document_type
                ))
            })?;

        // Calculate allocation pattern
        let allocation_pattern = self
            .calculate_allocation_pattern(model_profile, doc_profile, request)
            .await?;

        // Predict usage and peak
        let predicted_usage_gb = self.calculate_predicted_usage(&allocation_pattern);
        let peak_usage_gb = self.calculate_peak_usage(&allocation_pattern, doc_profile);

        // Analyze fragmentation risk
        let fragmentation_risk = self
            .calculate_fragmentation_risk(model, &allocation_pattern, doc_profile)
            .await?;

        // Calculate GC pressure
        let gc_pressure = self.calculate_gc_pressure(&allocation_pattern, request);

        // Estimate bandwidth utilization
        let bandwidth_utilization = self
            .estimate_bandwidth_utilization(model, &allocation_pattern, request)
            .await?;

        // Generate pool recommendations
        let pool_recommendations = self
            .generate_pool_recommendations(model, &allocation_pattern, predicted_usage_gb)
            .await?;

        // Generate optimization suggestions
        let optimization_suggestions = self
            .generate_optimization_suggestions(
                model,
                &allocation_pattern,
                fragmentation_risk,
                gc_pressure,
            )
            .await;

        let prediction = MemoryPrediction {
            model: model.clone(),
            predicted_usage_gb,
            peak_usage_gb,
            allocation_pattern,
            fragmentation_risk,
            gc_pressure,
            bandwidth_utilization,
            pool_recommendations,
            optimization_suggestions,
        };

        // Cache the prediction
        self.cache_prediction(cache_key, prediction.clone()).await;

        let prediction_time = start_time.elapsed();
        debug!("Memory prediction completed in {}ms for model {} (predicted: {:.1} GB, peak: {:.1} GB)", 
               prediction_time.as_millis(), model.name(), predicted_usage_gb, peak_usage_gb);

        Ok(prediction)
    }

    /// Analyze memory bottlenecks for given scenario
    pub async fn analyze_bottlenecks(
        &self,
        model: &Qwen3Model,
        request: &MLRequest,
        current_memory_state: Option<&MemoryStats>,
    ) -> Result<Vec<BottleneckAnalysis>> {
        let prediction = self.predict_memory_usage(model, request).await?;
        let mut bottlenecks = Vec::new();

        // Check for insufficient memory
        let available_memory = if let Some(ref memory_manager) = self.memory_manager {
            let system_info = memory_manager.get_system_info();
            system_info.total_memory_gb as f64
        } else {
            128.0 // Assume M3 Max
        };

        if prediction.predicted_usage_gb > available_memory * 0.9 {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::InsufficientMemory,
                severity: (prediction.predicted_usage_gb / available_memory).min(1.0),
                estimated_impact: Duration::from_secs(60), // Fallback to slower processing
                mitigation_strategies: vec![
                    "Consider using smaller model variant".to_string(),
                    "Enable memory optimization flags".to_string(),
                    "Implement gradient checkpointing".to_string(),
                ],
            });
        }

        // Check for bandwidth limitations
        if prediction.bandwidth_utilization > 0.85 {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::BandwidthLimited,
                severity: prediction.bandwidth_utilization,
                estimated_impact: Duration::from_millis(
                    (prediction.bandwidth_utilization * 1000.0) as u64,
                ),
                mitigation_strategies: vec![
                    "Optimize data layout for better cache locality".to_string(),
                    "Use memory pooling to reduce allocation overhead".to_string(),
                    "Enable prefetching for sequential access patterns".to_string(),
                ],
            });
        }

        // Check for high fragmentation
        if prediction.fragmentation_risk > 0.7 {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::FragmentationHigh,
                severity: prediction.fragmentation_risk,
                estimated_impact: Duration::from_millis(
                    (prediction.fragmentation_risk * 500.0) as u64,
                ),
                mitigation_strategies: vec![
                    "Use larger allocation chunks".to_string(),
                    "Implement memory compaction".to_string(),
                    "Pre-allocate memory pools".to_string(),
                ],
            });
        }

        // Check for GC pressure
        if prediction.gc_pressure > 0.6 {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::GarbageCollection,
                severity: prediction.gc_pressure,
                estimated_impact: Duration::from_millis((prediction.gc_pressure * 200.0) as u64),
                mitigation_strategies: vec![
                    "Reduce temporary object allocation".to_string(),
                    "Use object pooling".to_string(),
                    "Optimize cleanup patterns".to_string(),
                ],
            });
        }

        // If no bottlenecks found
        if bottlenecks.is_empty() {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::None,
                severity: 0.0,
                estimated_impact: Duration::from_secs(0),
                mitigation_strategies: vec!["System operating optimally".to_string()],
            });
        }

        Ok(bottlenecks)
    }

    /// Calculate allocation pattern for model and request
    async fn calculate_allocation_pattern(
        &self,
        model_profile: &ModelMemoryProfile,
        doc_profile: &DocumentMemoryProfile,
        request: &MLRequest,
    ) -> Result<AllocationPattern> {
        // Base model memory (weights, embeddings, etc.)
        let initial_load_gb = model_profile.base_model_size_gb * doc_profile.base_memory_multiplier;

        // Working set memory (activations, attention states)
        let context_size_factor = (request.document_size_bytes as f64 / 10000.0).min(10.0); // Normalize
        let working_set_gb =
            initial_load_gb * model_profile.activation_memory_ratio * context_size_factor;

        // Peak temporary memory (forward/backward passes)
        let peak_temporary_gb = working_set_gb * doc_profile.peak_memory_multiplier;

        // Gradient memory (if fine-tuning or training)
        let gradient_memory_gb = if matches!(request.priority, Priority::Critical) {
            // Assume critical requests might involve model adaptation
            initial_load_gb
                * model_profile.gradient_memory_ratio
                * doc_profile.gradient_memory_ratio
        } else {
            0.0
        };

        // Cache memory (KV cache, intermediate results)
        let cache_memory_gb =
            initial_load_gb * model_profile.kv_cache_ratio * doc_profile.cache_efficiency;

        // System overhead (Python runtime, CUDA contexts, etc.)
        let overhead_gb = (initial_load_gb + working_set_gb) * 0.15; // 15% overhead

        Ok(AllocationPattern {
            initial_load_gb,
            working_set_gb,
            peak_temporary_gb,
            gradient_memory_gb,
            cache_memory_gb,
            overhead_gb,
        })
    }

    /// Calculate predicted total usage
    fn calculate_predicted_usage(&self, pattern: &AllocationPattern) -> f64 {
        pattern.initial_load_gb
            + pattern.working_set_gb
            + pattern.cache_memory_gb
            + pattern.overhead_gb
            + pattern.gradient_memory_gb * 0.5 // Gradient memory not always fully allocated
    }

    /// Calculate peak usage during processing
    fn calculate_peak_usage(
        &self,
        pattern: &AllocationPattern,
        doc_profile: &DocumentMemoryProfile,
    ) -> f64 {
        let base_usage = self.calculate_predicted_usage(pattern);
        base_usage + pattern.peak_temporary_gb * doc_profile.peak_memory_multiplier
    }

    /// Calculate fragmentation risk
    async fn calculate_fragmentation_risk(
        &self,
        model: &Qwen3Model,
        pattern: &AllocationPattern,
        doc_profile: &DocumentMemoryProfile,
    ) -> Result<f64> {
        // Base fragmentation from document processing characteristics
        let mut fragmentation_risk = doc_profile.fragmentation_tendency;

        // Model size impact - larger models have more fragmentation risk
        let model_factor = match model {
            Qwen3Model::Qwen3_1_7B => 0.1,
            Qwen3Model::Qwen3_7B => 0.2,
            Qwen3Model::Qwen3_30B => 0.4,
        };
        fragmentation_risk += model_factor;

        // Allocation pattern impact
        let allocation_variance =
            (pattern.peak_temporary_gb - pattern.working_set_gb) / pattern.working_set_gb.max(1.0);
        fragmentation_risk += allocation_variance * 0.3;

        // Historical data adjustment
        if let Some(ref memory_manager) = self.memory_manager {
            let stats = memory_manager.get_stats().await?;
            fragmentation_risk = (fragmentation_risk + stats.fragmentation_ratio) / 2.0;
        }

        Ok(fragmentation_risk.min(1.0))
    }

    /// Calculate garbage collection pressure
    fn calculate_gc_pressure(&self, pattern: &AllocationPattern, request: &MLRequest) -> f64 {
        // Base GC pressure from temporary allocations
        let temp_ratio = pattern.peak_temporary_gb / pattern.working_set_gb.max(1.0);
        let mut gc_pressure = temp_ratio * 0.4;

        // Document size impact - larger documents create more temporary objects
        let size_factor = (request.document_size_bytes as f64 / 1_000_000.0).min(1.0);
        gc_pressure += size_factor * 0.3;

        // Processing complexity impact
        gc_pressure += request.complexity_score * 0.3;

        gc_pressure.min(1.0)
    }

    /// Estimate memory bandwidth utilization
    async fn estimate_bandwidth_utilization(
        &self,
        model: &Qwen3Model,
        pattern: &AllocationPattern,
        request: &MLRequest,
    ) -> Result<f64> {
        let total_memory_gb = self.calculate_predicted_usage(pattern);

        // M3 Max unified memory bandwidth: ~400 GB/s
        let max_bandwidth_gbps = 400.0;

        // Estimate memory access patterns
        let model_specs = model.specs();
        let tokens_per_second = model_specs.tokens_per_second as f64;

        // Rough estimate of memory bandwidth needed
        // Each token requires multiple memory accesses for weights, activations, etc.
        let memory_accesses_per_token = total_memory_gb * 0.1; // Conservative estimate
        let required_bandwidth = tokens_per_second * memory_accesses_per_token;

        let utilization = (required_bandwidth / max_bandwidth_gbps).min(1.0);

        // Adjust for document complexity
        let complexity_multiplier = 1.0 + request.complexity_score * 0.5;

        Ok((utilization * complexity_multiplier).min(1.0))
    }

    /// Generate pool recommendations for M3 Max
    async fn generate_pool_recommendations(
        &self,
        model: &Qwen3Model,
        pattern: &AllocationPattern,
        total_usage_gb: f64,
    ) -> Result<Vec<PoolRecommendation>> {
        let mut recommendations = Vec::new();

        // System pool for main model weights
        if pattern.initial_load_gb > 0.0 {
            recommendations.push(PoolRecommendation {
                pool_type: "System".to_string(),
                recommended_size_gb: (pattern.initial_load_gb * 1.2).ceil(), // 20% buffer
                priority: 1,
                optimization_flags: "AMX+Prefetch".to_string(),
            });
        }

        // High-bandwidth pool for activations and working memory
        if pattern.working_set_gb > 0.0 {
            recommendations.push(PoolRecommendation {
                pool_type: "HighBandwidth".to_string(),
                recommended_size_gb: (pattern.working_set_gb * 1.5).ceil(), // 50% buffer for peaks
                priority: 2,
                optimization_flags: "Prefetch+CacheOptimized".to_string(),
            });
        }

        // GPU pool for large models that can benefit from Metal acceleration
        if matches!(model, Qwen3Model::Qwen3_30B) || total_usage_gb > 25.0 {
            recommendations.push(PoolRecommendation {
                pool_type: "Gpu".to_string(),
                recommended_size_gb: (total_usage_gb * 0.3).ceil(), // 30% for GPU acceleration
                priority: 3,
                optimization_flags: "Metal+GPUOptimized".to_string(),
            });
        }

        // Cache pool for KV cache and intermediate results
        if pattern.cache_memory_gb > 0.0 {
            recommendations.push(PoolRecommendation {
                pool_type: "Cache".to_string(),
                recommended_size_gb: (pattern.cache_memory_gb * 2.0).ceil(), // Large cache buffer
                priority: 4,
                optimization_flags: "CacheOptimized+FastAccess".to_string(),
            });
        }

        // Sort by priority
        recommendations.sort_by_key(|r| r.priority);
        Ok(recommendations)
    }

    /// Generate optimization suggestions
    async fn generate_optimization_suggestions(
        &self,
        model: &Qwen3Model,
        pattern: &AllocationPattern,
        fragmentation_risk: f64,
        gc_pressure: f64,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Model-specific suggestions
        match model {
            Qwen3Model::Qwen3_1_7B => {
                suggestions
                    .push("Enable Neural Engine acceleration for faster inference".to_string());
                suggestions.push("Use int8 quantization to reduce memory usage".to_string());
            }
            Qwen3Model::Qwen3_7B => {
                suggestions
                    .push("Balance CPU and GPU utilization for optimal performance".to_string());
                suggestions.push("Consider dynamic batching for multiple requests".to_string());
            }
            Qwen3Model::Qwen3_30B => {
                suggestions.push("Enable model parallelism across CPU and GPU".to_string());
                suggestions.push("Use gradient checkpointing to reduce memory peaks".to_string());
            }
        }

        // Memory pattern suggestions
        if fragmentation_risk > 0.6 {
            suggestions.push("Pre-allocate large contiguous memory blocks".to_string());
            suggestions.push("Enable memory compaction during idle periods".to_string());
        }

        if gc_pressure > 0.6 {
            suggestions.push("Implement object pooling for temporary allocations".to_string());
            suggestions.push("Reduce allocation frequency in hot paths".to_string());
        }

        if pattern.peak_temporary_gb / pattern.working_set_gb > 2.0 {
            suggestions.push("Implement streaming processing to reduce peak memory".to_string());
            suggestions.push("Use memory mapping for large data structures".to_string());
        }

        // M3 Max specific suggestions
        suggestions
            .push("Leverage unified memory architecture for zero-copy operations".to_string());
        suggestions
            .push("Enable Metal Performance Shaders for compute-heavy operations".to_string());

        suggestions
    }

    /// Generate cache key for prediction
    fn generate_cache_key(&self, model: &Qwen3Model, request: &MLRequest) -> String {
        format!(
            "{}:{}:{}:{:.1}:{}",
            model.name(),
            request.document_type as u8,
            request.document_size_bytes / 1024, // KB granularity
            request.complexity_score,
            request.priority as u8
        )
    }

    /// Check prediction cache
    async fn check_prediction_cache(&self, key: &str) -> Option<MemoryPrediction> {
        let cache = self.prediction_cache.read();
        if let Some((prediction, timestamp)) = cache.get(key) {
            // Cache valid for 10 minutes
            if timestamp.elapsed() < Duration::from_secs(600) {
                return Some(prediction.clone());
            }
        }
        None
    }

    /// Cache prediction result
    async fn cache_prediction(&self, key: String, prediction: MemoryPrediction) {
        let mut cache = self.prediction_cache.write();
        cache.insert(key, (prediction, Instant::now()));

        // Limit cache size
        if cache.len() > 500 {
            // Remove oldest entries
            let cutoff = Instant::now() - Duration::from_secs(1200); // 20 minutes
            cache.retain(|_, (_, timestamp)| *timestamp > cutoff);
        }
    }

    /// Update prediction accuracy
    pub async fn update_accuracy(&self, was_accurate: bool) {
        if was_accurate {
            self.accurate_predictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get memory predictor statistics
    pub async fn get_statistics(&self) -> Result<MemoryPredictorStats> {
        let total_predictions = self.total_predictions.load(Ordering::Relaxed);
        let accurate_predictions = self.accurate_predictions.load(Ordering::Relaxed);
        let accuracy_rate = if total_predictions > 0 {
            accurate_predictions as f64 / total_predictions as f64
        } else {
            0.0
        };

        let cache_size = self.prediction_cache.read().len();
        let historical_entries = self.historical_data.read().len();

        Ok(MemoryPredictorStats {
            total_predictions,
            accurate_predictions,
            accuracy_rate,
            cache_size,
            historical_entries,
        })
    }
}

/// Memory predictor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPredictorStats {
    pub total_predictions: u64,
    pub accurate_predictions: u64,
    pub accuracy_rate: f64,
    pub cache_size: usize,
    pub historical_entries: usize,
}

/// Initialize memory predictor
pub async fn initialize() -> Result<()> {
    info!("Initializing Memory Predictor for Qwen3 models");

    let predictor = MemoryPredictor::new();

    MEMORY_PREDICTOR.set(Arc::new(predictor)).map_err(|_| {
        PipelineError::Optimization("Failed to initialize memory predictor".to_string())
    })?;

    info!("Memory Predictor initialized successfully");
    Ok(())
}

/// Predict memory usage for model and request
pub async fn predict_memory_usage(
    model: &Qwen3Model,
    request: &MLRequest,
) -> Result<MemoryPrediction> {
    let predictor = MEMORY_PREDICTOR.get().ok_or_else(|| {
        PipelineError::Optimization("Memory predictor not initialized".to_string())
    })?;

    predictor.predict_memory_usage(model, request).await
}

/// Analyze memory bottlenecks
pub async fn analyze_bottlenecks(
    model: &Qwen3Model,
    request: &MLRequest,
    current_memory_state: Option<&MemoryStats>,
) -> Result<Vec<BottleneckAnalysis>> {
    let predictor = MEMORY_PREDICTOR.get().ok_or_else(|| {
        PipelineError::Optimization("Memory predictor not initialized".to_string())
    })?;

    predictor
        .analyze_bottlenecks(model, request, current_memory_state)
        .await
}

/// Get memory predictor statistics
pub async fn get_statistics() -> Result<MemoryPredictorStats> {
    let predictor = MEMORY_PREDICTOR.get().ok_or_else(|| {
        PipelineError::Optimization("Memory predictor not initialized".to_string())
    })?;

    predictor.get_statistics().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::QualityRequirements;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_memory_predictor_initialization() {
        initialize().await.unwrap();
        let stats = get_statistics().await.unwrap();
        assert_eq!(stats.total_predictions, 0);
    }

    #[tokio::test]
    async fn test_memory_prediction() {
        initialize().await.unwrap();

        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::Technical,
            document_size_bytes: 100_000,
            complexity_score: 0.7,
            priority: Priority::High,
            quality_requirements: QualityRequirements {
                min_score: 0.8,
                consistency_target: 0.85,
                accuracy_threshold: 0.9,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(120)),
        };

        let prediction = predict_memory_usage(&Qwen3Model::Qwen3_7B, &request)
            .await
            .unwrap();

        assert!(prediction.predicted_usage_gb > 0.0);
        assert!(prediction.peak_usage_gb >= prediction.predicted_usage_gb);
        assert!(prediction.fragmentation_risk >= 0.0 && prediction.fragmentation_risk <= 1.0);
        assert!(prediction.gc_pressure >= 0.0 && prediction.gc_pressure <= 1.0);
        assert!(prediction.bandwidth_utilization >= 0.0 && prediction.bandwidth_utilization <= 1.0);
        assert!(!prediction.pool_recommendations.is_empty());
        assert!(!prediction.optimization_suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_bottleneck_analysis() {
        initialize().await.unwrap();

        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::Pdf,
            document_size_bytes: 10_000_000, // Large PDF
            complexity_score: 0.9,
            priority: Priority::Critical,
            quality_requirements: QualityRequirements {
                min_score: 0.9,
                consistency_target: 0.95,
                accuracy_threshold: 0.95,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(60)),
        };

        let bottlenecks = analyze_bottlenecks(&Qwen3Model::Qwen3_30B, &request, None)
            .await
            .unwrap();

        assert!(!bottlenecks.is_empty());
        for bottleneck in bottlenecks {
            assert!(bottleneck.severity >= 0.0 && bottleneck.severity <= 1.0);
            assert!(!bottleneck.mitigation_strategies.is_empty());
        }
    }
}
