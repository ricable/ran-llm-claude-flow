/*!
# Performance Benchmark System for Qwen3 Model Selection

Real-time performance benchmarking and model switching optimization system.
Continuously monitors and benchmarks model performance to improve selection accuracy.
*/

use crate::ml::{MLRequest, MLResponse, Qwen3Model, DocumentType, Priority, ProcessingStatus};
use crate::{Result, PipelineError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub model: Qwen3Model,
    pub document_type: DocumentType,
    pub throughput_docs_per_hour: f64,
    pub average_latency: Duration,
    pub quality_score: f64,
    pub memory_efficiency: f64,
    pub success_rate: f64,
    pub resource_utilization: ResourceUtilization,
    pub timestamp: u64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_gb: f64,
    pub gpu_usage_percent: f64,
    pub neural_engine_usage_percent: f64,
    pub memory_bandwidth_gbps: f64,
}

/// Model switching event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSwitchEvent {
    pub from_model: Option<Qwen3Model>,
    pub to_model: Qwen3Model,
    pub switching_time: Duration,
    pub reason: String,
    pub performance_impact: f64,
    pub timestamp: u64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub model: Qwen3Model,
    pub document_type: DocumentType,
    pub trend_direction: TrendDirection,
    pub trend_magnitude: f64,
    pub confidence: f64,
    pub prediction_accuracy: f64,
    pub sample_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub enabled: bool,
    pub warmup_samples: u32,
    pub min_samples_for_trend: u32,
    pub benchmark_interval: Duration,
    pub trend_analysis_window: Duration,
    pub performance_threshold: f64,
    pub automatic_switching: bool,
    pub max_switching_frequency: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            warmup_samples: 5,
            min_samples_for_trend: 10,
            benchmark_interval: Duration::from_secs(60),
            trend_analysis_window: Duration::from_secs(3600), // 1 hour
            performance_threshold: 0.8,
            automatic_switching: true,
            max_switching_frequency: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Performance Benchmarker
pub struct PerformanceBenchmark {
    config: Arc<RwLock<BenchmarkConfig>>,
    benchmark_results: Arc<RwLock<HashMap<String, Vec<BenchmarkResult>>>>,
    switching_history: Arc<RwLock<Vec<ModelSwitchEvent>>>,
    performance_trends: Arc<RwLock<HashMap<String, PerformanceTrend>>>,
    model_warmup_state: Arc<RwLock<HashMap<Qwen3Model, WarmupState>>>,
    active_benchmarks: Arc<Mutex<HashMap<String, ActiveBenchmark>>>,
    total_benchmarks: AtomicU64,
    total_switches: AtomicU64,
    last_switch_time: Arc<Mutex<Option<Instant>>>,
}

#[derive(Debug, Clone)]
struct WarmupState {
    samples_collected: u32,
    is_warmed_up: bool,
    warmup_start_time: Instant,
    average_warmup_time: Duration,
}

#[derive(Debug, Clone)]
struct ActiveBenchmark {
    model: Qwen3Model,
    start_time: Instant,
    samples: Vec<BenchmarkSample>,
    expected_completion: Instant,
}

#[derive(Debug, Clone)]
struct BenchmarkSample {
    latency: Duration,
    quality_score: f64,
    memory_used_gb: f64,
    success: bool,
    timestamp: Instant,
}

static PERFORMANCE_BENCHMARK: std::sync::OnceLock<Arc<PerformanceBenchmark>> = std::sync::OnceLock::new();

impl PerformanceBenchmark {
    /// Create new performance benchmark system
    pub fn new(config: BenchmarkConfig) -> Self {
        let mut model_warmup_state = HashMap::new();
        for model in Qwen3Model::all_models() {
            model_warmup_state.insert(model, WarmupState {
                samples_collected: 0,
                is_warmed_up: false,
                warmup_start_time: Instant::now(),
                average_warmup_time: Duration::from_secs(0),
            });
        }

        Self {
            config: Arc::new(RwLock::new(config)),
            benchmark_results: Arc::new(RwLock::new(HashMap::new())),
            switching_history: Arc::new(RwLock::new(Vec::new())),
            performance_trends: Arc::new(RwLock::new(HashMap::new())),
            model_warmup_state: Arc::new(RwLock::new(model_warmup_state)),
            active_benchmarks: Arc::new(Mutex::new(HashMap::new())),
            total_benchmarks: AtomicU64::new(0),
            total_switches: AtomicU64::new(0),
            last_switch_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Record ML processing result for benchmarking
    pub async fn record_result(&self, request: &MLRequest, response: &MLResponse) -> Result<()> {
        if !self.config.read().enabled {
            return Ok(());
        }

        let benchmark_key = self.generate_benchmark_key(&response.model_used, &request.document_type);
        
        debug!("Recording benchmark result for {} on {}", 
               response.model_used.name(), request.document_type as u8);

        // Update warmup state
        self.update_warmup_state(&response.model_used, response).await;

        // Create benchmark sample
        let sample = BenchmarkSample {
            latency: response.processing_time,
            quality_score: response.quality_score,
            memory_used_gb: response.memory_used_mb as f64 / 1024.0,
            success: matches!(response.status, ProcessingStatus::Success),
            timestamp: Instant::now(),
        };

        // Add to active benchmark if exists
        {
            let mut active = self.active_benchmarks.lock();
            if let Some(benchmark) = active.get_mut(&benchmark_key) {
                benchmark.samples.push(sample.clone());
            }
        }

        // Calculate resource utilization (simulated for now)
        let resource_utilization = self.calculate_resource_utilization(&response.model_used).await;

        // Create benchmark result
        let benchmark_result = BenchmarkResult {
            model: response.model_used.clone(),
            document_type: request.document_type.clone(),
            throughput_docs_per_hour: self.calculate_throughput(response.processing_time),
            average_latency: response.processing_time,
            quality_score: response.quality_score,
            memory_efficiency: response.quality_score * 1000.0 / (response.memory_used_mb as f64).max(1.0),
            success_rate: if matches!(response.status, ProcessingStatus::Success) { 1.0 } else { 0.0 },
            resource_utilization,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        // Store benchmark result
        {
            let mut results = self.benchmark_results.write();
            let entry = results.entry(benchmark_key.clone()).or_insert_with(Vec::new);
            entry.push(benchmark_result);

            // Limit history size
            if entry.len() > 1000 {
                entry.drain(0..100); // Remove oldest 100 entries
            }
        }

        // Update performance trends
        self.update_performance_trends(&benchmark_key).await?;

        // Check for automatic model switching opportunity
        if self.config.read().automatic_switching {
            self.evaluate_switching_opportunity(&request.document_type).await?;
        }

        self.total_benchmarks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Start continuous benchmarking for a model
    pub async fn start_benchmark(&self, model: Qwen3Model, document_type: DocumentType) -> Result<String> {
        let benchmark_key = self.generate_benchmark_key(&model, &document_type);
        let config = self.config.read();
        
        let benchmark = ActiveBenchmark {
            model: model.clone(),
            start_time: Instant::now(),
            samples: Vec::new(),
            expected_completion: Instant::now() + config.benchmark_interval,
        };

        {
            let mut active = self.active_benchmarks.lock();
            active.insert(benchmark_key.clone(), benchmark);
        }

        info!("Started benchmark for {} on {:?}", model.name(), document_type);
        Ok(benchmark_key)
    }

    /// Complete and analyze benchmark
    pub async fn complete_benchmark(&self, benchmark_key: &str) -> Result<BenchmarkResult> {
        let benchmark = {
            let mut active = self.active_benchmarks.lock();
            active.remove(benchmark_key)
                .ok_or_else(|| PipelineError::Optimization(format!("No active benchmark: {}", benchmark_key)))?
        };

        if benchmark.samples.is_empty() {
            return Err(PipelineError::Optimization("No samples collected for benchmark".to_string()));
        }

        // Analyze collected samples
        let total_samples = benchmark.samples.len() as f64;
        let successful_samples = benchmark.samples.iter().filter(|s| s.success).count() as f64;
        
        let average_latency = Duration::from_nanos(
            (benchmark.samples.iter().map(|s| s.latency.as_nanos()).sum::<u128>() / benchmark.samples.len() as u128) as u64
        );
        
        let average_quality = benchmark.samples.iter().map(|s| s.quality_score).sum::<f64>() / total_samples;
        let average_memory = benchmark.samples.iter().map(|s| s.memory_used_gb).sum::<f64>() / total_samples;
        let success_rate = successful_samples / total_samples;
        
        let throughput = self.calculate_throughput(average_latency);
        let memory_efficiency = average_quality * 1000.0 / (average_memory * 1024.0).max(1.0);
        let resource_utilization = self.calculate_resource_utilization(&benchmark.model).await;

        let result = BenchmarkResult {
            model: benchmark.model,
            document_type: DocumentType::PlainText, // TODO: Extract from benchmark key
            throughput_docs_per_hour,
            average_latency,
            quality_score: average_quality,
            memory_efficiency,
            success_rate,
            resource_utilization,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        info!("Completed benchmark: {} docs/hour, {:.3} quality, {:.1}% success rate", 
              result.throughput_docs_per_hour, result.quality_score, result.success_rate * 100.0);

        Ok(result)
    }

    /// Get current performance metrics for model
    pub async fn get_performance_metrics(&self, model: &Qwen3Model, document_type: &DocumentType) -> Result<Option<BenchmarkResult>> {
        let benchmark_key = self.generate_benchmark_key(model, document_type);
        let results = self.benchmark_results.read();
        
        if let Some(results_list) = results.get(&benchmark_key) {
            if let Some(latest) = results_list.last() {
                return Ok(Some(latest.clone()));
            }
        }
        
        Ok(None)
    }

    /// Compare model performance
    pub async fn compare_models(&self, models: &[Qwen3Model], document_type: &DocumentType) -> Result<Vec<ModelComparison>> {
        let mut comparisons = Vec::new();
        
        for model in models {
            if let Some(metrics) = self.get_performance_metrics(model, document_type).await? {
                // Calculate composite performance score
                let performance_score = self.calculate_composite_score(&metrics);
                
                comparisons.push(ModelComparison {
                    model: model.clone(),
                    performance_score,
                    metrics,
                });
            }
        }

        // Sort by performance score
        comparisons.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap());
        
        Ok(comparisons)
    }

    /// Record model switch event
    pub async fn record_model_switch(
        &self, 
        from_model: Option<Qwen3Model>, 
        to_model: Qwen3Model, 
        reason: String
    ) -> Result<()> {
        let switching_time = {
            let mut last_switch = self.last_switch_time.lock();
            let now = Instant::now();
            let switching_time = if let Some(last_time) = *last_switch {
                now.duration_since(last_time)
            } else {
                Duration::from_secs(0)
            };
            *last_switch = Some(now);
            switching_time
        };

        let performance_impact = self.calculate_switching_impact(&from_model, &to_model).await;

        let switch_event = ModelSwitchEvent {
            from_model,
            to_model,
            switching_time,
            reason,
            performance_impact,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        {
            let mut history = self.switching_history.write();
            history.push(switch_event);
            
            // Limit history size
            if history.len() > 500 {
                history.drain(0..50); // Remove oldest 50 entries
            }
        }

        self.total_switches.fetch_add(1, Ordering::Relaxed);
        
        info!("Recorded model switch from {:?} to {} - Reason: {}", 
              from_model.as_ref().map(|m| m.name()), to_model.name(), reason);
        
        Ok(())
    }

    /// Analyze performance trends
    pub async fn analyze_trends(&self) -> Result<HashMap<String, PerformanceTrend>> {
        let config = self.config.read();
        let results = self.benchmark_results.read();
        let mut trends = HashMap::new();

        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - config.trend_analysis_window.as_secs();

        for (key, result_list) in results.iter() {
            let recent_results: Vec<_> = result_list.iter()
                .filter(|r| r.timestamp > cutoff_time)
                .collect();

            if recent_results.len() < config.min_samples_for_trend as usize {
                continue;
            }

            let trend = self.calculate_trend(&recent_results);
            trends.insert(key.clone(), trend);
        }

        Ok(trends)
    }

    /// Get switching statistics
    pub async fn get_switching_stats(&self) -> Result<SwitchingStats> {
        let history = self.switching_history.read();
        let total_switches = self.total_switches.load(Ordering::Relaxed);
        
        let mut model_switch_counts = HashMap::new();
        let mut reasons = HashMap::new();
        let mut average_switching_time = Duration::from_secs(0);

        if !history.is_empty() {
            let total_time: u128 = history.iter().map(|s| s.switching_time.as_nanos()).sum();
            average_switching_time = Duration::from_nanos((total_time / history.len() as u128) as u64);

            for switch in history.iter() {
                *model_switch_counts.entry(switch.to_model.clone()).or_insert(0u64) += 1;
                *reasons.entry(switch.reason.clone()).or_insert(0u64) += 1;
            }
        }

        Ok(SwitchingStats {
            total_switches,
            model_switch_counts,
            switching_reasons: reasons,
            average_switching_time,
            recent_switches: history.iter().rev().take(10).cloned().collect(),
        })
    }

    // Helper methods

    /// Generate benchmark key
    fn generate_benchmark_key(&self, model: &Qwen3Model, document_type: &DocumentType) -> String {
        format!("{}:{}", model.name(), document_type as u8)
    }

    /// Update model warmup state
    async fn update_warmup_state(&self, model: &Qwen3Model, response: &MLResponse) {
        let mut warmup_state = self.model_warmup_state.write();
        if let Some(state) = warmup_state.get_mut(model) {
            if !state.is_warmed_up {
                state.samples_collected += 1;
                if state.samples_collected >= self.config.read().warmup_samples {
                    state.is_warmed_up = true;
                    state.average_warmup_time = state.warmup_start_time.elapsed();
                    debug!("Model {} warmed up after {} samples in {:?}", 
                           model.name(), state.samples_collected, state.average_warmup_time);
                }
            }
        }
    }

    /// Calculate throughput from latency
    fn calculate_throughput(&self, latency: Duration) -> f64 {
        if latency.as_secs_f64() > 0.0 {
            3600.0 / latency.as_secs_f64() // Documents per hour
        } else {
            0.0
        }
    }

    /// Calculate resource utilization (simulated)
    async fn calculate_resource_utilization(&self, model: &Qwen3Model) -> ResourceUtilization {
        // In a real implementation, this would query actual system metrics
        let specs = model.specs();
        
        ResourceUtilization {
            cpu_usage_percent: match model {
                Qwen3Model::Qwen3_1_7B => 25.0,
                Qwen3Model::Qwen3_7B => 50.0,
                Qwen3Model::Qwen3_30B => 80.0,
            },
            memory_usage_gb: specs.memory_gb as f64,
            gpu_usage_percent: match model {
                Qwen3Model::Qwen3_1_7B => 10.0,
                Qwen3Model::Qwen3_7B => 30.0,
                Qwen3Model::Qwen3_30B => 70.0,
            },
            neural_engine_usage_percent: match model {
                Qwen3Model::Qwen3_1_7B => 60.0,
                Qwen3Model::Qwen3_7B => 40.0,
                Qwen3Model::Qwen3_30B => 20.0,
            },
            memory_bandwidth_gbps: match model {
                Qwen3Model::Qwen3_1_7B => 50.0,
                Qwen3Model::Qwen3_7B => 150.0,
                Qwen3Model::Qwen3_30B => 300.0,
            },
        }
    }

    /// Update performance trends
    async fn update_performance_trends(&self, benchmark_key: &str) -> Result<()> {
        let results = self.benchmark_results.read();
        if let Some(result_list) = results.get(benchmark_key) {
            if result_list.len() >= 5 { // Need at least 5 samples for trend analysis
                let trend = self.calculate_trend(result_list);
                let mut trends = self.performance_trends.write();
                trends.insert(benchmark_key.to_string(), trend);
            }
        }
        Ok(())
    }

    /// Calculate performance trend from results
    fn calculate_trend(&self, results: &[&BenchmarkResult]) -> PerformanceTrend {
        if results.is_empty() {
            return PerformanceTrend {
                model: Qwen3Model::Qwen3_7B,
                document_type: DocumentType::PlainText,
                trend_direction: TrendDirection::Stable,
                trend_magnitude: 0.0,
                confidence: 0.0,
                prediction_accuracy: 0.0,
                sample_count: 0,
            };
        }

        let first_result = results[0];
        let model = first_result.model.clone();
        let document_type = first_result.document_type.clone();

        // Calculate linear regression for throughput trend
        let n = results.len() as f64;
        let x_values: Vec<f64> = (0..results.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = results.iter().map(|r| r.throughput_docs_per_hour).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = y_values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(&y_values).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let trend_magnitude = slope.abs();

        let trend_direction = if slope > 0.1 {
            TrendDirection::Improving
        } else if slope < -0.1 {
            TrendDirection::Declining
        } else if trend_magnitude > 0.5 {
            TrendDirection::Volatile
        } else {
            TrendDirection::Stable
        };

        // Calculate confidence based on R-squared
        let mean_y = sum_y / n;
        let ss_res: f64 = y_values.iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = slope * i as f64 + (sum_y - slope * sum_x) / n;
                (y - predicted).powi(2)
            })
            .sum();
        let ss_tot: f64 = y_values.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let r_squared = 1.0 - (ss_res / ss_tot.max(1.0));
        let confidence = r_squared.max(0.0).min(1.0);

        PerformanceTrend {
            model,
            document_type,
            trend_direction,
            trend_magnitude,
            confidence,
            prediction_accuracy: confidence, // Simplified
            sample_count: results.len() as u64,
        }
    }

    /// Calculate composite performance score
    fn calculate_composite_score(&self, metrics: &BenchmarkResult) -> f64 {
        // Weighted combination of performance metrics
        let throughput_score = (metrics.throughput_docs_per_hour / 100.0).min(1.0);
        let quality_score = metrics.quality_score;
        let efficiency_score = (metrics.memory_efficiency / 10.0).min(1.0);
        let reliability_score = metrics.success_rate;

        (throughput_score * 0.3 + quality_score * 0.4 + efficiency_score * 0.2 + reliability_score * 0.1)
    }

    /// Evaluate switching opportunity
    async fn evaluate_switching_opportunity(&self, document_type: &DocumentType) -> Result<()> {
        // Get recent performance for all models
        let all_models = Qwen3Model::all_models();
        let comparisons = self.compare_models(&all_models, document_type).await?;

        if comparisons.len() < 2 {
            return Ok(());
        }

        let best_model = &comparisons[0].model;
        let best_score = comparisons[0].performance_score;
        let second_best_score = comparisons.get(1).map(|c| c.performance_score).unwrap_or(0.0);

        // Check if there's significant improvement opportunity
        let improvement_threshold = 0.1; // 10% improvement required
        if (best_score - second_best_score) > improvement_threshold {
            // Check if we haven't switched too recently
            let last_switch = self.last_switch_time.lock();
            let can_switch = if let Some(last_time) = *last_switch {
                last_time.elapsed() > self.config.read().max_switching_frequency
            } else {
                true
            };

            if can_switch {
                let reason = format!("Performance improvement: {:.1}% better throughput", 
                                   (best_score - second_best_score) * 100.0);
                
                info!("Automatic switching opportunity detected: {}", reason);
                // Note: Actual switching would be handled by the model selector
            }
        }

        Ok(())
    }

    /// Calculate performance impact of model switching
    async fn calculate_switching_impact(&self, from_model: &Option<Qwen3Model>, to_model: &Qwen3Model) -> f64 {
        if let Some(from) = from_model {
            let from_specs = from.specs();
            let to_specs = to_model.specs();
            
            // Calculate relative performance change
            let throughput_change = (to_specs.tokens_per_second as f64 - from_specs.tokens_per_second as f64) 
                                  / from_specs.tokens_per_second as f64;
            let quality_change = to_specs.quality_score - from_specs.quality_score;
            
            // Weighted impact score
            (throughput_change * 0.6 + quality_change * 0.4)
        } else {
            0.0 // No baseline to compare against
        }
    }
}

/// Model comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model: Qwen3Model,
    pub performance_score: f64,
    pub metrics: BenchmarkResult,
}

/// Switching statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingStats {
    pub total_switches: u64,
    pub model_switch_counts: HashMap<Qwen3Model, u64>,
    pub switching_reasons: HashMap<String, u64>,
    pub average_switching_time: Duration,
    pub recent_switches: Vec<ModelSwitchEvent>,
}

/// Initialize performance benchmark system
pub async fn initialize() -> Result<()> {
    info!("Initializing Performance Benchmark system for Qwen3 models");
    
    let config = BenchmarkConfig::default();
    let benchmark = PerformanceBenchmark::new(config);
    
    PERFORMANCE_BENCHMARK.set(Arc::new(benchmark))
        .map_err(|_| PipelineError::Optimization("Failed to initialize performance benchmark".to_string()))?;
    
    info!("Performance Benchmark system initialized successfully");
    Ok(())
}

/// Record ML processing result for benchmarking
pub async fn record_result(request: &MLRequest, response: &MLResponse) -> Result<()> {
    let benchmark = PERFORMANCE_BENCHMARK.get()
        .ok_or_else(|| PipelineError::Optimization("Performance benchmark not initialized".to_string()))?;
    
    benchmark.record_result(request, response).await
}

/// Get performance metrics for model and document type
pub async fn get_performance_metrics(model: &Qwen3Model, document_type: &DocumentType) -> Result<Option<BenchmarkResult>> {
    let benchmark = PERFORMANCE_BENCHMARK.get()
        .ok_or_else(|| PipelineError::Optimization("Performance benchmark not initialized".to_string()))?;
    
    benchmark.get_performance_metrics(model, document_type).await
}

/// Compare model performance
pub async fn compare_models(models: &[Qwen3Model], document_type: &DocumentType) -> Result<Vec<ModelComparison>> {
    let benchmark = PERFORMANCE_BENCHMARK.get()
        .ok_or_else(|| PipelineError::Optimization("Performance benchmark not initialized".to_string()))?;
    
    benchmark.compare_models(models, document_type).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use crate::ml::QualityRequirements;

    #[tokio::test]
    async fn test_benchmark_initialization() {
        initialize().await.unwrap();
        
        let stats = PERFORMANCE_BENCHMARK.get().unwrap()
            .get_switching_stats().await.unwrap();
        
        assert_eq!(stats.total_switches, 0);
    }

    #[tokio::test]
    async fn test_record_benchmark_result() {
        initialize().await.unwrap();
        
        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::PlainText,
            document_size_bytes: 1000,
            complexity_score: 0.5,
            priority: Priority::Medium,
            quality_requirements: QualityRequirements {
                min_score: 0.7,
                consistency_target: 0.75,
                accuracy_threshold: 0.8,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(30)),
        };

        let response = MLResponse {
            request_id: request.request_id,
            model_used: Qwen3Model::Qwen3_7B,
            processing_time: Duration::from_millis(500),
            quality_score: 0.8,
            memory_used_mb: 14000,
            status: ProcessingStatus::Success,
            error_message: None,
        };

        record_result(&request, &response).await.unwrap();
        
        let metrics = get_performance_metrics(&Qwen3Model::Qwen3_7B, &DocumentType::PlainText)
            .await.unwrap();
        
        assert!(metrics.is_some());
        let metrics = metrics.unwrap();
        assert_eq!(metrics.model, Qwen3Model::Qwen3_7B);
        assert!(metrics.throughput_docs_per_hour > 0.0);
    }
}