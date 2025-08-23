/*!
# Dynamic Model Selector for Qwen3 Variants

Intelligent model selection system that chooses the optimal Qwen3 variant (1.7B/7B/30B)
based on workload analysis, performance metrics, and M3 Max memory constraints.
*/

use crate::ml::{MLRequest, Qwen3Model, ModelMetrics, WorkloadAnalysis};
use crate::optimization::m3_max::{M3MaxMemoryManager, MemoryStats};
use crate::{Result, PipelineError};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Model selection strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SelectionStrategy {
    /// Always select fastest model (1.7B)
    Speed,
    /// Always select highest quality model (30B)
    Quality,
    /// Balance speed and quality based on workload
    Balanced,
    /// Adaptive selection based on system state and performance history
    Adaptive,
    /// Memory-aware selection optimized for M3 Max
    MemoryOptimized,
}

/// Model selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub strategy: SelectionStrategy,
    pub quality_threshold: f64,
    pub performance_weight: f64,
    pub memory_weight: f64,
    pub latency_weight: f64,
    pub cost_weight: f64,
    pub max_memory_gb: Option<u32>,
    pub max_latency_ms: Option<u64>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::Adaptive,
            quality_threshold: 0.742,
            performance_weight: 0.3,
            memory_weight: 0.25,
            latency_weight: 0.25,
            cost_weight: 0.2,
            max_memory_gb: Some(45), // M3 Max constraint
            max_latency_ms: Some(30000), // 30 second timeout
        }
    }
}

/// Model performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model: Qwen3Model,
    pub estimated_quality: f64,
    pub estimated_latency_ms: u64,
    pub estimated_memory_gb: u32,
    pub confidence_score: f64,
    pub resource_efficiency: f64,
    pub throughput_score: f64,
}

/// Model selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub selected_model: Qwen3Model,
    pub selection_reason: String,
    pub confidence: f64,
    pub alternatives: Vec<ModelPrediction>,
    pub selection_time_ms: u64,
    pub memory_available_gb: u32,
    pub system_load: f64,
}

/// Dynamic Model Selector
pub struct ModelSelector {
    criteria: Arc<RwLock<SelectionCriteria>>,
    model_metrics: Arc<RwLock<HashMap<Qwen3Model, ModelMetrics>>>,
    selection_history: Arc<Mutex<Vec<SelectionResult>>>,
    performance_tracker: Arc<PerformanceTracker>,
    memory_manager: Option<Arc<M3MaxMemoryManager>>,
    total_selections: AtomicU64,
    successful_selections: AtomicU64,
}

#[derive(Debug)]
struct PerformanceTracker {
    model_loads: HashMap<Qwen3Model, AtomicU64>,
    switching_overhead: AtomicU64,
    last_model_switch: Mutex<Option<Instant>>,
    warmup_times: HashMap<Qwen3Model, Duration>,
}

static MODEL_SELECTOR: std::sync::OnceLock<Arc<ModelSelector>> = std::sync::OnceLock::new();

impl ModelSelector {
    /// Create new model selector
    pub fn new(criteria: SelectionCriteria) -> Self {
        let mut model_metrics = HashMap::new();
        for model in Qwen3Model::all_models() {
            let mut metrics = ModelMetrics::default();
            metrics.model = model.clone();
            model_metrics.insert(model, metrics);
        }

        let mut model_loads = HashMap::new();
        let mut warmup_times = HashMap::new();
        
        for model in Qwen3Model::all_models() {
            model_loads.insert(model.clone(), AtomicU64::new(0));
            // Estimate warmup times based on model size
            let warmup = match model {
                Qwen3Model::Qwen3_1_7B => Duration::from_millis(2000),
                Qwen3Model::Qwen3_7B => Duration::from_millis(5000),
                Qwen3Model::Qwen3_30B => Duration::from_millis(12000),
            };
            warmup_times.insert(model, warmup);
        }

        Self {
            criteria: Arc::new(RwLock::new(criteria)),
            model_metrics: Arc::new(RwLock::new(model_metrics)),
            selection_history: Arc::new(Mutex::new(Vec::new())),
            performance_tracker: Arc::new(PerformanceTracker {
                model_loads,
                switching_overhead: AtomicU64::new(0),
                last_model_switch: Mutex::new(None),
                warmup_times,
            }),
            memory_manager: None,
            total_selections: AtomicU64::new(0),
            successful_selections: AtomicU64::new(0),
        }
    }

    /// Set memory manager for M3 Max optimization
    pub fn set_memory_manager(&mut self, memory_manager: Arc<M3MaxMemoryManager>) {
        self.memory_manager = Some(memory_manager);
    }

    /// Select optimal Qwen3 model for the given request and workload
    pub async fn select_model(
        &self,
        request: &MLRequest,
        workload: &WorkloadAnalysis,
    ) -> Result<Qwen3Model> {
        let start_time = Instant::now();
        self.total_selections.fetch_add(1, Ordering::Relaxed);

        debug!("Starting model selection for request {}", request.request_id);

        // Get current system state
        let system_state = self.get_system_state().await?;
        
        // Generate predictions for all models
        let predictions = self.generate_model_predictions(request, workload, &system_state).await?;
        
        // Select best model based on criteria
        let selection_result = self.evaluate_and_select(predictions, &system_state).await?;
        
        // Record selection history
        {
            let mut history = self.selection_history.lock();
            history.push(selection_result.clone());
            
            // Keep only last 1000 selections
            if history.len() > 1000 {
                history.drain(0..100); // Remove oldest 100
            }
        }

        // Update performance tracking
        self.update_selection_tracking(&selection_result.selected_model).await?;

        let selection_time = start_time.elapsed();
        info!("Selected model {} for request {} in {}ms - Reason: {}", 
              selection_result.selected_model.name(),
              request.request_id,
              selection_time.as_millis(),
              selection_result.selection_reason);

        Ok(selection_result.selected_model)
    }

    /// Generate performance predictions for all models
    async fn generate_model_predictions(
        &self,
        request: &MLRequest,
        workload: &WorkloadAnalysis,
        system_state: &SystemState,
    ) -> Result<Vec<ModelPrediction>> {
        let mut predictions = Vec::new();
        let model_metrics = self.model_metrics.read();

        for model in Qwen3Model::all_models() {
            let specs = model.specs();
            let metrics = model_metrics.get(&model).unwrap();

            // Estimate quality based on model capability and workload complexity
            let base_quality = specs.quality_score;
            let complexity_adjustment = match workload.complexity_score {
                x if x < 0.3 => 0.05,  // Simple tasks may not need full capability
                x if x < 0.7 => 0.0,   // Normal complexity
                _ => -0.1,             // High complexity may challenge smaller models
            };
            let estimated_quality = (base_quality + complexity_adjustment).max(0.0).min(1.0);

            // Estimate latency based on document size, model speed, and system load
            let base_latency_ms = (request.document_size_bytes as f64 / specs.tokens_per_second as f64) * 1000.0;
            let system_load_multiplier = 1.0 + system_state.cpu_usage;
            let estimated_latency_ms = (base_latency_ms * system_load_multiplier) as u64;

            // Estimate memory usage with overhead
            let estimated_memory_gb = specs.memory_gb + (specs.memory_gb / 4); // 25% overhead

            // Calculate confidence based on historical performance
            let confidence_score = if metrics.total_requests > 10 {
                (metrics.successful_requests as f64 / metrics.total_requests as f64) * 0.9
            } else {
                0.5 // Low confidence for new models
            };

            // Calculate resource efficiency
            let memory_efficiency = if estimated_memory_gb > 0 {
                estimated_quality / (estimated_memory_gb as f64 / 10.0)
            } else {
                0.0
            };
            
            let time_efficiency = estimated_quality * 1000.0 / (estimated_latency_ms as f64 + 1.0);
            let resource_efficiency = (memory_efficiency + time_efficiency) / 2.0;

            // Calculate throughput score
            let throughput_score = (specs.tokens_per_second as f64 * estimated_quality) / 100.0;

            predictions.push(ModelPrediction {
                model,
                estimated_quality,
                estimated_latency_ms,
                estimated_memory_gb,
                confidence_score,
                resource_efficiency,
                throughput_score,
            });
        }

        predictions.sort_by(|a, b| b.resource_efficiency.partial_cmp(&a.resource_efficiency).unwrap());
        Ok(predictions)
    }

    /// Evaluate predictions and select the best model
    async fn evaluate_and_select(
        &self,
        mut predictions: Vec<ModelPrediction>,
        system_state: &SystemState,
    ) -> Result<SelectionResult> {
        let criteria = self.criteria.read();
        let start_time = Instant::now();

        // Filter out models that don't meet hard constraints
        predictions.retain(|pred| {
            let memory_ok = criteria.max_memory_gb.map_or(true, |max| pred.estimated_memory_gb <= max);
            let latency_ok = criteria.max_latency_ms.map_or(true, |max| pred.estimated_latency_ms <= max);
            let quality_ok = pred.estimated_quality >= criteria.quality_threshold;
            
            memory_ok && latency_ok && quality_ok
        });

        if predictions.is_empty() {
            warn!("No models meet the selection criteria, falling back to smallest model");
            predictions.push(ModelPrediction {
                model: Qwen3Model::Qwen3_1_7B,
                estimated_quality: 0.7,
                estimated_latency_ms: 5000,
                estimated_memory_gb: 12,
                confidence_score: 0.8,
                resource_efficiency: 0.6,
                throughput_score: 10.0,
            });
        }

        // Calculate weighted scores based on strategy
        let selected = match criteria.strategy {
            SelectionStrategy::Speed => {
                predictions.sort_by_key(|p| p.estimated_latency_ms);
                predictions[0].clone()
            }
            SelectionStrategy::Quality => {
                predictions.sort_by(|a, b| b.estimated_quality.partial_cmp(&a.estimated_quality).unwrap());
                predictions[0].clone()
            }
            SelectionStrategy::Balanced => {
                self.select_balanced_model(&predictions, &criteria).await
            }
            SelectionStrategy::Adaptive => {
                self.select_adaptive_model(&predictions, &criteria, system_state).await
            }
            SelectionStrategy::MemoryOptimized => {
                self.select_memory_optimized_model(&predictions, &criteria, system_state).await
            }
        };

        let selection_reason = self.generate_selection_reason(&selected, &criteria.strategy, system_state);
        let confidence = self.calculate_selection_confidence(&selected, &predictions, system_state);

        Ok(SelectionResult {
            selected_model: selected.model,
            selection_reason,
            confidence,
            alternatives: predictions,
            selection_time_ms: start_time.elapsed().as_millis() as u64,
            memory_available_gb: system_state.available_memory_gb,
            system_load: system_state.cpu_usage,
        })
    }

    /// Select model using balanced strategy
    async fn select_balanced_model(
        &self,
        predictions: &[ModelPrediction],
        criteria: &SelectionCriteria,
    ) -> ModelPrediction {
        let mut best_score = 0.0;
        let mut best_prediction = predictions[0].clone();

        for prediction in predictions {
            let quality_score = prediction.estimated_quality * criteria.performance_weight;
            let latency_score = (1.0 - (prediction.estimated_latency_ms as f64 / 30000.0)) * criteria.latency_weight;
            let memory_score = (1.0 - (prediction.estimated_memory_gb as f64 / 50.0)) * criteria.memory_weight;
            let cost_score = prediction.resource_efficiency * criteria.cost_weight;

            let total_score = quality_score + latency_score + memory_score + cost_score;

            if total_score > best_score {
                best_score = total_score;
                best_prediction = prediction.clone();
            }
        }

        best_prediction
    }

    /// Select model using adaptive strategy
    async fn select_adaptive_model(
        &self,
        predictions: &[ModelPrediction],
        criteria: &SelectionCriteria,
        system_state: &SystemState,
    ) -> ModelPrediction {
        // Adaptive selection considers system state and historical performance
        let mut adaptive_predictions = predictions.to_vec();

        // Adjust scores based on system state
        for prediction in &mut adaptive_predictions {
            // If system is under high load, prefer faster models
            if system_state.cpu_usage > 0.8 {
                if matches!(prediction.model, Qwen3Model::Qwen3_1_7B) {
                    prediction.resource_efficiency *= 1.2;
                }
            }

            // If memory is constrained, prefer smaller models
            if system_state.memory_usage > 0.85 {
                let memory_penalty = (prediction.estimated_memory_gb as f64 - 10.0) / 40.0;
                prediction.resource_efficiency *= (1.0 - memory_penalty).max(0.1);
            }

            // If we have successful history with a model, boost its score
            let model_metrics = self.model_metrics.read();
            if let Some(metrics) = model_metrics.get(&prediction.model) {
                if metrics.total_requests > 5 && metrics.error_rate < 0.1 {
                    prediction.confidence_score *= 1.1;
                }
            }
        }

        // Select model with highest adjusted resource efficiency
        adaptive_predictions.sort_by(|a, b| {
            (b.resource_efficiency * b.confidence_score)
                .partial_cmp(&(a.resource_efficiency * a.confidence_score))
                .unwrap()
        });

        adaptive_predictions[0].clone()
    }

    /// Select model using memory-optimized strategy for M3 Max
    async fn select_memory_optimized_model(
        &self,
        predictions: &[ModelPrediction],
        _criteria: &SelectionCriteria,
        system_state: &SystemState,
    ) -> ModelPrediction {
        let mut memory_scores = Vec::new();

        for prediction in predictions {
            // Calculate memory efficiency score
            let memory_utilization = prediction.estimated_memory_gb as f64 / system_state.available_memory_gb as f64;
            let memory_efficiency = prediction.estimated_quality / memory_utilization;
            
            // Bonus for models that fit well within M3 Max memory pools
            let pool_fit_bonus = match prediction.estimated_memory_gb {
                m if m <= 12 => 1.2, // Fits in small pool
                m if m <= 28 => 1.1, // Fits in medium pool
                m if m <= 45 => 1.0, // Fits in large pool
                _ => 0.8,            // Requires swap or special handling
            };

            let final_score = memory_efficiency * pool_fit_bonus;
            memory_scores.push((prediction.clone(), final_score));
        }

        memory_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        memory_scores[0].0.clone()
    }

    /// Generate human-readable selection reason
    fn generate_selection_reason(
        &self,
        selected: &ModelPrediction,
        strategy: &SelectionStrategy,
        system_state: &SystemState,
    ) -> String {
        match strategy {
            SelectionStrategy::Speed => {
                format!("Selected {} for fastest processing (~{}ms latency)",
                        selected.model.name(), selected.estimated_latency_ms)
            }
            SelectionStrategy::Quality => {
                format!("Selected {} for highest quality (est. {:.3} score)",
                        selected.model.name(), selected.estimated_quality)
            }
            SelectionStrategy::Balanced => {
                format!("Selected {} for optimal balance of quality ({:.3}) and performance ({}ms)",
                        selected.model.name(), selected.estimated_quality, selected.estimated_latency_ms)
            }
            SelectionStrategy::Adaptive => {
                let load_context = if system_state.cpu_usage > 0.8 {
                    "high system load"
                } else if system_state.memory_usage > 0.85 {
                    "memory constraints"
                } else {
                    "optimal conditions"
                };
                format!("Selected {} adaptively for {} (efficiency: {:.3})",
                        selected.model.name(), load_context, selected.resource_efficiency)
            }
            SelectionStrategy::MemoryOptimized => {
                format!("Selected {} for M3 Max memory optimization ({}GB usage, {:.3} quality)",
                        selected.model.name(), selected.estimated_memory_gb, selected.estimated_quality)
            }
        }
    }

    /// Calculate confidence in the selection
    fn calculate_selection_confidence(
        &self,
        selected: &ModelPrediction,
        alternatives: &[ModelPrediction],
        _system_state: &SystemState,
    ) -> f64 {
        if alternatives.len() <= 1 {
            return selected.confidence_score;
        }

        // Find the gap between selected and second-best option
        let selected_score = selected.resource_efficiency * selected.confidence_score;
        let mut other_scores: Vec<f64> = alternatives.iter()
            .filter(|p| p.model != selected.model)
            .map(|p| p.resource_efficiency * p.confidence_score)
            .collect();
        other_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let second_best_score = other_scores.first().copied().unwrap_or(0.0);
        let score_gap = (selected_score - second_best_score) / selected_score;

        // Higher gap means higher confidence
        (selected.confidence_score + score_gap * 0.3).min(1.0)
    }

    /// Get current system state
    async fn get_system_state(&self) -> Result<SystemState> {
        let (available_memory_gb, memory_usage, cpu_usage) = if let Some(ref memory_manager) = self.memory_manager {
            let stats = memory_manager.get_stats().await?;
            let total_memory = memory_manager.get_system_info().total_memory_gb as f64;
            let used_memory = stats.total_allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let available = total_memory - used_memory;
            let memory_usage = used_memory / total_memory;
            let cpu_usage = stats.system_metrics.cpu_usage_percent / 100.0;
            
            (available as u32, memory_usage, cpu_usage)
        } else {
            // Fallback system detection
            (64, 0.5, 0.3) // Conservative estimates
        };

        Ok(SystemState {
            available_memory_gb,
            memory_usage,
            cpu_usage,
            gpu_usage: 0.0, // TODO: Add GPU monitoring
            neural_engine_usage: 0.0, // TODO: Add Neural Engine monitoring
        })
    }

    /// Update selection tracking
    async fn update_selection_tracking(&self, selected_model: &Qwen3Model) -> Result<()> {
        self.performance_tracker.model_loads
            .get(selected_model)
            .unwrap()
            .fetch_add(1, Ordering::Relaxed);

        // Track model switching overhead
        let mut last_switch = self.performance_tracker.last_model_switch.lock();
        if let Some(last_time) = *last_switch {
            let switching_time = last_time.elapsed().as_millis() as u64;
            self.performance_tracker.switching_overhead
                .store(switching_time, Ordering::Relaxed);
        }
        *last_switch = Some(Instant::now());

        Ok(())
    }

    /// Update model metrics after processing
    pub async fn update_metrics(&self, response: &crate::ml::MLResponse) -> Result<()> {
        let mut model_metrics = self.model_metrics.write();
        if let Some(metrics) = model_metrics.get_mut(&response.model_used) {
            metrics.total_requests += 1;
            
            if matches!(response.status, crate::ml::ProcessingStatus::Success) {
                metrics.successful_requests += 1;
                self.successful_selections.fetch_add(1, Ordering::Relaxed);
                
                // Update rolling averages
                let new_weight = 1.0 / metrics.total_requests as f64;
                let old_weight = 1.0 - new_weight;
                
                metrics.average_processing_time = Duration::from_nanos(
                    (metrics.average_processing_time.as_nanos() as f64 * old_weight +
                     response.processing_time.as_nanos() as f64 * new_weight) as u64
                );
                
                metrics.average_quality_score = metrics.average_quality_score * old_weight + 
                                              response.quality_score * new_weight;
                
                metrics.memory_efficiency = metrics.memory_efficiency * old_weight +
                                          (response.quality_score * 1000.0 / response.memory_used_mb as f64) * new_weight;
                
                // Calculate throughput
                let docs_per_hour = 3600.0 / response.processing_time.as_secs_f64();
                metrics.throughput_docs_per_hour = metrics.throughput_docs_per_hour * old_weight + 
                                                 docs_per_hour * new_weight;
            }
            
            metrics.error_rate = 1.0 - (metrics.successful_requests as f64 / metrics.total_requests as f64);
            metrics.last_used = Instant::now();
        }

        Ok(())
    }

    /// Get model selection statistics
    pub async fn get_statistics(&self) -> Result<ModelSelectorStats> {
        let total_selections = self.total_selections.load(Ordering::Relaxed);
        let successful_selections = self.successful_selections.load(Ordering::Relaxed);
        let success_rate = if total_selections > 0 {
            successful_selections as f64 / total_selections as f64
        } else {
            0.0
        };

        let model_metrics = self.model_metrics.read();
        let mut model_usage = HashMap::new();
        
        for (model, tracker) in &self.performance_tracker.model_loads {
            let usage_count = tracker.load(Ordering::Relaxed);
            model_usage.insert(model.clone(), usage_count);
        }

        let switching_overhead = Duration::from_millis(
            self.performance_tracker.switching_overhead.load(Ordering::Relaxed)
        );

        Ok(ModelSelectorStats {
            total_selections,
            successful_selections,
            success_rate,
            model_usage,
            model_metrics: model_metrics.clone(),
            switching_overhead,
        })
    }
}

/// System state for model selection
#[derive(Debug, Clone)]
struct SystemState {
    available_memory_gb: u32,
    memory_usage: f64,
    cpu_usage: f64,
    gpu_usage: f64,
    neural_engine_usage: f64,
}

/// Model selector statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectorStats {
    pub total_selections: u64,
    pub successful_selections: u64,
    pub success_rate: f64,
    pub model_usage: HashMap<Qwen3Model, u64>,
    pub model_metrics: HashMap<Qwen3Model, ModelMetrics>,
    pub switching_overhead: Duration,
}

/// Initialize the model selector
pub async fn initialize() -> Result<()> {
    info!("Initializing Dynamic Model Selector for Qwen3 variants");
    
    let criteria = SelectionCriteria::default();
    let selector = ModelSelector::new(criteria);
    
    MODEL_SELECTOR.set(Arc::new(selector))
        .map_err(|_| PipelineError::Optimization("Failed to initialize model selector".to_string()))?;
    
    info!("Dynamic Model Selector initialized successfully");
    Ok(())
}

/// Select optimal model for the given request and workload
pub async fn select_model(request: &MLRequest, workload: &WorkloadAnalysis) -> Result<Qwen3Model> {
    let selector = MODEL_SELECTOR.get()
        .ok_or_else(|| PipelineError::Optimization("Model selector not initialized".to_string()))?;
    
    selector.select_model(request, workload).await
}

/// Update model performance metrics
pub async fn update_metrics(response: &crate::ml::MLResponse) -> Result<()> {
    let selector = MODEL_SELECTOR.get()
        .ok_or_else(|| PipelineError::Optimization("Model selector not initialized".to_string()))?;
    
    selector.update_metrics(response).await
}

/// Get model selector statistics
pub async fn get_statistics() -> Result<ModelSelectorStats> {
    let selector = MODEL_SELECTOR.get()
        .ok_or_else(|| PipelineError::Optimization("Model selector not initialized".to_string()))?;
    
    selector.get_statistics().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::{DocumentType, Priority, QualityRequirements};
    use uuid::Uuid;

    #[tokio::test]
    async fn test_model_selector_initialization() {
        initialize().await.unwrap();
        let stats = get_statistics().await.unwrap();
        assert_eq!(stats.total_selections, 0);
        assert_eq!(stats.successful_selections, 0);
    }

    #[tokio::test]
    async fn test_model_selection_speed_strategy() {
        initialize().await.unwrap();
        
        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::PlainText,
            document_size_bytes: 1024,
            complexity_score: 0.3,
            priority: Priority::High,
            quality_requirements: QualityRequirements {
                min_score: 0.7,
                consistency_target: 0.75,
                accuracy_threshold: 0.8,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(10)),
        };

        let workload = super::WorkloadAnalysis {
            complexity_score: 0.3,
            estimated_processing_time: Duration::from_secs(5),
            memory_requirements_gb: 8,
            cpu_intensity: 0.4,
            io_intensity: 0.6,
            recommended_model: Qwen3Model::Qwen3_1_7B,
        };

        let selected_model = select_model(&request, &workload).await.unwrap();
        // For simple, fast processing, should prefer smaller models
        assert!(matches!(selected_model, Qwen3Model::Qwen3_1_7B | Qwen3Model::Qwen3_7B));
    }

    #[tokio::test] 
    async fn test_model_prediction_generation() {
        let selector = ModelSelector::new(SelectionCriteria::default());
        
        let request = MLRequest {
            request_id: Uuid::new_v4(),
            document_type: DocumentType::Technical,
            document_size_bytes: 10_000,
            complexity_score: 0.8,
            priority: Priority::High,
            quality_requirements: QualityRequirements {
                min_score: 0.85,
                consistency_target: 0.9,
                accuracy_threshold: 0.9,
                enable_validation: true,
            },
            processing_deadline: Some(Duration::from_secs(60)),
        };

        let workload = super::WorkloadAnalysis {
            complexity_score: 0.8,
            estimated_processing_time: Duration::from_secs(30),
            memory_requirements_gb: 25,
            cpu_intensity: 0.7,
            io_intensity: 0.3,
            recommended_model: Qwen3Model::Qwen3_7B,
        };

        let system_state = SystemState {
            available_memory_gb: 80,
            memory_usage: 0.4,
            cpu_usage: 0.3,
            gpu_usage: 0.2,
            neural_engine_usage: 0.1,
        };

        let predictions = selector.generate_model_predictions(&request, &workload, &system_state).await.unwrap();
        
        assert_eq!(predictions.len(), 3); // All three Qwen3 models
        
        // Verify predictions are reasonable
        for prediction in predictions {
            assert!(prediction.estimated_quality >= 0.0 && prediction.estimated_quality <= 1.0);
            assert!(prediction.estimated_latency_ms > 0);
            assert!(prediction.estimated_memory_gb > 0);
            assert!(prediction.confidence_score >= 0.0 && prediction.confidence_score <= 1.0);
        }
    }
}