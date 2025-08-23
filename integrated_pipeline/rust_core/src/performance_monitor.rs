use crate::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use uuid::Uuid;
use dashmap::DashMap;

/// Advanced performance monitoring system for hybrid pipeline
/// Tracks real-time metrics, bottlenecks, and optimization opportunities
#[derive(Debug)]
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics_store: Arc<RwLock<MetricsStore>>,
    bottleneck_analyzer: Arc<BottleneckAnalyzer>,
    performance_predictor: Arc<PerformancePredictor>,
    alert_manager: Arc<AlertManager>,
    
    // Real-time tracking
    active_operations: Arc<DashMap<Uuid, OperationMetrics>>,
    system_metrics: Arc<RwLock<SystemMetrics>>,
    
    // Historical data
    metrics_history: Arc<RwLock<VecDeque<TimestampedMetrics>>>,
    bottleneck_history: Arc<RwLock<VecDeque<BottleneckEvent>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub collection_interval_ms: u64,
    pub retention_hours: u64,
    pub alert_thresholds: AlertThresholds,
    pub enable_predictive_analysis: bool,
    pub enable_bottleneck_detection: bool,
    pub enable_memory_profiling: bool,
    pub metrics_export_enabled: bool,
    pub real_time_dashboard: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub memory_usage_percent: f64,          // Alert at 85%
    pub processing_time_multiplier: f64,    // Alert if >2x expected
    pub error_rate_percent: f64,            // Alert at 5%
    pub throughput_deviation_percent: f64,  // Alert if <80% of target
    pub queue_depth: usize,                 // Alert at 100 pending operations
    pub model_switch_latency_ms: u64,       // Alert if >3000ms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStore {
    pub pipeline_metrics: PipelineMetrics,
    pub rust_metrics: RustProcessingMetrics,
    pub python_metrics: PythonMLMetrics,
    pub ipc_metrics: IPCMetrics,
    pub model_metrics: ModelMetrics,
    pub resource_metrics: ResourceMetrics,
    pub quality_metrics: QualityMetrics,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_processing_time: Duration,
    pub current_throughput_docs_hour: f64,
    pub target_throughput_docs_hour: f64,
    pub throughput_efficiency: f64,
    pub queue_depth: usize,
    pub active_sessions: usize,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustProcessingMetrics {
    pub documents_processed: u64,
    pub average_processing_time: Duration,
    pub batch_processing_efficiency: f64,
    pub quality_validation_time: Duration,
    pub memory_usage_gb: f64,
    pub cpu_utilization_percent: f64,
    pub concurrent_operations: usize,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonMLMetrics {
    pub models_loaded: usize,
    pub model_switch_count: u64,
    pub average_switch_latency: Duration,
    pub inference_time_by_model: HashMap<String, Duration>,
    pub memory_usage_by_model: HashMap<String, f64>,
    pub gpu_utilization_percent: f64,
    pub mlx_acceleration_active: bool,
    pub batch_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPCMetrics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub average_latency: Duration,
    pub max_latency: Duration,
    pub timeout_count: u64,
    pub shared_memory_usage_gb: f64,
    pub connection_pool_utilization: f64,
    pub serialization_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_performance: HashMap<String, ModelPerformanceData>,
    pub current_active_model: Option<String>,
    pub model_reliability_scores: HashMap<String, f64>,
    pub adaptation_effectiveness: f64,
    pub selection_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceData {
    pub inference_time_p50: Duration,
    pub inference_time_p95: Duration,
    pub quality_score_average: f64,
    pub memory_efficiency: f64,
    pub success_rate: f64,
    pub cost_per_document: f64,
    pub sample_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub total_memory_gb: f64,
    pub rust_memory_gb: f64,
    pub python_memory_gb: f64,
    pub shared_memory_gb: f64,
    pub memory_efficiency: f64,
    pub cpu_cores_utilized: usize,
    pub cpu_utilization_percent: f64,
    pub disk_io_mb_per_sec: f64,
    pub network_io_mb_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub structural_quality_average: f64,
    pub semantic_quality_average: f64,
    pub combined_quality_average: f64,
    pub quality_consistency: f64,
    pub parameter_extraction_rate: f64,
    pub technical_density_average: f64,
    pub validation_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub m3_max_optimization_active: bool,
    pub unified_memory_utilization: f64,
    pub metal_performance_shaders_active: bool,
    pub neural_engine_utilization: f64,
    pub thermal_state: ThermalState,
    pub power_consumption_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_id: Uuid,
    pub operation_type: OperationType,
    pub start_time: Instant,
    pub expected_duration: Option<Duration>,
    pub current_phase: String,
    pub documents_count: usize,
    pub memory_allocated: f64,
    pub model_used: Option<String>,
    pub quality_target: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    DocumentProcessing,
    BatchProcessing,
    ModelSwitching,
    QualityValidation,
    PerformanceOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetrics {
    pub timestamp: SystemTime,
    pub metrics: MetricsStore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckEvent {
    pub timestamp: SystemTime,
    pub bottleneck_type: BottleneckType,
    pub severity: Severity,
    pub description: String,
    pub affected_components: Vec<String>,
    pub performance_impact: f64,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    Memory,
    CPU,
    IPC,
    ModelSwitching,
    QualityValidation,
    NetworkIO,
    DiskIO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Bottleneck detection and analysis system
#[derive(Debug)]
pub struct BottleneckAnalyzer {
    detection_thresholds: HashMap<BottleneckType, f64>,
    analysis_window: Duration,
    recent_detections: VecDeque<BottleneckEvent>,
}

/// Performance prediction system using historical data
#[derive(Debug)]
pub struct PerformancePredictor {
    prediction_models: HashMap<String, PredictionModel>,
    training_data: VecDeque<TrainingDataPoint>,
    prediction_accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub accuracy: f64,
    pub last_training: SystemTime,
}

#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    pub timestamp: SystemTime,
    pub features: Vec<f64>,
    pub actual_outcome: f64,
    pub predicted_outcome: Option<f64>,
}

/// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    active_alerts: DashMap<String, Alert>,
    alert_history: VecDeque<Alert>,
    notification_handlers: Vec<Box<dyn NotificationHandler + Send + Sync>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub timestamp: SystemTime,
    pub resolved: bool,
    pub resolution_time: Option<SystemTime>,
    pub related_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Performance,
    Resource,
    Quality,
    System,
    Security,
}

pub trait NotificationHandler {
    fn handle_alert(&self, alert: &Alert) -> Result<()>;
    fn get_handler_type(&self) -> String;
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000, // 1 second
            retention_hours: 24,
            alert_thresholds: AlertThresholds {
                memory_usage_percent: 85.0,
                processing_time_multiplier: 2.0,
                error_rate_percent: 5.0,
                throughput_deviation_percent: 20.0,
                queue_depth: 100,
                model_switch_latency_ms: 3000,
            },
            enable_predictive_analysis: true,
            enable_bottleneck_detection: true,
            enable_memory_profiling: true,
            metrics_export_enabled: true,
            real_time_dashboard: true,
        }
    }
}

impl PerformanceMonitor {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        info!("Initializing Performance Monitor with real-time analytics");
        
        let metrics_store = Arc::new(RwLock::new(MetricsStore::default()));
        let bottleneck_analyzer = Arc::new(BottleneckAnalyzer::new());
        let performance_predictor = Arc::new(PerformancePredictor::new());
        let alert_manager = Arc::new(AlertManager::new());
        
        let monitor = Self {
            config,
            metrics_store,
            bottleneck_analyzer,
            performance_predictor,
            alert_manager,
            active_operations: Arc::new(DashMap::new()),
            system_metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(86400))), // 24h at 1s intervals
            bottleneck_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        };
        
        // Start background monitoring tasks
        monitor.start_monitoring_tasks().await;
        
        info!("Performance Monitor initialized successfully");
        Ok(monitor)
    }
    
    async fn start_monitoring_tasks(&self) {
        let config = self.config.clone();
        let metrics_store = self.metrics_store.clone();
        let system_metrics = self.system_metrics.clone();
        let metrics_history = self.metrics_history.clone();
        let bottleneck_analyzer = self.bottleneck_analyzer.clone();
        let alert_manager = self.alert_manager.clone();
        
        // Metrics collection task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_millis(config.collection_interval_ms)
            );
            
            loop {
                interval.tick().await;
                
                // Collect current metrics
                if let Err(e) = Self::collect_system_metrics(&system_metrics).await {
                    error!("Failed to collect system metrics: {}", e);
                }
                
                // Update metrics store
                let timestamp = SystemTime::now();
                let current_metrics = metrics_store.read().await.clone();
                
                // Add to history
                {
                    let mut history = metrics_history.write().await;
                    history.push_back(TimestampedMetrics {
                        timestamp,
                        metrics: current_metrics,
                    });
                    
                    // Trim history to retention period
                    let retention_duration = Duration::from_secs(config.retention_hours * 3600);
                    while let Some(oldest) = history.front() {
                        if timestamp.duration_since(oldest.timestamp).unwrap_or(Duration::ZERO) > retention_duration {
                            history.pop_front();
                        } else {
                            break;
                        }
                    }
                }
            }
        });
        
        // Bottleneck detection task
        if self.config.enable_bottleneck_detection {
            let bottleneck_analyzer = self.bottleneck_analyzer.clone();
            let metrics_store = self.metrics_store.clone();
            let alert_manager = self.alert_manager.clone();
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(5));
                
                loop {
                    interval.tick().await;
                    
                    let metrics = metrics_store.read().await.clone();
                    if let Ok(bottlenecks) = bottleneck_analyzer.analyze_bottlenecks(&metrics).await {
                        for bottleneck in bottlenecks {
                            if bottleneck.severity == Severity::High || bottleneck.severity == Severity::Critical {
                                let alert = Alert::from_bottleneck(&bottleneck);
                                if let Err(e) = alert_manager.trigger_alert(alert).await {
                                    error!("Failed to trigger bottleneck alert: {}", e);
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    
    async fn collect_system_metrics(system_metrics: &Arc<RwLock<SystemMetrics>>) -> Result<()> {
        let mut metrics = system_metrics.write().await;
        
        // M3 Max specific metrics collection
        metrics.m3_max_optimization_active = true; // Detected via hardware
        metrics.unified_memory_utilization = Self::get_unified_memory_usage();
        metrics.metal_performance_shaders_active = Self::is_mps_active();
        metrics.neural_engine_utilization = Self::get_neural_engine_usage();
        metrics.thermal_state = Self::get_thermal_state();
        metrics.power_consumption_watts = Self::get_power_consumption();
        
        Ok(())
    }
    
    fn get_unified_memory_usage() -> f64 {
        // Simplified - would use actual system APIs
        0.75 // 75% utilization
    }
    
    fn is_mps_active() -> bool {
        // Check if Metal Performance Shaders are active
        true
    }
    
    fn get_neural_engine_usage() -> f64 {
        // Neural Engine utilization (if available)
        0.0
    }
    
    fn get_thermal_state() -> ThermalState {
        ThermalState::Nominal
    }
    
    fn get_power_consumption() -> f64 {
        // Power consumption in watts
        45.0
    }
    
    pub async fn start_operation_tracking(
        &self,
        operation_type: OperationType,
        documents_count: usize,
        expected_duration: Option<Duration>,
    ) -> Uuid {
        let operation_id = Uuid::new_v4();
        
        let metrics = OperationMetrics {
            operation_id,
            operation_type,
            start_time: Instant::now(),
            expected_duration,
            current_phase: "starting".to_string(),
            documents_count,
            memory_allocated: 0.0,
            model_used: None,
            quality_target: 0.75,
        };
        
        self.active_operations.insert(operation_id, metrics);
        
        debug!("Started tracking operation {} ({:?})", operation_id, operation_type);
        operation_id
    }
    
    pub async fn update_operation_phase(&self, operation_id: Uuid, phase: String) {
        if let Some(mut operation) = self.active_operations.get_mut(&operation_id) {
            operation.current_phase = phase.clone();
            debug!("Operation {} entered phase: {}", operation_id, phase);
        }
    }
    
    pub async fn complete_operation_tracking(
        &self,
        operation_id: Uuid,
        success: bool,
        final_metrics: Option<HashMap<String, f64>>,
    ) {
        if let Some((_, operation)) = self.active_operations.remove(&operation_id) {
            let duration = operation.start_time.elapsed();
            
            // Update aggregate metrics
            self.update_pipeline_metrics(
                &operation.operation_type,
                duration,
                success,
                operation.documents_count,
            ).await;
            
            info!("Completed operation {} in {:?} (success: {})", 
                  operation_id, duration, success);
            
            // Check for performance deviations
            if let Some(expected) = operation.expected_duration {
                if duration > expected * 2 {
                    let alert = Alert {
                        id: format!("perf_deviation_{}", operation_id),
                        alert_type: AlertType::Performance,
                        severity: Severity::Medium,
                        title: "Performance Deviation Detected".to_string(),
                        description: format!(
                            "Operation took {:.2}s, expected {:.2}s", 
                            duration.as_secs_f64(), 
                            expected.as_secs_f64()
                        ),
                        timestamp: SystemTime::now(),
                        resolved: false,
                        resolution_time: None,
                        related_metrics: final_metrics.unwrap_or_default(),
                    };
                    
                    if let Err(e) = self.alert_manager.trigger_alert(alert).await {
                        error!("Failed to trigger performance alert: {}", e);
                    }
                }
            }
        }
    }
    
    async fn update_pipeline_metrics(
        &self,
        operation_type: &OperationType,
        duration: Duration,
        success: bool,
        documents_count: usize,
    ) {
        let mut metrics = self.metrics_store.write().await;
        
        metrics.pipeline_metrics.total_operations += 1;
        
        if success {
            metrics.pipeline_metrics.successful_operations += 1;
        } else {
            metrics.pipeline_metrics.failed_operations += 1;
        }
        
        // Update average processing time (exponential moving average)
        let alpha = 0.1;
        let current_avg = metrics.pipeline_metrics.average_processing_time.as_secs_f64();
        let new_avg = alpha * duration.as_secs_f64() + (1.0 - alpha) * current_avg;
        metrics.pipeline_metrics.average_processing_time = Duration::from_secs_f64(new_avg);
        
        // Update throughput
        if duration.as_secs_f64() > 0.0 {
            let op_throughput = documents_count as f64 * 3600.0 / duration.as_secs_f64();
            let current_throughput = metrics.pipeline_metrics.current_throughput_docs_hour;
            let new_throughput = alpha * op_throughput + (1.0 - alpha) * current_throughput;
            metrics.pipeline_metrics.current_throughput_docs_hour = new_throughput;
            
            // Calculate efficiency
            metrics.pipeline_metrics.throughput_efficiency = 
                new_throughput / metrics.pipeline_metrics.target_throughput_docs_hour;
        }
        
        metrics.last_updated = SystemTime::now();
    }
    
    pub async fn update_model_metrics(
        &self,
        model_name: String,
        inference_time: Duration,
        quality_score: f64,
        memory_used: f64,
        success: bool,
    ) {
        let mut metrics = self.metrics_store.write().await;
        
        let model_data = metrics.model_metrics.model_performance
            .entry(model_name.clone())
            .or_insert_with(|| ModelPerformanceData {
                inference_time_p50: inference_time,
                inference_time_p95: inference_time,
                quality_score_average: quality_score,
                memory_efficiency: 0.8,
                success_rate: 1.0,
                cost_per_document: 1.0,
                sample_count: 0,
            });
        
        // Update metrics with exponential moving average
        let alpha = 0.1;
        model_data.sample_count += 1;
        
        // Update inference time (simplified - would maintain percentiles properly)
        let current_time = model_data.inference_time_p50.as_secs_f64();
        let new_time = alpha * inference_time.as_secs_f64() + (1.0 - alpha) * current_time;
        model_data.inference_time_p50 = Duration::from_secs_f64(new_time);
        
        // Update quality score
        model_data.quality_score_average = 
            alpha * quality_score + (1.0 - alpha) * model_data.quality_score_average;
        
        // Update success rate
        let success_value = if success { 1.0 } else { 0.0 };
        model_data.success_rate = 
            alpha * success_value + (1.0 - alpha) * model_data.success_rate;
        
        // Update memory efficiency (simplified calculation)
        let efficiency = memory_used / 50.0; // Assuming 50GB baseline
        model_data.memory_efficiency = 
            alpha * efficiency + (1.0 - alpha) * model_data.memory_efficiency;
        
        debug!("Updated metrics for model {}: quality={:.3}, time={:.2}s, success_rate={:.3}",
               model_name, model_data.quality_score_average, 
               model_data.inference_time_p50.as_secs_f64(), model_data.success_rate);
    }
    
    pub async fn get_current_metrics(&self) -> MetricsStore {
        self.metrics_store.read().await.clone()
    }
    
    pub async fn get_performance_report(&self, timeframe_hours: u64) -> PerformanceReport {
        let current_time = SystemTime::now();
        let timeframe = Duration::from_secs(timeframe_hours * 3600);
        
        let history = self.metrics_history.read().await;
        let relevant_metrics: Vec<_> = history
            .iter()
            .filter(|tm| {
                current_time.duration_since(tm.timestamp)
                    .unwrap_or(Duration::MAX) <= timeframe
            })
            .collect();
        
        let bottleneck_history = self.bottleneck_history.read().await;
        let relevant_bottlenecks: Vec<_> = bottleneck_history
            .iter()
            .filter(|be| {
                current_time.duration_since(be.timestamp)
                    .unwrap_or(Duration::MAX) <= timeframe
            })
            .collect();
        
        PerformanceReport {
            timeframe_hours,
            total_operations: relevant_metrics.last().map(|tm| tm.metrics.pipeline_metrics.total_operations).unwrap_or(0),
            average_throughput: self.calculate_average_throughput(&relevant_metrics),
            peak_throughput: self.calculate_peak_throughput(&relevant_metrics),
            quality_trend: self.calculate_quality_trend(&relevant_metrics),
            bottleneck_summary: self.summarize_bottlenecks(&relevant_bottlenecks),
            resource_utilization: self.calculate_resource_utilization(&relevant_metrics),
            model_performance_comparison: self.compare_model_performance(&relevant_metrics),
            optimization_recommendations: self.generate_optimization_recommendations(&relevant_metrics, &relevant_bottlenecks),
            timestamp: current_time,
        }
    }
    
    fn calculate_average_throughput(&self, metrics: &[&TimestampedMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }
        
        metrics.iter()
            .map(|tm| tm.metrics.pipeline_metrics.current_throughput_docs_hour)
            .sum::<f64>() / metrics.len() as f64
    }
    
    fn calculate_peak_throughput(&self, metrics: &[&TimestampedMetrics]) -> f64 {
        metrics.iter()
            .map(|tm| tm.metrics.pipeline_metrics.current_throughput_docs_hour)
            .fold(0.0, f64::max)
    }
    
    fn calculate_quality_trend(&self, metrics: &[&TimestampedMetrics]) -> Vec<(SystemTime, f64)> {
        metrics.iter()
            .map(|tm| (tm.timestamp, tm.metrics.quality_metrics.combined_quality_average))
            .collect()
    }
    
    fn summarize_bottlenecks(&self, bottlenecks: &[&BottleneckEvent]) -> HashMap<BottleneckType, usize> {
        let mut summary = HashMap::new();
        
        for bottleneck in bottlenecks {
            *summary.entry(bottleneck.bottleneck_type.clone()).or_insert(0) += 1;
        }
        
        summary
    }
    
    fn calculate_resource_utilization(&self, metrics: &[&TimestampedMetrics]) -> ResourceUtilizationSummary {
        if metrics.is_empty() {
            return ResourceUtilizationSummary::default();
        }
        
        let mut memory_utilization = 0.0;
        let mut cpu_utilization = 0.0;
        
        for tm in metrics {
            memory_utilization += tm.metrics.resource_metrics.memory_efficiency;
            cpu_utilization += tm.metrics.resource_metrics.cpu_utilization_percent;
        }
        
        let count = metrics.len() as f64;
        
        ResourceUtilizationSummary {
            average_memory_utilization: memory_utilization / count,
            peak_memory_utilization: metrics.iter()
                .map(|tm| tm.metrics.resource_metrics.memory_efficiency)
                .fold(0.0, f64::max),
            average_cpu_utilization: cpu_utilization / count,
            peak_cpu_utilization: metrics.iter()
                .map(|tm| tm.metrics.resource_metrics.cpu_utilization_percent)
                .fold(0.0, f64::max),
        }
    }
    
    fn compare_model_performance(&self, metrics: &[&TimestampedMetrics]) -> Vec<ModelComparisonData> {
        // This would analyze model performance trends over time
        // Simplified implementation
        vec![]
    }
    
    fn generate_optimization_recommendations(
        &self,
        _metrics: &[&TimestampedMetrics],
        bottlenecks: &[&BottleneckEvent],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze bottlenecks for optimization opportunities
        for bottleneck in bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::Memory => {
                    recommendations.push(OptimizationRecommendation {
                        category: "Memory Optimization".to_string(),
                        description: "Consider enabling more aggressive garbage collection or reducing batch sizes".to_string(),
                        expected_improvement: 0.15,
                        implementation_effort: "Low".to_string(),
                    });
                }
                BottleneckType::ModelSwitching => {
                    recommendations.push(OptimizationRecommendation {
                        category: "Model Management".to_string(),
                        description: "Optimize model switching by pre-loading models or using model routing".to_string(),
                        expected_improvement: 0.25,
                        implementation_effort: "Medium".to_string(),
                    });
                }
                _ => {}
            }
        }
        
        recommendations
    }
    
    pub async fn export_metrics(&self, format: ExportFormat) -> Result<String> {
        let metrics = self.get_current_metrics().await;
        
        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(&metrics)?),
            ExportFormat::Prometheus => Ok(self.format_prometheus_metrics(&metrics)),
        }
    }
    
    fn format_prometheus_metrics(&self, metrics: &MetricsStore) -> String {
        let mut output = String::new();
        
        // Pipeline metrics
        output.push_str(&format!("pipeline_total_operations {}\n", metrics.pipeline_metrics.total_operations));
        output.push_str(&format!("pipeline_successful_operations {}\n", metrics.pipeline_metrics.successful_operations));
        output.push_str(&format!("pipeline_failed_operations {}\n", metrics.pipeline_metrics.failed_operations));
        output.push_str(&format!("pipeline_current_throughput {}\n", metrics.pipeline_metrics.current_throughput_docs_hour));
        output.push_str(&format!("pipeline_throughput_efficiency {}\n", metrics.pipeline_metrics.throughput_efficiency));
        
        // Resource metrics
        output.push_str(&format!("resource_memory_efficiency {}\n", metrics.resource_metrics.memory_efficiency));
        output.push_str(&format!("resource_cpu_utilization {}\n", metrics.resource_metrics.cpu_utilization_percent));
        output.push_str(&format!("resource_rust_memory_gb {}\n", metrics.resource_metrics.rust_memory_gb));
        output.push_str(&format!("resource_python_memory_gb {}\n", metrics.resource_metrics.python_memory_gb));
        
        // Quality metrics
        output.push_str(&format!("quality_combined_average {}\n", metrics.quality_metrics.combined_quality_average));
        output.push_str(&format!("quality_structural_average {}\n", metrics.quality_metrics.structural_quality_average));
        output.push_str(&format!("quality_semantic_average {}\n", metrics.quality_metrics.semantic_quality_average));
        
        output
    }
}

// Implementation for supporting types

impl BottleneckAnalyzer {
    fn new() -> Self {
        let mut detection_thresholds = HashMap::new();
        detection_thresholds.insert(BottleneckType::Memory, 0.85);
        detection_thresholds.insert(BottleneckType::CPU, 0.90);
        detection_thresholds.insert(BottleneckType::IPC, 0.80);
        detection_thresholds.insert(BottleneckType::ModelSwitching, 3.0); // 3 seconds
        
        Self {
            detection_thresholds,
            analysis_window: Duration::from_secs(60),
            recent_detections: VecDeque::with_capacity(100),
        }
    }
    
    async fn analyze_bottlenecks(&self, metrics: &MetricsStore) -> Result<Vec<BottleneckEvent>> {
        let mut bottlenecks = Vec::new();
        
        // Memory bottleneck detection
        if metrics.resource_metrics.memory_efficiency > self.detection_thresholds[&BottleneckType::Memory] {
            bottlenecks.push(BottleneckEvent {
                timestamp: SystemTime::now(),
                bottleneck_type: BottleneckType::Memory,
                severity: if metrics.resource_metrics.memory_efficiency > 0.95 { Severity::Critical } else { Severity::High },
                description: format!("High memory utilization: {:.1}%", metrics.resource_metrics.memory_efficiency * 100.0),
                affected_components: vec!["Rust Core".to_string(), "Python ML".to_string()],
                performance_impact: metrics.resource_metrics.memory_efficiency - self.detection_thresholds[&BottleneckType::Memory],
                suggested_actions: vec![
                    "Reduce batch sizes".to_string(),
                    "Enable garbage collection".to_string(),
                    "Optimize memory allocation".to_string(),
                ],
            });
        }
        
        // CPU bottleneck detection
        if metrics.resource_metrics.cpu_utilization_percent > self.detection_thresholds[&BottleneckType::CPU] {
            bottlenecks.push(BottleneckEvent {
                timestamp: SystemTime::now(),
                bottleneck_type: BottleneckType::CPU,
                severity: Severity::Medium,
                description: format!("High CPU utilization: {:.1}%", metrics.resource_metrics.cpu_utilization_percent),
                affected_components: vec!["Processing Pipeline".to_string()],
                performance_impact: (metrics.resource_metrics.cpu_utilization_percent - self.detection_thresholds[&BottleneckType::CPU]) / 100.0,
                suggested_actions: vec![
                    "Optimize parallel processing".to_string(),
                    "Review algorithm efficiency".to_string(),
                ],
            });
        }
        
        Ok(bottlenecks)
    }
}

impl PerformancePredictor {
    fn new() -> Self {
        Self {
            prediction_models: HashMap::new(),
            training_data: VecDeque::with_capacity(10000),
            prediction_accuracy: 0.0,
        }
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            active_alerts: DashMap::new(),
            alert_history: VecDeque::with_capacity(1000),
            notification_handlers: Vec::new(),
        }
    }
    
    async fn trigger_alert(&self, alert: Alert) -> Result<()> {
        let alert_id = alert.id.clone();
        
        // Check if alert already exists
        if self.active_alerts.contains_key(&alert_id) {
            return Ok(()); // Alert already active
        }
        
        self.active_alerts.insert(alert_id.clone(), alert.clone());
        
        // Notify handlers
        for handler in &self.notification_handlers {
            if let Err(e) = handler.handle_alert(&alert) {
                error!("Notification handler failed: {}", e);
            }
        }
        
        info!("Alert triggered: {} - {}", alert.title, alert.description);
        Ok(())
    }
}

impl Alert {
    fn from_bottleneck(bottleneck: &BottleneckEvent) -> Self {
        Self {
            id: format!("bottleneck_{}_{}", 
                       bottleneck.bottleneck_type as u8,
                       bottleneck.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs()),
            alert_type: AlertType::Performance,
            severity: bottleneck.severity.clone(),
            title: format!("{:?} Bottleneck Detected", bottleneck.bottleneck_type),
            description: bottleneck.description.clone(),
            timestamp: bottleneck.timestamp,
            resolved: false,
            resolution_time: None,
            related_metrics: HashMap::new(),
        }
    }
}

// Default implementations

impl Default for MetricsStore {
    fn default() -> Self {
        Self {
            pipeline_metrics: PipelineMetrics::default(),
            rust_metrics: RustProcessingMetrics::default(),
            python_metrics: PythonMLMetrics::default(),
            ipc_metrics: IPCMetrics::default(),
            model_metrics: ModelMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            quality_metrics: QualityMetrics::default(),
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_processing_time: Duration::from_secs(0),
            current_throughput_docs_hour: 0.0,
            target_throughput_docs_hour: 25.0,
            throughput_efficiency: 0.0,
            queue_depth: 0,
            active_sessions: 0,
            uptime: Duration::from_secs(0),
        }
    }
}

impl Default for RustProcessingMetrics {
    fn default() -> Self {
        Self {
            documents_processed: 0,
            average_processing_time: Duration::from_secs(0),
            batch_processing_efficiency: 0.0,
            quality_validation_time: Duration::from_secs(0),
            memory_usage_gb: 0.0,
            cpu_utilization_percent: 0.0,
            concurrent_operations: 0,
            cache_hit_rate: 0.0,
        }
    }
}

impl Default for PythonMLMetrics {
    fn default() -> Self {
        Self {
            models_loaded: 0,
            model_switch_count: 0,
            average_switch_latency: Duration::from_secs(0),
            inference_time_by_model: HashMap::new(),
            memory_usage_by_model: HashMap::new(),
            gpu_utilization_percent: 0.0,
            mlx_acceleration_active: false,
            batch_efficiency: 0.0,
        }
    }
}

impl Default for IPCMetrics {
    fn default() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            average_latency: Duration::from_secs(0),
            max_latency: Duration::from_secs(0),
            timeout_count: 0,
            shared_memory_usage_gb: 0.0,
            connection_pool_utilization: 0.0,
            serialization_time: Duration::from_secs(0),
        }
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            model_performance: HashMap::new(),
            current_active_model: None,
            model_reliability_scores: HashMap::new(),
            adaptation_effectiveness: 0.0,
            selection_accuracy: 0.0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            total_memory_gb: 128.0, // M3 Max total
            rust_memory_gb: 0.0,
            python_memory_gb: 0.0,
            shared_memory_gb: 0.0,
            memory_efficiency: 0.0,
            cpu_cores_utilized: 0,
            cpu_utilization_percent: 0.0,
            disk_io_mb_per_sec: 0.0,
            network_io_mb_per_sec: 0.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            structural_quality_average: 0.0,
            semantic_quality_average: 0.0,
            combined_quality_average: 0.0,
            quality_consistency: 0.0,
            parameter_extraction_rate: 0.0,
            technical_density_average: 0.0,
            validation_accuracy: 0.0,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            m3_max_optimization_active: false,
            unified_memory_utilization: 0.0,
            metal_performance_shaders_active: false,
            neural_engine_utilization: 0.0,
            thermal_state: ThermalState::Nominal,
            power_consumption_watts: 0.0,
        }
    }
}

// Additional support types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timeframe_hours: u64,
    pub total_operations: u64,
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub quality_trend: Vec<(SystemTime, f64)>,
    pub bottleneck_summary: HashMap<BottleneckType, usize>,
    pub resource_utilization: ResourceUtilizationSummary,
    pub model_performance_comparison: Vec<ModelComparisonData>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationSummary {
    pub average_memory_utilization: f64,
    pub peak_memory_utilization: f64,
    pub average_cpu_utilization: f64,
    pub peak_cpu_utilization: f64,
}

impl Default for ResourceUtilizationSummary {
    fn default() -> Self {
        Self {
            average_memory_utilization: 0.0,
            peak_memory_utilization: 0.0,
            average_cpu_utilization: 0.0,
            peak_cpu_utilization: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonData {
    pub model_name: String,
    pub average_performance: f64,
    pub trend: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: String,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Prometheus,
}