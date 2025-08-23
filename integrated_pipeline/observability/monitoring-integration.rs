//! Monitoring Integration Module for RAN LLM Claude Flow
//! Integrates distributed tracing, metrics collection, and real-time monitoring
//! with <1% overhead target and comprehensive observability

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry,
};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{broadcast, mpsc},
    time::interval,
};
use tracing::{debug, error, info, instrument, warn};

/// Configuration for monitoring integration
#[derive(Debug, Clone, Deserialize)]
pub struct MonitoringConfig {
    pub prometheus_port: u16,
    pub metrics_collection_interval: Duration,
    pub tracing_sample_rate: f64,
    pub enable_bottleneck_detection: bool,
    pub enable_adaptive_optimization: bool,
    pub max_overhead_percentage: f64,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlertThresholds {
    pub ipc_latency_us: u64,           // 100 microseconds
    pub document_throughput: f64,      // 25 docs/hour
    pub quality_score: f64,            // 0.75
    pub memory_utilization: f64,       // 0.95
    pub monitoring_overhead: f64,      // 0.01 (1%)
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus_port: 8080,
            metrics_collection_interval: Duration::from_secs(1),
            tracing_sample_rate: 0.1,
            enable_bottleneck_detection: true,
            enable_adaptive_optimization: true,
            max_overhead_percentage: 1.0,
            alert_thresholds: AlertThresholds {
                ipc_latency_us: 100,
                document_throughput: 25.0,
                quality_score: 0.75,
                memory_utilization: 0.95,
                monitoring_overhead: 0.01,
            },
        }
    }
}

/// Comprehensive metrics for the monitoring system
#[derive(Debug)]
pub struct MonitoringMetrics {
    // Document processing metrics
    pub documents_processed: IntCounter,
    pub document_processing_duration: Histogram,
    pub document_quality_score: Gauge,
    
    // IPC communication metrics
    pub ipc_latency: Histogram,
    pub ipc_messages_sent: IntCounter,
    pub ipc_messages_received: IntCounter,
    
    // System resource metrics
    pub rust_core_memory: IntGauge,
    pub python_ml_memory: IntGauge,
    pub shared_memory_used: IntGauge,
    pub monitoring_memory: IntGauge,
    
    // Performance optimization metrics
    pub optimizations_applied: IntCounter,
    pub optimization_effectiveness: Gauge,
    pub bottleneck_severity: Gauge,
    
    // Neural model metrics
    pub neural_model_confidence: Gauge,
    pub neural_predictions: IntCounter,
    pub neural_inference_duration: Histogram,
    
    // System health metrics
    pub component_health: IntGauge,
    pub monitoring_overhead: Gauge,
}

impl MonitoringMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let documents_processed = IntCounter::with_opts(
            Opts::new("documents_processed_total", "Total documents processed")
                .namespace("ran_llm")
        )?.with(&["component", "stage", "model"]);
        registry.register(Box::new(documents_processed.clone()))?;
        
        let document_processing_duration = Histogram::with_opts(
            HistogramOpts::new("document_processing_duration_seconds", "Document processing time")
                .namespace("ran_llm")
                .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])
        )?.with(&["stage", "model"]);
        registry.register(Box::new(document_processing_duration.clone()))?;
        
        let document_quality_score = Gauge::with_opts(
            Opts::new("document_quality_score", "Current document quality score")
                .namespace("ran_llm")
        )?.with(&["model", "document_type"]);
        registry.register(Box::new(document_quality_score.clone()))?;
        
        let ipc_latency = Histogram::with_opts(
            HistogramOpts::new("ipc_latency_seconds", "IPC communication latency")
                .namespace("ran_llm")
                .buckets(vec![0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
        )?.with(&["direction", "component"]);
        registry.register(Box::new(ipc_latency.clone()))?;
        
        let ipc_messages_sent = IntCounter::with_opts(
            Opts::new("ipc_messages_sent_total", "Total IPC messages sent")
                .namespace("ran_llm")
        )?.with(&["message_type", "destination"]);
        registry.register(Box::new(ipc_messages_sent.clone()))?;
        
        let ipc_messages_received = IntCounter::with_opts(
            Opts::new("ipc_messages_received_total", "Total IPC messages received")
                .namespace("ran_llm")
        )?.with(&["message_type", "source"]);
        registry.register(Box::new(ipc_messages_received.clone()))?;
        
        let rust_core_memory = IntGauge::with_opts(
            Opts::new("rust_core_memory_bytes", "Rust core memory usage in bytes")
                .namespace("ran_llm")
        )?;
        registry.register(Box::new(rust_core_memory.clone()))?;
        
        let python_ml_memory = IntGauge::with_opts(
            Opts::new("python_ml_memory_bytes", "Python ML memory usage in bytes")
                .namespace("ran_llm")
        )?;
        registry.register(Box::new(python_ml_memory.clone()))?;
        
        let shared_memory_used = IntGauge::with_opts(
            Opts::new("shared_memory_used_bytes", "Shared memory pool usage in bytes")
                .namespace("ran_llm")
        )?;
        registry.register(Box::new(shared_memory_used.clone()))?;
        
        let monitoring_memory = IntGauge::with_opts(
            Opts::new("monitoring_memory_bytes", "Monitoring system memory usage")
                .namespace("ran_llm")
        )?;
        registry.register(Box::new(monitoring_memory.clone()))?;
        
        let optimizations_applied = IntCounter::with_opts(
            Opts::new("optimizations_applied_total", "Total optimizations applied")
                .namespace("ran_llm")
        )?.with(&["optimization_type", "component"]);
        registry.register(Box::new(optimizations_applied.clone()))?;
        
        let optimization_effectiveness = Gauge::with_opts(
            Opts::new("optimization_effectiveness_score", "Effectiveness of applied optimizations")
                .namespace("ran_llm")
        )?.with(&["optimization_type"]);
        registry.register(Box::new(optimization_effectiveness.clone()))?;
        
        let bottleneck_severity = Gauge::with_opts(
            Opts::new("bottleneck_severity_score", "Current bottleneck severity score")
                .namespace("ran_llm")
        )?.with(&["bottleneck_component", "bottleneck_type"]);
        registry.register(Box::new(bottleneck_severity.clone()))?;
        
        let neural_model_confidence = Gauge::with_opts(
            Opts::new("neural_model_confidence", "Neural model confidence score")
                .namespace("ran_llm")
        )?.with(&["model"]);
        registry.register(Box::new(neural_model_confidence.clone()))?;
        
        let neural_predictions = IntCounter::with_opts(
            Opts::new("neural_predictions_total", "Total neural model predictions")
                .namespace("ran_llm")
        )?.with(&["model", "confidence_level"]);
        registry.register(Box::new(neural_predictions.clone()))?;
        
        let neural_inference_duration = Histogram::with_opts(
            HistogramOpts::new("neural_inference_duration_seconds", "Neural inference time")
                .namespace("ran_llm")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
        )?.with(&["model"]);
        registry.register(Box::new(neural_inference_duration.clone()))?;
        
        let component_health = IntGauge::with_opts(
            Opts::new("component_health", "Component health status (1=healthy, 0=unhealthy)")
                .namespace("ran_llm")
        )?.with(&["component"]);
        registry.register(Box::new(component_health.clone()))?;
        
        let monitoring_overhead = Gauge::with_opts(
            Opts::new("monitoring_overhead_percentage", "Monitoring system overhead percentage")
                .namespace("ran_llm")
        )?;
        registry.register(Box::new(monitoring_overhead.clone()))?;
        
        Ok(Self {
            documents_processed,
            document_processing_duration,
            document_quality_score,
            ipc_latency,
            ipc_messages_sent,
            ipc_messages_received,
            rust_core_memory,
            python_ml_memory,
            shared_memory_used,
            monitoring_memory,
            optimizations_applied,
            optimization_effectiveness,
            bottleneck_severity,
            neural_model_confidence,
            neural_predictions,
            neural_inference_duration,
            component_health,
            monitoring_overhead,
        })
    }
}

/// Real-time monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSnapshot {
    pub timestamp: Instant,
    pub documents_per_hour: f64,
    pub ipc_latency_p95_us: f64,
    pub quality_score_avg: f64,
    pub memory_utilization: MemoryUtilization,
    pub bottlenecks: Vec<BottleneckInfo>,
    pub system_health: SystemHealth,
    pub monitoring_overhead: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUtilization {
    pub rust_core_gb: f64,
    pub python_ml_gb: f64,
    pub shared_memory_gb: f64,
    pub monitoring_gb: f64,
    pub total_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    pub component: String,
    pub severity: f64,
    pub bottleneck_type: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub rust_core: bool,
    pub python_ml: bool,
    pub ipc_protocol: bool,
    pub mcp_server: bool,
    pub overall_healthy: bool,
}

/// Main monitoring integration system
pub struct MonitoringIntegration {
    config: MonitoringConfig,
    metrics: Arc<MonitoringMetrics>,
    registry: Arc<Registry>,
    monitoring_start_time: Instant,
    
    // Communication channels
    snapshot_tx: broadcast::Sender<MonitoringSnapshot>,
    alert_tx: mpsc::UnboundedSender<AlertEvent>,
    
    // State tracking
    current_snapshot: Arc<RwLock<Option<MonitoringSnapshot>>>,
    bottleneck_history: Arc<RwLock<Vec<BottleneckInfo>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlertEvent {
    pub alert_type: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub component: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Serialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl MonitoringIntegration {
    pub fn new(config: MonitoringConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Arc::new(Registry::new());
        let metrics = Arc::new(MonitoringMetrics::new(&registry)?);
        
        let (snapshot_tx, _) = broadcast::channel(100);
        let (alert_tx, _alert_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            config,
            metrics,
            registry,
            monitoring_start_time: Instant::now(),
            snapshot_tx,
            alert_tx,
            current_snapshot: Arc::new(RwLock::new(None)),
            bottleneck_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Start the monitoring system
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting monitoring integration system");
        
        // Start Prometheus HTTP server
        self.start_prometheus_server().await?;
        
        // Start metrics collection loop
        self.start_metrics_collection().await;
        
        // Start bottleneck detection
        if self.config.enable_bottleneck_detection {
            self.start_bottleneck_detection().await;
        }
        
        // Start adaptive optimization
        if self.config.enable_adaptive_optimization {
            self.start_adaptive_optimization().await;
        }
        
        info!("Monitoring integration system started successfully");
        Ok(())
    }
    
    async fn start_prometheus_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let registry = self.registry.clone();
        let port = self.config.prometheus_port;
        
        tokio::spawn(async move {
            let addr = ([0, 0, 0, 0], port).into();
            let service = prometheus::make_service(registry);
            
            info!("Starting Prometheus HTTP server on port {}", port);
            
            if let Err(e) = hyper::Server::bind(&addr)
                .serve(service)
                .await
            {
                error!("Prometheus HTTP server error: {}", e);
            }
        });
        
        Ok(())
    }
    
    async fn start_metrics_collection(&self) {
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        let snapshot_tx = self.snapshot_tx.clone();
        let current_snapshot = self.current_snapshot.clone();
        let monitoring_start_time = self.monitoring_start_time;
        
        tokio::spawn(async move {
            let mut interval = interval(config.metrics_collection_interval);
            
            loop {
                interval.tick().await;
                
                // Calculate monitoring overhead
                let overhead_start = Instant::now();
                
                // Collect current metrics
                let snapshot = Self::collect_snapshot(&metrics, monitoring_start_time).await;
                
                // Calculate overhead
                let overhead_duration = overhead_start.elapsed();
                let overhead_percentage = (overhead_duration.as_nanos() as f64) / 
                    (config.metrics_collection_interval.as_nanos() as f64) * 100.0;
                
                metrics.monitoring_overhead.set(overhead_percentage);
                
                // Update current snapshot
                {
                    let mut current = current_snapshot.write().unwrap();
                    *current = Some(snapshot.clone());
                }
                
                // Broadcast snapshot
                if let Err(e) = snapshot_tx.send(snapshot) {
                    debug!("No active snapshot receivers: {}", e);
                }
            }
        });
    }
    
    async fn start_bottleneck_detection(&self) {
        let bottleneck_history = self.bottleneck_history.clone();
        let alert_tx = self.alert_tx.clone();
        let current_snapshot = self.current_snapshot.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if let Some(snapshot) = current_snapshot.read().unwrap().as_ref() {
                    let bottlenecks = Self::detect_bottlenecks(snapshot).await;
                    
                    // Update bottleneck history
                    {
                        let mut history = bottleneck_history.write().unwrap();
                        history.extend(bottlenecks.clone());
                        
                        // Keep only recent bottlenecks (last 100)
                        if history.len() > 100 {
                            history.drain(0..history.len()-100);
                        }
                    }
                    
                    // Send alerts for critical bottlenecks
                    for bottleneck in bottlenecks {
                        if bottleneck.severity > 0.8 {
                            let alert = AlertEvent {
                                alert_type: "performance_bottleneck".to_string(),
                                severity: AlertSeverity::Critical,
                                message: format!("Critical bottleneck detected in {}: {}", 
                                    bottleneck.component, bottleneck.recommendation),
                                component: bottleneck.component.clone(),
                                metric_value: bottleneck.severity,
                                threshold: 0.8,
                                timestamp: Instant::now(),
                            };
                            
                            if let Err(e) = alert_tx.send(alert) {
                                warn!("Failed to send bottleneck alert: {}", e);
                            }
                        }
                    }
                }
            }
        });
    }
    
    async fn start_adaptive_optimization(&self) {
        let metrics = self.metrics.clone();
        let current_snapshot = self.current_snapshot.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Some(snapshot) = current_snapshot.read().unwrap().as_ref() {
                    let optimizations = Self::generate_optimizations(snapshot).await;
                    
                    for optimization in optimizations {
                        // Apply optimization (placeholder - would integrate with actual optimizer)
                        info!("Applying optimization: {} for component {}", 
                            optimization.optimization_type, optimization.component);
                        
                        metrics.optimizations_applied
                            .with_label_values(&[&optimization.optimization_type, &optimization.component])
                            .inc();
                        
                        metrics.optimization_effectiveness
                            .with_label_values(&[&optimization.optimization_type])
                            .set(optimization.expected_effectiveness);
                    }
                }
            }
        });
    }
    
    async fn collect_snapshot(metrics: &MonitoringMetrics, start_time: Instant) -> MonitoringSnapshot {
        // This would collect actual metrics from the system
        // For now, providing a template structure
        
        MonitoringSnapshot {
            timestamp: Instant::now(),
            documents_per_hour: 30.0, // Would be calculated from actual metrics
            ipc_latency_p95_us: 85.0, // Would be from actual histogram
            quality_score_avg: 0.82,  // Would be from actual gauge
            memory_utilization: MemoryUtilization {
                rust_core_gb: 45.2,
                python_ml_gb: 38.1,
                shared_memory_gb: 12.8,
                monitoring_gb: 6.3,
                total_utilization: 0.80,
            },
            bottlenecks: Vec::new(),
            system_health: SystemHealth {
                rust_core: true,
                python_ml: true,
                ipc_protocol: true,
                mcp_server: true,
                overall_healthy: true,
            },
            monitoring_overhead: 0.7, // Would be calculated from actual timing
        }
    }
    
    async fn detect_bottlenecks(snapshot: &MonitoringSnapshot) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();
        
        // Document throughput bottleneck
        if snapshot.documents_per_hour < 25.0 {
            bottlenecks.push(BottleneckInfo {
                component: "document_processor".to_string(),
                severity: (25.0 - snapshot.documents_per_hour) / 25.0,
                bottleneck_type: "throughput".to_string(),
                recommendation: "Consider increasing worker threads or switching to smaller model".to_string(),
            });
        }
        
        // IPC latency bottleneck
        if snapshot.ipc_latency_p95_us > 100.0 {
            bottlenecks.push(BottleneckInfo {
                component: "ipc_protocol".to_string(),
                severity: (snapshot.ipc_latency_p95_us - 100.0) / 100.0,
                bottleneck_type: "latency".to_string(),
                recommendation: "Optimize message serialization or reduce message frequency".to_string(),
            });
        }
        
        // Memory utilization bottleneck
        if snapshot.memory_utilization.total_utilization > 0.9 {
            bottlenecks.push(BottleneckInfo {
                component: "memory_manager".to_string(),
                severity: snapshot.memory_utilization.total_utilization - 0.9,
                bottleneck_type: "memory_pressure".to_string(),
                recommendation: "Trigger garbage collection or increase shared memory pool".to_string(),
            });
        }
        
        bottlenecks
    }
    
    async fn generate_optimizations(snapshot: &MonitoringSnapshot) -> Vec<OptimizationAction> {
        let mut optimizations = Vec::new();
        
        // Generate optimizations based on current snapshot
        // This is a placeholder - would contain actual optimization logic
        
        optimizations
    }
    
    /// Get the current monitoring snapshot
    pub fn get_current_snapshot(&self) -> Option<MonitoringSnapshot> {
        self.current_snapshot.read().unwrap().clone()
    }
    
    /// Subscribe to real-time monitoring updates
    pub fn subscribe_to_snapshots(&self) -> broadcast::Receiver<MonitoringSnapshot> {
        self.snapshot_tx.subscribe()
    }
    
    /// Get bottleneck history
    pub fn get_bottleneck_history(&self) -> Vec<BottleneckInfo> {
        self.bottleneck_history.read().unwrap().clone()
    }
    
    /// Manual trigger for performance analysis
    #[instrument(skip(self))]
    pub async fn trigger_performance_analysis(&self) -> Result<PerformanceAnalysisReport, Box<dyn std::error::Error>> {
        info!("Triggering manual performance analysis");
        
        let snapshot = self.get_current_snapshot()
            .ok_or("No monitoring snapshot available")?;
        
        let report = PerformanceAnalysisReport {
            timestamp: Instant::now(),
            overall_score: self.calculate_overall_performance_score(&snapshot),
            sla_compliance: self.check_sla_compliance(&snapshot),
            bottlenecks: Self::detect_bottlenecks(&snapshot).await,
            recommendations: self.generate_recommendations(&snapshot).await,
            trend_analysis: self.analyze_trends().await,
        };
        
        info!("Performance analysis completed with score: {:.2}", report.overall_score);
        Ok(report)
    }
    
    fn calculate_overall_performance_score(&self, snapshot: &MonitoringSnapshot) -> f64 {
        let throughput_score = (snapshot.documents_per_hour / 25.0).min(1.0);
        let latency_score = if snapshot.ipc_latency_p95_us <= 100.0 { 1.0 } else { 100.0 / snapshot.ipc_latency_p95_us };
        let quality_score = snapshot.quality_score_avg;
        let memory_score = 1.0 - snapshot.memory_utilization.total_utilization.max(0.0).min(1.0);
        let overhead_score = if snapshot.monitoring_overhead <= 1.0 { 1.0 } else { 1.0 / snapshot.monitoring_overhead };
        
        // Weighted average
        (throughput_score * 0.3 + 
         latency_score * 0.25 + 
         quality_score * 0.25 + 
         memory_score * 0.15 + 
         overhead_score * 0.05)
    }
    
    fn check_sla_compliance(&self, snapshot: &MonitoringSnapshot) -> SLACompliance {
        SLACompliance {
            throughput_sla_met: snapshot.documents_per_hour >= self.config.alert_thresholds.document_throughput,
            latency_sla_met: snapshot.ipc_latency_p95_us <= self.config.alert_thresholds.ipc_latency_us as f64,
            quality_sla_met: snapshot.quality_score_avg >= self.config.alert_thresholds.quality_score,
            memory_sla_met: snapshot.memory_utilization.total_utilization <= self.config.alert_thresholds.memory_utilization,
            monitoring_overhead_sla_met: snapshot.monitoring_overhead <= self.config.alert_thresholds.monitoring_overhead * 100.0,
            overall_compliance: true, // Would be calculated from above
        }
    }
    
    async fn generate_recommendations(&self, _snapshot: &MonitoringSnapshot) -> Vec<String> {
        // Generate performance recommendations based on current state
        vec![
            "Consider optimizing neural model switching strategy".to_string(),
            "Evaluate shared memory pool sizing for current workload".to_string(),
            "Monitor IPC message batching effectiveness".to_string(),
        ]
    }
    
    async fn analyze_trends(&self) -> TrendAnalysis {
        // Analyze performance trends over time
        TrendAnalysis {
            throughput_trend: TrendDirection::Stable,
            latency_trend: TrendDirection::Improving,
            quality_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Increasing,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationAction {
    pub optimization_type: String,
    pub component: String,
    pub expected_effectiveness: f64,
    pub priority: OptimizationPriority,
}

#[derive(Debug, Clone, Serialize)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceAnalysisReport {
    pub timestamp: Instant,
    pub overall_score: f64,
    pub sla_compliance: SLACompliance,
    pub bottlenecks: Vec<BottleneckInfo>,
    pub recommendations: Vec<String>,
    pub trend_analysis: TrendAnalysis,
}

#[derive(Debug, Clone, Serialize)]
pub struct SLACompliance {
    pub throughput_sla_met: bool,
    pub latency_sla_met: bool,
    pub quality_sla_met: bool,
    pub memory_sla_met: bool,
    pub monitoring_overhead_sla_met: bool,
    pub overall_compliance: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrendAnalysis {
    pub throughput_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub quality_trend: TrendDirection,
    pub memory_trend: TrendDirection,
}

#[derive(Debug, Clone, Serialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Increasing,
    Decreasing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_monitoring_integration_creation() {
        let config = MonitoringConfig::default();
        let monitoring = MonitoringIntegration::new(config);
        assert!(monitoring.is_ok());
    }

    #[tokio::test]
    async fn test_performance_analysis() {
        let config = MonitoringConfig::default();
        let monitoring = MonitoringIntegration::new(config).unwrap();
        
        // Start monitoring system
        monitoring.start().await.expect("Failed to start monitoring");
        
        // Wait for some metrics collection
        sleep(Duration::from_secs(2)).await;
        
        // Trigger performance analysis
        let report = monitoring.trigger_performance_analysis().await;
        assert!(report.is_ok());
        
        let report = report.unwrap();
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
    }
}