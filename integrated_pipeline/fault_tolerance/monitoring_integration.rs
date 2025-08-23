//! Fault Tolerance Integration with Monitoring System
//! 
//! Bridges the fault tolerance components with the existing bottleneck analyzer
//! and alert system for comprehensive system monitoring and response.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    sync::{broadcast, mpsc},
    time,
};
use tracing::{error, info, warn};

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerError, CircuitState},
    fault_detector::{FaultDetector, FailureRecord, FailureType, FailureSeverity},
    recovery_manager::{RecoveryManager, RecoveryConfig},
    isolation::{FailureIsolationManager, IsolationReason},
};

/// Integration bridge between fault tolerance and monitoring systems
pub struct FaultToleranceMonitoringBridge {
    // Core components
    fault_detector: Arc<FaultDetector>,
    recovery_manager: Arc<RecoveryManager>,
    isolation_manager: Arc<FailureIsolationManager>,
    
    // Communication channels
    bottleneck_receiver: broadcast::Receiver<Vec<MonitoringBottleneck>>,
    alert_sender: mpsc::UnboundedSender<FaultToleranceAlert>,
    
    // State management
    running: AtomicBool,
    component_health: Arc<dashmap::DashMap<String, ComponentHealth>>,
    
    // Configuration
    integration_config: IntegrationConfig,
}

impl FaultToleranceMonitoringBridge {
    pub fn new(
        fault_detector: Arc<FaultDetector>,
        recovery_manager: Arc<RecoveryManager>, 
        isolation_manager: Arc<FailureIsolationManager>,
        bottleneck_receiver: broadcast::Receiver<Vec<MonitoringBottleneck>>,
        alert_sender: mpsc::UnboundedSender<FaultToleranceAlert>,
        integration_config: IntegrationConfig,
    ) -> Self {
        Self {
            fault_detector,
            recovery_manager,
            isolation_manager,
            bottleneck_receiver,
            alert_sender,
            running: AtomicBool::new(false),
            component_health: Arc::new(dashmap::DashMap::new()),
            integration_config,
        }
    }
    
    pub async fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }
        
        self.running.store(true, Ordering::Release);
        info!("Starting fault tolerance monitoring bridge");
        
        // Initialize component health tracking
        self.initialize_component_health().await?;
        
        // Start monitoring loop
        let bridge_clone = self.clone_for_task();
        tokio::spawn(async move {
            bridge_clone.monitoring_loop().await;
        });
        
        // Start bottleneck processing
        let bottleneck_processor = self.clone_for_task();
        tokio::spawn(async move {
            bottleneck_processor.bottleneck_processing_loop().await;
        });
        
        // Start health assessment loop
        let health_assessor = self.clone_for_task();
        tokio::spawn(async move {
            health_assessor.health_assessment_loop().await;
        });
        
        Ok(())
    }
    
    async fn initialize_component_health(&self) -> Result<()> {
        let components = vec![
            "rust_core",
            "python_ml", 
            "ipc_manager",
            "quality_validator",
            "document_processor",
            "semantic_processor",
            "mlx_accelerator",
        ];
        
        for component in components {
            self.component_health.insert(
                component.to_string(),
                ComponentHealth {
                    component_name: component.to_string(),
                    health_status: HealthStatus::Healthy,
                    last_failure: None,
                    failure_count: 0,
                    circuit_breaker_state: CircuitState::Closed,
                    last_updated: Utc::now(),
                    metrics: ComponentMetrics::default(),
                }
            );
        }
        
        info!("Initialized health tracking for {} components", self.component_health.len());
        Ok(())
    }
    
    async fn monitoring_loop(&self) {
        let mut interval = time::interval(Duration::from_millis(
            self.integration_config.monitoring_interval_ms
        ));
        
        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.perform_monitoring_cycle().await {
                error!("Monitoring cycle failed: {}", e);
            }
        }
    }
    
    async fn perform_monitoring_cycle(&self) -> Result<()> {
        // Analyze current fault patterns
        let fault_patterns = self.fault_detector.detect_current_bottlenecks().await?;
        
        // Update component health based on patterns
        for pattern in &fault_patterns {
            self.update_component_health_from_pattern(pattern).await?;
        }
        
        // Check for components requiring intervention
        self.check_component_interventions().await?;
        
        // Generate health summary alerts
        self.generate_health_summary_alerts().await?;
        
        Ok(())
    }
    
    async fn bottleneck_processing_loop(&self) {
        while self.running.load(Ordering::Acquire) {
            match self.bottleneck_receiver.recv().await {
                Ok(bottlenecks) => {
                    if let Err(e) = self.process_monitoring_bottlenecks(bottlenecks).await {
                        error!("Failed to process bottlenecks: {}", e);
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {
                    warn!("Bottleneck channel closed, stopping bottleneck processing");
                    break;
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    warn!("Bottleneck processing lagged, skipped {} messages", skipped);
                }
            }
        }
    }
    
    async fn process_monitoring_bottlenecks(&self, bottlenecks: Vec<MonitoringBottleneck>) -> Result<()> {
        for bottleneck in bottlenecks {
            // Convert monitoring bottleneck to fault tolerance failure record
            let failure_record = self.convert_bottleneck_to_failure(&bottleneck)?;
            
            // Record failure in fault detector
            self.fault_detector.record_failure(failure_record).await;
            
            // Determine if immediate intervention is needed
            if self.should_trigger_immediate_intervention(&bottleneck).await? {
                self.trigger_immediate_intervention(&bottleneck).await?;
            }
            
            // Update component health
            self.update_component_health_from_bottleneck(&bottleneck).await?;
        }
        
        Ok(())
    }
    
    fn convert_bottleneck_to_failure(&self, bottleneck: &MonitoringBottleneck) -> Result<FailureRecord> {
        let failure_type = match bottleneck.bottleneck_type {
            MonitoringBottleneckType::CpuHighUtilization => FailureType::ResourceExhaustion,
            MonitoringBottleneckType::MemoryPressure => FailureType::ResourceExhaustion,
            MonitoringBottleneckType::IpcLatency => FailureType::Timeout,
            MonitoringBottleneckType::PipelineThroughput => FailureType::PerformanceDegradation,
            MonitoringBottleneckType::PipelineErrors => FailureType::ProcessingError,
            _ => FailureType::ProcessingError,
        };
        
        let severity = match bottleneck.severity {
            MonitoringBottleneckSeverity::Critical => FailureSeverity::Critical,
            MonitoringBottleneckSeverity::High => FailureSeverity::High,
            MonitoringBottleneckSeverity::Medium => FailureSeverity::Medium,
            MonitoringBottleneckSeverity::Low => FailureSeverity::Low,
        };
        
        let component = self.determine_component_from_bottleneck(bottleneck);
        
        Ok(FailureRecord {
            timestamp: bottleneck.detected_at,
            failure_type,
            severity,
            error_message: bottleneck.description.clone(),
            duration_ms: (bottleneck.impact_score * 5000.0) as u64, // Estimate duration from impact
            component,
            context: {
                let mut context = HashMap::new();
                context.insert("bottleneck_id".to_string(), bottleneck.id.clone());
                context.insert("impact_score".to_string(), bottleneck.impact_score.to_string());
                context.insert("suggested_actions".to_string(), bottleneck.suggested_actions.join(";"));
                context
            },
        })
    }
    
    fn determine_component_from_bottleneck(&self, bottleneck: &MonitoringBottleneck) -> String {
        match bottleneck.bottleneck_type {
            MonitoringBottleneckType::CpuHighUtilization | 
            MonitoringBottleneckType::CpuThermal => "rust_core".to_string(),
            MonitoringBottleneckType::PythonMemoryLimit => "python_ml".to_string(),
            MonitoringBottleneckType::RustMemoryLimit => "rust_core".to_string(),
            MonitoringBottleneckType::IpcLatency | 
            MonitoringBottleneckType::IpcBandwidth => "ipc_manager".to_string(),
            MonitoringBottleneckType::ModelInference => "python_ml".to_string(),
            MonitoringBottleneckType::QualityValidation => "quality_validator".to_string(),
            MonitoringBottleneckType::PipelineThroughput |
            MonitoringBottleneckType::PipelineErrors => "document_processor".to_string(),
            _ => "system".to_string(),
        }
    }
    
    async fn should_trigger_immediate_intervention(&self, bottleneck: &MonitoringBottleneck) -> Result<bool> {
        // Trigger immediate intervention for critical bottlenecks
        if bottleneck.severity == MonitoringBottleneckSeverity::Critical {
            return Ok(true);
        }
        
        // Trigger if impact score is very high
        if bottleneck.impact_score > 0.8 {
            return Ok(true);
        }
        
        // Trigger for cascading failure patterns
        if matches!(
            bottleneck.bottleneck_type,
            MonitoringBottleneckType::IpcLatency | 
            MonitoringBottleneckType::MemoryPressure |
            MonitoringBottleneckType::PipelineErrors
        ) && bottleneck.impact_score > 0.6 {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    async fn trigger_immediate_intervention(&self, bottleneck: &MonitoringBottleneck) -> Result<()> {
        let component = self.determine_component_from_bottleneck(bottleneck);
        
        info!("Triggering immediate intervention for component: {}", component);
        
        // Determine intervention strategy
        match bottleneck.bottleneck_type {
            MonitoringBottleneckType::MemoryPressure |
            MonitoringBottleneckType::PythonMemoryLimit |
            MonitoringBottleneckType::RustMemoryLimit => {
                // Initiate isolation to prevent memory exhaustion cascade
                let isolation_id = self.isolation_manager.initiate_isolation(
                    &component,
                    IsolationReason::ResourceExhaustion,
                    Some("Memory pressure detected - isolating to prevent cascade".to_string())
                ).await?;
                
                info!("Initiated memory pressure isolation: {}", isolation_id);
            }
            
            MonitoringBottleneckType::IpcLatency |
            MonitoringBottleneckType::IpcBandwidth => {
                // Initiate recovery for IPC issues
                let recovery_id = self.recovery_manager.initiate_recovery(
                    &component,
                    FailureType::Timeout,
                    Some("IPC performance degradation - initiating recovery".to_string())
                ).await?;
                
                info!("Initiated IPC recovery: {}", recovery_id);
            }
            
            MonitoringBottleneckType::PipelineErrors |
            MonitoringBottleneckType::ModelInference => {
                // Initiate processing recovery
                let recovery_id = self.recovery_manager.initiate_recovery(
                    &component,
                    FailureType::ModelInferenceError,
                    Some("Processing errors detected - initiating recovery".to_string())
                ).await?;
                
                info!("Initiated processing recovery: {}", recovery_id);
            }
            
            _ => {
                // General recovery for other issues
                let recovery_id = self.recovery_manager.initiate_recovery(
                    &component,
                    FailureType::ProcessingError,
                    Some(format!("Critical bottleneck detected: {}", bottleneck.description))
                ).await?;
                
                info!("Initiated general recovery: {}", recovery_id);
            }
        }
        
        // Send alert about intervention
        let alert = FaultToleranceAlert {
            alert_type: AlertType::ImmediateIntervention,
            component: component.clone(),
            severity: AlertSeverity::High,
            message: format!("Immediate intervention triggered for {} due to {}", component, bottleneck.description),
            details: serde_json::json!({
                "bottleneck_id": bottleneck.id,
                "bottleneck_type": format!("{:?}", bottleneck.bottleneck_type),
                "impact_score": bottleneck.impact_score,
                "intervention_time": Utc::now()
            }),
            timestamp: Utc::now(),
        };
        
        self.alert_sender.send(alert).ok();
        
        Ok(())
    }
    
    async fn update_component_health_from_bottleneck(&self, bottleneck: &MonitoringBottleneck) -> Result<()> {
        let component = self.determine_component_from_bottleneck(bottleneck);
        
        if let Some(mut health) = self.component_health.get_mut(&component) {
            // Update health status based on bottleneck severity
            health.health_status = match bottleneck.severity {
                MonitoringBottleneckSeverity::Critical => HealthStatus::Critical,
                MonitoringBottleneckSeverity::High => HealthStatus::Degraded,
                MonitoringBottleneckSeverity::Medium => HealthStatus::Warning,
                MonitoringBottleneckSeverity::Low => HealthStatus::Healthy,
            };
            
            health.last_updated = Utc::now();
            health.failure_count += 1;
            
            // Update metrics based on bottleneck
            match bottleneck.bottleneck_type {
                MonitoringBottleneckType::CpuHighUtilization => {
                    health.metrics.cpu_utilization = Some(bottleneck.impact_score * 100.0);
                }
                MonitoringBottleneckType::MemoryPressure |
                MonitoringBottleneckType::PythonMemoryLimit |
                MonitoringBottleneckType::RustMemoryLimit => {
                    health.metrics.memory_utilization = Some(bottleneck.impact_score * 100.0);
                }
                MonitoringBottleneckType::IpcLatency => {
                    health.metrics.ipc_latency = Some(bottleneck.impact_score * 50.0); // Scale to milliseconds
                }
                MonitoringBottleneckType::PipelineThroughput => {
                    health.metrics.throughput = Some((1.0 - bottleneck.impact_score) * 30.0); // Docs/hour
                }
                MonitoringBottleneckType::PipelineErrors => {
                    health.metrics.error_rate = Some(bottleneck.impact_score * 10.0); // Percentage
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    async fn update_component_health_from_pattern(&self, pattern: &FaultPattern) -> Result<()> {
        if let Some(mut health) = self.component_health.get_mut(&pattern.component) {
            health.last_failure = Some(pattern.last_occurrence);
            
            // Determine health status from pattern severity
            health.health_status = match pattern.severity {
                FailureSeverity::Critical => HealthStatus::Critical,
                FailureSeverity::High => HealthStatus::Degraded,
                FailureSeverity::Medium => HealthStatus::Warning,
                FailureSeverity::Low => HealthStatus::Healthy,
            };
            
            health.last_updated = Utc::now();
        }
        
        Ok(())
    }
    
    async fn check_component_interventions(&self) -> Result<()> {
        for entry in self.component_health.iter() {
            let (component, health) = (entry.key(), entry.value());
            
            // Check if component needs intervention
            if health.health_status == HealthStatus::Critical && 
               health.failure_count > self.integration_config.intervention_threshold {
                
                warn!("Component {} requires intervention: {:?} failures", component, health.failure_count);
                
                // Trigger appropriate intervention
                match component.as_str() {
                    "rust_core" | "document_processor" => {
                        self.trigger_component_recovery(component, FailureType::ProcessingError).await?;
                    }
                    "python_ml" | "semantic_processor" | "mlx_accelerator" => {
                        self.trigger_component_recovery(component, FailureType::ModelInferenceError).await?;
                    }
                    "ipc_manager" => {
                        self.trigger_component_recovery(component, FailureType::Timeout).await?;
                    }
                    "quality_validator" => {
                        self.trigger_component_recovery(component, FailureType::ValidationError).await?;
                    }
                    _ => {
                        self.trigger_component_recovery(component, FailureType::ProcessingError).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn trigger_component_recovery(&self, component: &str, failure_type: FailureType) -> Result<()> {
        let recovery_id = self.recovery_manager.initiate_recovery(
            component,
            failure_type,
            Some(format!("Automated recovery triggered for component health intervention"))
        ).await?;
        
        info!("Triggered component recovery for {}: {}", component, recovery_id);
        
        // Reset failure count after triggering recovery
        if let Some(mut health) = self.component_health.get_mut(component) {
            health.failure_count = 0;
            health.last_updated = Utc::now();
        }
        
        Ok(())
    }
    
    async fn health_assessment_loop(&self) {
        let mut interval = time::interval(Duration::from_secs(30)); // Health assessment every 30 seconds
        
        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.perform_health_assessment().await {
                error!("Health assessment failed: {}", e);
            }
        }
    }
    
    async fn perform_health_assessment(&self) -> Result<()> {
        let mut overall_health = HealthStatus::Healthy;
        let mut critical_components = Vec::new();
        let mut degraded_components = Vec::new();
        
        for entry in self.component_health.iter() {
            let (component, health) = (entry.key(), entry.value());
            
            match health.health_status {
                HealthStatus::Critical => {
                    critical_components.push(component.clone());
                    overall_health = HealthStatus::Critical;
                }
                HealthStatus::Degraded => {
                    degraded_components.push(component.clone());
                    if overall_health == HealthStatus::Healthy {
                        overall_health = HealthStatus::Degraded;
                    }
                }
                HealthStatus::Warning => {
                    if overall_health == HealthStatus::Healthy {
                        overall_health = HealthStatus::Warning;
                    }
                }
                HealthStatus::Healthy => {}
            }
        }
        
        // Generate health summary alert if needed
        if overall_health != HealthStatus::Healthy {
            let alert = FaultToleranceAlert {
                alert_type: AlertType::HealthSummary,
                component: "system".to_string(),
                severity: match overall_health {
                    HealthStatus::Critical => AlertSeverity::Critical,
                    HealthStatus::Degraded => AlertSeverity::High,
                    HealthStatus::Warning => AlertSeverity::Medium,
                    HealthStatus::Healthy => AlertSeverity::Low,
                },
                message: format!(
                    "System health summary: Overall={:?}, Critical={}, Degraded={}", 
                    overall_health,
                    critical_components.len(),
                    degraded_components.len()
                ),
                details: serde_json::json!({
                    "overall_health": format!("{:?}", overall_health),
                    "critical_components": critical_components,
                    "degraded_components": degraded_components,
                    "total_components": self.component_health.len()
                }),
                timestamp: Utc::now(),
            };
            
            self.alert_sender.send(alert).ok();
        }
        
        Ok(())
    }
    
    async fn generate_health_summary_alerts(&self) -> Result<()> {
        // Generate periodic health summary
        let healthy_count = self.component_health.iter()
            .filter(|entry| entry.value().health_status == HealthStatus::Healthy)
            .count();
        
        let total_count = self.component_health.len();
        let health_percentage = (healthy_count as f64 / total_count as f64) * 100.0;
        
        if health_percentage < self.integration_config.health_alert_threshold {
            let alert = FaultToleranceAlert {
                alert_type: AlertType::HealthSummary,
                component: "system".to_string(),
                severity: if health_percentage < 50.0 {
                    AlertSeverity::Critical
                } else if health_percentage < 75.0 {
                    AlertSeverity::High
                } else {
                    AlertSeverity::Medium
                },
                message: format!("System health below threshold: {:.1}% healthy components", health_percentage),
                details: serde_json::json!({
                    "health_percentage": health_percentage,
                    "healthy_components": healthy_count,
                    "total_components": total_count,
                    "threshold": self.integration_config.health_alert_threshold
                }),
                timestamp: Utc::now(),
            };
            
            self.alert_sender.send(alert).ok();
        }
        
        Ok(())
    }
    
    pub fn get_component_health_summary(&self) -> HashMap<String, ComponentHealth> {
        self.component_health.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
    
    pub fn get_integration_stats(&self) -> IntegrationStats {
        let total_components = self.component_health.len();
        let healthy_components = self.component_health.iter()
            .filter(|entry| entry.value().health_status == HealthStatus::Healthy)
            .count();
        let critical_components = self.component_health.iter()
            .filter(|entry| entry.value().health_status == HealthStatus::Critical)
            .count();
        
        IntegrationStats {
            total_components,
            healthy_components,
            critical_components,
            health_percentage: (healthy_components as f64 / total_components as f64) * 100.0,
            last_assessment: Utc::now(),
        }
    }
    
    pub fn clone_for_task(&self) -> Self {
        Self {
            fault_detector: self.fault_detector.clone(),
            recovery_manager: self.recovery_manager.clone(),
            isolation_manager: self.isolation_manager.clone(),
            bottleneck_receiver: self.bottleneck_receiver.resubscribe(),
            alert_sender: self.alert_sender.clone(),
            running: AtomicBool::new(self.running.load(Ordering::Acquire)),
            component_health: self.component_health.clone(),
            integration_config: self.integration_config.clone(),
        }
    }
    
    pub async fn shutdown(&mut self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        info!("Fault tolerance monitoring bridge shutdown complete");
        Ok(())
    }
}

// Shared data structures and enums

#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub monitoring_interval_ms: u64,
    pub intervention_threshold: u32,
    pub health_alert_threshold: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 5000, // 5 seconds
            intervention_threshold: 3,    // 3 failures before intervention
            health_alert_threshold: 80.0, // Alert if less than 80% components healthy
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Degraded,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub component_name: String,
    pub health_status: HealthStatus,
    pub last_failure: Option<DateTime<Utc>>,
    pub failure_count: u32,
    pub circuit_breaker_state: CircuitState,
    pub last_updated: DateTime<Utc>,
    pub metrics: ComponentMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct ComponentMetrics {
    pub cpu_utilization: Option<f64>,
    pub memory_utilization: Option<f64>,
    pub ipc_latency: Option<f64>,
    pub throughput: Option<f64>,
    pub error_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStats {
    pub total_components: usize,
    pub healthy_components: usize,
    pub critical_components: usize,
    pub health_percentage: f64,
    pub last_assessment: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceAlert {
    pub alert_type: AlertType,
    pub component: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub details: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertType {
    ImmediateIntervention,
    HealthSummary,
    ComponentRecovery,
    SystemStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Placeholder structures for integration with monitoring system
// These would be replaced with actual imports from the monitoring module

#[derive(Debug, Clone)]
pub struct MonitoringBottleneck {
    pub id: String,
    pub bottleneck_type: MonitoringBottleneckType,
    pub severity: MonitoringBottleneckSeverity,
    pub detected_at: DateTime<Utc>,
    pub description: String,
    pub impact_score: f64,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringBottleneckType {
    CpuHighUtilization,
    CpuThermal,
    MemoryPressure,
    RustMemoryLimit,
    PythonMemoryLimit,
    DiskSpace,
    DiskIoHigh,
    IpcLatency,
    IpcBandwidth,
    PipelineThroughput,
    PipelineErrors,
    ModelInference,
    QualityValidation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringBottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Placeholder for fault pattern (would be imported from fault_detector)
#[derive(Debug, Clone)]
pub struct FaultPattern {
    pub pattern_type: String,
    pub component: String,
    pub severity: FailureSeverity,
    pub last_occurrence: DateTime<Utc>,
}