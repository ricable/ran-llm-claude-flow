// Dynamic Scaling Module Integration
// Phase 2 MCP Advanced Features - Core Integration Module
// Coordinates all scaling components and provides unified interface

pub mod dynamic_scaler;
pub mod workload_analyzer;
pub mod resource_manager;

use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, sleep};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

// Re-export main types for easier access
pub use dynamic_scaler::{DynamicScaler, ScalingMetrics, ScalingDecision, ScalingConfig};
pub use workload_analyzer::{WorkloadAnalyzer, WorkloadPattern, WorkloadMetrics, AnalysisConfig};
pub use resource_manager::{ResourceManager, ResourceAllocation, ResourceUtilization, SystemConstraints};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingSystemConfig {
    pub enable_auto_scaling: bool,
    pub enable_workload_analysis: bool,
    pub enable_resource_optimization: bool,
    pub monitoring_interval_seconds: u64,
    pub scaling_decision_interval_seconds: u64,
    pub emergency_threshold_memory_percent: f64,
    pub emergency_threshold_cpu_percent: f64,
    pub max_agents: usize,
    pub min_agents: usize,
    pub target_throughput_docs_per_hour: f64,
}

impl Default for ScalingSystemConfig {
    fn default() -> Self {
        Self {
            enable_auto_scaling: true,
            enable_workload_analysis: true,
            enable_resource_optimization: true,
            monitoring_interval_seconds: 10,
            scaling_decision_interval_seconds: 30,
            emergency_threshold_memory_percent: 95.0,
            emergency_threshold_cpu_percent: 98.0,
            max_agents: 12,
            min_agents: 2,
            target_throughput_docs_per_hour: 25.0,
        }
    }
}

/// Unified scaling system that coordinates all scaling components
pub struct ScalingSystem {
    scaler: Arc<RwLock<DynamicScaler>>,
    workload_analyzer: Arc<RwLock<WorkloadAnalyzer>>,
    resource_manager: Arc<ResourceManager>,
    config: ScalingSystemConfig,
    is_running: Arc<RwLock<bool>>,
    mcp_integration: Option<Arc<McpScalingIntegration>>,
}

impl ScalingSystem {
    pub fn new(config: ScalingSystemConfig) -> Self {
        let scaling_config = ScalingConfig {
            max_agents: config.max_agents,
            min_agents: config.min_agents,
            target_cpu_utilization: 85.0,
            target_memory_utilization: 90.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 50.0,
            cooldown_duration: Duration::from_secs(30),
            prediction_window_size: 20,
            memory_constraint_gb: 128.0,
            min_throughput_docs_per_hour: config.target_throughput_docs_per_hour,
            max_ipc_latency_ms: 5.0,
        };

        let analysis_config = AnalysisConfig {
            history_window_minutes: 60,
            pattern_detection_sensitivity: 0.7,
            min_confidence_threshold: 0.6,
            burst_detection_threshold: 2.0,
            steady_state_variance_threshold: 0.2,
            memory_optimization_threshold: 0.85,
        };

        Self {
            scaler: Arc::new(RwLock::new(DynamicScaler::new(scaling_config))),
            workload_analyzer: Arc::new(RwLock::new(WorkloadAnalyzer::new(analysis_config))),
            resource_manager: Arc::new(ResourceManager::new()),
            config,
            is_running: Arc::new(RwLock::new(false)),
            mcp_integration: None,
        }
    }

    /// Initialize MCP integration for external monitoring
    pub fn with_mcp_integration(mut self, integration: Arc<McpScalingIntegration>) -> Self {
        self.mcp_integration = Some(integration);
        self
    }

    /// Start the scaling system
    pub async fn start(&self) -> Result<(), ScalingError> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(ScalingError::AlreadyRunning);
        }
        *running = true;
        drop(running);

        // Start monitoring task
        let monitoring_task = self.start_monitoring_task();
        
        // Start scaling decision task
        let scaling_task = self.start_scaling_task();

        // Start MCP integration if available
        if let Some(ref mcp_integration) = self.mcp_integration {
            tokio::spawn(mcp_integration.clone().start_mcp_reporting());
        }

        // Run both tasks concurrently
        tokio::select! {
            result = monitoring_task => {
                eprintln!("Monitoring task ended: {:?}", result);
            }
            result = scaling_task => {
                eprintln!("Scaling task ended: {:?}", result);
            }
        }

        Ok(())
    }

    /// Stop the scaling system
    pub async fn stop(&self) {
        *self.is_running.write().await = false;
    }

    async fn start_monitoring_task(&self) -> Result<(), ScalingError> {
        let mut interval = interval(Duration::from_secs(self.config.monitoring_interval_seconds));
        let resource_manager = self.resource_manager.clone();
        let is_running = self.is_running.clone();

        loop {
            interval.tick().await;
            
            if !*is_running.read().await {
                break;
            }

            // Collect system metrics
            match self.collect_system_metrics().await {
                Ok(metrics) => {
                    // Update resource utilization
                    let resource_utilizations = self.convert_to_resource_utilizations(&metrics);
                    resource_manager.update_utilization(resource_utilizations).await;

                    // Check for emergency conditions
                    if metrics.memory_utilization > self.config.emergency_threshold_memory_percent ||
                       metrics.cpu_utilization > self.config.emergency_threshold_cpu_percent {
                        self.handle_emergency(&metrics).await;
                    }
                }
                Err(e) => {
                    eprintln!("Failed to collect system metrics: {:?}", e);
                }
            }
        }

        Ok(())
    }

    async fn start_scaling_task(&self) -> Result<(), ScalingError> {
        let mut interval = interval(Duration::from_secs(self.config.scaling_decision_interval_seconds));
        let is_running = self.is_running.clone();
        let scaler = self.scaler.clone();
        let workload_analyzer = self.workload_analyzer.clone();

        loop {
            interval.tick().await;
            
            if !*is_running.read().await {
                break;
            }

            // Collect current metrics
            match self.collect_system_metrics().await {
                Ok(system_metrics) => {
                    let scaling_metrics = self.convert_to_scaling_metrics(&system_metrics);
                    let workload_metrics = self.convert_to_workload_metrics(&system_metrics);

                    // Analyze workload pattern
                    if self.config.enable_workload_analysis {
                        let mut analyzer = workload_analyzer.write().await;
                        let pattern = analyzer.analyze_workload(workload_metrics).await;
                        
                        // Log pattern for monitoring
                        eprintln!("Detected workload pattern: {:?} (confidence: {:.2})", 
                                 pattern.pattern_type, pattern.confidence_score);
                    }

                    // Make scaling decisions
                    if self.config.enable_auto_scaling {
                        let mut scaler_guard = scaler.write().await;
                        if let Some(decision) = scaler_guard.analyze_and_scale(scaling_metrics).await {
                            eprintln!("Scaling decision: {:?} -> {} agents (confidence: {:.2})", 
                                     decision.action, decision.target_agents, decision.confidence);

                            // Apply scaling decision (would integrate with actual process management)
                            self.apply_scaling_decision(&decision).await;
                            
                            // Notify MCP server
                            if let Some(ref mcp_integration) = self.mcp_integration {
                                mcp_integration.notify_scaling_decision(&decision).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to collect metrics for scaling: {:?}", e);
                }
            }
        }

        Ok(())
    }

    async fn collect_system_metrics(&self) -> Result<SystemMetrics, ScalingError> {
        // This would integrate with the actual monitoring system
        // For now, simulate metrics collection
        
        let memory_info = self.get_memory_info().await?;
        let cpu_info = self.get_cpu_info().await?;
        let ipc_info = self.get_ipc_info().await?;
        let throughput_info = self.get_throughput_info().await?;

        Ok(SystemMetrics {
            memory_total_gb: memory_info.total_gb,
            memory_used_gb: memory_info.used_gb,
            memory_utilization: (memory_info.used_gb / memory_info.total_gb) * 100.0,
            cpu_utilization: cpu_info.utilization_percent,
            cpu_cores: cpu_info.cores,
            queue_depth: throughput_info.queue_depth,
            throughput_docs_per_hour: throughput_info.docs_per_hour,
            ipc_latency_ms: ipc_info.latency_ms,
            active_agents: throughput_info.active_agents,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    async fn get_memory_info(&self) -> Result<MemoryInfo, ScalingError> {
        // Integrate with system monitoring
        Ok(MemoryInfo {
            total_gb: 128.0,
            used_gb: 54.0, // From baseline metrics
        })
    }

    async fn get_cpu_info(&self) -> Result<CpuInfo, ScalingError> {
        Ok(CpuInfo {
            cores: 16,
            utilization_percent: 25.0, // From baseline metrics
        })
    }

    async fn get_ipc_info(&self) -> Result<IpcInfo, ScalingError> {
        Ok(IpcInfo {
            latency_ms: 2.5,
            throughput_mbps: 1200.0,
        })
    }

    async fn get_throughput_info(&self) -> Result<ThroughputInfo, ScalingError> {
        Ok(ThroughputInfo {
            docs_per_hour: 25.5,
            queue_depth: 3,
            active_agents: 2,
        })
    }

    fn convert_to_scaling_metrics(&self, system_metrics: &SystemMetrics) -> ScalingMetrics {
        ScalingMetrics {
            cpu_utilization: system_metrics.cpu_utilization,
            memory_utilization: system_metrics.memory_utilization,
            queue_depth: system_metrics.queue_depth,
            throughput_docs_per_hour: system_metrics.throughput_docs_per_hour,
            ipc_latency_ms: system_metrics.ipc_latency_ms,
            active_agents: system_metrics.active_agents,
            timestamp: system_metrics.timestamp,
            rust_memory_gb: 60.0, // Would be dynamically determined
            python_memory_gb: 45.0,
            shared_memory_gb: 15.0,
        }
    }

    fn convert_to_workload_metrics(&self, system_metrics: &SystemMetrics) -> WorkloadMetrics {
        use workload_analyzer::{MemoryBreakdown};
        use std::collections::HashMap;

        WorkloadMetrics {
            processing_queue_depth: system_metrics.queue_depth,
            average_processing_time_ms: 200.0, // Would be measured
            documents_per_minute: system_metrics.throughput_docs_per_hour / 60.0,
            cpu_utilization_per_core: vec![system_metrics.cpu_utilization; system_metrics.cpu_cores],
            memory_utilization_breakdown: MemoryBreakdown {
                rust_processes_gb: 60.0,
                python_processes_gb: 45.0,
                shared_memory_gb: 15.0,
                cached_data_gb: 5.0,
                free_memory_gb: system_metrics.memory_total_gb - system_metrics.memory_used_gb,
                memory_fragmentation_percent: 5.0,
            },
            ipc_throughput_mbps: 1200.0,
            error_rate_percent: 0.5,
            agent_utilization: HashMap::new(),
            timestamp: system_metrics.timestamp,
            peak_hour_indicator: false,
        }
    }

    fn convert_to_resource_utilizations(&self, system_metrics: &SystemMetrics) -> Vec<ResourceUtilization> {
        vec![
            ResourceUtilization {
                process_id: "rust_core_primary".to_string(),
                memory_used_gb: 58.0,
                memory_utilization_percent: 96.7, // 58/60
                cpu_utilization_percent: system_metrics.cpu_utilization,
                ipc_throughput_mbps: system_metrics.ipc_latency_ms,
                timestamp: system_metrics.timestamp,
                is_healthy: true,
                performance_score: 0.85,
            },
            ResourceUtilization {
                process_id: "python_ml_primary".to_string(),
                memory_used_gb: 42.0,
                memory_utilization_percent: 93.3, // 42/45
                cpu_utilization_percent: system_metrics.cpu_utilization * 0.8,
                ipc_throughput_mbps: system_metrics.ipc_latency_ms,
                timestamp: system_metrics.timestamp,
                is_healthy: true,
                performance_score: 0.82,
            },
        ]
    }

    async fn handle_emergency(&self, metrics: &SystemMetrics) {
        eprintln!("EMERGENCY: System under severe stress - Memory: {:.1}%, CPU: {:.1}%", 
                 metrics.memory_utilization, metrics.cpu_utilization);

        // Emergency scaling down
        let mut scaler = self.scaler.write().await;
        let scaling_metrics = self.convert_to_scaling_metrics(metrics);
        
        if let Some(emergency_decision) = scaler.emergency_scale(&scaling_metrics).await {
            eprintln!("Emergency scaling decision: {:?}", emergency_decision);
            self.apply_scaling_decision(&emergency_decision).await;
        }

        // Emergency resource reclaim
        if let Ok(reclaimed) = self.resource_manager.emergency_resource_reclaim().await {
            eprintln!("Emergency resource reclaim: {:.1}GB freed", reclaimed);
        }
    }

    async fn apply_scaling_decision(&self, decision: &ScalingDecision) {
        // This would integrate with the actual process management system
        // For now, just log the decision
        eprintln!("Applying scaling decision: {} -> {} agents", 
                 decision.action.action_name(), decision.target_agents);
        
        // Simulate application delay
        sleep(Duration::from_millis(100)).await;
    }

    /// Get current scaling statistics
    pub async fn get_scaling_statistics(&self) -> ScalingStatistics {
        let scaler = self.scaler.read().await;
        let resource_stats = self.resource_manager.get_resource_statistics().await;
        
        ScalingStatistics {
            current_agents: scaler.get_current_agents(),
            scaling_effectiveness: scaler.calculate_scaling_effectiveness(),
            total_scaling_decisions: scaler.get_scaling_history().len(),
            memory_utilization_percent: resource_stats.memory_utilization_percent,
            cpu_utilization_percent: resource_stats.average_memory_efficiency, // Placeholder
            active_processes: resource_stats.active_processes,
            uptime_seconds: 0, // Would track actual uptime
        }
    }
}

// MCP Integration for external monitoring and control
pub struct McpScalingIntegration {
    scaling_system: Arc<RwLock<ScalingSystem>>,
}

impl McpScalingIntegration {
    pub fn new(scaling_system: Arc<RwLock<ScalingSystem>>) -> Self {
        Self { scaling_system }
    }

    pub async fn start_mcp_reporting(self: Arc<Self>) {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Report scaling statistics to MCP server
            let system = self.scaling_system.read().await;
            let stats = system.get_scaling_statistics().await;
            
            // This would send to MCP server
            eprintln!("MCP Report: {} agents, {:.1}% memory, {} decisions", 
                     stats.current_agents, stats.memory_utilization_percent, stats.total_scaling_decisions);
        }
    }

    pub async fn notify_scaling_decision(&self, decision: &ScalingDecision) {
        // Notify MCP server about scaling decisions
        eprintln!("MCP Notification: Scaling decision - {}", decision.reasoning);
    }
}

// Helper types
#[derive(Debug, Clone)]
struct SystemMetrics {
    memory_total_gb: f64,
    memory_used_gb: f64,
    memory_utilization: f64,
    cpu_utilization: f64,
    cpu_cores: usize,
    queue_depth: usize,
    throughput_docs_per_hour: f64,
    ipc_latency_ms: f64,
    active_agents: usize,
    timestamp: u64,
}

#[derive(Debug)]
struct MemoryInfo {
    total_gb: f64,
    used_gb: f64,
}

#[derive(Debug)]
struct CpuInfo {
    cores: usize,
    utilization_percent: f64,
}

#[derive(Debug)]
struct IpcInfo {
    latency_ms: f64,
    throughput_mbps: f64,
}

#[derive(Debug)]
struct ThroughputInfo {
    docs_per_hour: f64,
    queue_depth: usize,
    active_agents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingStatistics {
    pub current_agents: usize,
    pub scaling_effectiveness: f64,
    pub total_scaling_decisions: usize,
    pub memory_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub active_processes: usize,
    pub uptime_seconds: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ScalingError {
    #[error("Scaling system is already running")]
    AlreadyRunning,
    
    #[error("Failed to collect system metrics: {0}")]
    MetricsCollection(String),
    
    #[error("Resource allocation failed: {0}")]
    ResourceAllocation(String),
    
    #[error("Scaling decision failed: {0}")]
    ScalingDecision(String),
}

// Extension trait for action names
impl crate::scaling::dynamic_scaler::ScalingAction {
    pub fn action_name(&self) -> &'static str {
        match self {
            Self::ScaleUp => "scale_up",
            Self::ScaleDown => "scale_down", 
            Self::Maintain => "maintain",
            Self::Rebalance => "rebalance",
            Self::OptimizeMemory => "optimize_memory",
        }
    }
}