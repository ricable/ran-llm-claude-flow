//! Configuration management for the monitoring system

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Comprehensive monitoring system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection interval in milliseconds
    pub collection_interval_ms: u64,
    
    /// Bottleneck analysis interval in milliseconds  
    pub analysis_interval_ms: u64,
    
    /// Maximum number of metric snapshots to keep in memory
    pub max_history_size: usize,
    
    /// Dashboard configuration
    pub dashboard: DashboardConfig,
    
    /// Alert configuration
    pub alerts: AlertConfig,
    
    /// Performance targets
    pub targets: PerformanceTargets,
    
    /// Optimization settings
    pub optimization: OptimizationConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 100, // 100ms for high-frequency monitoring
            analysis_interval_ms: 1000,  // 1s for bottleneck analysis
            max_history_size: 10000,
            dashboard: DashboardConfig::default(),
            alerts: AlertConfig::default(),
            targets: PerformanceTargets::default(),
            optimization: OptimizationConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl MonitoringConfig {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from environment or default
    pub fn load() -> Self {
        if let Ok(config_path) = std::env::var("MONITORING_CONFIG_PATH") {
            Self::load_from_file(config_path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load config from {}: {}. Using defaults.", 
                              std::env::var("MONITORING_CONFIG_PATH").unwrap_or_default(), e);
                Self::default()
            })
        } else {
            Self::default()
        }
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<()> {
        if self.collection_interval_ms == 0 {
            return Err(anyhow::anyhow!("Collection interval must be > 0"));
        }
        
        if self.analysis_interval_ms == 0 {
            return Err(anyhow::anyhow!("Analysis interval must be > 0"));
        }
        
        if self.max_history_size == 0 {
            return Err(anyhow::anyhow!("Max history size must be > 0"));
        }

        self.targets.validate()?;
        self.optimization.validate()?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable real-time dashboard
    pub enabled: bool,
    
    /// Dashboard server port
    pub port: u16,
    
    /// Dashboard bind address
    pub bind_address: String,
    
    /// Enable WebSocket for real-time updates
    pub websocket_enabled: bool,
    
    /// Update interval for dashboard in seconds
    pub update_interval_seconds: u64,
    
    /// Enable historical charts
    pub historical_charts: bool,
    
    /// Chart time ranges in minutes
    pub chart_time_ranges: Vec<u32>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8080,
            bind_address: "0.0.0.0".to_string(),
            websocket_enabled: true,
            update_interval_seconds: 1,
            historical_charts: true,
            chart_time_ranges: vec![5, 15, 60, 240], // 5min, 15min, 1hr, 4hr
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting system
    pub enabled: bool,
    
    /// Alert severity thresholds
    pub severity_thresholds: AlertThresholds,
    
    /// Alert channels configuration
    pub channels: Vec<AlertChannel>,
    
    /// Cooldown period between similar alerts (seconds)
    pub cooldown_seconds: u64,
    
    /// Maximum alerts per minute to prevent spam
    pub max_alerts_per_minute: u32,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            severity_thresholds: AlertThresholds::default(),
            channels: vec![AlertChannel::default()],
            cooldown_seconds: 300, // 5 minutes
            max_alerts_per_minute: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU utilization threshold (percentage)
    pub cpu_utilization: f32,
    
    /// Memory utilization threshold (percentage) 
    pub memory_utilization: f64,
    
    /// IPC latency threshold (milliseconds)
    pub ipc_latency_ms: f64,
    
    /// Document processing rate threshold (docs/hour)
    pub document_rate: f64,
    
    /// Error rate threshold (percentage)
    pub error_rate: f64,
    
    /// Disk space threshold (percentage)
    pub disk_usage: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_utilization: 85.0,
            memory_utilization: 90.0,
            ipc_latency_ms: 10.0,
            document_rate: 20.0,
            error_rate: 1.0,
            disk_usage: 90.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertChannel {
    /// Channel type (log, webhook, email, etc.)
    pub channel_type: String,
    
    /// Channel configuration
    pub config: serde_json::Value,
    
    /// Minimum severity level for this channel
    pub min_severity: String,
}

impl Default for AlertChannel {
    fn default() -> Self {
        Self {
            channel_type: "log".to_string(),
            config: serde_json::json!({
                "level": "warn"
            }),
            min_severity: "medium".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target document processing rate (docs/hour)
    pub document_processing_rate: f64,
    
    /// Target Rust memory usage (GB)
    pub rust_memory_limit: f64,
    
    /// Target Python memory usage (GB)
    pub python_memory_limit: f64,
    
    /// Target shared memory usage (GB)
    pub shared_memory_limit: f64,
    
    /// Target IPC latency P99 (milliseconds)
    pub ipc_latency_p99: f64,
    
    /// Target error rate (percentage)
    pub max_error_rate: f64,
    
    /// Target CPU utilization (percentage)
    pub max_cpu_utilization: f32,
    
    /// Target model inference time (milliseconds)
    pub model_inference_time: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            document_processing_rate: 25.0, // 20-30 target range
            rust_memory_limit: 60.0,
            python_memory_limit: 45.0,
            shared_memory_limit: 15.0,
            ipc_latency_p99: 10.0,
            max_error_rate: 1.0,
            max_cpu_utilization: 85.0,
            model_inference_time: 500.0,
        }
    }
}

impl PerformanceTargets {
    pub fn validate(&self) -> Result<()> {
        if self.document_processing_rate <= 0.0 {
            return Err(anyhow::anyhow!("Document processing rate must be > 0"));
        }
        
        if self.rust_memory_limit <= 0.0 || self.python_memory_limit <= 0.0 {
            return Err(anyhow::anyhow!("Memory limits must be > 0"));
        }
        
        if self.max_error_rate < 0.0 || self.max_error_rate > 100.0 {
            return Err(anyhow::anyhow!("Error rate must be 0-100%"));
        }
        
        if self.max_cpu_utilization <= 0.0 || self.max_cpu_utilization > 100.0 {
            return Err(anyhow::anyhow!("CPU utilization must be 0-100%"));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable automatic optimization
    pub enabled: bool,
    
    /// Minimum improvement threshold to apply optimization
    pub min_improvement_threshold: f64,
    
    /// Maximum implementation cost acceptable
    pub max_implementation_cost: f64,
    
    /// Enable predictive optimizations
    pub predictive_enabled: bool,
    
    /// Learning rate for ML-based optimizations
    pub learning_rate: f64,
    
    /// Optimization strategies to enable
    pub enabled_strategies: Vec<String>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_improvement_threshold: 0.1, // 10% minimum improvement
            max_implementation_cost: 0.3,   // 30% maximum cost
            predictive_enabled: true,
            learning_rate: 0.01,
            enabled_strategies: vec![
                "concurrency_adjustment".to_string(),
                "memory_rebalancing".to_string(),
                "ipc_optimization".to_string(),
                "pipeline_tuning".to_string(),
                "model_selection".to_string(),
            ],
        }
    }
}

impl OptimizationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.min_improvement_threshold < 0.0 || self.min_improvement_threshold > 1.0 {
            return Err(anyhow::anyhow!("Min improvement threshold must be 0.0-1.0"));
        }
        
        if self.max_implementation_cost < 0.0 || self.max_implementation_cost > 1.0 {
            return Err(anyhow::anyhow!("Max implementation cost must be 0.0-1.0"));
        }
        
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(anyhow::anyhow!("Learning rate must be 0.0-1.0"));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    
    /// Enable structured logging (JSON format)
    pub structured: bool,
    
    /// Log file path (optional, logs to stdout if not specified)
    pub file_path: Option<PathBuf>,
    
    /// Maximum log file size in MB
    pub max_file_size_mb: u64,
    
    /// Number of log files to rotate
    pub max_files: u32,
    
    /// Enable performance metrics in logs
    pub include_metrics: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: true,
            file_path: None,
            max_file_size_mb: 100,
            max_files: 5,
            include_metrics: false,
        }
    }
}