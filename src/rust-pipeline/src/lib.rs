//! RAN-LLM Rust Pipeline Library
//!
//! High-performance Rust I/O pipeline optimized for M3 Max architecture,
//! featuring real-time performance monitoring, IPC communication, and MCP integration.

pub mod core;
pub mod io;
pub mod ipc;
pub mod mcp;
pub mod monitoring;
pub mod optimization;

// Re-export main modules
pub use ipc::{message_queue, process_manager, shared_memory};
pub use monitoring::{
    generate_performance_report, get_current_m3_metrics, get_current_system_metrics,
    get_current_throughput_metrics, initialize_monitoring, is_system_healthy,
    log_performance_metrics, start_dashboard, M3MaxMetrics, PerformanceDashboard, SystemMetrics,
    ThroughputMetrics,
};
pub use optimization::m3_max;

// Add missing ML module
pub mod ml;

use std::error::Error;
use std::fmt;

/// Main library error type
#[derive(Debug, Clone)]
pub enum PipelineError {
    Io(String),
    IoError(String),
    Ipc(String),
    IpcError(String),
    Mcp(String),
    McpError(String),
    MonitoringError(String),
    ConfigError(String),
    Optimization(String),
    Processing(String),
    Initialization(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "IO error: {}", e),
            PipelineError::IoError(e) => write!(f, "IO error: {}", e),
            PipelineError::Ipc(e) => write!(f, "IPC error: {}", e),
            PipelineError::IpcError(e) => write!(f, "IPC error: {}", e),
            PipelineError::Mcp(e) => write!(f, "MCP error: {}", e),
            PipelineError::McpError(e) => write!(f, "MCP error: {}", e),
            PipelineError::MonitoringError(e) => write!(f, "Monitoring error: {}", e),
            PipelineError::ConfigError(e) => write!(f, "Configuration error: {}", e),
            PipelineError::Optimization(e) => write!(f, "Optimization error: {}", e),
            PipelineError::Processing(e) => write!(f, "Processing error: {}", e),
            PipelineError::Initialization(e) => write!(f, "Initialization error: {}", e),
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(error: std::io::Error) -> Self {
        PipelineError::IoError(error.to_string())
    }
}

/// Result type for pipeline operations
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Pipeline configuration
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    pub m3_max: M3MaxConfig,
}

/// M3 Max configuration
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct M3MaxConfig {
    pub memory_pools: MemoryPoolConfig,
}

/// Memory pool configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryPoolConfig {
    pub processing: u32,
    pub ipc: u32,
    pub cache: u32,
}


impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            processing: 32, // 32GB for processing
            ipc: 8,         // 8GB for IPC
            cache: 4,       // 4GB for cache
        }
    }
}

/// Initialize the entire pipeline system
pub async fn initialize() -> Result<()> {
    println!("🚀 Initializing RAN-LLM Rust Pipeline...");

    // Initialize performance monitoring
    monitoring::initialize_monitoring()
        .await
        .map_err(|e| PipelineError::MonitoringError(e.to_string()))?;

    println!("✅ Pipeline initialization complete");
    Ok(())
}

/// Start the performance dashboard server
pub async fn start_performance_dashboard(port: u16) -> Result<()> {
    monitoring::start_dashboard(port)
        .await
        .map_err(|e| PipelineError::MonitoringError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_initialization() {
        let result = initialize().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let pipeline_error: PipelineError = io_error.into();

        match pipeline_error {
            PipelineError::IoError(_) => assert!(true),
            _ => assert!(false, "Expected IoError"),
        }
    }
}
