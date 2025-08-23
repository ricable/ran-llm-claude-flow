//! RAN-LLM Rust Pipeline Library
//! 
//! High-performance Rust I/O pipeline optimized for M3 Max architecture,
//! featuring real-time performance monitoring, IPC communication, and MCP integration.

pub mod core;
pub mod io;
pub mod ipc;
pub mod mcp;
pub mod optimization;
pub mod monitoring;

// Re-export main modules
pub use optimization::m3_max;
pub use ipc::{message_queue, shared_memory, process_manager};
pub use monitoring::{
    PerformanceDashboard, 
    M3MaxMetrics, 
    SystemMetrics, 
    ThroughputMetrics,
    initialize_monitoring,
    start_dashboard,
    get_current_m3_metrics,
    get_current_system_metrics,
    get_current_throughput_metrics,
    is_system_healthy,
    generate_performance_report,
    log_performance_metrics
};

use std::error::Error;
use std::fmt;

/// Main library error type
#[derive(Debug)]
pub enum PipelineError {
    IoError(std::io::Error),
    IpcError(String),
    MonitoringError(String),
    ConfigError(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::IoError(e) => write!(f, "IO error: {}", e),
            PipelineError::IpcError(e) => write!(f, "IPC error: {}", e),
            PipelineError::MonitoringError(e) => write!(f, "Monitoring error: {}", e),
            PipelineError::ConfigError(e) => write!(f, "Configuration error: {}", e),
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PipelineError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(error: std::io::Error) -> Self {
        PipelineError::IoError(error)
    }
}

/// Result type for pipeline operations
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Initialize the entire pipeline system
pub async fn initialize() -> Result<()> {
    println!("🚀 Initializing RAN-LLM Rust Pipeline...");
    
    // Initialize performance monitoring
    monitoring::initialize_monitoring().await
        .map_err(|e| PipelineError::MonitoringError(e.to_string()))?;
    
    println!("✅ Pipeline initialization complete");
    Ok(())
}

/// Start the performance dashboard server
pub async fn start_performance_dashboard(port: u16) -> Result<()> {
    monitoring::start_dashboard(port).await
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