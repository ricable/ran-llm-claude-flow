//! RAN-LLM Rust Pipeline Library - Minimal Version for Testing
//! 
//! High-performance Rust I/O pipeline optimized for M3 Max architecture,
//! featuring real-time performance monitoring, IPC communication, and MCP integration.

// Temporarily comment out problematic modules for testing
// pub mod core;
// pub mod io;
// pub mod ipc;
// pub mod mcp;
// pub mod optimization;
// pub mod monitoring;
// pub mod ml;

use std::error::Error;
use std::fmt;

/// Main library error type
#[derive(Debug)]
pub enum PipelineError {
    Io(String),
    IoError(std::io::Error),
    Ipc(String),
    IpcError(String),
    Mcp(String),
    McpError(String),
    MonitoringError(String),
    ConfigError(String),
    Optimization(String),
    Processing(String),
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

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub m3_max: M3MaxConfig,
}

/// M3 Max configuration
#[derive(Debug, Clone)]
pub struct M3MaxConfig {
    pub memory_pools: MemoryPoolConfig,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub processing: u32,
    pub ipc: u32,
    pub cache: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            m3_max: M3MaxConfig::default(),
        }
    }
}

impl Default for M3MaxConfig {
    fn default() -> Self {
        Self {
            memory_pools: MemoryPoolConfig::default(),
        }
    }
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