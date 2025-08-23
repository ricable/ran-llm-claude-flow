/*!
# Rust-Python Hybrid Pipeline

Ultra-high-performance document processing pipeline optimized for M3 Max with 128GB unified memory.
Implements hybrid architecture with Rust I/O processing and Python ML inference.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Rust-Python Hybrid Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│  Rust Core (I/O + Processing)    │    Python ML (Inference)    │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────┐ │
│  │ • Document I/O              │◄─┼─►│ • Qwen3 Model Pool     │ │
│  │ • Parallel Processing       │  │  │ • MLX Acceleration     │ │
│  │ • Memory Management         │  │  │ • LangExtract          │ │
│  │ • Quality Validation        │  │  │ • Quality Scoring      │ │
│  └─────────────────────────────┘  │  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      MCP Protocol Layer                         │
│  • Server: Pipeline coordination and status                     │
│  • Client: Inter-process communication                          │
│  • Host: Orchestration and load balancing                       │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Targets
- **Processing Throughput**: 20-30 documents/hour (4-5x improvement)
- **Memory Utilization**: 95% efficiency of 128GB unified memory
- **Model Switching**: <5 second latency
- **Quality Consistency**: >0.742 quality score target

## Features
- M3 Max optimized with Metal Performance Shaders
- IPC communication between Rust and Python processes
- Dynamic Qwen3 model selection (1.7B, 7B, 30B variants)
- Circuit breaker patterns for reliability
- Real-time performance monitoring
*/

pub mod core;
pub mod io;
pub mod python;
pub mod mcp;
pub mod ipc;
pub mod optimization;
pub mod orchestration;
pub mod integration;

use std::error::Error;
use std::fmt;

/// Pipeline result type
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Pipeline error types
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    
    #[error("IPC communication error: {0}")]
    Ipc(String),
    
    #[error("Python integration error: {0}")]
    Python(String),
    
    #[error("MCP protocol error: {0}")]
    Mcp(String),
    
    #[error("Performance optimization error: {0}")]
    Optimization(String),
    
    #[error("Quality validation error: {0}")]
    Quality(String),
}

/// Pipeline configuration
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PipelineConfig {
    /// M3 Max specific settings
    pub m3_max: M3MaxConfig,
    
    /// Python process configuration
    pub python: PythonConfig,
    
    /// MCP protocol settings
    pub mcp: McpConfig,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
    
    /// Quality control settings
    pub quality: QualityConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct M3MaxConfig {
    /// Total unified memory in GB (default: 128)
    pub total_memory_gb: u64,
    
    /// Memory allocation strategy
    pub memory_pools: MemoryPools,
    
    /// CPU core utilization (8P + 4E cores)
    pub cpu_cores: CpuCoreConfig,
    
    /// GPU utilization (40-core GPU)
    pub gpu_cores: u32,
    
    /// Neural Engine utilization (16-core, 15.8 TOPS)
    pub neural_engine: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MemoryPools {
    /// Memory for document processing (GB)
    pub processing: u64,
    
    /// Memory for IPC communication (GB) 
    pub ipc: u64,
    
    /// Memory for caching (GB)
    pub cache: u64,
    
    /// Reserved system memory (GB)
    pub system: u64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CpuCoreConfig {
    /// Performance cores for heavy processing
    pub performance_cores: u8,
    
    /// Efficiency cores for background tasks
    pub efficiency_cores: u8,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PythonConfig {
    /// Python executable path
    pub python_path: String,
    
    /// ML worker processes count
    pub worker_count: u8,
    
    /// Model configuration
    pub models: ModelConfig,
    
    /// Quality scoring settings
    pub quality: QualityConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelConfig {
    /// Qwen3 1.7B settings for fast processing
    pub qwen3_1_7b: ModelVariant,
    
    /// Qwen3 7B settings for balanced processing
    pub qwen3_7b: ModelVariant,
    
    /// Qwen3 30B settings for high-quality processing
    pub qwen3_30b: ModelVariant,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelVariant {
    /// Model path
    pub path: String,
    
    /// Maximum memory usage in GB
    pub max_memory_gb: u32,
    
    /// Target throughput (items/minute)
    pub target_throughput: u32,
    
    /// Use cases for this variant
    pub use_cases: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct McpConfig {
    /// MCP server port
    pub server_port: u16,
    
    /// Client connection timeout (seconds)
    pub client_timeout: u64,
    
    /// Host orchestration settings
    pub orchestration: OrchestrationConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OrchestrationConfig {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: u32,
    
    /// Task timeout (seconds)
    pub task_timeout: u64,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    ResourceAware,
    Adaptive,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct OptimizationConfig {
    /// Parallel processing settings
    pub parallel: ParallelConfig,
    
    /// Caching configuration
    pub caching: CachingConfig,
    
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ParallelConfig {
    /// Document processing workers
    pub document_workers: u8,
    
    /// I/O worker threads
    pub io_workers: u8,
    
    /// Batch size for parallel processing
    pub batch_size: u32,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CachingConfig {
    /// Cache size in MB
    pub cache_size_mb: u64,
    
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    
    /// Enable intelligent cache warming
    pub intelligent_warming: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: u32,
    
    /// Timeout before attempting recovery (seconds)
    pub timeout_seconds: u64,
    
    /// Success threshold for closing circuit
    pub success_threshold: u32,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct QualityConfig {
    /// Minimum quality score threshold
    pub min_quality_score: f64,
    
    /// Enable quality validation
    pub enable_validation: bool,
    
    /// Quality consistency target
    pub consistency_target: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            m3_max: M3MaxConfig {
                total_memory_gb: 128,
                memory_pools: MemoryPools {
                    processing: 50,
                    ipc: 20,
                    cache: 30,
                    system: 28,
                },
                cpu_cores: CpuCoreConfig {
                    performance_cores: 8,
                    efficiency_cores: 4,
                },
                gpu_cores: 40,
                neural_engine: true,
            },
            python: PythonConfig {
                python_path: "python".to_string(),
                worker_count: 4,
                models: ModelConfig {
                    qwen3_1_7b: ModelVariant {
                        path: "qwen3-1.7b".to_string(),
                        max_memory_gb: 12,
                        target_throughput: 2000,
                        use_cases: vec!["embedding".to_string(), "simple_extraction".to_string()],
                    },
                    qwen3_7b: ModelVariant {
                        path: "qwen3-7b".to_string(),
                        max_memory_gb: 28,
                        target_throughput: 300,
                        use_cases: vec!["balanced_processing".to_string(), "conversation_gen".to_string()],
                    },
                    qwen3_30b: ModelVariant {
                        path: "qwen3-30b".to_string(),
                        max_memory_gb: 45,
                        target_throughput: 100,
                        use_cases: vec!["complex_analysis".to_string(), "quality_assessment".to_string()],
                    },
                },
                quality: QualityConfig {
                    min_quality_score: 0.742,
                    enable_validation: true,
                    consistency_target: 0.742,
                },
            },
            mcp: McpConfig {
                server_port: 8700,
                client_timeout: 30,
                orchestration: OrchestrationConfig {
                    max_concurrent_tasks: 10,
                    task_timeout: 300,
                    load_balancing: LoadBalancingStrategy::Adaptive,
                },
            },
            optimization: OptimizationConfig {
                parallel: ParallelConfig {
                    document_workers: 8,
                    io_workers: 4,
                    batch_size: 100,
                },
                caching: CachingConfig {
                    cache_size_mb: 2048,
                    ttl_seconds: 3600,
                    intelligent_warming: true,
                },
                circuit_breaker: CircuitBreakerConfig {
                    failure_threshold: 5,
                    timeout_seconds: 60,
                    success_threshold: 3,
                },
            },
            quality: QualityConfig {
                min_quality_score: 0.742,
                enable_validation: true,
                consistency_target: 0.742,
            },
        }
    }
}

/// Initialize the pipeline with M3 Max optimizations
pub async fn initialize_pipeline(config: PipelineConfig) -> Result<()> {
    tracing::info!("Initializing Rust-Python hybrid pipeline for M3 Max");
    
    // Initialize M3 Max optimizations
    optimization::m3_max::initialize(&config.m3_max)?;
    
    // Setup IPC communication
    ipc::initialize(&config)?;
    
    // Start MCP server
    mcp::server::start(config.mcp.server_port).await?;
    
    tracing::info!("Pipeline initialization complete");
    Ok(())
}

/// Shutdown pipeline gracefully
pub async fn shutdown_pipeline() -> Result<()> {
    tracing::info!("Shutting down pipeline");
    
    // Stop MCP server
    mcp::server::stop().await?;
    
    // Cleanup IPC resources
    ipc::cleanup().await?;
    
    tracing::info!("Pipeline shutdown complete");
    Ok(())
}