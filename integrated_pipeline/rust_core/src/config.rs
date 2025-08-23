use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use anyhow::Result;

/// Configuration for the processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub processing: ProcessingSettings,
    pub ipc: IpcSettings,
    pub memory: MemorySettings,
    pub quality: QualitySettings,
    pub m3_optimization: M3OptimizationSettings,
}

/// Core processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSettings {
    /// Maximum concurrent documents to process
    pub max_concurrent_docs: usize,
    /// Timeout for individual document processing
    pub document_timeout: Duration,
    /// Supported file extensions
    pub supported_extensions: Vec<String>,
    /// Enable adaptive concurrency based on system load
    pub adaptive_concurrency: bool,
    /// Batch size for group processing
    pub batch_size: usize,
}

/// Inter-process communication settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcSettings {
    /// Size of shared memory pool in GB
    pub shared_memory_size_gb: usize,
    /// Named pipe buffer size in bytes
    pub pipe_buffer_size: usize,
    /// Timeout for IPC operations
    pub timeout_seconds: u64,
    /// Maximum number of concurrent IPC connections
    pub max_connections: usize,
    /// Enable checksum validation for data integrity
    pub enable_checksum_validation: bool,
}

/// Memory management settings optimized for M3 Max
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    /// Total memory limit in GB
    pub memory_limit_gb: usize,
    /// Memory pool for document buffers (percentage of total)
    pub document_buffer_pool_percent: f64,
    /// Memory pool for IPC operations (percentage of total)
    pub ipc_pool_percent: f64,
    /// Enable memory mapping for large files
    pub enable_memory_mapping: bool,
    /// Garbage collection frequency
    pub gc_frequency_seconds: u64,
}

/// Quality validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Minimum quality threshold for acceptance
    pub quality_threshold: f64,
    /// Enable structural validation
    pub enable_structural_validation: bool,
    /// Minimum number of parameters required
    pub min_parameters: usize,
    /// Minimum technical term density
    pub min_technical_density: f64,
    /// Enable early quality filtering
    pub enable_early_filtering: bool,
}

/// M3 Max specific optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3OptimizationSettings {
    /// Use all 16 performance cores
    pub use_all_performance_cores: bool,
    /// Enable unified memory optimization
    pub unified_memory_optimization: bool,
    /// Thread pool size (0 = auto-detect)
    pub thread_pool_size: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    /// CPU affinity settings
    pub cpu_affinity: Vec<usize>,
}

impl ProcessingConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ProcessingConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Create default configuration optimized for M3 Max
    pub fn default_m3_max(max_concurrent: usize, memory_limit_gb: usize) -> Self {
        Self {
            processing: ProcessingSettings {
                max_concurrent_docs: max_concurrent,
                document_timeout: Duration::from_secs(300),
                supported_extensions: vec![
                    ".md".to_string(),
                    ".html".to_string(),
                    ".pdf".to_string(),
                    ".csv".to_string(),
                    ".txt".to_string(),
                ],
                adaptive_concurrency: true,
                batch_size: 8,
            },
            ipc: IpcSettings {
                shared_memory_size_gb: 15, // 15GB shared memory pool
                pipe_buffer_size: 1024 * 1024, // 1MB pipe buffer
                timeout_seconds: 300,
                max_connections: 32,
                enable_checksum_validation: true,
            },
            memory: MemorySettings {
                memory_limit_gb,
                document_buffer_pool_percent: 0.6, // 60% for documents
                ipc_pool_percent: 0.25,           // 25% for IPC
                enable_memory_mapping: true,
                gc_frequency_seconds: 60,
            },
            quality: QualitySettings {
                quality_threshold: 0.75,
                enable_structural_validation: true,
                min_parameters: 1,
                min_technical_density: 0.1,
                enable_early_filtering: true,
            },
            m3_optimization: M3OptimizationSettings {
                use_all_performance_cores: true,
                unified_memory_optimization: true,
                thread_pool_size: 0, // Auto-detect
                enable_simd: true,
                enable_prefetching: true,
                cpu_affinity: (0..16).collect(), // All 16 cores
            },
        }
    }
    
    /// Validate configuration settings
    pub fn validate(&self) -> Result<()> {
        if self.processing.max_concurrent_docs == 0 {
            anyhow::bail!("max_concurrent_docs must be greater than 0");
        }
        
        if self.ipc.shared_memory_size_gb == 0 {
            anyhow::bail!("shared_memory_size_gb must be greater than 0");
        }
        
        if self.memory.memory_limit_gb == 0 {
            anyhow::bail!("memory_limit_gb must be greater than 0");
        }
        
        if self.quality.quality_threshold < 0.0 || self.quality.quality_threshold > 1.0 {
            anyhow::bail!("quality_threshold must be between 0.0 and 1.0");
        }
        
        let total_memory_percent = self.memory.document_buffer_pool_percent + 
                                  self.memory.ipc_pool_percent;
        if total_memory_percent > 0.95 {
            anyhow::bail!("Total memory pool allocation exceeds 95%");
        }
        
        Ok(())
    }
    
    /// Get optimal thread count based on configuration
    pub fn get_optimal_thread_count(&self) -> usize {
        if self.m3_optimization.thread_pool_size > 0 {
            self.m3_optimization.thread_pool_size
        } else if self.m3_optimization.use_all_performance_cores {
            num_cpus::get() // Use all available cores
        } else {
            std::cmp::max(1, num_cpus::get() / 2) // Use half the cores
        }
    }
    
    /// Calculate memory allocations in bytes
    pub fn get_memory_allocations(&self) -> MemoryAllocations {
        let total_bytes = (self.memory.memory_limit_gb as u64) * 1024 * 1024 * 1024;
        let document_pool = (total_bytes as f64 * self.memory.document_buffer_pool_percent) as u64;
        let ipc_pool = (total_bytes as f64 * self.memory.ipc_pool_percent) as u64;
        let system_reserve = total_bytes - document_pool - ipc_pool;
        
        MemoryAllocations {
            total_bytes,
            document_pool_bytes: document_pool,
            ipc_pool_bytes: ipc_pool,
            system_reserve_bytes: system_reserve,
        }
    }
}

/// Memory allocation breakdown
#[derive(Debug, Clone)]
pub struct MemoryAllocations {
    pub total_bytes: u64,
    pub document_pool_bytes: u64,
    pub ipc_pool_bytes: u64,
    pub system_reserve_bytes: u64,
}

impl MemoryAllocations {
    pub fn document_pool_gb(&self) -> f64 {
        self.document_pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    
    pub fn ipc_pool_gb(&self) -> f64 {
        self.ipc_pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
    
    pub fn system_reserve_gb(&self) -> f64 {
        self.system_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Default configuration template
pub const DEFAULT_CONFIG_TOML: &str = r#"
[processing]
max_concurrent_docs = 16
document_timeout = "5m"
supported_extensions = [".md", ".html", ".pdf", ".csv", ".txt"]
adaptive_concurrency = true
batch_size = 8

[ipc]
shared_memory_size_gb = 15
pipe_buffer_size = 1048576
timeout_seconds = 300
max_connections = 32
enable_checksum_validation = true

[memory]
memory_limit_gb = 60
document_buffer_pool_percent = 0.6
ipc_pool_percent = 0.25
enable_memory_mapping = true
gc_frequency_seconds = 60

[quality]
quality_threshold = 0.75
enable_structural_validation = true
min_parameters = 1
min_technical_density = 0.1
enable_early_filtering = true

[m3_optimization]
use_all_performance_cores = true
unified_memory_optimization = true
thread_pool_size = 0
enable_simd = true
enable_prefetching = true
cpu_affinity = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_m3_max_config() {
        let config = ProcessingConfig::default_m3_max(16, 60);
        assert!(config.validate().is_ok());
        assert_eq!(config.processing.max_concurrent_docs, 16);
        assert_eq!(config.memory.memory_limit_gb, 60);
        assert_eq!(config.ipc.shared_memory_size_gb, 15);
    }

    #[test]
    fn test_memory_allocations() {
        let config = ProcessingConfig::default_m3_max(16, 60);
        let allocations = config.get_memory_allocations();
        
        assert_eq!(allocations.document_pool_gb(), 36.0); // 60 * 0.6
        assert_eq!(allocations.ipc_pool_gb(), 15.0);      // 60 * 0.25
        assert_eq!(allocations.system_reserve_gb(), 9.0); // Remainder
    }

    #[test]
    fn test_config_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", DEFAULT_CONFIG_TOML).unwrap();
        
        let config = ProcessingConfig::from_file(temp_file.path()).unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(config.processing.max_concurrent_docs, 16);
    }

    #[test]
    fn test_optimal_thread_count() {
        let config = ProcessingConfig::default_m3_max(16, 60);
        let thread_count = config.get_optimal_thread_count();
        assert!(thread_count > 0);
        assert!(thread_count <= num_cpus::get());
    }
}