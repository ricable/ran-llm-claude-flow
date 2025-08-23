use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::{Result, Context};
use uuid::Uuid;
use tracing::{info, warn, error};

/// Comprehensive IPC configuration for M3 Max optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfiguration {
    /// Shared memory configuration
    pub shared_memory: SharedMemoryConfig,
    /// Named pipe configuration
    pub named_pipes: NamedPipeConfig,
    /// Connection pool configuration
    pub connection_pool: ConnectionPoolConfig,
    /// Performance tuning parameters
    pub performance: PerformanceConfig,
    /// M3 Max specific optimizations
    pub m3_optimizations: M3OptimizationConfig,
    /// Monitoring and logging configuration
    pub monitoring: MonitoringConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
}

/// Shared memory configuration for 15GB pool strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryConfig {
    /// Total shared memory pool size in bytes
    pub pool_size_bytes: usize,
    /// Ring buffer size for streaming data
    pub ring_buffer_size_bytes: usize,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    /// Garbage collection threshold percentage
    pub gc_threshold_percent: f64,
    /// Automatic garbage collection interval
    pub auto_gc_interval_seconds: u64,
    /// Memory pool allocation strategies
    pub allocation_strategies: AllocationStrategies,
    /// Cache line optimization
    pub cache_line_optimization: CacheLineConfig,
}

/// Memory allocation strategies for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStrategies {
    /// Strategy for large documents (>1MB)
    pub large_documents: AllocationStrategy,
    /// Strategy for medium documents (64KB-1MB)
    pub medium_documents: AllocationStrategy,
    /// Strategy for small documents (<64KB)
    pub small_documents: AllocationStrategy,
    /// Strategy for streaming buffers
    pub streaming_buffers: AllocationStrategy,
    /// Strategy for temporary processing buffers
    pub temporary_buffers: AllocationStrategy,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First-fit allocation
    FirstFit,
    /// Best-fit allocation
    BestFit,
    /// Worst-fit allocation
    WorstFit,
    /// Buddy system allocation
    BuddySystem,
    /// Slab allocation for fixed sizes
    Slab { block_sizes: Vec<usize> },
}

/// Cache line optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLineConfig {
    /// Cache line size in bytes (64 for M3 Max)
    pub cache_line_size: usize,
    /// Enable cache-conscious data structures
    pub enable_cache_conscious_layouts: bool,
    /// Prefetch distance in cache lines
    pub prefetch_distance: usize,
    /// Enable memory locality optimizations
    pub enable_locality_optimizations: bool,
}

/// Named pipe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedPipeConfig {
    /// Base path for named pipes
    pub base_path: PathBuf,
    /// Pipe buffer size in bytes
    pub buffer_size: usize,
    /// Enable non-blocking I/O
    pub non_blocking: bool,
    /// Pipe permissions (Unix only)
    pub permissions: u32,
    /// Enable pipe multiplexing
    pub enable_multiplexing: bool,
    /// Number of pipe workers
    pub worker_count: usize,
    /// Message serialization format
    pub serialization_format: SerializationFormat,
}

/// Serialization formats for IPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// JSON format (human readable)
    Json,
    /// MessagePack format (compact binary)
    MessagePack,
    /// Protocol Buffers
    ProtocolBuffers,
    /// CBOR format
    Cbor,
    /// Custom binary format
    Custom(String),
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Maximum total connections
    pub max_total_connections: usize,
    /// Maximum connections per endpoint
    pub max_connections_per_endpoint: usize,
    /// Minimum connections to maintain
    pub min_connections_per_endpoint: usize,
    /// Connection idle timeout
    pub idle_timeout_seconds: u64,
    /// Connection establishment timeout
    pub connect_timeout_seconds: u64,
    /// Health check interval
    pub health_check_interval_seconds: u64,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Connection retry configuration
    pub retry_config: RetryConfig,
    /// Enable connection pooling
    pub enable_pooling: bool,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections first
    LeastConnections,
    /// Weighted round-robin with endpoint weights
    WeightedRoundRobin { weights: HashMap<String, f32> },
    /// Response time based selection
    ResponseTimeBased,
    /// Random selection
    Random,
    /// Consistent hashing
    ConsistentHash,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial retry delay in milliseconds
    pub initial_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Enable exponential backoff with jitter
    pub enable_jitter: bool,
    /// Retryable error codes
    pub retryable_errors: Vec<String>,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target latency for small messages in microseconds
    pub target_latency_us: u64,
    /// Target throughput in MB/s
    pub target_throughput_mbps: f64,
    /// Batch processing configuration
    pub batching: BatchingConfig,
    /// Concurrency limits
    pub concurrency: ConcurrencyConfig,
    /// I/O optimization settings
    pub io_optimization: IoOptimizationConfig,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimizationConfig,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Enable batch processing
    pub enabled: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Adaptive batch sizing
    pub adaptive_sizing: bool,
    /// Batch size adjustment factors
    pub size_adjustment_factors: BatchSizeFactors,
}

/// Batch size adjustment factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeFactors {
    /// Factor for high load conditions
    pub high_load_factor: f64,
    /// Factor for low load conditions
    pub low_load_factor: f64,
    /// Factor for error conditions
    pub error_condition_factor: f64,
    /// Load threshold for adjustments
    pub load_threshold: f64,
}

/// Concurrency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,
    /// Worker thread count
    pub worker_thread_count: usize,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinityConfig,
    /// Queue size limits
    pub queue_limits: QueueLimitsConfig,
}

/// Thread affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// Enable CPU affinity
    pub enable_affinity: bool,
    /// CPU cores to use
    pub cpu_cores: Vec<usize>,
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// Preferred NUMA nodes
    pub preferred_numa_nodes: Vec<usize>,
}

/// Queue limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueLimitsConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// High watermark for backpressure
    pub high_watermark: usize,
    /// Low watermark for flow control
    pub low_watermark: usize,
    /// Queue timeout in milliseconds
    pub queue_timeout_ms: u64,
}

/// I/O optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOptimizationConfig {
    /// Enable zero-copy operations
    pub enable_zero_copy: bool,
    /// Buffer size for I/O operations
    pub io_buffer_size: usize,
    /// Enable vectored I/O
    pub enable_vectored_io: bool,
    /// Direct I/O settings
    pub direct_io: DirectIoConfig,
    /// Async I/O configuration
    pub async_io: AsyncIoConfig,
}

/// Direct I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectIoConfig {
    /// Enable direct I/O for large transfers
    pub enabled: bool,
    /// Minimum size for direct I/O
    pub min_size_bytes: usize,
    /// Alignment requirements
    pub alignment_bytes: usize,
}

/// Async I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncIoConfig {
    /// Enable async I/O
    pub enabled: bool,
    /// I/O queue depth
    pub queue_depth: usize,
    /// Polling vs interrupt driven
    pub polling_mode: bool,
    /// Submission queue size
    pub submission_queue_size: usize,
    /// Completion queue size
    pub completion_queue_size: usize,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable memory compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Memory deduplication settings
    pub deduplication: MemoryDeduplicationConfig,
    /// Memory pressure handling
    pub pressure_handling: MemoryPressureConfig,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstd compression (balanced)
    Zstd,
    /// Zlib compression (high ratio)
    Zlib,
    /// Custom compression
    Custom(String),
}

/// Memory deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDeduplicationConfig {
    /// Enable memory deduplication
    pub enabled: bool,
    /// Hash algorithm for deduplication
    pub hash_algorithm: String,
    /// Minimum block size for deduplication
    pub min_block_size: usize,
    /// Deduplication window size
    pub window_size: usize,
}

/// Memory pressure handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureConfig {
    /// Enable memory pressure monitoring
    pub enabled: bool,
    /// High pressure threshold (percentage)
    pub high_pressure_threshold: f64,
    /// Critical pressure threshold (percentage)
    pub critical_pressure_threshold: f64,
    /// Pressure relief strategies
    pub relief_strategies: Vec<PressureReliefStrategy>,
}

/// Memory pressure relief strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PressureReliefStrategy {
    /// Force garbage collection
    ForceGarbageCollection,
    /// Reduce buffer sizes
    ReduceBufferSizes { factor: f64 },
    /// Increase compression
    IncreaseCompression,
    /// Trigger cache eviction
    TriggerCacheEviction,
    /// Throttle incoming requests
    ThrottleRequests { rate_limit: f64 },
}

/// M3 Max specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3OptimizationConfig {
    /// Enable unified memory optimizations
    pub unified_memory_optimization: bool,
    /// Use all 16 performance cores
    pub use_all_performance_cores: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable Apple Silicon specific features
    pub apple_silicon_features: AppleSiliconFeatures,
    /// Memory bandwidth optimization
    pub memory_bandwidth: MemoryBandwidthConfig,
    /// Cache optimization
    pub cache_optimization: CacheOptimizationConfig,
}

/// Apple Silicon specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleSiliconFeatures {
    /// Enable AMX (Apple Matrix eXtensions) if available
    pub enable_amx: bool,
    /// Enable Neural Engine integration
    pub enable_neural_engine: bool,
    /// Use hardware AES acceleration
    pub hardware_aes: bool,
    /// Use hardware compression/decompression
    pub hardware_compression: bool,
}

/// Memory bandwidth optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBandwidthConfig {
    /// Target memory bandwidth utilization (percentage)
    pub target_bandwidth_utilization: f64,
    /// Enable memory interleaving
    pub enable_memory_interleaving: bool,
    /// Optimize for sequential access patterns
    pub optimize_sequential_access: bool,
    /// Memory striping configuration
    pub memory_striping: MemoryStripingConfig,
}

/// Memory striping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStripingConfig {
    /// Enable memory striping
    pub enabled: bool,
    /// Stripe size in bytes
    pub stripe_size: usize,
    /// Number of stripes
    pub stripe_count: usize,
}

/// Cache optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOptimizationConfig {
    /// L1 cache optimization
    pub l1_optimization: bool,
    /// L2 cache optimization
    pub l2_optimization: bool,
    /// L3 cache optimization
    pub l3_optimization: bool,
    /// Cache warming strategies
    pub cache_warming: CacheWarmingConfig,
    /// Cache partitioning
    pub cache_partitioning: CachePartitioningConfig,
}

/// Cache warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarmingConfig {
    /// Enable cache warming
    pub enabled: bool,
    /// Warm-up data patterns
    pub warmup_patterns: Vec<String>,
    /// Warm-up schedule
    pub warmup_schedule: String,
}

/// Cache partitioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePartitioningConfig {
    /// Enable cache partitioning
    pub enabled: bool,
    /// Partition sizes (as percentages)
    pub partition_sizes: HashMap<String, f64>,
    /// Partition priorities
    pub partition_priorities: HashMap<String, u32>,
}

/// Monitoring and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable health monitoring
    pub enable_health_monitoring: bool,
    /// Metrics collection configuration
    pub metrics: MetricsConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval_seconds: u64,
    /// Metrics retention period
    pub retention_period_seconds: u64,
    /// Export configuration
    pub export: MetricsExportConfig,
    /// Custom metrics
    pub custom_metrics: Vec<CustomMetric>,
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export format
    pub format: MetricsFormat,
    /// Export destination
    pub destination: String,
    /// Export interval
    pub interval_seconds: u64,
    /// Enable real-time export
    pub real_time_export: bool,
}

/// Metrics formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsFormat {
    /// Prometheus format
    Prometheus,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Custom format
    Custom(String),
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Collection function
    pub collection_function: String,
    /// Labels
    pub labels: HashMap<String, String>,
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Counter metric
    Counter,
    /// Gauge metric
    Gauge,
    /// Histogram metric
    Histogram,
    /// Summary metric
    Summary,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Log format
    pub format: LogFormat,
    /// Log outputs
    pub outputs: Vec<LogOutput>,
    /// Structured logging
    pub structured: bool,
    /// Log sampling rate
    pub sampling_rate: f64,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warn level
    Warn,
    /// Error level
    Error,
}

/// Log formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Logfmt format
    Logfmt,
    /// Custom format
    Custom(String),
}

/// Log output destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    /// Console output
    Console,
    /// File output
    File { path: PathBuf, rotation: Option<FileRotationConfig> },
    /// Syslog output
    Syslog { facility: String },
    /// Network output
    Network { endpoint: String },
}

/// File rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRotationConfig {
    /// Maximum file size
    pub max_size_mb: usize,
    /// Maximum number of files
    pub max_files: usize,
    /// Rotation interval
    pub rotation_interval: RotationInterval,
}

/// Rotation intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationInterval {
    /// Hourly rotation
    Hourly,
    /// Daily rotation
    Daily,
    /// Weekly rotation
    Weekly,
    /// Monthly rotation
    Monthly,
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub channels: Vec<NotificationChannel>,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Threshold value
    pub threshold: f64,
    /// Evaluation interval
    pub evaluation_interval_seconds: u64,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Critical alert
    Critical,
    /// Warning alert
    Warning,
    /// Info alert
    Info,
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email { recipients: Vec<String> },
    /// Slack notification
    Slack { webhook_url: String, channel: String },
    /// Webhook notification
    Webhook { url: String, headers: HashMap<String, String> },
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption
    pub enable_encryption: bool,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Authorization configuration
    pub authorization: AuthorizationConfig,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key size in bits
    pub key_size: usize,
    /// Key derivation function
    pub kdf: String,
    /// Key rotation interval
    pub key_rotation_interval_seconds: u64,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Token expiration time
    pub token_expiration_seconds: u64,
    /// Enable mutual authentication
    pub mutual_authentication: bool,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// No authentication
    None,
    /// Token-based authentication
    Token { secret_key: String },
    /// Certificate-based authentication
    Certificate { ca_cert_path: PathBuf },
    /// Custom authentication
    Custom { method: String },
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    /// Enable authorization
    pub enabled: bool,
    /// Authorization rules
    pub rules: Vec<AuthorizationRule>,
    /// Default access policy
    pub default_policy: AccessPolicy,
}

/// Authorization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRule {
    /// Rule name
    pub name: String,
    /// Resource pattern
    pub resource: String,
    /// Action pattern
    pub action: String,
    /// Subject pattern
    pub subject: String,
    /// Access policy
    pub policy: AccessPolicy,
}

/// Access policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPolicy {
    /// Allow access
    Allow,
    /// Deny access
    Deny,
    /// Conditional access
    Conditional { condition: String },
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable fault tolerance
    pub enabled: bool,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Timeout configuration
    pub timeouts: TimeoutConfig,
    /// Recovery configuration
    pub recovery: RecoveryConfig,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Success threshold for recovery
    pub success_threshold: usize,
    /// Timeout duration
    pub timeout_duration_seconds: u64,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Request timeout
    pub request_timeout_seconds: u64,
    /// Connection timeout
    pub connection_timeout_seconds: u64,
    /// Health check timeout
    pub health_check_timeout_seconds: u64,
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Recovery strategies
    pub strategies: Vec<RecoveryStrategy>,
    /// Recovery attempt interval
    pub attempt_interval_seconds: u64,
    /// Maximum recovery attempts
    pub max_attempts: u32,
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart process
    RestartProcess,
    /// Recreate connections
    RecreateConnections,
    /// Reset shared memory
    ResetSharedMemory,
    /// Failover to backup
    Failover { backup_endpoint: String },
    /// Custom recovery
    Custom { script: String },
}

impl IpcConfiguration {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let config: IpcConfiguration = toml::from_str(&content)
            .context("Failed to parse TOML configuration")?;
        
        config.validate()?;
        
        info!("Loaded IPC configuration from {:?}", path.as_ref());
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration to TOML")?;
        
        std::fs::write(path.as_ref(), content)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;
        
        info!("Saved IPC configuration to {:?}", path.as_ref());
        Ok(())
    }

    /// Create default configuration optimized for M3 Max with 128GB unified memory
    pub fn default_m3_max_128gb() -> Self {
        Self {
            shared_memory: SharedMemoryConfig {
                pool_size_bytes: 15 * 1024 * 1024 * 1024, // 15GB
                ring_buffer_size_bytes: 256 * 1024 * 1024, // 256MB
                memory_alignment: 64, // M3 Max cache line size
                enable_prefetching: true,
                gc_threshold_percent: 15.0,
                auto_gc_interval_seconds: 300, // 5 minutes
                allocation_strategies: AllocationStrategies {
                    large_documents: AllocationStrategy::BestFit,
                    medium_documents: AllocationStrategy::FirstFit,
                    small_documents: AllocationStrategy::Slab { 
                        block_sizes: vec![64, 128, 256, 512, 1024, 2048, 4096] 
                    },
                    streaming_buffers: AllocationStrategy::BuddySystem,
                    temporary_buffers: AllocationStrategy::FirstFit,
                },
                cache_line_optimization: CacheLineConfig {
                    cache_line_size: 64,
                    enable_cache_conscious_layouts: true,
                    prefetch_distance: 8,
                    enable_locality_optimizations: true,
                },
            },
            named_pipes: NamedPipeConfig {
                base_path: PathBuf::from("/tmp/claude_flow_ipc"),
                buffer_size: 1024 * 1024, // 1MB
                non_blocking: true,
                permissions: 0o600,
                enable_multiplexing: true,
                worker_count: 8,
                serialization_format: SerializationFormat::MessagePack,
            },
            connection_pool: ConnectionPoolConfig {
                max_total_connections: 128,
                max_connections_per_endpoint: 32,
                min_connections_per_endpoint: 4,
                idle_timeout_seconds: 300,
                connect_timeout_seconds: 10,
                health_check_interval_seconds: 30,
                load_balancing_strategy: LoadBalancingStrategy::LeastConnections,
                retry_config: RetryConfig {
                    max_retries: 3,
                    initial_delay_ms: 100,
                    backoff_multiplier: 2.0,
                    max_delay_ms: 30000,
                    enable_jitter: true,
                    retryable_errors: vec![
                        "ConnectionRefused".to_string(),
                        "Timeout".to_string(),
                        "TemporaryFailure".to_string(),
                    ],
                },
                enable_pooling: true,
            },
            performance: PerformanceConfig {
                target_latency_us: 100, // <100μs for small messages
                target_throughput_mbps: 1000.0, // 1GB/s target
                batching: BatchingConfig {
                    enabled: true,
                    max_batch_size: 32,
                    batch_timeout_ms: 10,
                    adaptive_sizing: true,
                    size_adjustment_factors: BatchSizeFactors {
                        high_load_factor: 1.5,
                        low_load_factor: 0.7,
                        error_condition_factor: 0.5,
                        load_threshold: 0.8,
                    },
                },
                concurrency: ConcurrencyConfig {
                    max_concurrent_operations: 256,
                    worker_thread_count: 16, // All M3 Max performance cores
                    enable_work_stealing: true,
                    thread_affinity: ThreadAffinityConfig {
                        enable_affinity: true,
                        cpu_cores: (0..16).collect(), // All performance cores
                        numa_aware: false, // M3 Max has unified memory
                        preferred_numa_nodes: vec![0],
                    },
                    queue_limits: QueueLimitsConfig {
                        max_queue_size: 10000,
                        high_watermark: 8000,
                        low_watermark: 2000,
                        queue_timeout_ms: 5000,
                    },
                },
                io_optimization: IoOptimizationConfig {
                    enable_zero_copy: true,
                    io_buffer_size: 64 * 1024, // 64KB
                    enable_vectored_io: true,
                    direct_io: DirectIoConfig {
                        enabled: true,
                        min_size_bytes: 1024 * 1024, // 1MB
                        alignment_bytes: 4096,
                    },
                    async_io: AsyncIoConfig {
                        enabled: true,
                        queue_depth: 64,
                        polling_mode: false, // Interrupt-driven for efficiency
                        submission_queue_size: 256,
                        completion_queue_size: 256,
                    },
                },
                memory_optimization: MemoryOptimizationConfig {
                    enable_compression: true,
                    compression_algorithm: CompressionAlgorithm::Lz4,
                    deduplication: MemoryDeduplicationConfig {
                        enabled: true,
                        hash_algorithm: "SHA-256".to_string(),
                        min_block_size: 4096,
                        window_size: 1024 * 1024,
                    },
                    pressure_handling: MemoryPressureConfig {
                        enabled: true,
                        high_pressure_threshold: 80.0,
                        critical_pressure_threshold: 95.0,
                        relief_strategies: vec![
                            PressureReliefStrategy::ForceGarbageCollection,
                            PressureReliefStrategy::IncreaseCompression,
                            PressureReliefStrategy::ReduceBufferSizes { factor: 0.5 },
                        ],
                    },
                },
            },
            m3_optimizations: M3OptimizationConfig {
                unified_memory_optimization: true,
                use_all_performance_cores: true,
                enable_simd: true,
                apple_silicon_features: AppleSiliconFeatures {
                    enable_amx: true,
                    enable_neural_engine: false, // Not used for IPC
                    hardware_aes: true,
                    hardware_compression: true,
                },
                memory_bandwidth: MemoryBandwidthConfig {
                    target_bandwidth_utilization: 85.0,
                    enable_memory_interleaving: true,
                    optimize_sequential_access: true,
                    memory_striping: MemoryStripingConfig {
                        enabled: true,
                        stripe_size: 64 * 1024, // 64KB stripes
                        stripe_count: 8,
                    },
                },
                cache_optimization: CacheOptimizationConfig {
                    l1_optimization: true,
                    l2_optimization: true,
                    l3_optimization: true,
                    cache_warming: CacheWarmingConfig {
                        enabled: true,
                        warmup_patterns: vec![
                            "sequential".to_string(),
                            "random".to_string(),
                        ],
                        warmup_schedule: "startup".to_string(),
                    },
                    cache_partitioning: CachePartitioningConfig {
                        enabled: false, // Let OS handle cache partitioning
                        partition_sizes: HashMap::new(),
                        partition_priorities: HashMap::new(),
                    },
                },
            },
            monitoring: MonitoringConfig {
                enable_performance_monitoring: true,
                enable_health_monitoring: true,
                metrics: MetricsConfig {
                    enabled: true,
                    collection_interval_seconds: 10,
                    retention_period_seconds: 3600 * 24 * 7, // 1 week
                    export: MetricsExportConfig {
                        format: MetricsFormat::Prometheus,
                        destination: "http://localhost:9090".to_string(),
                        interval_seconds: 30,
                        real_time_export: false,
                    },
                    custom_metrics: Vec::new(),
                },
                logging: LoggingConfig {
                    level: LogLevel::Info,
                    format: LogFormat::Json,
                    outputs: vec![
                        LogOutput::Console,
                        LogOutput::File {
                            path: PathBuf::from("/tmp/claude_flow_ipc.log"),
                            rotation: Some(FileRotationConfig {
                                max_size_mb: 100,
                                max_files: 10,
                                rotation_interval: RotationInterval::Daily,
                            }),
                        },
                    ],
                    structured: true,
                    sampling_rate: 1.0,
                },
                alerting: AlertingConfig {
                    enabled: true,
                    rules: vec![
                        AlertRule {
                            name: "high_memory_usage".to_string(),
                            condition: "memory_utilization > threshold".to_string(),
                            severity: AlertSeverity::Warning,
                            threshold: 90.0,
                            evaluation_interval_seconds: 60,
                        },
                        AlertRule {
                            name: "connection_failures".to_string(),
                            condition: "connection_error_rate > threshold".to_string(),
                            severity: AlertSeverity::Critical,
                            threshold: 10.0,
                            evaluation_interval_seconds: 30,
                        },
                    ],
                    channels: Vec::new(),
                },
            },
            security: SecurityConfig {
                enable_encryption: false, // Local IPC doesn't need encryption by default
                encryption: EncryptionConfig {
                    algorithm: "AES-256-GCM".to_string(),
                    key_size: 256,
                    kdf: "PBKDF2".to_string(),
                    key_rotation_interval_seconds: 3600 * 24, // Daily
                },
                authentication: AuthenticationConfig {
                    method: AuthenticationMethod::None, // Local processes
                    token_expiration_seconds: 3600,
                    mutual_authentication: false,
                },
                authorization: AuthorizationConfig {
                    enabled: false, // Local processes
                    rules: Vec::new(),
                    default_policy: AccessPolicy::Allow,
                },
            },
            fault_tolerance: FaultToleranceConfig {
                enabled: true,
                circuit_breaker: CircuitBreakerConfig {
                    enabled: true,
                    failure_threshold: 5,
                    success_threshold: 3,
                    timeout_duration_seconds: 60,
                },
                timeouts: TimeoutConfig {
                    request_timeout_seconds: 30,
                    connection_timeout_seconds: 10,
                    health_check_timeout_seconds: 5,
                },
                recovery: RecoveryConfig {
                    auto_recovery: true,
                    strategies: vec![
                        RecoveryStrategy::RecreateConnections,
                        RecoveryStrategy::ResetSharedMemory,
                    ],
                    attempt_interval_seconds: 10,
                    max_attempts: 3,
                },
            },
        }
    }

    /// Validate configuration for consistency and feasibility
    pub fn validate(&self) -> Result<()> {
        // Validate shared memory configuration
        if self.shared_memory.pool_size_bytes == 0 {
            anyhow::bail!("Shared memory pool size cannot be zero");
        }
        
        if self.shared_memory.pool_size_bytes < 1024 * 1024 {
            warn!("Shared memory pool size is very small: {} bytes", self.shared_memory.pool_size_bytes);
        }

        if self.shared_memory.ring_buffer_size_bytes > self.shared_memory.pool_size_bytes / 2 {
            anyhow::bail!("Ring buffer size cannot exceed half of the shared memory pool");
        }

        // Validate connection pool configuration
        if self.connection_pool.max_connections_per_endpoint > self.connection_pool.max_total_connections {
            anyhow::bail!("Max connections per endpoint cannot exceed max total connections");
        }

        if self.connection_pool.min_connections_per_endpoint > self.connection_pool.max_connections_per_endpoint {
            anyhow::bail!("Min connections per endpoint cannot exceed max connections per endpoint");
        }

        // Validate performance configuration
        if self.performance.target_latency_us == 0 {
            anyhow::bail!("Target latency cannot be zero");
        }

        if self.performance.concurrency.worker_thread_count == 0 {
            anyhow::bail!("Worker thread count cannot be zero");
        }

        // Validate thread affinity
        let max_cores = num_cpus::get();
        for &core in &self.performance.concurrency.thread_affinity.cpu_cores {
            if core >= max_cores {
                warn!("CPU core {} specified but only {} cores available", core, max_cores);
            }
        }

        // Validate M3 optimizations
        if self.m3_optimizations.use_all_performance_cores && 
           self.performance.concurrency.worker_thread_count != 16 {
            warn!("M3 Max has 16 performance cores but worker_thread_count is {}", 
                  self.performance.concurrency.worker_thread_count);
        }

        // Validate monitoring configuration
        if self.monitoring.metrics.collection_interval_seconds == 0 {
            anyhow::bail!("Metrics collection interval cannot be zero");
        }

        // Validate fault tolerance timeouts
        if self.fault_tolerance.timeouts.connection_timeout_seconds > 
           self.fault_tolerance.timeouts.request_timeout_seconds {
            warn!("Connection timeout is greater than request timeout");
        }

        info!("Configuration validation completed successfully");
        Ok(())
    }

    /// Get memory allocations breakdown
    pub fn get_memory_allocations(&self) -> MemoryAllocations {
        let pool_size = self.shared_memory.pool_size_bytes;
        let ring_buffer_size = self.shared_memory.ring_buffer_size_bytes;
        let remaining = pool_size.saturating_sub(ring_buffer_size);

        MemoryAllocations {
            total_bytes: pool_size,
            ring_buffer_bytes: ring_buffer_size,
            document_pool_bytes: remaining * 80 / 100, // 80% for documents
            metadata_bytes: remaining * 15 / 100,      // 15% for metadata
            system_reserve_bytes: remaining * 5 / 100, // 5% system reserve
        }
    }
}

/// Memory allocation breakdown
#[derive(Debug, Clone)]
pub struct MemoryAllocations {
    pub total_bytes: usize,
    pub ring_buffer_bytes: usize,
    pub document_pool_bytes: usize,
    pub metadata_bytes: usize,
    pub system_reserve_bytes: usize,
}

impl MemoryAllocations {
    pub fn ring_buffer_gb(&self) -> f64 {
        self.ring_buffer_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn document_pool_gb(&self) -> f64 {
        self.document_pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Configuration template for different use cases
pub const M3_MAX_TEMPLATE: &str = include_str!("../templates/m3_max_config.toml");
pub const HIGH_THROUGHPUT_TEMPLATE: &str = include_str!("../templates/high_throughput_config.toml");
pub const LOW_LATENCY_TEMPLATE: &str = include_str!("../templates/low_latency_config.toml");

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_m3_max_config_validation() {
        let config = IpcConfiguration::default_m3_max_128gb();
        assert!(config.validate().is_ok());
        
        let allocations = config.get_memory_allocations();
        assert_eq!(allocations.total_gb(), 15.0);
        assert!(allocations.ring_buffer_gb() > 0.0);
        assert!(allocations.document_pool_gb() > 0.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = IpcConfiguration::default_m3_max_128gb();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(!toml_str.is_empty());
        
        let deserialized: IpcConfiguration = toml::from_str(&toml_str).unwrap();
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_config_file_operations() {
        let config = IpcConfiguration::default_m3_max_128gb();
        let temp_file = NamedTempFile::new().unwrap();
        
        config.to_file(temp_file.path()).unwrap();
        let loaded_config = IpcConfiguration::from_file(temp_file.path()).unwrap();
        
        assert!(loaded_config.validate().is_ok());
        assert_eq!(loaded_config.shared_memory.pool_size_bytes, config.shared_memory.pool_size_bytes);
    }
}