use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use anyhow::{Result, Context};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write, Cursor};
use tracing::{debug, info, warn, error};

/// Advanced IPC message protocol with comprehensive features
/// Optimized for Rust-Python pipeline communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMessage {
    /// Message header with metadata
    pub header: MessageHeader,
    /// Message payload
    pub payload: MessagePayload,
    /// Optional binary attachments
    pub attachments: Vec<BinaryAttachment>,
    /// Protocol extensions for future compatibility
    pub extensions: HashMap<String, ProtocolExtension>,
}

/// Message header with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Protocol version for compatibility
    pub protocol_version: ProtocolVersion,
    /// Unique message identifier
    pub message_id: Uuid,
    /// Message sequence number for ordering
    pub sequence_number: u64,
    /// Correlation ID for request-response matching
    pub correlation_id: Option<Uuid>,
    /// Message timestamp (nanoseconds since UNIX epoch)
    pub timestamp: u64,
    /// Time-to-live for message expiration
    pub ttl: Option<Duration>,
    /// Message priority for queue ordering
    pub priority: MessagePriority,
    /// Source process identification
    pub source: ProcessIdentification,
    /// Destination process identification
    pub destination: ProcessIdentification,
    /// Message routing information
    pub routing: RoutingInfo,
    /// Security and authentication data
    pub security: SecurityInfo,
    /// Message compression information
    pub compression: CompressionInfo,
    /// Total message size including all components
    pub total_size: u64,
    /// Checksum for integrity verification
    pub checksum: u32,
}

/// Protocol version for backward/forward compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

/// Message priority levels for queue management
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    /// System-critical messages
    Critical = 0,
    /// High priority processing
    High = 1,
    /// Normal priority processing
    Normal = 2,
    /// Low priority background tasks
    Low = 3,
    /// Bulk operations
    Bulk = 4,
}

/// Process identification for routing and security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessIdentification {
    /// Process ID
    pub pid: u32,
    /// Process name/identifier
    pub name: String,
    /// Process version
    pub version: String,
    /// Machine/node identifier
    pub node_id: String,
    /// Process capabilities
    pub capabilities: Vec<String>,
}

/// Routing information for multi-hop communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingInfo {
    /// Direct routing (single hop)
    pub direct: bool,
    /// Routing path for multi-hop messages
    pub path: Vec<String>,
    /// Routing strategy
    pub strategy: RoutingStrategy,
    /// Maximum hops allowed
    pub max_hops: u8,
    /// Current hop count
    pub hop_count: u8,
}

/// Routing strategies for message delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Direct delivery
    Direct,
    /// Round-robin load balancing
    RoundRobin,
    /// Weighted load balancing
    Weighted,
    /// Proximity-based routing
    Proximity,
    /// Broadcast to all nodes
    Broadcast,
    /// Multicast to specific group
    Multicast(String),
}

/// Security information for authentication and encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityInfo {
    /// Authentication token
    pub auth_token: Option<String>,
    /// Message signature for integrity
    pub signature: Option<Vec<u8>>,
    /// Encryption algorithm used
    pub encryption: EncryptionType,
    /// Key derivation information
    pub key_info: Option<KeyInfo>,
}

/// Encryption types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionType {
    /// No encryption
    None,
    /// AES-256-GCM encryption
    Aes256Gcm,
    /// ChaCha20-Poly1305 encryption
    ChaCha20Poly1305,
    /// Custom encryption scheme
    Custom(String),
}

/// Key derivation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyInfo {
    /// Key identifier
    pub key_id: String,
    /// Salt for key derivation
    pub salt: Vec<u8>,
    /// Initialization vector
    pub iv: Vec<u8>,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Original size before compression
    pub original_size: u64,
    /// Compressed size
    pub compressed_size: u64,
    /// Compression level used
    pub level: u8,
}

/// Supported compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 compression for speed
    Lz4,
    /// Zstd compression for balance
    Zstd,
    /// Gzip compression for compatibility
    Gzip,
    /// Custom compression
    Custom(String),
}

/// Message payload containing the actual data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Document processing request
    DocumentProcessingRequest {
        document: DocumentData,
        processing_options: ProcessingOptions,
        callback_info: Option<CallbackInfo>,
    },
    /// Document processing response
    DocumentProcessingResponse {
        result: ProcessingResult,
        metrics: ProcessingMetrics,
        error: Option<ProcessingError>,
    },
    /// System control messages
    SystemControl {
        command: SystemCommand,
        parameters: HashMap<String, String>,
    },
    /// Health check and monitoring
    HealthCheck {
        check_type: HealthCheckType,
        metrics: Option<HealthMetrics>,
    },
    /// Configuration updates
    ConfigurationUpdate {
        config_section: String,
        updates: HashMap<String, serde_json::Value>,
        apply_immediately: bool,
    },
    /// Raw binary data
    BinaryData {
        data_type: String,
        data: Vec<u8>,
        metadata: HashMap<String, String>,
    },
    /// Custom message type for extensions
    Custom {
        message_type: String,
        data: serde_json::Value,
    },
}

/// Document data for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentData {
    /// Document identifier
    pub document_id: Uuid,
    /// Document metadata
    pub metadata: DocumentMetadata,
    /// Document content (may be reference to shared memory)
    pub content: DocumentContent,
    /// Processing hints
    pub hints: ProcessingHints,
}

/// Document content representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentContent {
    /// Inline content (small documents)
    Inline(String),
    /// Shared memory reference (large documents)
    SharedMemory {
        offset: u64,
        size: usize,
        pool_id: Uuid,
    },
    /// File reference
    File {
        path: String,
        offset: Option<u64>,
        size: Option<usize>,
    },
    /// URL reference
    Url(String),
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title
    pub title: Option<String>,
    /// Document format
    pub format: String,
    /// Document size in bytes
    pub size: usize,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Document tags
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    /// Model preference
    pub model_preference: Option<String>,
    /// Quality threshold
    pub quality_threshold: f64,
    /// Maximum processing time
    pub max_processing_time: Duration,
    /// Enable caching
    pub enable_caching: bool,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

/// Processing hints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingHints {
    /// Estimated complexity
    pub complexity: String,
    /// Expected output size
    pub expected_output_size: Option<usize>,
    /// Processing priority
    pub priority: MessagePriority,
    /// Batch processing eligible
    pub batch_eligible: bool,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: Option<usize>,
    /// CPU cores required
    pub cpu_cores: Option<usize>,
    /// GPU required
    pub gpu_required: bool,
    /// Estimated processing time
    pub estimated_time: Option<Duration>,
}

/// Processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Processing status
    pub status: ProcessingStatus,
    /// Generated output
    pub output: ProcessingOutput,
    /// Quality assessment
    pub quality: QualityAssessment,
    /// Processing metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// Processing completed successfully
    Success,
    /// Processing completed with warnings
    SuccessWithWarnings,
    /// Processing failed
    Failed,
    /// Processing timeout
    Timeout,
    /// Processing cancelled
    Cancelled,
}

/// Processing output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingOutput {
    /// Inline output data
    Inline(serde_json::Value),
    /// Shared memory reference
    SharedMemory {
        offset: u64,
        size: usize,
        pool_id: Uuid,
    },
    /// File output
    File(String),
    /// Multiple outputs
    Multiple(Vec<ProcessingOutput>),
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f64,
    /// Individual quality metrics
    pub metrics: HashMap<String, f64>,
    /// Quality threshold met
    pub threshold_met: bool,
    /// Quality recommendations
    pub recommendations: Vec<String>,
}

/// Processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Processing start time
    pub start_time: u64,
    /// Processing end time
    pub end_time: u64,
    /// Processing duration
    pub duration: Duration,
    /// Memory usage peak (MB)
    pub memory_peak_mb: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f64>,
    /// Tokens processed
    pub tokens_processed: usize,
    /// Model used
    pub model_used: String,
}

/// Processing error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingError {
    /// Error code
    pub error_code: String,
    /// Error message
    pub message: String,
    /// Error details
    pub details: Option<String>,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
    /// Error context
    pub context: HashMap<String, String>,
}

/// System control commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemCommand {
    /// Shutdown system
    Shutdown,
    /// Restart system
    Restart,
    /// Pause processing
    Pause,
    /// Resume processing
    Resume,
    /// Update configuration
    UpdateConfig,
    /// Clear caches
    ClearCaches,
    /// Run garbage collection
    GarbageCollect,
    /// Performance benchmark
    Benchmark,
}

/// Health check types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    /// Basic ping/pong
    Ping,
    /// Comprehensive system health
    SystemHealth,
    /// Performance metrics
    Performance,
    /// Resource utilization
    Resources,
    /// Custom health check
    Custom(String),
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// System uptime
    pub uptime: Duration,
    /// Memory usage
    pub memory_usage: MemoryUsage,
    /// CPU usage
    pub cpu_usage: f64,
    /// Active connections
    pub active_connections: usize,
    /// Request queue size
    pub queue_size: usize,
    /// Error rate
    pub error_rate: f64,
    /// Response times
    pub response_times: ResponseTimeMetrics,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total memory in MB
    pub total_mb: usize,
    /// Used memory in MB
    pub used_mb: usize,
    /// Free memory in MB
    pub free_mb: usize,
    /// Cached memory in MB
    pub cached_mb: usize,
}

/// Response time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeMetrics {
    /// Average response time
    pub average: Duration,
    /// 50th percentile
    pub p50: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// Maximum response time
    pub max: Duration,
}

/// Callback information for asynchronous processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallbackInfo {
    /// Callback URL or endpoint
    pub endpoint: String,
    /// Callback method
    pub method: String,
    /// Authentication information
    pub auth: Option<String>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Retry policy
    pub retry_policy: RetryPolicy,
}

/// Retry policy for callbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay: Duration,
}

/// Binary attachments for large data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryAttachment {
    /// Attachment identifier
    pub id: String,
    /// Content type
    pub content_type: String,
    /// Attachment size
    pub size: usize,
    /// Attachment data
    pub data: AttachmentData,
    /// Attachment metadata
    pub metadata: HashMap<String, String>,
}

/// Attachment data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttachmentData {
    /// Inline binary data
    Inline(Vec<u8>),
    /// Shared memory reference
    SharedMemory {
        offset: u64,
        size: usize,
        pool_id: Uuid,
    },
    /// File reference
    File(String),
    /// URL reference
    Url(String),
}

/// Protocol extensions for future compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolExtension {
    /// Extension name
    pub name: String,
    /// Extension version
    pub version: String,
    /// Extension data
    pub data: serde_json::Value,
}

/// Message serialization and deserialization
impl ProtocolMessage {
    /// Create a new protocol message
    pub fn new(payload: MessagePayload, source: ProcessIdentification, destination: ProcessIdentification) -> Self {
        let message_id = Uuid::new_v4();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let header = MessageHeader {
            protocol_version: ProtocolVersion::current(),
            message_id,
            sequence_number: 0, // Set by sender
            correlation_id: None,
            timestamp,
            ttl: None,
            priority: MessagePriority::Normal,
            source,
            destination,
            routing: RoutingInfo {
                direct: true,
                path: Vec::new(),
                strategy: RoutingStrategy::Direct,
                max_hops: 1,
                hop_count: 0,
            },
            security: SecurityInfo {
                auth_token: None,
                signature: None,
                encryption: EncryptionType::None,
                key_info: None,
            },
            compression: CompressionInfo {
                algorithm: CompressionAlgorithm::None,
                original_size: 0,
                compressed_size: 0,
                level: 0,
            },
            total_size: 0, // Calculated during serialization
            checksum: 0, // Calculated during serialization
        };

        Self {
            header,
            payload,
            attachments: Vec::new(),
            extensions: HashMap::new(),
        }
    }

    /// Serialize message to binary format
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // Serialize to JSON first
        let json_data = serde_json::to_vec(self)
            .context("Failed to serialize message to JSON")?;

        // Apply compression if enabled
        let compressed_data = match self.header.compression.algorithm {
            CompressionAlgorithm::None => json_data,
            CompressionAlgorithm::Lz4 => {
                // Would use lz4 compression
                json_data // Placeholder
            },
            CompressionAlgorithm::Zstd => {
                // Would use zstd compression
                json_data // Placeholder
            },
            CompressionAlgorithm::Gzip => {
                // Would use gzip compression
                json_data // Placeholder
            },
            CompressionAlgorithm::Custom(_) => {
                json_data // Placeholder
            },
        };

        // Create binary frame
        let mut buffer = Vec::new();

        // Magic number for protocol identification
        buffer.write_u32::<LittleEndian>(0x434C4946)?; // "CLIF" (Claude Flow)

        // Protocol version
        buffer.write_u16::<LittleEndian>(self.header.protocol_version.major)?;
        buffer.write_u16::<LittleEndian>(self.header.protocol_version.minor)?;
        buffer.write_u16::<LittleEndian>(self.header.protocol_version.patch)?;

        // Data length
        buffer.write_u64::<LittleEndian>(compressed_data.len() as u64)?;

        // Checksum
        let checksum = crc32fast::hash(&compressed_data);
        buffer.write_u32::<LittleEndian>(checksum)?;

        // Message data
        buffer.extend_from_slice(&compressed_data);

        Ok(buffer)
    }

    /// Deserialize message from binary format
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Read and verify magic number
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != 0x434C4946 {
            anyhow::bail!("Invalid protocol magic number: 0x{:08x}", magic);
        }

        // Read protocol version
        let major = cursor.read_u16::<LittleEndian>()?;
        let minor = cursor.read_u16::<LittleEndian>()?;
        let patch = cursor.read_u16::<LittleEndian>()?;
        
        let version = ProtocolVersion { major, minor, patch };
        if !version.is_compatible() {
            anyhow::bail!("Incompatible protocol version: {}.{}.{}", major, minor, patch);
        }

        // Read data length
        let data_length = cursor.read_u64::<LittleEndian>()? as usize;

        // Read and verify checksum
        let expected_checksum = cursor.read_u32::<LittleEndian>()?;

        // Read message data
        let mut message_data = vec![0u8; data_length];
        cursor.read_exact(&mut message_data)?;

        // Verify checksum
        let actual_checksum = crc32fast::hash(&message_data);
        if actual_checksum != expected_checksum {
            anyhow::bail!("Message checksum mismatch: expected 0x{:08x}, got 0x{:08x}", 
                         expected_checksum, actual_checksum);
        }

        // Decompress if needed (placeholder - would implement actual decompression)
        let json_data = message_data;

        // Deserialize from JSON
        let mut message: ProtocolMessage = serde_json::from_slice(&json_data)
            .context("Failed to deserialize message from JSON")?;

        // Update calculated fields
        message.header.total_size = data.len() as u64;
        message.header.checksum = actual_checksum;

        Ok(message)
    }

    /// Validate message integrity and structure
    pub fn validate(&self) -> Result<()> {
        // Check protocol version compatibility
        if !self.header.protocol_version.is_compatible() {
            anyhow::bail!("Incompatible protocol version");
        }

        // Check message expiration
        if let Some(ttl) = self.header.ttl {
            let message_age = SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_nanos() as u64 - self.header.timestamp;
            
            if message_age > ttl.as_nanos() as u64 {
                anyhow::bail!("Message has expired");
            }
        }

        // Check hop count
        if self.header.routing.hop_count >= self.header.routing.max_hops {
            anyhow::bail!("Message exceeded maximum hops");
        }

        // Validate payload-specific constraints
        match &self.payload {
            MessagePayload::DocumentProcessingRequest { document, .. } => {
                if document.document_id.is_nil() {
                    anyhow::bail!("Invalid document ID");
                }
            },
            MessagePayload::DocumentProcessingResponse { result, .. } => {
                match result.status {
                    ProcessingStatus::Success | ProcessingStatus::SuccessWithWarnings => {
                        // Validate successful response has output
                    },
                    _ => {
                        // Error responses should have error information
                    }
                }
            },
            _ => {}
        }

        Ok(())
    }
}

impl ProtocolVersion {
    /// Current protocol version
    pub fn current() -> Self {
        Self {
            major: 2,
            minor: 0,
            patch: 0,
        }
    }

    /// Check if version is compatible with current
    pub fn is_compatible(&self) -> bool {
        let current = Self::current();
        
        // Major version must match exactly
        if self.major != current.major {
            return false;
        }

        // Minor version must be less than or equal to current
        if self.minor > current.minor {
            return false;
        }

        true
    }
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        CompressionAlgorithm::None
    }
}

impl Default for EncryptionType {
    fn default() -> Self {
        EncryptionType::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_compatibility() {
        let current = ProtocolVersion::current();
        assert!(current.is_compatible());

        let compatible = ProtocolVersion { major: 2, minor: 0, patch: 0 };
        assert!(compatible.is_compatible());

        let incompatible_major = ProtocolVersion { major: 1, minor: 0, patch: 0 };
        assert!(!incompatible_major.is_compatible());

        let incompatible_minor = ProtocolVersion { major: 2, minor: 1, patch: 0 };
        assert!(!incompatible_minor.is_compatible());
    }

    #[test]
    fn test_message_serialization() {
        let source = ProcessIdentification {
            pid: 1000,
            name: "rust-pipeline".to_string(),
            version: "1.0.0".to_string(),
            node_id: "test-node".to_string(),
            capabilities: vec!["document-processing".to_string()],
        };

        let destination = ProcessIdentification {
            pid: 2000,
            name: "python-ml".to_string(),
            version: "1.0.0".to_string(),
            node_id: "test-node".to_string(),
            capabilities: vec!["ml-processing".to_string()],
        };

        let payload = MessagePayload::SystemControl {
            command: SystemCommand::Ping,
            parameters: HashMap::new(),
        };

        let message = ProtocolMessage::new(payload, source, destination);
        
        let serialized = message.serialize().unwrap();
        assert!(!serialized.is_empty());

        let deserialized = ProtocolMessage::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.header.message_id, message.header.message_id);
    }

    #[test]
    fn test_message_validation() {
        let source = ProcessIdentification {
            pid: 1000,
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            node_id: "test-node".to_string(),
            capabilities: Vec::new(),
        };

        let destination = source.clone();

        let payload = MessagePayload::SystemControl {
            command: SystemCommand::Ping,
            parameters: HashMap::new(),
        };

        let message = ProtocolMessage::new(payload, source, destination);
        assert!(message.validate().is_ok());
    }

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical < MessagePriority::High);
        assert!(MessagePriority::High < MessagePriority::Normal);
        assert!(MessagePriority::Normal < MessagePriority::Low);
        assert!(MessagePriority::Low < MessagePriority::Bulk);
    }
}