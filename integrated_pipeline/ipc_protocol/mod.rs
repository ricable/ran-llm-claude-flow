//! Advanced IPC Protocol Module
//! 
//! This module provides a comprehensive Inter-Process Communication protocol
//! optimized for high-throughput Rust-Python pipeline communication with
//! zero-copy semantics and M3 Max unified memory architecture support.

pub mod message_protocol;
pub mod connection_pool;

pub use message_protocol::{
    ProtocolMessage,
    MessagePayload,
    MessageHeader,
    MessagePriority,
    ProcessIdentification,
    DocumentData,
    DocumentContent,
    ProcessingOptions,
    ProcessingResult,
    ProtocolVersion,
};

pub use connection_pool::{
    ConnectionPool,
    PoolConfig as ConnectionPoolConfig,
    ConnectionLease,
    PoolStatistics,
    LoadBalancingStrategy,
    RetryConfig,
};

use anyhow::Result;
use std::sync::Arc;
use tracing::{info, error};
use uuid::Uuid;

/// Unified IPC Protocol Manager
/// Integrates message protocol with connection pooling for complete IPC solution
pub struct IpcProtocolManager {
    /// Connection pool for managing network connections
    connection_pool: Arc<ConnectionPool>,
    /// Protocol configuration
    config: IpcProtocolConfig,
    /// Process identification for this instance
    process_id: ProcessIdentification,
}

/// Configuration for IPC Protocol Manager
#[derive(Debug, Clone)]
pub struct IpcProtocolConfig {
    /// Connection pool configuration
    pub connection_pool: ConnectionPoolConfig,
    /// Message serialization preferences
    pub message_serialization: MessageSerializationConfig,
    /// Protocol optimization settings
    pub optimizations: ProtocolOptimizations,
    /// Security settings
    pub security: ProtocolSecurity,
}

/// Message serialization configuration
#[derive(Debug, Clone)]
pub struct MessageSerializationConfig {
    /// Enable message compression
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Enable message validation
    pub enable_validation: bool,
    /// Enable checksum verification
    pub enable_checksums: bool,
}

/// Protocol optimization settings
#[derive(Debug, Clone)]
pub struct ProtocolOptimizations {
    /// Enable message batching
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable connection reuse
    pub enable_connection_reuse: bool,
    /// Enable protocol-level caching
    pub enable_protocol_caching: bool,
}

/// Protocol security settings
#[derive(Debug, Clone)]
pub struct ProtocolSecurity {
    /// Enable message encryption
    pub enable_encryption: bool,
    /// Enable message authentication
    pub enable_authentication: bool,
    /// Enable access control
    pub enable_access_control: bool,
    /// Authentication token (if used)
    pub auth_token: Option<String>,
}

impl IpcProtocolManager {
    /// Create new IPC Protocol Manager
    pub async fn new(config: IpcProtocolConfig, process_id: ProcessIdentification) -> Result<Self> {
        info!("Initializing IPC Protocol Manager for process: {}", process_id.name);

        // Initialize connection pool
        let connection_pool = Arc::new(ConnectionPool::new(config.connection_pool.clone()).await?);

        let manager = Self {
            connection_pool,
            config,
            process_id,
        };

        info!("IPC Protocol Manager initialized successfully");
        Ok(manager)
    }

    /// Send document processing request
    pub async fn send_document_processing_request(
        &self,
        document: DocumentData,
        processing_options: ProcessingOptions,
        endpoints: &[String],
    ) -> Result<ProcessingResult> {
        // Create message payload
        let payload = MessagePayload::DocumentProcessingRequest {
            document,
            processing_options,
            callback_info: None,
        };

        // Create destination process identification (would be determined by endpoint)
        let destination = ProcessIdentification {
            pid: 0, // Unknown
            name: "python-ml-processor".to_string(),
            version: "1.0.0".to_string(),
            node_id: "ml-node".to_string(),
            capabilities: vec!["ml-processing".to_string(), "qa-generation".to_string()],
        };

        // Create protocol message
        let message = ProtocolMessage::new(payload, self.process_id.clone(), destination);

        // Send message and wait for response
        let response_message = self.connection_pool.send_message(&message, endpoints).await?;

        // Extract processing result from response
        match response_message.payload {
            MessagePayload::DocumentProcessingResponse { result, error, .. } => {
                if let Some(error) = error {
                    anyhow::bail!("Processing failed: {} - {}", error.error_code, error.message);
                }
                Ok(result)
            },
            _ => {
                anyhow::bail!("Unexpected response message type");
            }
        }
    }

    /// Send system control command
    pub async fn send_system_control(
        &self,
        command: message_protocol::SystemCommand,
        endpoints: &[String],
    ) -> Result<()> {
        let payload = MessagePayload::SystemControl {
            command,
            parameters: std::collections::HashMap::new(),
        };

        let destination = ProcessIdentification {
            pid: 0,
            name: "system-controller".to_string(),
            version: "1.0.0".to_string(),
            node_id: "control-node".to_string(),
            capabilities: vec!["system-control".to_string()],
        };

        let message = ProtocolMessage::new(payload, self.process_id.clone(), destination);
        let _response = self.connection_pool.send_message(&message, endpoints).await?;

        info!("System control command sent successfully");
        Ok(())
    }

    /// Perform health check on endpoints
    pub async fn health_check(&self, endpoints: &[String]) -> Result<message_protocol::HealthMetrics> {
        let payload = MessagePayload::HealthCheck {
            check_type: message_protocol::HealthCheckType::SystemHealth,
            metrics: None,
        };

        let destination = ProcessIdentification {
            pid: 0,
            name: "health-monitor".to_string(),
            version: "1.0.0".to_string(),
            node_id: "monitor-node".to_string(),
            capabilities: vec!["health-monitoring".to_string()],
        };

        let message = ProtocolMessage::new(payload, self.process_id.clone(), destination);
        let response = self.connection_pool.send_message(&message, endpoints).await?;

        match response.payload {
            MessagePayload::HealthCheck { metrics: Some(metrics), .. } => Ok(metrics),
            _ => anyhow::bail!("Invalid health check response"),
        }
    }

    /// Get connection pool statistics
    pub async fn get_connection_statistics(&self) -> connection_pool::PoolStatistics {
        self.connection_pool.get_statistics().await
    }

    /// Shutdown the protocol manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down IPC Protocol Manager");
        self.connection_pool.shutdown().await?;
        info!("IPC Protocol Manager shutdown complete");
        Ok(())
    }
}

impl Default for IpcProtocolConfig {
    fn default() -> Self {
        Self {
            connection_pool: ConnectionPoolConfig::default(),
            message_serialization: MessageSerializationConfig {
                enable_compression: true,
                compression_threshold: 1024, // Compress messages > 1KB
                enable_validation: true,
                enable_checksums: true,
            },
            optimizations: ProtocolOptimizations {
                enable_batching: true,
                max_batch_size: 32,
                batch_timeout_ms: 10,
                enable_connection_reuse: true,
                enable_protocol_caching: true,
            },
            security: ProtocolSecurity {
                enable_encryption: false, // Disabled for local IPC by default
                enable_authentication: false,
                enable_access_control: false,
                auth_token: None,
            },
        }
    }
}

impl IpcProtocolConfig {
    /// Create configuration optimized for M3 Max performance
    pub fn for_m3_max() -> Self {
        Self {
            connection_pool: ConnectionPoolConfig {
                max_total_connections: 128,
                max_connections_per_endpoint: 32,
                min_connections_per_endpoint: 4,
                idle_timeout_seconds: 300,
                connect_timeout_seconds: 10,
                health_check_interval_seconds: 30,
                load_balancing_strategy: LoadBalancingStrategy::LeastConnections,
                enable_pooling: true,
                ..ConnectionPoolConfig::default()
            },
            message_serialization: MessageSerializationConfig {
                enable_compression: true,
                compression_threshold: 4096, // Compress larger messages
                enable_validation: true,
                enable_checksums: true,
            },
            optimizations: ProtocolOptimizations {
                enable_batching: true,
                max_batch_size: 64, // Larger batches for better throughput
                batch_timeout_ms: 5, // Lower timeout for better latency
                enable_connection_reuse: true,
                enable_protocol_caching: true,
            },
            security: ProtocolSecurity {
                enable_encryption: false,
                enable_authentication: false,
                enable_access_control: false,
                auth_token: None,
            },
        }
    }

    /// Create configuration for high-security environments
    pub fn for_secure_deployment() -> Self {
        Self {
            connection_pool: ConnectionPoolConfig::default(),
            message_serialization: MessageSerializationConfig {
                enable_compression: false, // Disable compression for security
                compression_threshold: usize::MAX,
                enable_validation: true,
                enable_checksums: true,
            },
            optimizations: ProtocolOptimizations {
                enable_batching: false, // Disable batching for security
                max_batch_size: 1,
                batch_timeout_ms: 0,
                enable_connection_reuse: false,
                enable_protocol_caching: false,
            },
            security: ProtocolSecurity {
                enable_encryption: true,
                enable_authentication: true,
                enable_access_control: true,
                auth_token: None, // Would be set externally
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipc_protocol_manager_creation() {
        let config = IpcProtocolConfig::default();
        let process_id = ProcessIdentification {
            pid: 1000,
            name: "test-process".to_string(),
            version: "1.0.0".to_string(),
            node_id: "test-node".to_string(),
            capabilities: vec!["testing".to_string()],
        };

        let manager = IpcProtocolManager::new(config, process_id).await.unwrap();
        let stats = manager.get_connection_statistics().await;
        
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = IpcProtocolConfig::default();
        assert!(config.message_serialization.enable_compression);
        assert!(config.optimizations.enable_batching);
        assert!(!config.security.enable_encryption);
    }

    #[test]
    fn test_m3_max_config() {
        let config = IpcProtocolConfig::for_m3_max();
        assert_eq!(config.connection_pool.max_total_connections, 128);
        assert_eq!(config.optimizations.max_batch_size, 64);
    }

    #[test]
    fn test_secure_config() {
        let config = IpcProtocolConfig::for_secure_deployment();
        assert!(config.security.enable_encryption);
        assert!(!config.message_serialization.enable_compression);
        assert!(!config.optimizations.enable_batching);
    }
}