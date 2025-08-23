/*!
# MCP Protocol Implementation

Core protocol definitions and message handling for the Model Context Protocol.
Provides standardized communication patterns and error handling.
*/

use crate::{PipelineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// MCP Protocol version
pub const MCP_VERSION: &str = "1.0.0";

/// Standard MCP message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpEnvelope {
    pub version: String,
    pub message_id: Uuid,
    pub timestamp: u64,
    pub sender_id: String,
    pub recipient_id: Option<String>,
    pub message_type: String,
    pub payload: serde_json::Value,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl McpEnvelope {
    /// Create a new MCP message envelope
    pub fn new(sender_id: String, message_type: String, payload: serde_json::Value) -> Self {
        Self {
            version: MCP_VERSION.to_string(),
            message_id: Uuid::new_v4(),
            timestamp: current_timestamp(),
            sender_id,
            recipient_id: None,
            message_type,
            payload,
            metadata: None,
        }
    }

    /// Create a response envelope
    pub fn create_response(&self, payload: serde_json::Value) -> Self {
        Self {
            version: self.version.clone(),
            message_id: Uuid::new_v4(),
            timestamp: current_timestamp(),
            sender_id: self.recipient_id.clone().unwrap_or_default(),
            recipient_id: Some(self.sender_id.clone()),
            message_type: format!("{}_response", self.message_type),
            payload,
            metadata: None,
        }
    }

    /// Add metadata to the envelope
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key, value);
        self
    }

    /// Set recipient
    pub fn to(mut self, recipient_id: String) -> Self {
        self.recipient_id = Some(recipient_id);
        self
    }
}

/// MCP handshake information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpHandshake {
    pub protocol_version: String,
    pub client_info: ClientInfo,
    pub capabilities: Vec<String>,
    pub supported_message_types: Vec<String>,
    pub authentication: Option<AuthenticationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub client_id: String,
    pub client_type: String,
    pub client_version: String,
    pub platform: String,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationInfo {
    pub auth_type: String,
    pub credentials: Option<serde_json::Value>,
    pub token: Option<String>,
    pub expires_at: Option<u64>,
}

/// MCP acknowledgment message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAcknowledgment {
    pub message_id: Uuid,
    pub status: AckStatus,
    pub timestamp: u64,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AckStatus {
    Received,
    Processing,
    Completed,
    Error(String),
}

/// MCP batch message for multiple operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBatch {
    pub batch_id: Uuid,
    pub messages: Vec<McpEnvelope>,
    pub require_ordered_processing: bool,
    pub timeout_seconds: Option<u64>,
}

/// MCP streaming message for large data transfers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpStream {
    pub stream_id: Uuid,
    pub sequence_number: u64,
    pub is_last_chunk: bool,
    pub chunk_data: Vec<u8>,
    pub checksum: Option<String>,
}

/// Protocol-level error codes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProtocolErrorCode {
    InvalidVersion = 1000,
    UnsupportedMessageType = 1001,
    InvalidMessage = 1002,
    AuthenticationRequired = 1003,
    AuthenticationFailed = 1004,
    InsufficientPrivileges = 1005,
    RateLimitExceeded = 1006,
    ResourceNotFound = 1007,
    ResourceConflict = 1008,
    InternalError = 1009,
    ServiceUnavailable = 1010,
    Timeout = 1011,
    PayloadTooLarge = 1012,
}

impl ProtocolErrorCode {
    pub fn to_error_message(&self) -> &'static str {
        match self {
            Self::InvalidVersion => "Unsupported protocol version",
            Self::UnsupportedMessageType => "Unsupported message type",
            Self::InvalidMessage => "Invalid message format",
            Self::AuthenticationRequired => "Authentication required",
            Self::AuthenticationFailed => "Authentication failed",
            Self::InsufficientPrivileges => "Insufficient privileges",
            Self::RateLimitExceeded => "Rate limit exceeded",
            Self::ResourceNotFound => "Resource not found",
            Self::ResourceConflict => "Resource conflict",
            Self::InternalError => "Internal server error",
            Self::ServiceUnavailable => "Service unavailable",
            Self::Timeout => "Operation timeout",
            Self::PayloadTooLarge => "Payload too large",
        }
    }
}

/// Protocol message validator
pub struct MessageValidator;

impl MessageValidator {
    /// Validate MCP envelope structure
    pub fn validate_envelope(envelope: &McpEnvelope) -> Result<()> {
        // Check version
        if envelope.version != MCP_VERSION {
            return Err(PipelineError::Mcp(format!(
                "Unsupported protocol version: {} (expected: {})",
                envelope.version, MCP_VERSION
            )));
        }

        // Check required fields
        if envelope.sender_id.is_empty() {
            return Err(PipelineError::Mcp("Sender ID is required".to_string()));
        }

        if envelope.message_type.is_empty() {
            return Err(PipelineError::Mcp("Message type is required".to_string()));
        }

        // Check timestamp (within reasonable bounds)
        let current_time = current_timestamp();
        if envelope.timestamp > current_time + 300 || envelope.timestamp < current_time - 3600 {
            return Err(PipelineError::Mcp("Invalid timestamp".to_string()));
        }

        Ok(())
    }

    /// Validate handshake message
    pub fn validate_handshake(handshake: &McpHandshake) -> Result<()> {
        if handshake.protocol_version != MCP_VERSION {
            return Err(PipelineError::Mcp(format!(
                "Unsupported protocol version in handshake: {}",
                handshake.protocol_version
            )));
        }

        if handshake.client_info.client_id.is_empty() {
            return Err(PipelineError::Mcp(
                "Client ID is required in handshake".to_string(),
            ));
        }

        if handshake.capabilities.is_empty() {
            return Err(PipelineError::Mcp(
                "At least one capability is required".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate batch message
    pub fn validate_batch(batch: &McpBatch) -> Result<()> {
        if batch.messages.is_empty() {
            return Err(PipelineError::Mcp("Batch cannot be empty".to_string()));
        }

        if batch.messages.len() > 100 {
            return Err(PipelineError::Mcp(
                "Batch size exceeds maximum limit (100)".to_string(),
            ));
        }

        // Validate each message in the batch
        for message in &batch.messages {
            Self::validate_envelope(message)?;
        }

        Ok(())
    }
}

/// Protocol message router
pub struct MessageRouter {
    handlers: HashMap<String, Box<dyn MessageHandler + Send + Sync>>,
}

/// Message handler trait for protocol-level routing
pub trait MessageHandler {
    fn can_handle(&self, message_type: &str) -> bool;
    fn handle(&self, envelope: McpEnvelope) -> Result<Option<McpEnvelope>>;
}

impl MessageRouter {
    /// Create new message router
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a message handler
    pub fn register_handler<H>(&mut self, handler: H)
    where
        H: MessageHandler + Send + Sync + 'static,
    {
        let handler_key = format!("handler_{}", self.handlers.len());
        self.handlers.insert(handler_key, Box::new(handler));
    }

    /// Route message to appropriate handler
    pub fn route_message(&self, envelope: McpEnvelope) -> Result<Option<McpEnvelope>> {
        // Validate the envelope first
        MessageValidator::validate_envelope(&envelope)?;

        // Find handler for this message type
        for handler in self.handlers.values() {
            if handler.can_handle(&envelope.message_type) {
                return handler.handle(envelope);
            }
        }

        // No handler found
        Err(PipelineError::Mcp(format!(
            "No handler found for message type: {}",
            envelope.message_type
        )))
    }
}

/// Default protocol handlers
pub struct HandshakeHandler;

impl MessageHandler for HandshakeHandler {
    fn can_handle(&self, message_type: &str) -> bool {
        message_type == "handshake"
    }

    fn handle(&self, envelope: McpEnvelope) -> Result<Option<McpEnvelope>> {
        // Parse handshake payload
        let handshake: McpHandshake = serde_json::from_value(envelope.payload.clone())
            .map_err(|e| PipelineError::Mcp(format!("Invalid handshake payload: {}", e)))?;

        // Validate handshake
        MessageValidator::validate_handshake(&handshake)?;

        // Create response
        let response_payload = serde_json::json!({
            "status": "accepted",
            "server_capabilities": [
                "task_orchestration",
                "pipeline_management",
                "performance_monitoring",
                "load_balancing"
            ],
            "server_info": {
                "server_id": "mcp_host_server",
                "server_version": env!("CARGO_PKG_VERSION"),
                "protocol_version": MCP_VERSION
            }
        });

        Ok(Some(envelope.create_response(response_payload)))
    }
}

pub struct PingHandler;

impl MessageHandler for PingHandler {
    fn can_handle(&self, message_type: &str) -> bool {
        message_type == "ping"
    }

    fn handle(&self, envelope: McpEnvelope) -> Result<Option<McpEnvelope>> {
        let response_payload = serde_json::json!({
            "status": "pong",
            "timestamp": current_timestamp(),
            "server_id": "mcp_host_server"
        });

        Ok(Some(envelope.create_response(response_payload)))
    }
}

/// Protocol utilities
pub struct ProtocolUtils;

impl ProtocolUtils {
    /// Create error response envelope
    pub fn create_error_response(
        original: &McpEnvelope,
        error_code: ProtocolErrorCode,
        details: Option<String>,
    ) -> McpEnvelope {
        let payload = serde_json::json!({
            "error": {
                "code": error_code as u32,
                "message": error_code.to_error_message(),
                "details": details
            }
        });

        original.create_response(payload)
    }

    /// Create success response envelope
    pub fn create_success_response(
        original: &McpEnvelope,
        data: Option<serde_json::Value>,
    ) -> McpEnvelope {
        let payload = serde_json::json!({
            "success": true,
            "data": data
        });

        original.create_response(payload)
    }

    /// Calculate message checksum
    pub fn calculate_checksum(data: &[u8]) -> String {
        // Use deterministic CRC32 hash for consistent results across platforms
        let checksum = crc32fast::hash(data);
        format!("{:08x}", checksum)
    }

    /// Serialize envelope to JSON
    pub fn serialize_envelope(envelope: &McpEnvelope) -> Result<String> {
        serde_json::to_string(envelope)
            .map_err(|e| PipelineError::Mcp(format!("Failed to serialize envelope: {}", e)))
    }

    /// Deserialize envelope from JSON
    pub fn deserialize_envelope(json: &str) -> Result<McpEnvelope> {
        serde_json::from_str(json)
            .map_err(|e| PipelineError::Mcp(format!("Failed to deserialize envelope: {}", e)))
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_creation() {
        let payload = serde_json::json!({"test": "data"});
        let envelope = McpEnvelope::new(
            "test_sender".to_string(),
            "test_message".to_string(),
            payload,
        );

        assert_eq!(envelope.sender_id, "test_sender");
        assert_eq!(envelope.message_type, "test_message");
        assert_eq!(envelope.version, MCP_VERSION);
    }

    #[test]
    fn test_response_creation() {
        let payload = serde_json::json!({"test": "data"});
        let envelope = McpEnvelope::new("sender".to_string(), "request".to_string(), payload)
            .to("recipient".to_string());

        let response_payload = serde_json::json!({"status": "ok"});
        let response = envelope.create_response(response_payload);

        assert_eq!(response.message_type, "request_response");
        assert_eq!(response.sender_id, "recipient");
        assert_eq!(response.recipient_id, Some("sender".to_string()));
    }

    #[test]
    fn test_message_validation() {
        let payload = serde_json::json!({"test": "data"});
        let envelope = McpEnvelope::new("sender".to_string(), "test".to_string(), payload);

        let result = MessageValidator::validate_envelope(&envelope);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_version() {
        let mut envelope = McpEnvelope::new(
            "sender".to_string(),
            "test".to_string(),
            serde_json::json!({}),
        );
        envelope.version = "0.5.0".to_string();

        let result = MessageValidator::validate_envelope(&envelope);
        assert!(result.is_err());
    }

    #[test]
    fn test_handshake_validation() {
        let handshake = McpHandshake {
            protocol_version: MCP_VERSION.to_string(),
            client_info: ClientInfo {
                client_id: "test_client".to_string(),
                client_type: "rust_core".to_string(),
                client_version: "1.0.0".to_string(),
                platform: "linux".to_string(),
                features: vec!["batching".to_string()],
            },
            capabilities: vec!["task_execution".to_string()],
            supported_message_types: vec!["task_submit".to_string()],
            authentication: None,
        };

        let result = MessageValidator::validate_handshake(&handshake);
        assert!(result.is_ok());
    }

    #[test]
    fn test_message_router() {
        let mut router = MessageRouter::new();
        router.register_handler(HandshakeHandler);

        let handshake = McpHandshake {
            protocol_version: MCP_VERSION.to_string(),
            client_info: ClientInfo {
                client_id: "test_client".to_string(),
                client_type: "rust_core".to_string(),
                client_version: "1.0.0".to_string(),
                platform: "linux".to_string(),
                features: vec![],
            },
            capabilities: vec!["test".to_string()],
            supported_message_types: vec!["handshake".to_string()],
            authentication: None,
        };

        let envelope = McpEnvelope::new(
            "client".to_string(),
            "handshake".to_string(),
            serde_json::to_value(handshake).unwrap(),
        );

        let result = router.route_message(envelope);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_checksum_calculation() {
        let data = b"hello world";
        let checksum = ProtocolUtils::calculate_checksum(data);
        assert_eq!(checksum, "0d4a1185");
    }
}
