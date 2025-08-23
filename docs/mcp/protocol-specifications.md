# MCP Protocol Specifications for Hybrid Rust-Python Pipeline

**System Architect: Agent 1**
**Version**: 2024-11-05 MCP Standard
**Scope**: Phase 1 Implementation

## Protocol Overview

This document defines the Model Context Protocol (MCP) specifications for the hybrid Rust-Python document processing pipeline, ensuring standardized communication while maintaining high-performance characteristics.

## Message Transport Layer

### 1. Transport Protocol
- **Primary**: Local Unix Domain Sockets
- **Fallback**: TCP/IP loopback (127.0.0.1)
- **Format**: JSON-RPC 2.0 with MCP extensions
- **Encoding**: UTF-8
- **Max Message Size**: 10MB (configurable)

### 2. Connection Management
```rust
#[derive(Debug, Clone)]
pub struct MCPConnectionConfig {
    pub socket_path: PathBuf,
    pub tcp_port: Option<u16>,
    pub connection_timeout: Duration,
    pub keepalive_interval: Duration,
    pub max_retries: u32,
    pub buffer_size: usize,
}

impl Default for MCPConnectionConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/claude_flow_mcp.sock"),
            tcp_port: Some(8700),
            connection_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(30),
            max_retries: 3,
            buffer_size: 1024 * 1024, // 1MB
        }
    }
}
```

## Core Protocol Messages

### 1. Initialization Handshake

#### Client → Server: Initialize
```json
{
  "jsonrpc": "2.0",
  "id": "init_001",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "sampling": {},
      "prompts": {},
      "resources": {},
      "tools": {}
    },
    "clientInfo": {
      "name": "claude-flow-python-ml",
      "version": "1.0.0"
    },
    "serverInfo": {
      "name": "claude-flow-rust-core",
      "version": "1.0.0"
    }
  }
}
```

#### Server → Client: Initialize Response
```json
{
  "jsonrpc": "2.0",
  "id": "init_001",
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "document_processing": {
        "supports_batch": true,
        "supports_streaming": true,
        "max_document_size": 10485760
      },
      "quality_validation": {
        "threshold_range": [0.0, 1.0],
        "supports_custom_criteria": true
      },
      "performance_monitoring": {
        "real_time_metrics": true,
        "historical_data": true
      }
    },
    "serverInfo": {
      "name": "claude-flow-rust-core",
      "version": "1.0.0",
      "implementation": "Rust tokio-based"
    }
  }
}
```

### 2. Service Discovery

#### Client → Server: List Services
```json
{
  "jsonrpc": "2.0",
  "id": "discovery_001",
  "method": "services/list",
  "params": {}
}
```

#### Server → Client: Services Response
```json
{
  "jsonrpc": "2.0",
  "id": "discovery_001",
  "result": {
    "services": [
      {
        "name": "document_processing",
        "version": "1.0.0",
        "description": "High-performance document processing with M3 optimization",
        "methods": [
          "process_document",
          "process_batch",
          "validate_quality"
        ],
        "status": "available",
        "load": 0.25,
        "max_concurrent": 16
      },
      {
        "name": "model_inference",
        "version": "1.0.0", 
        "description": "MLX-accelerated Qwen3 model inference",
        "methods": [
          "generate_qa_pairs",
          "assess_semantic_quality",
          "get_model_info"
        ],
        "status": "available",
        "load": 0.40,
        "models": ["qwen3-1.7b", "qwen3-7b", "qwen3-30b"]
      }
    ]
  }
}
```

## Document Processing Protocol

### 1. Single Document Processing

#### Request Schema
```json
{
  "jsonrpc": "2.0",
  "id": "proc_001",
  "method": "document_processing/process",
  "params": {
    "document": {
      "id": "doc_12345",
      "content": "Document content...",
      "metadata": {
        "source": "3gpp_spec",
        "feature": "handover_procedures",
        "complexity": 0.7
      }
    },
    "processing_options": {
      "quality_threshold": 0.8,
      "qa_target_count": 5,
      "model_preference": "qwen3-7b",
      "priority": "high",
      "timeout_seconds": 60
    },
    "shared_memory_hint": {
      "enabled": true,
      "region": "document_data",
      "offset": 1048576
    }
  }
}
```

#### Response Schema
```json
{
  "jsonrpc": "2.0",
  "id": "proc_001",
  "result": {
    "document_id": "doc_12345",
    "success": true,
    "processing_time_ms": 2340,
    "qa_pairs": [
      {
        "question": "What triggers an LTE handover?",
        "answer": "An LTE handover is triggered when...",
        "quality_score": 0.85,
        "technical_accuracy": 0.90
      }
    ],
    "semantic_quality": {
      "coherence_score": 0.88,
      "relevance_score": 0.92,
      "technical_accuracy_score": 0.89,
      "diversity_score": 0.75,
      "overall_score": 0.86
    },
    "processing_metadata": {
      "model_used": "qwen3-7b",
      "inference_time_ms": 1850,
      "tokens_processed": 2048,
      "memory_used_mb": 1250,
      "gpu_utilization": 0.68
    },
    "shared_memory_info": {
      "region": "document_data",
      "offset": 1048576,
      "size": 4096
    }
  }
}
```

### 2. Batch Processing

#### Request Schema
```json
{
  "jsonrpc": "2.0",
  "id": "batch_001",
  "method": "document_processing/batch",
  "params": {
    "batch_id": "batch_12345",
    "documents": [
      {
        "id": "doc_001",
        "shared_memory_ref": {
          "region": "document_data",
          "offset": 0,
          "size": 10240
        }
      }
    ],
    "processing_options": {
      "parallel_processing": true,
      "max_concurrent": 4,
      "quality_threshold": 0.75,
      "timeout_seconds": 300
    }
  }
}
```

### 3. Real-time Status Updates

#### Server → Client: Progress Notification
```json
{
  "jsonrpc": "2.0",
  "method": "processing/progress",
  "params": {
    "request_id": "proc_001",
    "stage": "model_inference",
    "progress": 0.65,
    "eta_seconds": 15,
    "current_metrics": {
      "memory_usage_mb": 1180,
      "cpu_usage": 0.85,
      "gpu_usage": 0.72
    }
  }
}
```

## Quality Validation Protocol

### 1. Quality Assessment Request
```json
{
  "jsonrpc": "2.0",
  "id": "qual_001",
  "method": "quality_validation/assess",
  "params": {
    "content": {
      "qa_pairs": [...],
      "document_metadata": {...}
    },
    "validation_criteria": {
      "min_coherence": 0.8,
      "min_relevance": 0.75,
      "min_technical_accuracy": 0.85,
      "min_diversity": 0.7
    },
    "validation_mode": "comprehensive"
  }
}
```

### 2. Quality Assessment Response
```json
{
  "jsonrpc": "2.0",
  "id": "qual_001",
  "result": {
    "overall_quality": 0.84,
    "meets_criteria": true,
    "detailed_scores": {
      "coherence": 0.86,
      "relevance": 0.89,
      "technical_accuracy": 0.87,
      "diversity": 0.74
    },
    "validation_details": {
      "total_pairs_assessed": 5,
      "pairs_meeting_threshold": 4,
      "improvement_suggestions": [
        "Increase technical terminology diversity",
        "Enhance question complexity"
      ]
    }
  }
}
```

## Performance Monitoring Protocol

### 1. Real-time Metrics
```json
{
  "jsonrpc": "2.0",
  "id": "perf_001", 
  "method": "performance/get_metrics",
  "params": {
    "scope": "system",
    "time_range": "last_5_minutes",
    "include_history": false
  }
}
```

#### Response
```json
{
  "jsonrpc": "2.0",
  "id": "perf_001",
  "result": {
    "timestamp": "2025-08-23T08:06:00Z",
    "system_metrics": {
      "document_processing_rate": 28.5,
      "rust_memory_utilization_gb": 58.2,
      "python_memory_utilization_gb": 43.8,
      "shared_memory_utilization_gb": 14.2,
      "ipc_latency_p99_ms": 0.08,
      "model_inference_time_avg_ms": 1850,
      "total_throughput_docs_sec": 0.0079,
      "error_rate_percent": 0.2,
      "cpu_utilization_avg": 0.68,
      "gpu_utilization_avg": 0.72
    },
    "bottlenecks": [],
    "optimizations_active": [
      "m3_max_memory_optimization",
      "mlx_acceleration",
      "zero_copy_ipc"
    ]
  }
}
```

## Error Handling Protocol

### 1. Error Classification
```rust
#[derive(Debug, Serialize, Deserialize)]
pub enum MCPErrorCode {
    // Standard JSON-RPC errors
    ParseError = -32700,
    InvalidRequest = -32600, 
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    
    // MCP-specific errors
    ProtocolError = -32000,
    InitializationFailed = -32001,
    ServiceUnavailable = -32002,
    
    // Domain-specific errors  
    DocumentProcessingFailed = -40000,
    QualityThresholdNotMet = -40001,
    ModelInferenceFailed = -40002,
    SharedMemoryError = -40003,
    ResourceExhausted = -40004,
    TimeoutExceeded = -40005,
}
```

### 2. Error Response Format
```json
{
  "jsonrpc": "2.0",
  "id": "proc_001",
  "error": {
    "code": -40001,
    "message": "Quality threshold not met",
    "data": {
      "threshold_required": 0.8,
      "actual_score": 0.72,
      "failed_criteria": ["technical_accuracy", "coherence"],
      "suggestion": "Consider adjusting quality threshold or improving source content",
      "retry_recommended": true,
      "retry_delay_seconds": 30
    }
  }
}
```

## Health Monitoring & Circuit Breaker

### 1. Health Check Protocol
```json
{
  "jsonrpc": "2.0",
  "id": "health_001",
  "method": "system/health",
  "params": {
    "include_detailed": true
  }
}
```

#### Response
```json
{
  "jsonrpc": "2.0",
  "id": "health_001",
  "result": {
    "status": "healthy",
    "uptime_seconds": 3600,
    "services": {
      "document_processing": {
        "status": "healthy",
        "load": 0.25,
        "last_error": null,
        "response_time_p95_ms": 2.1
      },
      "model_inference": {
        "status": "degraded", 
        "load": 0.85,
        "last_error": "2025-08-23T07:45:00Z",
        "response_time_p95_ms": 3.8,
        "warning": "High load detected"
      }
    },
    "resources": {
      "memory_usage": 0.82,
      "cpu_usage": 0.68,
      "disk_usage": 0.15,
      "shared_memory_usage": 0.89
    }
  }
}
```

### 2. Circuit Breaker States
```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests  
    HALF_OPEN = "half_open" # Testing if service recovered

# Circuit breaker notification
{
  "jsonrpc": "2.0",
  "method": "circuit_breaker/state_changed",
  "params": {
    "service": "model_inference",
    "previous_state": "closed",
    "current_state": "open", 
    "failure_count": 5,
    "failure_threshold": 5,
    "reset_timeout_seconds": 60,
    "reason": "Consecutive timeout errors"
  }
}
```

## Backward Compatibility

### 1. Dual Protocol Support
- **Primary**: MCP for new features and standardized communication
- **Fallback**: Existing IPC for performance-critical operations
- **Migration**: Gradual service migration without disruption

### 2. Legacy IPC Bridge
```rust
pub struct LegacyIPCBridge {
    mcp_server: Arc<MCPServer>,
    ipc_manager: Arc<IpcManager>,
    config: BridgeConfig,
}

impl LegacyIPCBridge {
    pub async fn handle_legacy_request(
        &self,
        ipc_message: IpcMessage
    ) -> Result<IpcMessage> {
        // Convert IPC message to MCP request
        let mcp_request = self.convert_to_mcp(ipc_message)?;
        
        // Process via MCP
        let mcp_response = self.mcp_server.handle_request(mcp_request).await?;
        
        // Convert back to IPC response
        self.convert_to_ipc(mcp_response)
    }
}
```

## Implementation Considerations

### 1. Performance Optimization
- **Message Batching**: Group related operations
- **Connection Pooling**: Reuse connections efficiently
- **Lazy Serialization**: Serialize only when necessary
- **Zero-Copy Paths**: Use shared memory for large data

### 2. Security Considerations
- **Authentication**: Service-level authentication tokens
- **Authorization**: Role-based access control
- **Encryption**: TLS for sensitive data (optional for local transport)
- **Input Validation**: Strict schema validation

### 3. Monitoring Integration
```rust
pub struct MCPMetrics {
    pub requests_total: Counter,
    pub requests_duration: Histogram, 
    pub active_connections: Gauge,
    pub errors_total: Counter,
    pub message_size: Histogram,
}
```

## Testing Protocol

### 1. Unit Tests
- Message serialization/deserialization
- Error handling scenarios
- Connection management
- Service discovery

### 2. Integration Tests  
- End-to-end document processing
- Performance validation
- Failover scenarios
- Load testing

### 3. Compliance Tests
- MCP 2024-11-05 specification compliance
- JSON-RPC 2.0 compliance
- Error handling standards
- Protocol version negotiation

---

**Next Phase**: Protocol implementation begins with Agent 2 (Implementation Specialist) using these specifications as the foundation for Phase 1 development.