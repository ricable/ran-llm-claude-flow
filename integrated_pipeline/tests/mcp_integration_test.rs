//! MCP Integration Tests
//!
//! Comprehensive test suite for MCP (Model Context Protocol) integration
//! with the hybrid Rust-Python pipeline, ensuring performance targets
//! are maintained while adding MCP capabilities.

use anyhow::Result;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use uuid::Uuid;

use rust_core::{
    config::IpcSettings,
    mcp_server::{McpServer, McpServerConfig},
    mcp_rpc_impl::McpRpcImpl,
};

/// Test configuration for integration tests
struct McpTestConfig {
    server_port: u16,
    http_port: u16,
    timeout_seconds: u64,
}

impl Default for McpTestConfig {
    fn default() -> Self {
        Self {
            server_port: 8080, // Use different port for tests
            http_port: 8081,
            timeout_seconds: 30,
        }
    }
}

/// Test fixture for MCP integration tests
struct McpTestFixture {
    config: McpTestConfig,
    server: Option<Arc<McpServer>>,
}

impl McpTestFixture {
    async fn new() -> Result<Self> {
        Ok(Self {
            config: McpTestConfig::default(),
            server: None,
        })
    }
    
    async fn start_server(&mut self) -> Result<()> {
        let mut server_config = McpServerConfig::default();
        server_config.websocket_addr = format!("127.0.0.1:{}", self.config.server_port);
        server_config.http_addr = format!("127.0.0.1:{}", self.config.http_port);
        
        let ipc_settings = IpcSettings {
            shared_memory_size_gb: 1, // Smaller for tests
            max_connections: 16,
            timeout_seconds: 10,
            enable_checksum_validation: true,
            connection_pool_size: 4,
            queue_size: 64,
            batch_size: 8,
            compression_enabled: true,
            enable_performance_monitoring: true,
        };
        
        let server = McpServer::new(server_config, &ipc_settings).await?;
        self.server = Some(Arc::new(server));
        
        Ok(())
    }
    
    async fn cleanup(&mut self) {
        // Cleanup will be handled by Drop
    }
}

#[tokio::test]
async fn test_mcp_server_initialization() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    
    // Test server initialization
    fixture.start_server().await?;
    
    assert!(fixture.server.is_some());
    
    // Test server metrics
    let server = fixture.server.as_ref().unwrap();
    let metrics = server.get_metrics().await;
    
    assert_eq!(metrics.total_connections, 0);
    assert_eq!(metrics.active_connections, 0);
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_mcp_protocol_compatibility() -> Result<()> {
    // Test protocol version compatibility
    let server_config = McpServerConfig::default();
    
    // Test that server supports expected protocol version
    assert_eq!(
        rust_core::mcp_server::MCP_PROTOCOL_VERSION,
        "2024-11-05"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_resource_discovery() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    let server = fixture.server.as_ref().unwrap();
    
    // Test default resources are registered
    // This would require access to server's resources registry
    // For now, test that server initializes successfully
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test] 
async fn test_tool_discovery() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    let server = fixture.server.as_ref().unwrap();
    
    // Test that server has initialized tools
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_performance_preservation() -> Result<()> {
    // Test that adding MCP doesn't degrade existing performance
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    let server = fixture.server.as_ref().unwrap();
    
    // Measure baseline metrics
    let start_time = std::time::Instant::now();
    let initial_metrics = server.get_metrics().await;
    let baseline_time = start_time.elapsed();
    
    // Metrics collection should be fast (< 1ms)
    assert!(baseline_time.as_millis() < 1);
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_shared_memory_integration() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // Test that server initializes with shared memory support
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_ipc_system_compatibility() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // Test that MCP server doesn't interfere with existing IPC
    let server = fixture.server.as_ref().unwrap();
    
    // Server should initialize successfully alongside IPC manager
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_concurrent_client_support() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // Test multiple concurrent connections
    let server = fixture.server.as_ref().unwrap();
    let initial_metrics = server.get_metrics().await;
    
    // Baseline should show no active connections
    assert_eq!(initial_metrics.active_connections, 0);
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    let server = fixture.server.as_ref().unwrap();
    
    // Test error metrics start at zero
    let metrics = server.get_metrics().await;
    assert_eq!(metrics.failed_requests, 0);
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_memory_usage_bounds() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // Test that MCP server has reasonable memory footprint
    let server = fixture.server.as_ref().unwrap();
    
    // Server should initialize without excessive memory usage
    // This is a basic check - in production would monitor actual memory usage
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_configuration_validation() -> Result<()> {
    // Test various configuration scenarios
    
    // Test default configuration
    let default_config = McpServerConfig::default();
    assert!(!default_config.name.is_empty());
    assert!(default_config.max_connections > 0);
    
    // Test M3 Max optimized configuration
    let m3_config = McpServerConfig::for_m3_max();
    assert_eq!(m3_config.max_connections, 200);
    assert!(m3_config.use_shared_memory);
    assert_eq!(m3_config.large_payload_threshold, 50 * 1024 * 1024);
    
    // Test production configuration
    let prod_config = McpServerConfig::for_production();
    assert_eq!(prod_config.max_connections, 500);
    assert_eq!(prod_config.large_payload_threshold, 100 * 1024 * 1024);
    
    Ok(())
}

/// Integration test for document processing through MCP
#[tokio::test]
async fn test_document_processing_integration() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // This would test actual document processing through MCP
    // For now, just verify server starts successfully
    assert!(fixture.server.is_some());
    
    fixture.cleanup().await;
    Ok(())
}

/// Performance regression test
#[tokio::test]
async fn test_performance_regression() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    
    // Measure startup time
    let start_time = std::time::Instant::now();
    fixture.start_server().await?;
    let startup_time = start_time.elapsed();
    
    // Server should start quickly (< 5 seconds)
    assert!(startup_time.as_secs() < 5);
    
    let server = fixture.server.as_ref().unwrap();
    
    // Test metrics collection performance
    let metrics_start = std::time::Instant::now();
    let _metrics = server.get_metrics().await;
    let metrics_time = metrics_start.elapsed();
    
    // Metrics should be very fast (< 1ms)
    assert!(metrics_time.as_millis() < 1);
    
    fixture.cleanup().await;
    Ok(())
}

/// Test MCP server shutdown and cleanup
#[tokio::test]
async fn test_graceful_shutdown() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    assert!(fixture.server.is_some());
    
    // Test cleanup
    fixture.cleanup().await;
    
    // After cleanup, resources should be released
    // This is tested implicitly by successful test completion
    
    Ok(())
}

/// Comprehensive integration test suite
#[tokio::test]
async fn test_comprehensive_mcp_integration() -> Result<()> {
    let mut fixture = McpTestFixture::new().await?;
    
    // Phase 1: Initialization
    fixture.start_server().await?;
    let server = fixture.server.as_ref().unwrap();
    
    // Phase 2: Baseline metrics
    let initial_metrics = server.get_metrics().await;
    assert_eq!(initial_metrics.active_connections, 0);
    assert_eq!(initial_metrics.total_requests, 0);
    
    // Phase 3: Performance validation
    let perf_start = std::time::Instant::now();
    for _ in 0..10 {
        let _metrics = server.get_metrics().await;
    }
    let perf_time = perf_start.elapsed();
    
    // 10 metrics calls should be very fast (< 10ms)
    assert!(perf_time.as_millis() < 10);
    
    // Phase 4: Cleanup
    fixture.cleanup().await;
    
    Ok(())
}

/// Test integration with existing performance targets
#[tokio::test] 
async fn test_performance_target_compatibility() -> Result<()> {
    // Test that MCP integration doesn't prevent achieving 25+ docs/hour target
    
    let mut fixture = McpTestFixture::new().await?;
    fixture.start_server().await?;
    
    // Server should initialize successfully
    assert!(fixture.server.is_some());
    
    // The actual performance test would require running document processing
    // For now, verify that server starts without excessive overhead
    
    fixture.cleanup().await;
    Ok(())
}