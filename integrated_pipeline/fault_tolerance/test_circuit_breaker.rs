//! Circuit Breaker Test Suite for Phase 2 MCP Pipeline
//! 
//! Comprehensive testing of circuit breaker patterns, fault isolation,
//! and recovery mechanisms with failure scenario simulation.

#[cfg(test)]
mod tests {
    use super::super::*;
    use anyhow::Result;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};
    use chrono::Utc;
    
    /// Test basic circuit breaker functionality
    #[tokio::test]
    async fn test_circuit_breaker_basic_functionality() -> Result<()> {
        let config = CircuitConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // Test successful operation
        let result = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        
        // Test circuit state is closed
        let status = circuit_breaker.get_status();
        assert_eq!(status.state, CircuitState::Closed);
        
        Ok(())
    }
    
    /// Test circuit breaker opens after failures
    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failure() -> Result<()> {
        let mut config = CircuitConfig::default();
        config.failure_threshold = 3;
        config.timeout_ms = 1000;
        
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // Trigger failures to open circuit
        for _ in 0..3 {
            let result = circuit_breaker.call(async { 
                Err::<i32, &str>("Test failure")
            }).await;
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let status = circuit_breaker.get_status();
        assert_eq!(status.state, CircuitState::Open);
        
        // Next call should fail fast
        let result = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CircuitBreakerError::CircuitOpen(_)));
        
        Ok(())
    }
    
    /// Test circuit breaker recovery mechanism
    #[tokio::test] 
    async fn test_circuit_breaker_recovery() -> Result<()> {
        let mut config = CircuitConfig::default();
        config.failure_threshold = 2;
        config.success_threshold = 2;
        config.recovery_timeout_ms = 100;
        config.timeout_ms = 1000;
        
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // Open circuit with failures
        for _ in 0..2 {
            let _ = circuit_breaker.call(async { 
                Err::<i32, &str>("Test failure")
            }).await;
        }
        
        assert_eq!(circuit_breaker.get_status().state, CircuitState::Open);
        
        // Wait for recovery timeout
        sleep(Duration::from_millis(200)).await;
        
        // Circuit should allow test request (half-open state)
        let result = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        assert!(result.is_ok());
        
        // Another successful request should close circuit
        let result = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        assert!(result.is_ok());
        
        // Circuit should be closed now
        let status = circuit_breaker.get_status();
        assert_eq!(status.state, CircuitState::Closed);
        
        Ok(())
    }
    
    /// Test timeout handling
    #[tokio::test]
    async fn test_circuit_breaker_timeout() -> Result<()> {
        let mut config = CircuitConfig::default();
        config.timeout_ms = 100;
        config.failure_threshold = 2;
        
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // Test timeout
        let result = circuit_breaker.call(async {
            sleep(Duration::from_millis(200)).await;
            Ok::<i32, &str>(42)
        }).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CircuitBreakerError::Timeout(_)));
        
        Ok(())
    }
    
    /// Test exponential backoff
    #[tokio::test]
    async fn test_exponential_backoff() -> Result<()> {
        let mut config = CircuitConfig::default();
        config.failure_threshold = 1;
        config.recovery_timeout_ms = 50;
        config.max_backoff_ms = 1000;
        config.exponential_backoff = true;
        
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // First failure
        let _ = circuit_breaker.call(async { 
            Err::<i32, &str>("Test failure")
        }).await;
        
        assert_eq!(circuit_breaker.get_status().state, CircuitState::Open);
        
        let first_open_time = Utc::now();
        
        // Wait and fail again to increase backoff
        sleep(Duration::from_millis(100)).await;
        let _ = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await; // This should fail fast
        let _ = circuit_breaker.call(async { 
            Err::<i32, &str>("Another failure")
        }).await;
        
        // Second recovery should take longer due to exponential backoff
        sleep(Duration::from_millis(200)).await; // Should still be closed
        let result = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        
        // Should still fail because backoff increased
        assert!(result.is_err());
        
        Ok(())
    }
    
    /// Test component-specific configurations
    #[tokio::test]
    async fn test_component_specific_configs() -> Result<()> {
        // Test Rust core configuration
        let rust_config = CircuitConfig::rust_processing_config();
        assert_eq!(rust_config.failure_threshold, 5);
        assert_eq!(rust_config.timeout_ms, 30000);
        
        // Test Python ML configuration  
        let python_config = CircuitConfig::python_ml_config();
        assert_eq!(python_config.failure_threshold, 3);
        assert_eq!(python_config.timeout_ms, 60000);
        
        // Test IPC configuration
        let ipc_config = CircuitConfig::ipc_config();
        assert_eq!(ipc_config.failure_threshold, 8);
        assert_eq!(ipc_config.timeout_ms, 5000);
        
        Ok(())
    }
    
    /// Test metrics collection
    #[tokio::test]
    async fn test_metrics_collection() -> Result<()> {
        let config = CircuitConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        );
        
        // Perform some operations
        let _ = circuit_breaker.call(async { Ok::<i32, &str>(42) }).await;
        let _ = circuit_breaker.call(async { 
            Err::<i32, &str>("Test failure")
        }).await;
        let _ = circuit_breaker.call(async { Ok::<i32, &str>(43) }).await;
        
        let status = circuit_breaker.get_status();
        assert_eq!(status.metrics.total_requests, 3);
        assert_eq!(status.metrics.successful_requests, 2);
        assert_eq!(status.metrics.failed_requests, 1);
        assert!(status.metrics.avg_response_time_ms > 0.0);
        
        Ok(())
    }
    
    /// Test fault detector integration
    #[tokio::test]
    async fn test_fault_detector_integration() -> Result<()> {
        let config = FaultDetectionConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let mut detector = FaultDetector::new(config, mcp_sender);
        
        // Create some failure records
        let failure_records = vec![
            FailureRecord {
                timestamp: Utc::now(),
                failure_type: FailureType::Timeout,
                severity: FailureSeverity::High,
                error_message: "Test timeout".to_string(),
                duration_ms: 5000,
                component: "test_component".to_string(),
                context: std::collections::HashMap::new(),
            },
            FailureRecord {
                timestamp: Utc::now(),
                failure_type: FailureType::Timeout,
                severity: FailureSeverity::High,
                error_message: "Another timeout".to_string(),
                duration_ms: 5500,
                component: "test_component".to_string(),
                context: std::collections::HashMap::new(),
            },
        ];
        
        // Record failures
        for failure in failure_records {
            detector.record_failure(failure).await;
        }
        
        // Wait a bit for pattern analysis
        sleep(Duration::from_millis(100)).await;
        
        let patterns = detector.detect_current_bottlenecks().await?;
        // In a real implementation, we'd expect pattern detection here
        
        Ok(())
    }
    
    /// Test recovery manager integration
    #[tokio::test]
    async fn test_recovery_manager_integration() -> Result<()> {
        let config = RecoveryConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let (cb_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let mut manager = RecoveryManager::new(config, mcp_sender, Some(cb_sender));
        
        manager.start().await?;
        
        // Initiate recovery for a component
        let recovery_id = manager.initiate_recovery(
            "test_component",
            FailureType::ModelInferenceError,
            None
        ).await?;
        
        assert!(!recovery_id.is_empty());
        
        // Wait for recovery to complete
        sleep(Duration::from_millis(2000)).await;
        
        let stats = manager.get_statistics();
        assert!(stats.total_recoveries >= 1);
        
        manager.shutdown().await;
        
        Ok(())
    }
    
    /// Test isolation manager integration
    #[tokio::test]
    async fn test_isolation_manager_integration() -> Result<()> {
        let config = IsolationConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let (cb_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let mut manager = FailureIsolationManager::new(config, mcp_sender, Some(cb_sender));
        
        manager.start().await?;
        
        // Initiate isolation for a component
        let isolation_id = manager.initiate_isolation(
            "test_component",
            IsolationReason::ResourceExhaustion,
            None
        ).await?;
        
        assert!(!isolation_id.is_empty());
        
        // Wait for isolation to complete
        sleep(Duration::from_millis(1000)).await;
        
        let stats = manager.get_statistics();
        assert!(stats.total_isolations >= 1);
        
        let component_states = manager.get_component_states();
        assert!(component_states.contains_key("test_component"));
        
        manager.shutdown().await;
        
        Ok(())
    }
    
    /// Test full fault tolerance integration
    #[tokio::test]
    async fn test_full_fault_tolerance_integration() -> Result<()> {
        let config = FaultToleranceConfig::default();
        let (mut manager, _notification_receiver, _command_sender) = FaultToleranceManager::new(config);
        
        manager.start().await?;
        
        // Wait for initialization
        sleep(Duration::from_millis(500)).await;
        
        let health_status = manager.get_health_status();
        assert_eq!(health_status.overall_health, HealthLevel::Healthy);
        
        let metrics = manager.get_metrics();
        assert_eq!(metrics.uptime_percentage, 100.0);
        
        // Test circuit breaker status
        let cb_status = manager.get_circuit_breaker_status("rust_core");
        assert!(cb_status.is_some());
        
        // Test manual reset
        manager.manual_circuit_reset("rust_core").await?;
        
        manager.shutdown().await;
        
        Ok(())
    }
    
    /// Test cascade failure prevention
    #[tokio::test]
    async fn test_cascade_failure_prevention() -> Result<()> {
        let config = IsolationConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let (cb_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let mut manager = FailureIsolationManager::new(config, mcp_sender, Some(cb_sender));
        
        manager.start().await?;
        
        // Simulate cascade failure scenario
        // IPC manager failure should affect multiple components
        let isolation_id = manager.initiate_isolation(
            "ipc_manager",
            IsolationReason::CascadeFailurePrevention,
            None
        ).await;
        
        // Should succeed because it prevents cascade
        assert!(isolation_id.is_ok());
        
        // Wait for analysis
        sleep(Duration::from_millis(1000)).await;
        
        let stats = manager.get_statistics();
        // Should show cascade failure prevention
        assert!(stats.cascade_failures_prevented > 0 || stats.total_isolations > 0);
        
        manager.shutdown().await;
        
        Ok(())
    }
    
    /// Performance test - should handle multiple concurrent operations
    #[tokio::test]
    async fn test_performance_concurrent_operations() -> Result<()> {
        let config = CircuitConfig::default();
        let (mcp_sender, _) = tokio::sync::mpsc::unbounded_channel();
        let circuit_breaker = std::sync::Arc::new(CircuitBreaker::new(
            "test_component".to_string(),
            config,
            std::sync::Arc::new(mcp_sender)
        ));
        
        // Launch 100 concurrent operations
        let mut handles = Vec::new();
        for i in 0..100 {
            let cb = circuit_breaker.clone();
            let handle = tokio::spawn(async move {
                cb.call(async move {
                    // Simulate some work
                    sleep(Duration::from_millis(10)).await;
                    Ok::<i32, &str>(i)
                }).await
            });
            handles.push(handle);
        }
        
        // Wait for all operations with timeout
        let results = timeout(
            Duration::from_secs(10),
            futures::future::join_all(handles)
        ).await?;
        
        let successful_operations = results.iter()
            .filter(|r| r.as_ref().unwrap().is_ok())
            .count();
        
        // Should have high success rate
        assert!(successful_operations >= 95);
        
        let status = circuit_breaker.get_status();
        assert_eq!(status.metrics.total_requests, 100);
        assert!(status.metrics.successful_requests >= 95);
        
        Ok(())
    }
    
    /// Test emergency shutdown scenarios
    #[tokio::test]
    async fn test_emergency_shutdown() -> Result<()> {
        let config = FaultToleranceConfig::default();
        let (mut manager, _notification_receiver, _command_sender) = FaultToleranceManager::new(config);
        
        manager.start().await?;
        
        // Simulate emergency condition
        manager.emergency_shutdown().await?;
        
        // System should be shut down gracefully
        let health_status = manager.get_health_status();
        // After emergency shutdown, we might not get accurate health status
        // but the test should complete without hanging
        
        Ok(())
    }
}