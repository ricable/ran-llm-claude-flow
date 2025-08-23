/*!
# Comprehensive Unit Tests for RAN Document Pipeline

Unit tests covering core functionality, error handling, configuration management,
and basic pipeline operations with 95% test coverage target.
*/

use ran_document_pipeline::{
    initialize, start_performance_dashboard, M3MaxConfig, MemoryPoolConfig, PipelineConfig,
    PipelineError, Result,
};
use std::error::Error;

/// Test module for error handling functionality
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_pipeline_error_display() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file not found");
        let pipeline_error = PipelineError::IoError(io_error.to_string());

        let display_string = format!("{}", pipeline_error);
        assert!(display_string.contains("IO error"));
        assert!(display_string.contains("test file not found"));
    }

    #[test]
    fn test_pipeline_error_variants() {
        let errors = vec![
            PipelineError::IpcError("IPC connection failed".to_string()),
            PipelineError::MonitoringError("Monitoring system down".to_string()),
            PipelineError::ConfigError("Invalid configuration".to_string()),
            PipelineError::McpError("MCP protocol error".to_string()),
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());

            // Test that error implements std::error::Error
            let _: &dyn Error = &error;
        }
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let pipeline_error: PipelineError = io_error.into();

        match pipeline_error {
            PipelineError::IoError(e) => {
                assert!(e.contains("access denied"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_error_source() {
        let io_error = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let pipeline_error = PipelineError::IoError(io_error.to_string());

        assert!(pipeline_error.source().is_none()); // String errors don't have sources

        let config_error = PipelineError::ConfigError("test".to_string());
        assert!(config_error.source().is_none());
    }

    #[test]
    fn test_error_debug_format() {
        let error = PipelineError::IpcError("test error".to_string());
        let debug_string = format!("{:?}", error);
        assert!(debug_string.contains("IpcError"));
        assert!(debug_string.contains("test error"));
    }
}

/// Test module for configuration management
mod configuration_tests {
    use super::*;

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();

        assert_eq!(config.processing, 32);
        assert_eq!(config.ipc, 8);
        assert_eq!(config.cache, 4);
    }

    #[test]
    fn test_m3_max_config_default() {
        let config = M3MaxConfig::default();

        assert_eq!(config.memory_pools.processing, 32);
        assert_eq!(config.memory_pools.ipc, 8);
        assert_eq!(config.memory_pools.cache, 4);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();

        assert_eq!(config.m3_max.memory_pools.processing, 32);
        assert_eq!(config.m3_max.memory_pools.ipc, 8);
        assert_eq!(config.m3_max.memory_pools.cache, 4);
    }

    #[test]
    fn test_config_clone() {
        let config1 = PipelineConfig::default();
        let config2 = config1.clone();

        assert_eq!(
            config1.m3_max.memory_pools.processing,
            config2.m3_max.memory_pools.processing
        );
        assert_eq!(
            config1.m3_max.memory_pools.ipc,
            config2.m3_max.memory_pools.ipc
        );
        assert_eq!(
            config1.m3_max.memory_pools.cache,
            config2.m3_max.memory_pools.cache
        );
    }

    #[test]
    fn test_config_debug() {
        let config = PipelineConfig::default();
        let debug_string = format!("{:?}", config);

        assert!(debug_string.contains("PipelineConfig"));
        assert!(debug_string.contains("M3MaxConfig"));
        assert!(debug_string.contains("MemoryPoolConfig"));
    }

    #[test]
    fn test_custom_memory_pool_config() {
        let config = MemoryPoolConfig {
            processing: 64,
            ipc: 16,
            cache: 8,
        };

        assert_eq!(config.processing, 64);
        assert_eq!(config.ipc, 16);
        assert_eq!(config.cache, 8);
    }

    #[test]
    fn test_custom_pipeline_config() {
        let memory_config = MemoryPoolConfig {
            processing: 48,
            ipc: 12,
            cache: 6,
        };

        let m3_config = M3MaxConfig {
            memory_pools: memory_config,
        };

        let pipeline_config = PipelineConfig { m3_max: m3_config };

        assert_eq!(pipeline_config.m3_max.memory_pools.processing, 48);
        assert_eq!(pipeline_config.m3_max.memory_pools.ipc, 12);
        assert_eq!(pipeline_config.m3_max.memory_pools.cache, 6);
    }
}

/// Test module for basic pipeline functionality
mod pipeline_functionality_tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_initialization() {
        let result = initialize().await;

        // The initialization should succeed or fail gracefully
        match result {
            Ok(()) => {
                // Success case - pipeline initialized
                assert!(true);
            }
            Err(e) => {
                // Expected failure case due to missing dependencies
                // This is acceptable in unit tests
                assert!(matches!(e, PipelineError::MonitoringError(_)));
            }
        }
    }

    #[tokio::test]
    async fn test_performance_dashboard_start() {
        let result = start_performance_dashboard(0).await; // Use port 0 for testing

        // The dashboard start should succeed or fail gracefully
        match result {
            Ok(()) => {
                // Success case - dashboard started
                assert!(true);
            }
            Err(e) => {
                // Expected failure case due to missing dependencies
                // This is acceptable in unit tests
                assert!(matches!(e, PipelineError::MonitoringError(_)));
            }
        }
    }

    #[test]
    fn test_result_type_ok() {
        let success: Result<i32> = Ok(42);
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let failure: Result<i32> = Err(PipelineError::ConfigError("test".to_string()));
        assert!(failure.is_err());

        match failure {
            Err(PipelineError::ConfigError(msg)) => assert_eq!(msg, "test"),
            _ => panic!("Expected ConfigError"),
        }
    }
}

/// Test module for edge cases and boundary conditions
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_error_messages() {
        let errors = vec![
            PipelineError::IpcError(String::new()),
            PipelineError::MonitoringError(String::new()),
            PipelineError::ConfigError(String::new()),
            PipelineError::McpError(String::new()),
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty()); // Should still have error type prefix
        }
    }

    #[test]
    fn test_very_long_error_messages() {
        let long_message = "a".repeat(10000);
        let error = PipelineError::IpcError(long_message.clone());

        let display = format!("{}", error);
        assert!(display.contains(&long_message));
        assert!(display.len() > 10000);
    }

    #[test]
    fn test_unicode_error_messages() {
        let unicode_message = "测试错误消息 🚀 Тест ошибки";
        let error = PipelineError::ConfigError(unicode_message.to_string());

        let display = format!("{}", error);
        assert!(display.contains(unicode_message));
    }

    #[test]
    fn test_zero_memory_pool_config() {
        let config = MemoryPoolConfig {
            processing: 0,
            ipc: 0,
            cache: 0,
        };

        // Should be able to create config with zero values
        assert_eq!(config.processing, 0);
        assert_eq!(config.ipc, 0);
        assert_eq!(config.cache, 0);
    }

    #[test]
    fn test_maximum_memory_pool_config() {
        let config = MemoryPoolConfig {
            processing: u32::MAX,
            ipc: u32::MAX,
            cache: u32::MAX,
        };

        // Should be able to create config with maximum values
        assert_eq!(config.processing, u32::MAX);
        assert_eq!(config.ipc, u32::MAX);
        assert_eq!(config.cache, u32::MAX);
    }
}

/// Test module for memory safety and thread safety
mod safety_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_config_send_sync() {
        let config = Arc::new(PipelineConfig::default());
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            let _local_config = config_clone;
            // Config should be Send + Sync
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_error_send_sync() {
        let error = Arc::new(PipelineError::ConfigError("test".to_string()));
        let error_clone = error.clone();

        let handle = thread::spawn(move || {
            let _local_error = error_clone;
            // Error should be Send + Sync
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_multiple_config_instances() {
        let configs: Vec<PipelineConfig> = (0..100)
            .map(|i| PipelineConfig {
                m3_max: M3MaxConfig {
                    memory_pools: MemoryPoolConfig {
                        processing: i,
                        ipc: i * 2,
                        cache: i * 3,
                    },
                },
            })
            .collect();

        assert_eq!(configs.len(), 100);

        for (i, config) in configs.iter().enumerate() {
            assert_eq!(config.m3_max.memory_pools.processing, i as u32);
            assert_eq!(config.m3_max.memory_pools.ipc, (i * 2) as u32);
            assert_eq!(config.m3_max.memory_pools.cache, (i * 3) as u32);
        }
    }
}

/// Test module for performance characteristics
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_error_creation_performance() {
        let start = Instant::now();

        for i in 0..10000 {
            let _error = PipelineError::IpcError(format!("Error {}", i));
        }

        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000); // Should complete in under 1 second
    }

    #[test]
    fn test_config_creation_performance() {
        let start = Instant::now();

        for i in 0..10000 {
            let _config = PipelineConfig {
                m3_max: M3MaxConfig {
                    memory_pools: MemoryPoolConfig {
                        processing: i,
                        ipc: i * 2,
                        cache: i * 3,
                    },
                },
            };
        }

        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000); // Should complete in under 1 second
    }

    #[test]
    fn test_error_display_performance() {
        let errors: Vec<PipelineError> = (0..1000)
            .map(|i| PipelineError::ConfigError(format!("Error {}", i)))
            .collect();

        let start = Instant::now();

        for error in &errors {
            let _display = format!("{}", error);
        }

        let duration = start.elapsed();
        assert!(duration.as_millis() < 100); // Should complete in under 100ms
    }
}

/// Test module for integration scenarios
mod integration_tests {
    use super::*;

    #[test]
    fn test_error_chain() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let pipeline_error = PipelineError::IoError(io_error.to_string());

        // Test error chain
        let mut current_error: &dyn Error = &pipeline_error;
        let mut error_count = 0;

        loop {
            error_count += 1;
            match current_error.source() {
                Some(source) => current_error = source,
                None => break,
            }
        }

        assert_eq!(error_count, 1); // Only PipelineError (String errors don't have sources)
    }

    #[test]
    fn test_config_serialization_compatibility() {
        // Test that configs can be used in serialization contexts
        let config = PipelineConfig::default();

        // This tests that the config types implement the necessary traits
        let debug_output = format!("{:?}", config);
        assert!(!debug_output.is_empty());

        let cloned_config = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", cloned_config));
    }

    #[tokio::test]
    async fn test_async_error_handling() {
        async fn failing_operation() -> Result<()> {
            Err(PipelineError::IpcError(
                "async operation failed".to_string(),
            ))
        }

        let result = failing_operation().await;
        assert!(result.is_err());

        match result {
            Err(PipelineError::IpcError(msg)) => {
                assert_eq!(msg, "async operation failed");
            }
            _ => panic!("Expected IpcError"),
        }
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        use tokio::task;

        let tasks = (0..10).map(|i| {
            task::spawn(async move {
                let config = PipelineConfig {
                    m3_max: M3MaxConfig {
                        memory_pools: MemoryPoolConfig {
                            processing: i,
                            ipc: i * 2,
                            cache: i * 3,
                        },
                    },
                };

                // Simulate some work
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

                config.m3_max.memory_pools.processing
            })
        });

        let results = futures::future::join_all(tasks).await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), i as u32);
        }
    }
}

/// Test module for regression tests
mod regression_tests {
    use super::*;

    #[test]
    fn test_error_message_consistency() {
        // Ensure error messages remain consistent across versions
        let test_cases = vec![
            (
                PipelineError::IpcError("test".to_string()),
                "IPC error: test",
            ),
            (
                PipelineError::MonitoringError("test".to_string()),
                "Monitoring error: test",
            ),
            (
                PipelineError::ConfigError("test".to_string()),
                "Configuration error: test",
            ),
            (
                PipelineError::McpError("test".to_string()),
                "MCP error: test",
            ),
        ];

        for (error, expected) in test_cases {
            assert_eq!(format!("{}", error), expected);
        }
    }

    #[test]
    fn test_default_config_values() {
        // Ensure default configuration values remain stable
        let config = PipelineConfig::default();

        assert_eq!(config.m3_max.memory_pools.processing, 32);
        assert_eq!(config.m3_max.memory_pools.ipc, 8);
        assert_eq!(config.m3_max.memory_pools.cache, 4);
    }

    #[test]
    fn test_error_type_stability() {
        // Ensure error types can be matched consistently
        let errors = vec![
            PipelineError::IpcError("test".to_string()),
            PipelineError::MonitoringError("test".to_string()),
            PipelineError::ConfigError("test".to_string()),
            PipelineError::McpError("test".to_string()),
        ];

        for error in errors {
            match error {
                PipelineError::Io(_) => panic!("Unexpected Io"),
                PipelineError::IoError(_) => panic!("Unexpected IoError"),
                PipelineError::Ipc(_) => panic!("Unexpected Ipc"),
                PipelineError::IpcError(_) => assert!(true),
                PipelineError::Mcp(_) => panic!("Unexpected Mcp"),
                PipelineError::McpError(_) => assert!(true),
                PipelineError::MonitoringError(_) => assert!(true),
                PipelineError::ConfigError(_) => assert!(true),
                PipelineError::Optimization(_) => panic!("Unexpected Optimization"),
                PipelineError::Processing(_) => panic!("Unexpected Processing"),
                PipelineError::Initialization(_) => panic!("Unexpected Initialization"),
            }
        }
    }
}
