//! Rust Test Suite Library
//!
//! Comprehensive testing framework for Rust components of the RAN LLM Pipeline.
//! This library provides test utilities, mock objects, and benchmarking tools
//! for testing the ML integration pipeline and high-performance document processing.

pub mod test_helpers;
pub mod mock_factories;

// Re-export commonly used testing utilities
pub use test_helpers::*;
pub use mock_factories::*;

/// Re-export test modules for external access
pub mod comprehensive_tests {
    //! Comprehensive unit tests for core Rust components
    // Include the test module content directly
    // include!("comprehensive_rust_tests.rs");
}

pub mod ipc_tests {
    //! IPC integration tests for Rust-Python communication  
    // Include the test module content directly
    // include!("ipc_integration_tests.rs");
}

/// Test configuration and utilities
pub mod test_utils {
    use uuid::Uuid;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    /// Performance test helper
    pub fn measure_performance<F, T>(operation: F) -> (T, std::time::Duration)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Memory usage estimation helper
    pub fn estimate_memory_usage(content_size: usize) -> usize {
        // Rough estimation of memory usage for a document
        content_size * 3 // Assume 3x overhead for processing
    }
    
    /// Generate test UUID
    pub fn generate_test_id() -> Uuid {
        Uuid::new_v4()
    }
    
    /// Get current timestamp
    pub fn current_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_library_integration() {
        // Test basic library functionality
        let id = test_utils::generate_test_id();
        assert!(!id.is_nil());
        
        let timestamp = test_utils::current_timestamp();
        assert!(timestamp > 0);
    }
    
    #[test]
    fn test_performance_measurement() {
        let (result, duration) = test_utils::measure_performance(|| {
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
    }
    
    #[test]
    fn test_memory_estimation() {
        let memory_usage = test_utils::estimate_memory_usage(1000);
        assert_eq!(memory_usage, 3000); // 3x overhead
    }
}