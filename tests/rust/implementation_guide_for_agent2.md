# Implementation Guide for Agent 2: Rust Test Implementation Specialist

## Mission Brief
You are tasked with implementing the comprehensive test suite designed by Agent 1 to achieve 98% coverage across 5 Rust IO modules.

## Architecture Summary
- **Total Test Cases**: 253 tests across all categories
- **Coverage Target**: 98% line coverage, 95% branch coverage  
- **Modules Under Test**: 5 (mod.rs, batch_processor.rs, document_reader.rs, file_handler.rs, memory_mapper.rs)
- **Test Categories**: Unit (168), Integration (35), Performance (18), Error Handling (35), Edge Cases (42)

## Priority Implementation Order

### Phase 1: Foundation Setup (IMMEDIATE PRIORITY)
1. **Create mock framework infrastructure** (tests/rust/mock_factories.rs)
2. **Set up test utilities** (tests/rust/test_helpers.rs)
3. **Configure async test runtime** 
4. **Implement coverage measurement setup**

### Phase 2: Core Unit Tests (HIGH PRIORITY)
1. **mod.rs tests** - 45 test cases (highest complexity)
2. **batch_processor.rs tests** - 32 test cases (parallel processing focus)
3. **document_reader.rs tests** - 38 test cases (format detection focus)
4. **file_handler.rs tests** - 25 test cases (filesystem operations)
5. **memory_mapper.rs tests** - 28 test cases (memory operations)

### Phase 3: Integration & Performance (MEDIUM PRIORITY)
1. **Integration tests** - 35 test cases (module interactions)
2. **Performance benchmarks** - 18 test cases (M3 Max optimization)

### Phase 4: Edge Cases & Final Coverage (LOW PRIORITY)
1. **Error handling tests** - 35 test cases (comprehensive error paths)
2. **Boundary condition tests** - 42 test cases (edge cases)

## Test Implementation Templates

### Mock Framework Template (START HERE)
```rust
// tests/rust/mock_factories.rs
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

pub struct MockFileSystem {
    pub files: HashMap<PathBuf, MockFile>,
    pub directories: HashMap<PathBuf, MockDirectory>, 
    pub error_scenarios: HashMap<String, PipelineError>,
}

pub struct MockFile {
    pub content: Vec<u8>,
    pub metadata: MockFileMetadata,
    pub should_fail: Option<PipelineError>,
}

impl MockFileSystem {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            directories: HashMap::new(),
            error_scenarios: HashMap::new(),
        }
    }
    
    pub fn add_file(&mut self, path: PathBuf, content: Vec<u8>) {
        self.files.insert(path, MockFile {
            content,
            metadata: MockFileMetadata::default(),
            should_fail: None,
        });
    }
    
    pub fn simulate_error(&mut self, path: &str, error: PipelineError) {
        self.error_scenarios.insert(path.to_string(), error);
    }
}
```

### Unit Test Template Example
```rust
// tests/rust/test_mod.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_factories::*;
    use tokio_test;
    
    #[test]
    fn test_io_metrics_calculation() {
        let mut metrics = IoMetrics {
            bytes_read: 0,
            bytes_written: 0, 
            files_processed: 0,
            throughput_mbps: 0.0,
            average_latency_ms: 0.0,
        };
        
        // Test metrics update logic
        // ... test implementation
        
        assert_eq!(metrics.files_processed, 1);
        assert!(metrics.throughput_mbps > 0.0);
    }
    
    #[tokio::test]
    async fn test_document_source_file_reading() {
        let mut mock_fs = MockFileSystem::new();
        mock_fs.add_file(
            PathBuf::from("test.txt"), 
            b"test content".to_vec()
        );
        
        let source = DocumentSource::File(PathBuf::from("test.txt"));
        let mut reader = DocumentReader::new(BatchConfig::default());
        
        let result = reader.read_document(source).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), b"test content");
    }
}
```

### Error Handling Test Template
```rust
#[tokio::test]
async fn test_file_not_found_error() {
    let source = DocumentSource::File(PathBuf::from("nonexistent.txt"));
    let mut reader = DocumentReader::new(BatchConfig::default());
    
    let result = reader.read_document(source).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        PipelineError::Io(msg) => assert!(msg.contains("File not found")),
        _ => panic!("Expected IoError"),
    }
}
```

### Performance Test Template
```rust
#[tokio::test]
async fn test_batch_processing_performance() {
    use std::time::Instant;
    
    let processor = BatchProcessor::new(100);
    let request = BatchRequest {
        batch_id: Uuid::new_v4(),
        documents: generate_test_documents(1000), // Helper function
        processing_options: BatchOptions::default(),
    };
    
    let start = Instant::now();
    let result = processor.process_batch(request).await.unwrap();
    let duration = start.elapsed();
    
    // Performance assertions
    assert!(duration.as_millis() < 100); // Sub-100ms requirement
    assert_eq!(result.processed_count, 1000);
    assert!(result.total_processing_time_ms < 100);
}
```

## Coverage Implementation Strategy

### Step 1: Identify All Code Paths
```rust
// Use cargo-tarpaulin to identify uncovered lines
// cargo tarpaulin --out Html --output-dir coverage/

// Focus on these critical coverage areas:
// 1. All match arms in enums (DocumentFormat, DocumentSource)
// 2. All Result<T> error paths  
// 3. All conditional branches (if/else, match)
// 4. All loop iterations (for, while)
// 5. All async operation completion paths
```

### Step 2: Mock External Dependencies
```rust
// Mock all external crate dependencies:
// - tokio::fs operations
// - std::fs operations  
// - zip::ZipArchive operations
// - Network operations (streams)
// - Memory mapping operations
```

### Step 3: Test All Error Scenarios
```rust
// Comprehensive error testing for each module:
#[tokio::test]
async fn test_all_error_paths() {
    let error_scenarios = vec![
        ("permission_denied", PipelineError::Io("Permission denied".to_string())),
        ("file_not_found", PipelineError::Io("File not found".to_string())),
        ("out_of_memory", PipelineError::Memory("Out of memory".to_string())),
        ("timeout", PipelineError::Timeout("Operation timeout".to_string())),
    ];
    
    for (scenario, expected_error) in error_scenarios {
        // Test each error scenario
        // Verify proper error propagation
        // Ensure no resource leaks
    }
}
```

## Async Testing Best Practices

### 1. Use Tokio Test Macros
```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_operations() {
    // Multi-threaded async test
}

#[tokio::test] 
async fn test_single_threaded_async() {
    // Single-threaded async test
}
```

### 2. Test Timeout Scenarios
```rust
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_operation_timeout() {
    let result = timeout(
        Duration::from_millis(50),
        long_running_operation()
    ).await;
    
    assert!(result.is_err()); // Should timeout
}
```

### 3. Resource Cleanup Testing
```rust
#[tokio::test]
async fn test_resource_cleanup() {
    let initial_resources = count_resources();
    
    {
        let _resource = create_resource().await;
        // Use resource
    } // Resource should be cleaned up here
    
    tokio::task::yield_now().await; // Allow cleanup
    let final_resources = count_resources();
    assert_eq!(initial_resources, final_resources);
}
```

## Test Execution Commands

```bash
# Run all tests with coverage
cargo tarpaulin --out Html --output-dir coverage/

# Run specific test module
cargo test test_mod -- --nocapture

# Run performance benchmarks  
cargo bench

# Run tests with specific features
cargo test --features "m3-max-optimization"

# Check test coverage percentage
cargo tarpaulin --skip-clean | grep "Coverage"
```

## Expected Deliverables

1. **Complete test implementation** in tests/rust/ directory
2. **Coverage report** showing 98%+ coverage
3. **Performance benchmarks** meeting latency requirements
4. **Documentation** of any uncovered code paths (with justification)

## Coordination with Agent 1

- **Use shared memory**: Access designs via hooks  
- **Report progress**: Update todos and notify via hooks
- **Ask questions**: Use swarm communication for clarification
- **Share results**: Store coverage reports and benchmark results

## Success Metrics
- ✅ 98%+ line coverage achieved
- ✅ All 253 test cases implemented and passing
- ✅ Performance benchmarks meet requirements  
- ✅ Zero test failures in CI/CD pipeline
- ✅ Test execution time < 30 seconds

Agent 2: Focus on Phase 1 foundation setup first, then systematically implement tests following this guide. The architecture is designed for success - execute methodically and achieve the 98% coverage target!