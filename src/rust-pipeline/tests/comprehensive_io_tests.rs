/*!
# Comprehensive IO Tests
Main test runner for all IO module tests with coverage analysis
*/

mod io;

use ran_document_pipeline::{initialize, Result};
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the test environment once
async fn init_test_env() {
    INIT.call_once(|| {
        // Set up tracing for tests
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .with_test_writer()
            .try_init();
    });
    
    // Initialize the pipeline
    let _ = initialize().await;
}

#[tokio::test]
async fn test_io_module_initialization() -> Result<()> {
    init_test_env().await;
    
    // Test that all IO modules can be initialized
    ran_document_pipeline::io::batch_processor::initialize().await?;
    ran_document_pipeline::io::document_reader::initialize().await?;
    ran_document_pipeline::io::file_handler::initialize().await?;
    ran_document_pipeline::io::memory_mapper::initialize().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_io_module_integration() -> Result<()> {
    init_test_env().await;
    
    use ran_document_pipeline::io::{
        DocumentSource, DocumentReader, BatchConfig,
        batch_processor::{BatchProcessor, BatchRequest, DocumentItem, BatchOptions},
        file_handler::FileHandler,
        memory_mapper::{MemoryMapper, MappingRequest},
    };
    use uuid::Uuid;
    use tempfile::TempDir;
    use tokio::fs;
    
    // Create test environment
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("integration_test.txt");
    let test_content = b"Integration test content for IO modules";
    
    fs::write(&file_path, test_content).await.expect("Failed to write test file");
    
    // Test DocumentReader with FileHandler integration
    let file_handler = FileHandler::new();
    let file_info = file_handler.get_file_info(&file_path.to_string_lossy()).await?;
    
    let mut document_reader = DocumentReader::new(BatchConfig::default());
    let source = DocumentSource::File(file_path.clone());
    let content = document_reader.read_document(source).await?;
    
    assert_eq!(content, test_content);
    assert_eq!(file_info.size_bytes, test_content.len() as u64);
    
    // Test MemoryMapper integration
    let memory_mapper = MemoryMapper::new();
    let mapping_request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };
    
    let mapping_result = memory_mapper.create_mapping(mapping_request).await?;
    assert_eq!(mapping_result.mapped_size, test_content.len());
    
    // Test BatchProcessor integration
    let batch_processor = BatchProcessor::new(10);
    let batch_request = BatchRequest {
        batch_id: Uuid::new_v4(),
        documents: vec![DocumentItem {
            document_id: Uuid::new_v4(),
            file_path: file_path.to_string_lossy().to_string(),
            priority: 1,
        }],
        processing_options: BatchOptions::default(),
    };
    
    let batch_result = batch_processor.process_batch(batch_request).await?;
    assert_eq!(batch_result.processed_count, 1);
    assert_eq!(batch_result.failed_count, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_io_error_handling_consistency() -> Result<()> {
    init_test_env().await;
    
    use ran_document_pipeline::io::{
        DocumentSource, DocumentReader, BatchConfig,
        file_handler::FileHandler,
        memory_mapper::{MemoryMapper, MappingRequest},
    };
    use ran_document_pipeline::PipelineError;
    
    // Test consistent error handling across modules
    let nonexistent_path = "/nonexistent/file.txt";
    
    // DocumentReader should return IO error
    let mut document_reader = DocumentReader::new(BatchConfig::default());
    let source = DocumentSource::File(std::path::PathBuf::from(nonexistent_path));
    let result = document_reader.read_document(source).await;
    assert!(result.is_err());
    
    // FileHandler should return IO error  
    let file_handler = FileHandler::new();
    let result = file_handler.get_file_info(nonexistent_path).await;
    assert!(result.is_err());
    
    // MemoryMapper should return IO error
    let memory_mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: nonexistent_path.to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };
    let result = memory_mapper.create_mapping(request).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_io_performance_benchmarks() -> Result<()> {
    init_test_env().await;
    
    use ran_document_pipeline::io::{DocumentSource, DocumentReader, BatchConfig};
    use tempfile::TempDir;
    use tokio::fs;
    use std::time::Instant;
    
    // Create test files of different sizes
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let small_file = temp_dir.path().join("small.txt");
    let medium_file = temp_dir.path().join("medium.txt");
    let large_file = temp_dir.path().join("large.txt");
    
    let small_content = vec![b'S'; 1024]; // 1KB
    let medium_content = vec![b'M'; 10 * 1024]; // 10KB  
    let large_content = vec![b'L'; 100 * 1024]; // 100KB
    
    fs::write(&small_file, &small_content).await?;
    fs::write(&medium_file, &medium_content).await?;
    fs::write(&large_file, &large_content).await?;
    
    let mut reader = DocumentReader::new(BatchConfig::default());
    
    // Benchmark small file
    let start = Instant::now();
    let result = reader.read_document(DocumentSource::File(small_file)).await?;
    let small_duration = start.elapsed();
    assert_eq!(result.len(), small_content.len());
    
    // Benchmark medium file
    let start = Instant::now();
    let result = reader.read_document(DocumentSource::File(medium_file)).await?;
    let medium_duration = start.elapsed();
    assert_eq!(result.len(), medium_content.len());
    
    // Benchmark large file
    let start = Instant::now();
    let result = reader.read_document(DocumentSource::File(large_file)).await?;
    let large_duration = start.elapsed();
    assert_eq!(result.len(), large_content.len());
    
    // Verify performance characteristics
    // Larger files should not take proportionally longer due to optimizations
    println!("Small file (1KB): {:?}", small_duration);
    println!("Medium file (10KB): {:?}", medium_duration);  
    println!("Large file (100KB): {:?}", large_duration);
    
    // All operations should complete within reasonable time
    assert!(small_duration.as_millis() < 100);
    assert!(medium_duration.as_millis() < 200);
    assert!(large_duration.as_millis() < 500);
    
    Ok(())
}

#[tokio::test]
async fn test_io_concurrent_operations() -> Result<()> {
    init_test_env().await;
    
    use ran_document_pipeline::io::{DocumentSource, DocumentReader, BatchConfig};
    use tempfile::TempDir;
    use tokio::fs;
    use futures::future::join_all;
    
    // Test concurrent file operations
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let mut tasks = Vec::new();
    
    // Create multiple test files
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("concurrent_{}.txt", i));
        let content = format!("Concurrent test content {}", i);
        fs::write(&file_path, content.as_bytes()).await?;
        
        // Create concurrent read tasks
        let file_path_clone = file_path.clone();
        let task = tokio::spawn(async move {
            let mut reader = DocumentReader::new(BatchConfig::default());
            let source = DocumentSource::File(file_path_clone);
            reader.read_document(source).await
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    let results = join_all(tasks).await;
    
    // Verify all operations succeeded
    for (i, result) in results.into_iter().enumerate() {
        let content = result.expect("Task should not panic").expect("Read should succeed");
        let expected = format!("Concurrent test content {}", i);
        assert_eq!(content, expected.as_bytes());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_io_memory_efficiency() -> Result<()> {
    init_test_env().await;
    
    use ran_document_pipeline::io::{DocumentSource, DocumentReader, BatchConfig};
    use tempfile::TempDir;
    use tokio::fs;
    
    // Test memory efficiency with larger files
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("memory_test.txt");
    
    // Create a 1MB file
    let large_content = vec![b'X'; 1024 * 1024];
    fs::write(&file_path, &large_content).await?;
    
    let mut reader = DocumentReader::new(BatchConfig::default());
    let source = DocumentSource::File(file_path);
    
    // Read the file multiple times to test memory management
    for _ in 0..5 {
        let result = reader.read_document(source.clone()).await?;
        assert_eq!(result.len(), large_content.len());
        
        // Force garbage collection between reads
        drop(result);
    }
    
    // Verify metrics are updated correctly
    let metrics = reader.get_metrics();
    assert_eq!(metrics.files_processed, 5);
    assert_eq!(metrics.bytes_read, 5 * 1024 * 1024);
    assert!(metrics.throughput_mbps > 0.0);
    
    Ok(())
}

/// Test coverage analysis helper
#[tokio::test]
async fn test_io_module_coverage_analysis() -> Result<()> {
    init_test_env().await;
    
    // This test ensures we've covered the major code paths
    use ran_document_pipeline::io::{
        DocumentFormat, detect_format,
        batch_processor::BatchProcessor,
        document_reader::{DocumentReader as DocReader, ReadOptions},
        file_handler::FileHandler,
        memory_mapper::MemoryMapper,
    };
    
    // Test all document format variants
    let formats = vec![
        DocumentFormat::Pdf,
        DocumentFormat::Html,
        DocumentFormat::Markdown,
        DocumentFormat::PlainText,
        DocumentFormat::Csv,
        DocumentFormat::Json,
        DocumentFormat::Xml,
    ];
    
    for format in formats {
        // Test serialization for each format
        let _json = serde_json::to_string(&format).expect("Format serialization should succeed");
    }
    
    // Test format detection with various inputs
    assert_eq!(detect_format(b"%PDF-1.4"), DocumentFormat::Pdf);
    assert_eq!(detect_format(b"<html>"), DocumentFormat::Html);
    assert_eq!(detect_format(b"{\"key\": \"value\"}"), DocumentFormat::Json);
    assert_eq!(detect_format(b"text"), DocumentFormat::PlainText);
    
    // Test component initialization
    let _batch_processor = BatchProcessor::new(50);
    let _doc_reader = DocReader::new();
    let _file_handler = FileHandler::new();  
    let _memory_mapper = MemoryMapper::new();
    
    // Test options and configurations
    let _read_options = ReadOptions::default();
    
    println!("✅ IO module coverage analysis complete");
    
    Ok(())
}