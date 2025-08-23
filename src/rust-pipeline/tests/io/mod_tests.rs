/*!
# IO Module Tests
Comprehensive unit tests for the IO module main interface
*/

use ran_document_pipeline::io::{
    IoMetrics, DocumentSource, DocumentFormat, BatchConfig, DocumentReader, detect_format
};
use ran_document_pipeline::{Result, PipelineError};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_io_metrics_default() {
    let metrics = IoMetrics {
        bytes_read: 0,
        bytes_written: 0,
        files_processed: 0,
        throughput_mbps: 0.0,
        average_latency_ms: 0.0,
    };
    
    assert_eq!(metrics.bytes_read, 0);
    assert_eq!(metrics.bytes_written, 0);
    assert_eq!(metrics.files_processed, 0);
    assert_eq!(metrics.throughput_mbps, 0.0);
    assert_eq!(metrics.average_latency_ms, 0.0);
}

#[tokio::test]
async fn test_document_source_variants() {
    let file_source = DocumentSource::File(PathBuf::from("/test/file.txt"));
    let dir_source = DocumentSource::Directory(PathBuf::from("/test/dir"));
    let zip_source = DocumentSource::Zip(PathBuf::from("/test/archive.zip"));
    let stream_source = DocumentSource::Stream("https://example.com/data".to_string());
    let memory_source = DocumentSource::Memory(vec![1, 2, 3, 4]);
    
    match file_source {
        DocumentSource::File(path) => assert_eq!(path, PathBuf::from("/test/file.txt")),
        _ => panic!("Expected File variant"),
    }
    
    match dir_source {
        DocumentSource::Directory(path) => assert_eq!(path, PathBuf::from("/test/dir")),
        _ => panic!("Expected Directory variant"),
    }
    
    match zip_source {
        DocumentSource::Zip(path) => assert_eq!(path, PathBuf::from("/test/archive.zip")),
        _ => panic!("Expected Zip variant"),
    }
    
    match stream_source {
        DocumentSource::Stream(url) => assert_eq!(url, "https://example.com/data"),
        _ => panic!("Expected Stream variant"),
    }
    
    match memory_source {
        DocumentSource::Memory(data) => assert_eq!(data, vec![1, 2, 3, 4]),
        _ => panic!("Expected Memory variant"),
    }
}

#[tokio::test]
async fn test_document_format_detection() {
    // PDF format
    let pdf_content = b"%PDF-1.4\n%random content";
    assert_eq!(detect_format(pdf_content), DocumentFormat::Pdf);
    
    // HTML format
    let html_content = b"<html><body>test</body></html>";
    assert_eq!(detect_format(html_content), DocumentFormat::Html);
    
    // JSON format
    let json_content = b"  {\"key\": \"value\"}";
    assert_eq!(detect_format(json_content), DocumentFormat::Json);
    
    // CSV format
    let csv_content = b"col1,col2,col3\nval1,val2,val3\nval4,val5,val6";
    assert_eq!(detect_format(csv_content), DocumentFormat::Csv);
    
    // Markdown format
    let markdown_content = b"# Header\n```code\ntest\n```";
    assert_eq!(detect_format(markdown_content), DocumentFormat::Markdown);
    
    // XML format
    let xml_content = b"<?xml version=\"1.0\"?><root></root>";
    assert_eq!(detect_format(xml_content), DocumentFormat::Xml);
    
    // Plain text format
    let text_content = b"This is plain text content";
    assert_eq!(detect_format(text_content), DocumentFormat::PlainText);
    
    // Unknown format (binary)
    let binary_content = b"\xFF\xFE\x00\x01\x02\x03";
    assert_eq!(detect_format(binary_content), DocumentFormat::Unknown);
}

#[tokio::test]
async fn test_batch_config_default() {
    let config = BatchConfig::default();
    
    assert_eq!(config.batch_size, 100);
    assert_eq!(config.max_concurrent, 8);
    assert_eq!(config.memory_limit_mb, 4096);
    assert!(config.enable_streaming);
}

#[tokio::test]
async fn test_batch_config_custom() {
    let config = BatchConfig {
        batch_size: 50,
        max_concurrent: 4,
        memory_limit_mb: 2048,
        enable_streaming: false,
    };
    
    assert_eq!(config.batch_size, 50);
    assert_eq!(config.max_concurrent, 4);
    assert_eq!(config.memory_limit_mb, 2048);
    assert!(!config.enable_streaming);
}

#[tokio::test]
async fn test_document_reader_creation() {
    let config = BatchConfig::default();
    let reader = DocumentReader::new(config.clone());
    
    let metrics = reader.get_metrics();
    assert_eq!(metrics.bytes_read, 0);
    assert_eq!(metrics.bytes_written, 0);
    assert_eq!(metrics.files_processed, 0);
    assert_eq!(metrics.throughput_mbps, 0.0);
    assert_eq!(metrics.average_latency_ms, 0.0);
}

#[tokio::test]
async fn test_document_reader_memory_source() -> Result<()> {
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let test_data = b"Hello, World!".to_vec();
    let source = DocumentSource::Memory(test_data.clone());
    
    let result = reader.read_document(source).await?;
    assert_eq!(result, test_data);
    
    let metrics = reader.get_metrics();
    assert_eq!(metrics.bytes_read, test_data.len() as u64);
    assert_eq!(metrics.files_processed, 1);
    assert!(metrics.throughput_mbps >= 0.0);
    assert!(metrics.average_latency_ms >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_document_reader_file_source() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let test_content = b"Test file content";
    
    fs::write(&file_path, test_content).await.expect("Failed to write test file");
    
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::File(file_path);
    let result = reader.read_document(source).await?;
    
    assert_eq!(result, test_content);
    
    let metrics = reader.get_metrics();
    assert_eq!(metrics.bytes_read, test_content.len() as u64);
    assert_eq!(metrics.files_processed, 1);
    
    Ok(())
}

#[tokio::test]
async fn test_document_reader_directory_source() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    // Create test files
    let file1_path = dir_path.join("file1.txt");
    let file2_path = dir_path.join("file2.txt");
    let content1 = b"Content 1";
    let content2 = b"Content 2";
    
    fs::write(&file1_path, content1).await.expect("Failed to write test file 1");
    fs::write(&file2_path, content2).await.expect("Failed to write test file 2");
    
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::Directory(dir_path.to_path_buf());
    let result = reader.read_document(source).await?;
    
    // Should combine all file contents
    assert!(!result.is_empty());
    
    let metrics = reader.get_metrics();
    assert_eq!(metrics.files_processed, 1);
    assert!(metrics.bytes_read > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_document_reader_nonexistent_file() {
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::File(PathBuf::from("/nonexistent/file.txt"));
    let result = reader.read_document(source).await;
    
    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(_)) => (),
        Err(PipelineError::IoError(_)) => (),
        _ => panic!("Expected IO error"),
    }
}

#[tokio::test]
async fn test_document_reader_stream_source() -> Result<()> {
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::Stream("https://example.com/data".to_string());
    let result = reader.read_document(source).await?;
    
    // Stream implementation returns empty for now
    assert!(result.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_format_detection_edge_cases() {
    // Empty content
    assert_eq!(detect_format(&[]), DocumentFormat::PlainText);
    
    // JSON array
    let json_array = b"[{\"key\": \"value\"}]";
    assert_eq!(detect_format(json_array), DocumentFormat::Json);
    
    // HTML without DOCTYPE
    let html_no_doctype = b"<HTML><body>test</body></html>";
    assert_eq!(detect_format(html_no_doctype), DocumentFormat::Html);
    
    // Single column CSV
    let single_csv = b"header\nvalue1\nvalue2";
    assert_eq!(detect_format(single_csv), DocumentFormat::PlainText);
    
    // Markdown with just headers
    let markdown_headers = b"# Header 1\n## Header 2\n### Header 3";
    assert_eq!(detect_format(markdown_headers), DocumentFormat::Markdown);
    
    // XML without declaration
    let xml_no_decl = b"<root><child>value</child></root>";
    assert_eq!(detect_format(xml_no_decl), DocumentFormat::PlainText);
}

#[tokio::test]
async fn test_csv_detection_consistency() {
    // Consistent CSV
    let csv_consistent = b"col1,col2,col3\nval1,val2,val3\nval4,val5,val6";
    assert_eq!(detect_format(csv_consistent), DocumentFormat::Csv);
    
    // Inconsistent CSV (different number of commas)
    let csv_inconsistent = b"col1,col2,col3\nval1,val2\nval4,val5,val6";
    assert_eq!(detect_format(csv_inconsistent), DocumentFormat::PlainText);
    
    // Single line (not CSV)
    let csv_single = b"col1,col2,col3";
    assert_eq!(detect_format(csv_single), DocumentFormat::PlainText);
}

#[tokio::test]
async fn test_large_file_threshold() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("small.txt");
    
    // Create a small file (under 100MB threshold)
    let small_content = vec![b'A'; 1024]; // 1KB
    fs::write(&file_path, &small_content).await.expect("Failed to write test file");
    
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::File(file_path);
    let result = reader.read_document(source).await?;
    
    assert_eq!(result, small_content);
    
    Ok(())
}

#[tokio::test]
async fn test_zip_source_handling() -> Result<()> {
    // Test ZIP source (will fail for non-existent file)
    let config = BatchConfig::default();
    let mut reader = DocumentReader::new(config);
    
    let source = DocumentSource::Zip(PathBuf::from("/nonexistent/archive.zip"));
    let result = reader.read_document(source).await;
    
    // Should return an error for non-existent ZIP
    assert!(result.is_err());
    
    Ok(())
}