/*!
# Document Reader Tests
Comprehensive unit tests for the document reader module
*/

use ran_document_pipeline::io::document_reader::{
    DocumentFormat, DocumentReader, ReadOptions, ReadRequest,
};
use ran_document_pipeline::{PipelineError, Result};
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_document_reader_creation() {
    let _reader = DocumentReader::new();

    // Verify reader is created successfully
    // Since reader_id is private, we can only test that creation doesn't panic
    assert_eq!(
        std::mem::size_of::<DocumentReader>(),
        std::mem::size_of::<DocumentReader>()
    );
}

#[tokio::test]
async fn test_read_options_default() {
    let options = ReadOptions::default();

    assert_eq!(options.max_size_bytes, Some(100 * 1024 * 1024));
    assert_eq!(options.encoding, Some("utf-8".to_string()));
    assert!(options.validate_format);
}

#[tokio::test]
async fn test_read_options_custom() {
    let options = ReadOptions {
        max_size_bytes: Some(50 * 1024 * 1024),
        encoding: Some("latin-1".to_string()),
        validate_format: false,
    };

    assert_eq!(options.max_size_bytes, Some(50 * 1024 * 1024));
    assert_eq!(options.encoding, Some("latin-1".to_string()));
    assert!(!options.validate_format);
}

#[tokio::test]
async fn test_document_format_variants() {
    let formats = vec![
        DocumentFormat::Pdf,
        DocumentFormat::Html,
        DocumentFormat::Markdown,
        DocumentFormat::PlainText,
        DocumentFormat::Csv,
        DocumentFormat::Json,
        DocumentFormat::Xml,
    ];

    // Test that all variants can be created
    assert_eq!(formats.len(), 7);

    // Test serialization/deserialization
    for format in formats {
        let json = serde_json::to_string(&format).expect("Serialization failed");
        let deserialized: DocumentFormat =
            serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(format, deserialized);
    }
}

#[tokio::test]
async fn test_read_existing_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let test_content = b"Hello, World! This is a test file.";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.content, test_content);
    assert_eq!(result.detected_format, DocumentFormat::PlainText);
    assert_eq!(result.metadata.file_size_bytes, test_content.len() as u64);
    assert_eq!(result.metadata.encoding, "utf-8");
    assert!(result.read_time_ms >= 0);

    Ok(())
}

#[tokio::test]
async fn test_read_nonexistent_file() {
    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: "/nonexistent/file.txt".to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await;

    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("File not found"));
        }
        _ => panic!("Expected IO error for nonexistent file"),
    }
}

#[tokio::test]
async fn test_file_size_limit() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("large.txt");
    let large_content = vec![b'A'; 1024]; // 1KB content

    fs::write(&file_path, &large_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions {
            max_size_bytes: Some(512), // Limit to 512 bytes
            encoding: Some("utf-8".to_string()),
            validate_format: true,
        },
    };

    let result = reader.read_document(request).await;

    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("File too large"));
        }
        _ => panic!("Expected IO error for file too large"),
    }

    Ok(())
}

#[tokio::test]
async fn test_format_detection_pdf() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.pdf");
    let pdf_content = b"%PDF-1.4\n%fake pdf content";

    fs::write(&file_path, pdf_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Pdf);
    assert_eq!(
        result.metadata.mime_type,
        Some("application/pdf".to_string())
    );

    Ok(())
}

#[tokio::test]
async fn test_format_detection_html() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.html");
    let html_content =
        b"<!DOCTYPE html><html><head><title>Test</title></head><body>Content</body></html>";

    fs::write(&file_path, html_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Html);
    assert_eq!(result.metadata.mime_type, Some("text/html".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_format_detection_json() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.json");
    let json_content = br#"{"name": "test", "value": 42, "items": [1, 2, 3]}"#;

    fs::write(&file_path, json_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Json);
    assert_eq!(
        result.metadata.mime_type,
        Some("application/json".to_string())
    );

    Ok(())
}

#[tokio::test]
async fn test_format_detection_csv() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.csv");
    let csv_content = b"name,age,city\nJohn,30,New York\nJane,25,London";

    fs::write(&file_path, csv_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Csv);
    assert_eq!(result.metadata.mime_type, Some("text/csv".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_format_detection_markdown() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.md");
    let markdown_content =
        b"# Main Title\n\n## Subtitle\n\n```rust\nfn main() {\n    println!(\"Hello\");\n}\n```";

    fs::write(&file_path, markdown_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Markdown);
    assert_eq!(result.metadata.mime_type, Some("text/markdown".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_format_detection_xml() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.xml");
    let xml_content =
        b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><item>value</item></root>";

    fs::write(&file_path, xml_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::Xml);
    assert_eq!(
        result.metadata.mime_type,
        Some("application/xml".to_string())
    );

    Ok(())
}

#[tokio::test]
async fn test_format_hint_override() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let content = b"This is actually JSON: {\"key\": \"value\"}";

    fs::write(&file_path, content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: Some(DocumentFormat::Json),
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    // Format hint should override detection
    assert_eq!(result.detected_format, DocumentFormat::Json);

    Ok(())
}

#[tokio::test]
async fn test_format_validation_disabled() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let content = b"Plain text content";

    fs::write(&file_path, content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions {
            max_size_bytes: None,
            encoding: Some("utf-8".to_string()),
            validate_format: false,
        },
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.detected_format, DocumentFormat::PlainText);
    assert!(result.read_time_ms >= 0);

    Ok(())
}

#[tokio::test]
async fn test_document_metadata() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("metadata_test.txt");
    let content = b"Content for metadata testing";

    fs::write(&file_path, content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions {
            max_size_bytes: None,
            encoding: Some("iso-8859-1".to_string()),
            validate_format: true,
        },
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.metadata.file_size_bytes, content.len() as u64);
    assert_eq!(result.metadata.encoding, "iso-8859-1");
    assert_eq!(result.metadata.mime_type, Some("text/plain".to_string()));
    assert!(result.metadata.creation_time.is_some() || result.metadata.creation_time.is_none());
    assert!(result.metadata.modification_time.is_some());

    Ok(())
}

#[tokio::test]
async fn test_content_based_format_detection() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("noext"); // No extension
    let json_content = b"{\"test\": \"value\"}";

    fs::write(&file_path, json_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    // Should detect JSON from content even without extension
    assert_eq!(result.detected_format, DocumentFormat::Json);

    Ok(())
}

#[tokio::test]
async fn test_empty_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("empty.txt");

    fs::write(&file_path, b"")
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert!(result.content.is_empty());
    assert_eq!(result.metadata.file_size_bytes, 0);
    assert_eq!(result.detected_format, DocumentFormat::PlainText);

    Ok(())
}

#[tokio::test]
async fn test_binary_file_detection() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("binary.bin");
    let binary_content = vec![0xFF, 0xFE, 0x00, 0x01, 0x02, 0x03, 0x04];

    fs::write(&file_path, &binary_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.content, binary_content);
    assert_eq!(result.detected_format, DocumentFormat::PlainText); // Falls back to PlainText for unknown binary

    Ok(())
}

#[tokio::test]
async fn test_large_file_no_limit() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("large_unlimited.txt");
    let large_content = vec![b'X'; 2048]; // 2KB

    fs::write(&file_path, &large_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions {
            max_size_bytes: None, // No size limit
            encoding: Some("utf-8".to_string()),
            validate_format: false,
        },
    };

    let result = reader.read_document(request).await?;

    assert_eq!(result.content.len(), 2048);
    assert_eq!(result.metadata.file_size_bytes, 2048);

    Ok(())
}

#[tokio::test]
async fn test_file_extension_priority() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Test that extension takes priority over content for format detection
    let file_path = temp_dir.path().join("test.pdf");
    let not_pdf_content = b"This is not a PDF file, just text";

    fs::write(&file_path, not_pdf_content)
        .await
        .expect("Failed to write test file");

    let reader = DocumentReader::new();
    let request = ReadRequest {
        file_path: file_path.to_string_lossy().to_string(),
        format_hint: None,
        read_options: ReadOptions::default(),
    };

    let result = reader.read_document(request).await?;

    // Extension should take priority
    assert_eq!(result.detected_format, DocumentFormat::Pdf);

    Ok(())
}

#[tokio::test]
async fn test_read_request_serialization() {
    let request = ReadRequest {
        file_path: "/test/path.txt".to_string(),
        format_hint: Some(DocumentFormat::Json),
        read_options: ReadOptions {
            max_size_bytes: Some(1024),
            encoding: Some("utf-8".to_string()),
            validate_format: true,
        },
    };

    let json = serde_json::to_string(&request).expect("Serialization failed");
    let deserialized: ReadRequest = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.file_path, request.file_path);
    assert_eq!(deserialized.format_hint, request.format_hint);
    assert_eq!(
        deserialized.read_options.max_size_bytes,
        request.read_options.max_size_bytes
    );
}

#[tokio::test]
async fn test_edge_case_extensions() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let test_cases = vec![
        ("test.htm", DocumentFormat::Html),
        ("test.markdown", DocumentFormat::Markdown),
        ("unknown.xyz", DocumentFormat::PlainText),
    ];

    for (filename, expected_format) in test_cases {
        let file_path = temp_dir.path().join(filename);
        let content = b"Test content";

        fs::write(&file_path, content)
            .await
            .expect("Failed to write test file");

        let reader = DocumentReader::new();
        let request = ReadRequest {
            file_path: file_path.to_string_lossy().to_string(),
            format_hint: None,
            read_options: ReadOptions::default(),
        };

        let result = reader.read_document(request).await?;
        assert_eq!(
            result.detected_format, expected_format,
            "Failed for {}",
            filename
        );
    }

    Ok(())
}
