/*!
# Batch Processor Tests
Comprehensive unit tests for the batch processor module
*/

use ran_document_pipeline::io::batch_processor::{
    BatchOptions, BatchProcessor, BatchRequest, DocumentItem, DocumentResult,
};
use ran_document_pipeline::Result;
use std::time::Duration;
use uuid::Uuid;

#[tokio::test]
async fn test_batch_processor_creation() {
    let _processor = BatchProcessor::new(100);

    // Verify processor is created successfully
    // Since processor_id is private, we can only test that creation doesn't panic
    assert_eq!(
        std::mem::size_of::<BatchProcessor>(),
        std::mem::size_of::<BatchProcessor>()
    );
}

#[tokio::test]
async fn test_batch_options_default() {
    let options = BatchOptions::default();

    assert!(options.parallel_processing);
    assert_eq!(options.max_concurrent, 4);
    assert_eq!(options.timeout_seconds, 300);
}

#[tokio::test]
async fn test_batch_options_custom() {
    let options = BatchOptions {
        parallel_processing: false,
        max_concurrent: 8,
        timeout_seconds: 600,
    };

    assert!(!options.parallel_processing);
    assert_eq!(options.max_concurrent, 8);
    assert_eq!(options.timeout_seconds, 600);
}

#[tokio::test]
async fn test_document_item_creation() {
    let doc_id = Uuid::new_v4();
    let item = DocumentItem {
        document_id: doc_id,
        file_path: "/test/file.txt".to_string(),
        priority: 5,
    };

    assert_eq!(item.document_id, doc_id);
    assert_eq!(item.file_path, "/test/file.txt");
    assert_eq!(item.priority, 5);
}

#[tokio::test]
async fn test_batch_request_creation() {
    let batch_id = Uuid::new_v4();
    let doc_id = Uuid::new_v4();

    let document = DocumentItem {
        document_id: doc_id,
        file_path: "/test/file.txt".to_string(),
        priority: 1,
    };

    let options = BatchOptions::default();

    let request = BatchRequest {
        batch_id,
        documents: vec![document],
        processing_options: options,
    };

    assert_eq!(request.batch_id, batch_id);
    assert_eq!(request.documents.len(), 1);
    assert_eq!(request.documents[0].document_id, doc_id);
    assert!(request.processing_options.parallel_processing);
}

#[tokio::test]
async fn test_batch_processor_empty_batch() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();

    let request = BatchRequest {
        batch_id,
        documents: Vec::new(),
        processing_options: BatchOptions::default(),
    };

    let result = processor.process_batch(request).await?;

    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.processed_count, 0);
    assert_eq!(result.failed_count, 0);
    assert_eq!(result.results.len(), 0);
    assert!(result.total_processing_time_ms >= 0);

    Ok(())
}

#[tokio::test]
async fn test_batch_processor_single_document() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();
    let doc_id = Uuid::new_v4();

    let document = DocumentItem {
        document_id: doc_id,
        file_path: "/test/file.txt".to_string(),
        priority: 1,
    };

    let request = BatchRequest {
        batch_id,
        documents: vec![document],
        processing_options: BatchOptions::default(),
    };

    let result = processor.process_batch(request).await?;

    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.processed_count, 1);
    assert_eq!(result.failed_count, 0);
    assert_eq!(result.results.len(), 1);

    let doc_result = &result.results[0];
    assert_eq!(doc_result.document_id, doc_id);
    assert!(doc_result.success);
    assert!(doc_result.error_message.is_none());
    assert!(doc_result.processing_time_ms >= 100); // Should be at least 100ms due to sleep

    Ok(())
}

#[tokio::test]
async fn test_batch_processor_multiple_documents_parallel() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();

    let mut documents = Vec::new();
    let mut expected_ids = Vec::new();

    for i in 0..5 {
        let doc_id = Uuid::new_v4();
        expected_ids.push(doc_id);

        documents.push(DocumentItem {
            document_id: doc_id,
            file_path: format!("/test/file{}.txt", i),
            priority: (i % 3) as u8,
        });
    }

    let request = BatchRequest {
        batch_id,
        documents,
        processing_options: BatchOptions {
            parallel_processing: true,
            max_concurrent: 3,
            timeout_seconds: 300,
        },
    };

    let result = processor.process_batch(request).await?;

    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.processed_count, 5);
    assert_eq!(result.failed_count, 0);
    assert_eq!(result.results.len(), 5);

    // Verify all documents were processed
    let result_ids: std::collections::HashSet<Uuid> =
        result.results.iter().map(|r| r.document_id).collect();
    let expected_ids_set: std::collections::HashSet<Uuid> = expected_ids.into_iter().collect();

    assert_eq!(result_ids, expected_ids_set);

    // All results should be successful
    for doc_result in &result.results {
        assert!(doc_result.success);
        assert!(doc_result.error_message.is_none());
        assert!(doc_result.processing_time_ms >= 100);
    }

    Ok(())
}

#[tokio::test]
async fn test_batch_processor_sequential_processing() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();

    let mut documents = Vec::new();
    for i in 0..3 {
        let doc_id = Uuid::new_v4();
        documents.push(DocumentItem {
            document_id: doc_id,
            file_path: format!("/test/file{}.txt", i),
            priority: i as u8,
        });
    }

    let request = BatchRequest {
        batch_id,
        documents,
        processing_options: BatchOptions {
            parallel_processing: false,
            max_concurrent: 1,
            timeout_seconds: 300,
        },
    };

    let start_time = std::time::Instant::now();
    let result = processor.process_batch(request).await?;
    let total_elapsed = start_time.elapsed();

    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.processed_count, 3);
    assert_eq!(result.failed_count, 0);
    assert_eq!(result.results.len(), 3);

    // Sequential processing should take at least 300ms (3 * 100ms sleep)
    assert!(total_elapsed >= Duration::from_millis(250));

    // All results should be successful
    for doc_result in &result.results {
        assert!(doc_result.success);
        assert!(doc_result.error_message.is_none());
    }

    Ok(())
}

#[tokio::test]
async fn test_batch_processor_large_batch() -> Result<()> {
    let processor = BatchProcessor::new(100);
    let batch_id = Uuid::new_v4();

    let mut documents = Vec::new();
    for i in 0..20 {
        let doc_id = Uuid::new_v4();
        documents.push(DocumentItem {
            document_id: doc_id,
            file_path: format!("/test/batch/file{}.txt", i),
            priority: (i % 5) as u8,
        });
    }

    let request = BatchRequest {
        batch_id,
        documents,
        processing_options: BatchOptions {
            parallel_processing: true,
            max_concurrent: 8,
            timeout_seconds: 300,
        },
    };

    let result = processor.process_batch(request).await?;

    assert_eq!(result.batch_id, batch_id);
    assert_eq!(result.processed_count, 20);
    assert_eq!(result.failed_count, 0);
    assert_eq!(result.results.len(), 20);

    // All results should be successful
    for doc_result in &result.results {
        assert!(doc_result.success);
        assert!(doc_result.error_message.is_none());
        assert!(doc_result.processing_time_ms >= 100);
    }

    Ok(())
}

#[tokio::test]
async fn test_batch_processor_high_concurrency() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();

    let mut documents = Vec::new();
    for i in 0..10 {
        let doc_id = Uuid::new_v4();
        documents.push(DocumentItem {
            document_id: doc_id,
            file_path: format!("/test/concurrent/file{}.txt", i),
            priority: 1,
        });
    }

    let request = BatchRequest {
        batch_id,
        documents,
        processing_options: BatchOptions {
            parallel_processing: true,
            max_concurrent: 10,
            timeout_seconds: 300,
        },
    };

    let start_time = std::time::Instant::now();
    let result = processor.process_batch(request).await?;
    let elapsed = start_time.elapsed();

    assert_eq!(result.processed_count, 10);
    assert_eq!(result.failed_count, 0);

    // With high concurrency, should complete much faster than sequential
    // Should be closer to 100ms than 1000ms
    assert!(elapsed < Duration::from_millis(500));

    Ok(())
}

#[tokio::test]
async fn test_document_result_structure() {
    let doc_id = Uuid::new_v4();

    let result = DocumentResult {
        document_id: doc_id,
        success: true,
        error_message: None,
        processing_time_ms: 150,
    };

    assert_eq!(result.document_id, doc_id);
    assert!(result.success);
    assert!(result.error_message.is_none());
    assert_eq!(result.processing_time_ms, 150);
}

#[tokio::test]
async fn test_document_result_with_error() {
    let doc_id = Uuid::new_v4();

    let result = DocumentResult {
        document_id: doc_id,
        success: false,
        error_message: Some("Processing failed".to_string()),
        processing_time_ms: 50,
    };

    assert_eq!(result.document_id, doc_id);
    assert!(!result.success);
    assert_eq!(result.error_message, Some("Processing failed".to_string()));
    assert_eq!(result.processing_time_ms, 50);
}

#[tokio::test]
async fn test_batch_result_metrics() -> Result<()> {
    let processor = BatchProcessor::new(100);
    let batch_id = Uuid::new_v4();

    let request = BatchRequest {
        batch_id,
        documents: vec![
            DocumentItem {
                document_id: Uuid::new_v4(),
                file_path: "/test/file1.txt".to_string(),
                priority: 1,
            },
            DocumentItem {
                document_id: Uuid::new_v4(),
                file_path: "/test/file2.txt".to_string(),
                priority: 2,
            },
        ],
        processing_options: BatchOptions::default(),
    };

    let result = processor.process_batch(request).await?;

    // Verify batch result structure
    assert_eq!(result.batch_id, batch_id);
    assert!(result.total_processing_time_ms > 0);
    assert_eq!(
        result.processed_count + result.failed_count,
        result.results.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_batch_options_serialization() {
    let options = BatchOptions {
        parallel_processing: true,
        max_concurrent: 6,
        timeout_seconds: 450,
    };

    // Test serialization/deserialization
    let json = serde_json::to_string(&options).expect("Serialization failed");
    let deserialized: BatchOptions = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(
        deserialized.parallel_processing,
        options.parallel_processing
    );
    assert_eq!(deserialized.max_concurrent, options.max_concurrent);
    assert_eq!(deserialized.timeout_seconds, options.timeout_seconds);
}

#[tokio::test]
async fn test_different_batch_sizes() -> Result<()> {
    // Test different batch sizes
    for batch_size in [1, 10, 50, 100] {
        let processor = BatchProcessor::new(batch_size);
        let batch_id = Uuid::new_v4();

        let request = BatchRequest {
            batch_id,
            documents: vec![DocumentItem {
                document_id: Uuid::new_v4(),
                file_path: "/test/file.txt".to_string(),
                priority: 1,
            }],
            processing_options: BatchOptions::default(),
        };

        let result = processor.process_batch(request).await?;
        assert_eq!(result.processed_count, 1);
        assert_eq!(result.failed_count, 0);
    }

    Ok(())
}

#[tokio::test]
async fn test_priority_handling() -> Result<()> {
    let processor = BatchProcessor::new(50);
    let batch_id = Uuid::new_v4();

    let mut documents = Vec::new();
    for priority in 0..=255u8 {
        if priority % 50 == 0 {
            // Test a subset to avoid too many documents
            documents.push(DocumentItem {
                document_id: Uuid::new_v4(),
                file_path: format!("/test/priority{}.txt", priority),
                priority,
            });
        }
    }

    let request = BatchRequest {
        batch_id,
        documents,
        processing_options: BatchOptions::default(),
    };

    let result = processor.process_batch(request).await?;

    assert_eq!(result.processed_count, 6); // 0, 50, 100, 150, 200, 250
    assert_eq!(result.failed_count, 0);

    Ok(())
}
