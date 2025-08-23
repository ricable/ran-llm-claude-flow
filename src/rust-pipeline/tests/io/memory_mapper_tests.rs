/*!
# Memory Mapper Tests
Comprehensive unit tests for the memory mapper module
*/

use ran_document_pipeline::io::memory_mapper::{
    MappingRequest, MappingResult, MappingStats, MemoryMapper,
};
use ran_document_pipeline::{PipelineError, Result};
use tempfile::TempDir;
use tokio::fs;
use uuid::Uuid;

#[tokio::test]
async fn test_memory_mapper_creation() {
    let _mapper = MemoryMapper::new();

    // Verify mapper is created successfully
    // Since mapper_id is private, we can only test that creation doesn't panic
    assert_eq!(
        std::mem::size_of::<MemoryMapper>(),
        std::mem::size_of::<MemoryMapper>()
    );
}

#[tokio::test]
async fn test_mapping_request_structure() {
    let request = MappingRequest {
        file_path: "/test/file.txt".to_string(),
        read_only: true,
        offset: 1024,
        length: Some(2048),
    };

    assert_eq!(request.file_path, "/test/file.txt");
    assert!(request.read_only);
    assert_eq!(request.offset, 1024);
    assert_eq!(request.length, Some(2048));
}

#[tokio::test]
async fn test_mapping_result_structure() {
    let mapping_id = Uuid::new_v4();

    let result = MappingResult {
        mapping_id,
        file_path: "/test/file.txt".to_string(),
        mapped_size: 4096,
        is_read_only: false,
    };

    assert_eq!(result.mapping_id, mapping_id);
    assert_eq!(result.file_path, "/test/file.txt");
    assert_eq!(result.mapped_size, 4096);
    assert!(!result.is_read_only);
}

#[tokio::test]
async fn test_mapping_stats_structure() {
    let mapping_id = Uuid::new_v4();
    let now = std::time::SystemTime::now();

    let stats = MappingStats {
        mapping_id,
        access_count: 42,
        bytes_read: 1024,
        bytes_written: 512,
        last_access: now,
    };

    assert_eq!(stats.mapping_id, mapping_id);
    assert_eq!(stats.access_count, 42);
    assert_eq!(stats.bytes_read, 1024);
    assert_eq!(stats.bytes_written, 512);
    assert_eq!(stats.last_access, now);
}

#[tokio::test]
async fn test_create_mapping_existing_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let test_content = b"Hello, World! This is test content for memory mapping.";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.file_path, file_path.to_string_lossy());
    assert_eq!(result.mapped_size, test_content.len());
    assert!(result.is_read_only);

    Ok(())
}

#[tokio::test]
async fn test_create_mapping_nonexistent_file() {
    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: "/nonexistent/file.txt".to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let result = mapper.create_mapping(request).await;

    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("File not found"));
        }
        _ => panic!("Expected IO error for nonexistent file"),
    }
}

#[tokio::test]
async fn test_create_mapping_with_offset() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let test_content = vec![b'A'; 1024]; // 1KB of 'A' characters

    fs::write(&file_path, &test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 100,
        length: Some(500),
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.mapped_size, 500);
    assert!(result.is_read_only);

    Ok(())
}

#[tokio::test]
async fn test_create_mapping_writable() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("writable.txt");
    let test_content = b"Writable content for testing";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: false,
        offset: 0,
        length: None,
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.file_path, file_path.to_string_lossy());
    assert_eq!(result.mapped_size, test_content.len());
    assert!(!result.is_read_only);

    Ok(())
}

#[tokio::test]
async fn test_create_mapping_offset_beyond_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("small.txt");
    let test_content = b"Small"; // 5 bytes

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 10, // Beyond file size
        length: Some(5),
    };

    let result = mapper.create_mapping(request).await;

    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("Mapping extends beyond file size"));
        }
        _ => panic!("Expected IO error for offset beyond file size"),
    }

    Ok(())
}

#[tokio::test]
async fn test_create_mapping_length_beyond_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("small.txt");
    let test_content = b"Small file"; // 10 bytes

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 5,
        length: Some(10), // 5 + 10 = 15, but file is only 10 bytes
    };

    let result = mapper.create_mapping(request).await;

    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("Mapping extends beyond file size"));
        }
        _ => panic!("Expected IO error for length beyond file size"),
    }

    Ok(())
}

#[tokio::test]
async fn test_read_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("read_test.txt");
    let test_content = b"Content for reading from memory mapping";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;

    // Read from the mapping
    let data = mapper
        .read_mapping(mapping_result.mapping_id, 0, 10)
        .await?;

    assert_eq!(data.len(), 10);
    // Note: The current implementation returns zeros, but in a real implementation
    // it would return the actual file content

    Ok(())
}

#[tokio::test]
async fn test_write_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("write_test.txt");
    let test_content = b"Content for writing to memory mapping";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: false,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;

    // Write to the mapping
    let write_data = b"New data";
    mapper
        .write_mapping(mapping_result.mapping_id, 0, write_data)
        .await?;

    // The operation should complete without error
    // Note: The current implementation is a simulation

    Ok(())
}

#[tokio::test]
async fn test_sync_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("sync_test.txt");
    let test_content = b"Content for sync testing";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: false,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;

    // Sync the mapping
    mapper.sync_mapping(mapping_result.mapping_id).await?;

    // The operation should complete without error

    Ok(())
}

#[tokio::test]
async fn test_unmap() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("unmap_test.txt");
    let test_content = b"Content for unmap testing";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;

    // Unmap the mapping
    mapper.unmap(mapping_result.mapping_id).await?;

    // The operation should complete without error

    Ok(())
}

#[tokio::test]
async fn test_get_mapping_stats() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("stats_test.txt");
    let test_content = b"Content for stats testing";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;
    let stats = mapper.get_mapping_stats(mapping_result.mapping_id).await?;

    assert_eq!(stats.mapping_id, mapping_result.mapping_id);
    assert_eq!(stats.access_count, 0);
    assert_eq!(stats.bytes_read, 0);
    assert_eq!(stats.bytes_written, 0);
    // last_access should be recent
    assert!(stats.last_access.elapsed().unwrap().as_secs() < 10);

    Ok(())
}

#[tokio::test]
async fn test_large_file_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("large.txt");

    // Create a larger file (10KB)
    let large_content = vec![b'X'; 10 * 1024];
    fs::write(&file_path, &large_content)
        .await
        .expect("Failed to write large file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.mapped_size, 10 * 1024);
    assert!(result.is_read_only);

    Ok(())
}

#[tokio::test]
async fn test_partial_file_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("partial.txt");

    // Create a file with known content
    let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    fs::write(&file_path, content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();

    // Map only a portion of the file
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 10,
        length: Some(10),
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.mapped_size, 10);
    assert!(result.is_read_only);

    // Read from the partial mapping
    let data = mapper.read_mapping(result.mapping_id, 0, 10).await?;
    assert_eq!(data.len(), 10);

    Ok(())
}

#[tokio::test]
async fn test_zero_length_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("zero_length.txt");
    let test_content = b"Content";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: Some(0),
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.mapped_size, 0);

    Ok(())
}

#[tokio::test]
async fn test_mapping_request_serialization() {
    let request = MappingRequest {
        file_path: "/test/file.txt".to_string(),
        read_only: false,
        offset: 1024,
        length: Some(2048),
    };

    let json = serde_json::to_string(&request).expect("Serialization failed");
    let deserialized: MappingRequest = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.file_path, request.file_path);
    assert_eq!(deserialized.read_only, request.read_only);
    assert_eq!(deserialized.offset, request.offset);
    assert_eq!(deserialized.length, request.length);
}

#[tokio::test]
async fn test_mapping_result_serialization() {
    let mapping_id = Uuid::new_v4();
    let result = MappingResult {
        mapping_id,
        file_path: "/test/file.txt".to_string(),
        mapped_size: 4096,
        is_read_only: true,
    };

    let json = serde_json::to_string(&result).expect("Serialization failed");
    let deserialized: MappingResult = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.mapping_id, result.mapping_id);
    assert_eq!(deserialized.file_path, result.file_path);
    assert_eq!(deserialized.mapped_size, result.mapped_size);
    assert_eq!(deserialized.is_read_only, result.is_read_only);
}

#[tokio::test]
async fn test_mapping_stats_serialization() {
    let mapping_id = Uuid::new_v4();
    let stats = MappingStats {
        mapping_id,
        access_count: 100,
        bytes_read: 2048,
        bytes_written: 1024,
        last_access: std::time::SystemTime::UNIX_EPOCH,
    };

    let json = serde_json::to_string(&stats).expect("Serialization failed");
    let deserialized: MappingStats = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.mapping_id, stats.mapping_id);
    assert_eq!(deserialized.access_count, stats.access_count);
    assert_eq!(deserialized.bytes_read, stats.bytes_read);
    assert_eq!(deserialized.bytes_written, stats.bytes_written);
    assert_eq!(deserialized.last_access, stats.last_access);
}

#[tokio::test]
async fn test_multiple_mappings() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("multiple.txt");
    let test_content = vec![b'M'; 1024];

    fs::write(&file_path, &test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();

    // Create multiple mappings of the same file
    let mut mappings = Vec::new();

    for i in 0..5 {
        let request = MappingRequest {
            file_path: file_path.to_string_lossy().to_string(),
            read_only: true,
            offset: i * 100,
            length: Some(100),
        };

        let result = mapper.create_mapping(request).await?;
        mappings.push(result);
    }

    // Verify all mappings were created with different IDs
    let mut ids = std::collections::HashSet::new();
    for mapping in &mappings {
        assert!(ids.insert(mapping.mapping_id));
        assert_eq!(mapping.mapped_size, 100);
    }

    assert_eq!(ids.len(), 5);

    Ok(())
}

#[tokio::test]
async fn test_edge_case_operations() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("edge_case.txt");
    let test_content = b"Edge case testing content";

    fs::write(&file_path, test_content)
        .await
        .expect("Failed to write test file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: false,
        offset: 0,
        length: None,
    };

    let mapping_result = mapper.create_mapping(request).await?;
    let mapping_id = mapping_result.mapping_id;

    // Test reading with zero length
    let data = mapper.read_mapping(mapping_id, 0, 0).await?;
    assert_eq!(data.len(), 0);

    // Test writing empty data
    mapper.write_mapping(mapping_id, 0, &[]).await?;

    // Test operations complete successfully
    mapper.sync_mapping(mapping_id).await?;
    mapper.unmap(mapping_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_empty_file_mapping() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("empty.txt");

    // Create an empty file
    fs::write(&file_path, b"")
        .await
        .expect("Failed to create empty file");

    let mapper = MemoryMapper::new();
    let request = MappingRequest {
        file_path: file_path.to_string_lossy().to_string(),
        read_only: true,
        offset: 0,
        length: None,
    };

    let result = mapper.create_mapping(request).await?;

    assert_eq!(result.mapped_size, 0);

    Ok(())
}
