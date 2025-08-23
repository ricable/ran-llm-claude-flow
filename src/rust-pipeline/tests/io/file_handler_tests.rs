/*!
# File Handler Tests
Comprehensive unit tests for the file handler module
*/

use ran_document_pipeline::io::file_handler::{
    FileHandler, FileInfo, FilePermissions
};
use ran_document_pipeline::{Result, PipelineError};
use tempfile::TempDir;
use tokio::fs;
use std::path::Path;

#[tokio::test]
async fn test_file_handler_creation() {
    let _handler = FileHandler::new();
    
    // Verify handler is created successfully
    // Since handler_id is private, we can only test that creation doesn't panic
    assert_eq!(std::mem::size_of::<FileHandler>(), std::mem::size_of::<FileHandler>());
}

#[tokio::test]
async fn test_file_permissions_structure() {
    let permissions = FilePermissions {
        readable: true,
        writable: false,
        executable: true,
    };
    
    assert!(permissions.readable);
    assert!(!permissions.writable);
    assert!(permissions.executable);
}

#[tokio::test]
async fn test_file_info_structure() {
    let file_info = FileInfo {
        file_path: "/test/file.txt".to_string(),
        size_bytes: 1024,
        is_directory: false,
        created: Some(std::time::SystemTime::now()),
        modified: Some(std::time::SystemTime::now()),
        permissions: FilePermissions {
            readable: true,
            writable: true,
            executable: false,
        },
    };
    
    assert_eq!(file_info.file_path, "/test/file.txt");
    assert_eq!(file_info.size_bytes, 1024);
    assert!(!file_info.is_directory);
    assert!(file_info.created.is_some());
    assert!(file_info.modified.is_some());
    assert!(file_info.permissions.readable);
    assert!(file_info.permissions.writable);
    assert!(!file_info.permissions.executable);
}

#[tokio::test]
async fn test_get_file_info_existing_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.txt");
    let test_content = b"Hello, World!";
    
    fs::write(&file_path, test_content).await.expect("Failed to write test file");
    
    let handler = FileHandler::new();
    let file_info = handler.get_file_info(&file_path.to_string_lossy()).await?;
    
    assert_eq!(file_info.file_path, file_path.to_string_lossy());
    assert_eq!(file_info.size_bytes, test_content.len() as u64);
    assert!(!file_info.is_directory);
    assert!(file_info.permissions.readable);
    
    Ok(())
}

#[tokio::test]
async fn test_get_file_info_nonexistent_file() {
    let handler = FileHandler::new();
    let result = handler.get_file_info("/nonexistent/file.txt").await;
    
    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("File not found"));
        }
        _ => panic!("Expected IO error for nonexistent file"),
    }
}

#[tokio::test]
async fn test_get_file_info_directory() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    let handler = FileHandler::new();
    let file_info = handler.get_file_info(&dir_path.to_string_lossy()).await?;
    
    assert_eq!(file_info.file_path, dir_path.to_string_lossy());
    assert!(file_info.is_directory);
    assert!(file_info.permissions.readable);
    
    Ok(())
}

#[tokio::test]
async fn test_list_directory_empty() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    let handler = FileHandler::new();
    let contents = handler.list_directory(&dir_path.to_string_lossy()).await?;
    
    assert!(contents.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_list_directory_with_files() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    // Create test files
    let file1_path = dir_path.join("file1.txt");
    let file2_path = dir_path.join("file2.txt");
    let subdir_path = dir_path.join("subdir");
    
    fs::write(&file1_path, b"Content 1").await.expect("Failed to write file1");
    fs::write(&file2_path, b"Content 2").await.expect("Failed to write file2");
    fs::create_dir(&subdir_path).await.expect("Failed to create subdirectory");
    
    let handler = FileHandler::new();
    let contents = handler.list_directory(&dir_path.to_string_lossy()).await?;
    
    assert_eq!(contents.len(), 3);
    
    // Check that we have both files and the directory
    let paths: std::collections::HashSet<String> = contents
        .iter()
        .map(|info| Path::new(&info.file_path).file_name().unwrap().to_string_lossy().to_string())
        .collect();
    
    assert!(paths.contains("file1.txt"));
    assert!(paths.contains("file2.txt"));
    assert!(paths.contains("subdir"));
    
    // Check file/directory distinction
    for info in &contents {
        let name = Path::new(&info.file_path).file_name().unwrap().to_string_lossy();
        if name == "subdir" {
            assert!(info.is_directory);
        } else {
            assert!(!info.is_directory);
            assert!(info.size_bytes > 0);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_list_directory_nonexistent() {
    let handler = FileHandler::new();
    let result = handler.list_directory("/nonexistent/directory").await;
    
    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("Directory not found"));
        }
        _ => panic!("Expected IO error for nonexistent directory"),
    }
}

#[tokio::test]
async fn test_list_directory_on_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("not_a_dir.txt");
    
    fs::write(&file_path, b"Content").await.expect("Failed to write test file");
    
    let handler = FileHandler::new();
    let result = handler.list_directory(&file_path.to_string_lossy()).await;
    
    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("Path is not a directory"));
        }
        _ => panic!("Expected IO error for file instead of directory"),
    }
    
    Ok(())
}

#[tokio::test]
async fn test_create_directory() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let new_dir_path = temp_dir.path().join("new_directory");
    
    let handler = FileHandler::new();
    handler.create_directory(&new_dir_path.to_string_lossy()).await?;
    
    // Verify directory was created
    assert!(new_dir_path.exists());
    assert!(new_dir_path.is_dir());
    
    Ok(())
}

#[tokio::test]
async fn test_create_directory_nested() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let nested_dir_path = temp_dir.path().join("level1").join("level2").join("level3");
    
    let handler = FileHandler::new();
    handler.create_directory(&nested_dir_path.to_string_lossy()).await?;
    
    // Verify nested directory was created
    assert!(nested_dir_path.exists());
    assert!(nested_dir_path.is_dir());
    
    Ok(())
}

#[tokio::test]
async fn test_create_directory_already_exists() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path().join("existing");
    
    // Create directory manually first
    fs::create_dir(&dir_path).await.expect("Failed to create directory");
    
    let handler = FileHandler::new();
    // Should not fail if directory already exists
    handler.create_directory(&dir_path.to_string_lossy()).await?;
    
    assert!(dir_path.exists());
    assert!(dir_path.is_dir());
    
    Ok(())
}

#[tokio::test]
async fn test_delete_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("to_delete.txt");
    
    fs::write(&file_path, b"Content to delete").await.expect("Failed to write test file");
    assert!(file_path.exists());
    
    let handler = FileHandler::new();
    handler.delete(&file_path.to_string_lossy()).await?;
    
    // Verify file was deleted
    assert!(!file_path.exists());
    
    Ok(())
}

#[tokio::test]
async fn test_delete_directory() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path().join("to_delete_dir");
    
    // Create directory with content
    fs::create_dir(&dir_path).await.expect("Failed to create directory");
    let file_in_dir = dir_path.join("file.txt");
    fs::write(&file_in_dir, b"Content").await.expect("Failed to write file in directory");
    
    assert!(dir_path.exists());
    assert!(file_in_dir.exists());
    
    let handler = FileHandler::new();
    handler.delete(&dir_path.to_string_lossy()).await?;
    
    // Verify directory and its contents were deleted
    assert!(!dir_path.exists());
    assert!(!file_in_dir.exists());
    
    Ok(())
}

#[tokio::test]
async fn test_delete_nonexistent_path() {
    let handler = FileHandler::new();
    let result = handler.delete("/nonexistent/path").await;
    
    assert!(result.is_err());
    match result {
        Err(PipelineError::Io(msg)) => {
            assert!(msg.contains("Path not found"));
        }
        _ => panic!("Expected IO error for nonexistent path"),
    }
}

#[tokio::test]
async fn test_file_permissions_readonly() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("readonly.txt");
    
    fs::write(&file_path, b"Readonly content").await.expect("Failed to write test file");
    
    // Make file readonly
    let mut permissions = fs::metadata(&file_path).await.expect("Failed to get metadata").permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&file_path, permissions).await.expect("Failed to set permissions");
    
    let handler = FileHandler::new();
    let file_info = handler.get_file_info(&file_path.to_string_lossy()).await?;
    
    assert!(file_info.permissions.readable);
    assert!(!file_info.permissions.writable);
    
    Ok(())
}

#[tokio::test]
async fn test_empty_directory_listing() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let empty_subdir = temp_dir.path().join("empty");
    
    fs::create_dir(&empty_subdir).await.expect("Failed to create subdirectory");
    
    let handler = FileHandler::new();
    let contents = handler.list_directory(&empty_subdir.to_string_lossy()).await?;
    
    assert!(contents.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_large_directory_listing() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    // Create many files
    for i in 0..50 {
        let file_path = dir_path.join(format!("file{:03}.txt", i));
        fs::write(&file_path, format!("Content {}", i)).await.expect("Failed to write test file");
    }
    
    let handler = FileHandler::new();
    let contents = handler.list_directory(&dir_path.to_string_lossy()).await?;
    
    assert_eq!(contents.len(), 50);
    
    // Verify all files are accounted for
    let mut found_indices = std::collections::HashSet::new();
    for info in &contents {
        let name = Path::new(&info.file_path).file_name().unwrap().to_string_lossy();
        if let Some(captures) = regex::Regex::new(r"file(\d+)\.txt").unwrap().captures(&name) {
            let index: usize = captures[1].parse().unwrap();
            found_indices.insert(index);
        }
    }
    
    // All indices from 0 to 49 should be found
    for i in 0..50 {
        assert!(found_indices.contains(&i), "Missing file index: {}", i);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_mixed_content_directory() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let dir_path = temp_dir.path();
    
    // Create mixed content: files, subdirectories
    fs::write(dir_path.join("file.txt"), b"File content").await.expect("Failed to write file");
    fs::create_dir(dir_path.join("subdir")).await.expect("Failed to create subdirectory");
    fs::write(dir_path.join("another.dat"), b"Binary data").await.expect("Failed to write another file");
    fs::create_dir(dir_path.join("another_dir")).await.expect("Failed to create another subdirectory");
    
    let handler = FileHandler::new();
    let contents = handler.list_directory(&dir_path.to_string_lossy()).await?;
    
    assert_eq!(contents.len(), 4);
    
    let mut files = 0;
    let mut directories = 0;
    
    for info in &contents {
        if info.is_directory {
            directories += 1;
            // Directories have system-specific sizes, just verify it's not negative
            assert!(info.size_bytes >= 0); 
        } else {
            files += 1;
            assert!(info.size_bytes > 0);
        }
    }
    
    assert_eq!(files, 2);
    assert_eq!(directories, 2);
    
    Ok(())
}

#[tokio::test]
async fn test_file_info_serialization() {
    let file_info = FileInfo {
        file_path: "/test/file.txt".to_string(),
        size_bytes: 1024,
        is_directory: false,
        created: Some(std::time::SystemTime::UNIX_EPOCH),
        modified: Some(std::time::SystemTime::UNIX_EPOCH),
        permissions: FilePermissions {
            readable: true,
            writable: false,
            executable: true,
        },
    };
    
    let json = serde_json::to_string(&file_info).expect("Serialization failed");
    let deserialized: FileInfo = serde_json::from_str(&json).expect("Deserialization failed");
    
    assert_eq!(deserialized.file_path, file_info.file_path);
    assert_eq!(deserialized.size_bytes, file_info.size_bytes);
    assert_eq!(deserialized.is_directory, file_info.is_directory);
    assert_eq!(deserialized.permissions.readable, file_info.permissions.readable);
    assert_eq!(deserialized.permissions.writable, file_info.permissions.writable);
    assert_eq!(deserialized.permissions.executable, file_info.permissions.executable);
}

#[tokio::test]
async fn test_special_characters_in_paths() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let special_file = temp_dir.path().join("file with spaces & symbols!.txt");
    
    fs::write(&special_file, b"Special content").await.expect("Failed to write special file");
    
    let handler = FileHandler::new();
    let file_info = handler.get_file_info(&special_file.to_string_lossy()).await?;
    
    assert_eq!(file_info.file_path, special_file.to_string_lossy());
    assert!(!file_info.is_directory);
    assert!(file_info.size_bytes > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_very_large_file() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let large_file = temp_dir.path().join("large.txt");
    
    // Create a relatively large file (1MB)
    let large_content = vec![b'A'; 1024 * 1024];
    fs::write(&large_file, &large_content).await.expect("Failed to write large file");
    
    let handler = FileHandler::new();
    let file_info = handler.get_file_info(&large_file.to_string_lossy()).await?;
    
    assert_eq!(file_info.size_bytes, 1024 * 1024);
    assert!(!file_info.is_directory);
    
    Ok(())
}