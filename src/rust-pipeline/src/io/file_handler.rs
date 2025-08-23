/*!
# File Handler

File system operations with M3 Max optimizations.
*/

use crate::{PipelineError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// File handler for optimized file operations
#[derive(Debug)]
pub struct FileHandler {
    handler_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub file_path: String,
    pub size_bytes: u64,
    pub is_directory: bool,
    pub created: Option<std::time::SystemTime>,
    pub modified: Option<std::time::SystemTime>,
    pub permissions: FilePermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePermissions {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

impl FileHandler {
    /// Create new file handler
    pub fn new() -> Self {
        Self {
            handler_id: Uuid::new_v4(),
        }
    }

    /// Get file information
    pub async fn get_file_info(&self, file_path: &str) -> Result<FileInfo> {
        let path = Path::new(file_path);

        if !path.exists() {
            return Err(PipelineError::Io(format!("File not found: {}", file_path)));
        }

        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|e| PipelineError::Io(format!("Failed to read metadata: {}", e)))?;

        Ok(FileInfo {
            file_path: file_path.to_string(),
            size_bytes: metadata.len(),
            is_directory: metadata.is_dir(),
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
            permissions: FilePermissions {
                readable: true, // Simplified for now
                writable: !metadata.permissions().readonly(),
                executable: false, // Simplified for now
            },
        })
    }

    /// List directory contents
    pub async fn list_directory(&self, dir_path: &str) -> Result<Vec<FileInfo>> {
        let path = Path::new(dir_path);

        if !path.exists() {
            return Err(PipelineError::Io(format!(
                "Directory not found: {}",
                dir_path
            )));
        }

        if !path.is_dir() {
            return Err(PipelineError::Io(format!(
                "Path is not a directory: {}",
                dir_path
            )));
        }

        let mut entries = tokio::fs::read_dir(path)
            .await
            .map_err(|e| PipelineError::Io(format!("Failed to read directory: {}", e)))?;

        let mut file_infos = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| PipelineError::Io(format!("Failed to read directory entry: {}", e)))?
        {
            let entry_path = entry.path();
            if let Some(path_str) = entry_path.to_str() {
                match self.get_file_info(path_str).await {
                    Ok(info) => file_infos.push(info),
                    Err(e) => tracing::warn!("Failed to get info for {}: {}", path_str, e),
                }
            }
        }

        Ok(file_infos)
    }

    /// Create directory
    pub async fn create_directory(&self, dir_path: &str) -> Result<()> {
        tokio::fs::create_dir_all(dir_path)
            .await
            .map_err(|e| PipelineError::Io(format!("Failed to create directory: {}", e)))?;

        tracing::info!("Created directory: {}", dir_path);
        Ok(())
    }

    /// Delete file or directory
    pub async fn delete(&self, path: &str) -> Result<()> {
        let path_obj = Path::new(path);

        if !path_obj.exists() {
            return Err(PipelineError::Io(format!("Path not found: {}", path)));
        }

        if path_obj.is_dir() {
            tokio::fs::remove_dir_all(path)
                .await
                .map_err(|e| PipelineError::Io(format!("Failed to delete directory: {}", e)))?;
        } else {
            tokio::fs::remove_file(path)
                .await
                .map_err(|e| PipelineError::Io(format!("Failed to delete file: {}", e)))?;
        }

        tracing::info!("Deleted: {}", path);
        Ok(())
    }
}

/// Initialize file handler
pub async fn initialize() -> Result<()> {
    tracing::info!("Initializing file handler");
    Ok(())
}
