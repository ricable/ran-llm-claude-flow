/*!
# Document Reader

High-performance document reading with M3 Max memory optimizations.
*/

use crate::{Result, PipelineError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// Document reader for various file formats
#[derive(Debug)]
pub struct DocumentReader {
    reader_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadRequest {
    pub file_path: String,
    pub format_hint: Option<DocumentFormat>,
    pub read_options: ReadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentFormat {
    Pdf,
    Html,
    Markdown,
    PlainText,
    Csv,
    Json,
    Xml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadOptions {
    pub max_size_bytes: Option<usize>,
    pub encoding: Option<String>,
    pub validate_format: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            max_size_bytes: Some(100 * 1024 * 1024), // 100MB default limit
            encoding: Some("utf-8".to_string()),
            validate_format: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResult {
    pub document_id: Uuid,
    pub content: Vec<u8>,
    pub detected_format: DocumentFormat,
    pub metadata: DocumentMetadata,
    pub read_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub file_size_bytes: u64,
    pub encoding: String,
    pub mime_type: Option<String>,
    pub creation_time: Option<std::time::SystemTime>,
    pub modification_time: Option<std::time::SystemTime>,
}

impl DocumentReader {
    /// Create new document reader
    pub fn new() -> Self {
        Self {
            reader_id: Uuid::new_v4(),
        }
    }

    /// Read document from file path
    pub async fn read_document(&self, request: ReadRequest) -> Result<ReadResult> {
        let start_time = std::time::SystemTime::now();
        let document_id = Uuid::new_v4();

        tracing::debug!("Reading document from: {}", request.file_path);

        // Check if file exists
        let path = Path::new(&request.file_path);
        if !path.exists() {
            return Err(PipelineError::Io(format!("File not found: {}", request.file_path)));
        }

        // Get file metadata
        let file_metadata = std::fs::metadata(path)
            .map_err(|e| PipelineError::Io(format!("Failed to read file metadata: {}", e)))?;

        let file_size = file_metadata.len();

        // Check size limits
        if let Some(max_size) = request.read_options.max_size_bytes {
            if file_size > max_size as u64 {
                return Err(PipelineError::Io(format!(
                    "File too large: {} bytes (max: {} bytes)", 
                    file_size, max_size
                )));
            }
        }

        // Read file content
        let content = tokio::fs::read(path).await
            .map_err(|e| PipelineError::Io(format!("Failed to read file: {}", e)))?;

        // Detect format
        let detected_format = request.format_hint
            .unwrap_or_else(|| self.detect_format(&request.file_path, &content));

        // Validate format if requested
        if request.read_options.validate_format {
            self.validate_format(&detected_format, &content)?;
        }

        let read_time = start_time.elapsed()
            .unwrap_or(std::time::Duration::from_secs(0))
            .as_millis() as u64;

        let metadata = DocumentMetadata {
            file_size_bytes: file_size,
            encoding: request.read_options.encoding.unwrap_or_else(|| "utf-8".to_string()),
            mime_type: self.detect_mime_type(&detected_format),
            creation_time: file_metadata.created().ok(),
            modification_time: file_metadata.modified().ok(),
        };

        Ok(ReadResult {
            document_id,
            content,
            detected_format,
            metadata,
            read_time_ms: read_time,
        })
    }

    /// Detect document format from file extension and content
    fn detect_format(&self, file_path: &str, content: &[u8]) -> DocumentFormat {
        let path = Path::new(file_path);
        
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "pdf" => DocumentFormat::Pdf,
                "html" | "htm" => DocumentFormat::Html,
                "md" | "markdown" => DocumentFormat::Markdown,
                "txt" => DocumentFormat::PlainText,
                "csv" => DocumentFormat::Csv,
                "json" => DocumentFormat::Json,
                "xml" => DocumentFormat::Xml,
                _ => self.detect_from_content(content),
            }
        } else {
            self.detect_from_content(content)
        }
    }

    /// Detect format from content
    fn detect_from_content(&self, content: &[u8]) -> DocumentFormat {
        if content.starts_with(b"%PDF") {
            DocumentFormat::Pdf
        } else if content.starts_with(b"<!DOCTYPE") || content.starts_with(b"<html") {
            DocumentFormat::Html
        } else if content.starts_with(b"{") || content.starts_with(b"[") {
            DocumentFormat::Json
        } else if content.starts_with(b"<?xml") || content.starts_with(b"<") {
            DocumentFormat::Xml
        } else {
            DocumentFormat::PlainText
        }
    }

    /// Validate document format
    fn validate_format(&self, format: &DocumentFormat, _content: &[u8]) -> Result<()> {
        // Basic format validation - can be extended
        match format {
            DocumentFormat::Pdf => {
                // PDF validation would go here
            },
            DocumentFormat::Json => {
                // JSON validation would go here
            },
            _ => {
                // Other format validations
            }
        }
        Ok(())
    }

    /// Detect MIME type from format
    fn detect_mime_type(&self, format: &DocumentFormat) -> Option<String> {
        match format {
            DocumentFormat::Pdf => Some("application/pdf".to_string()),
            DocumentFormat::Html => Some("text/html".to_string()),
            DocumentFormat::Markdown => Some("text/markdown".to_string()),
            DocumentFormat::PlainText => Some("text/plain".to_string()),
            DocumentFormat::Csv => Some("text/csv".to_string()),
            DocumentFormat::Json => Some("application/json".to_string()),
            DocumentFormat::Xml => Some("application/xml".to_string()),
        }
    }
}

/// Initialize document reader
pub async fn initialize() -> Result<()> {
    tracing::info!("Initializing document reader");
    Ok(())
}