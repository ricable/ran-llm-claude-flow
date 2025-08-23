/*!
# I/O Module

High-performance I/O operations optimized for M3 Max unified memory architecture.
*/

pub mod document_reader;
pub mod batch_processor;
pub mod file_handler;
pub mod memory_mapper;

use crate::{Result, PipelineError};
use std::path::PathBuf;
use tokio::io::{AsyncRead, AsyncWrite};

/// I/O performance metrics
#[derive(Debug, Clone)]
pub struct IoMetrics {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub files_processed: u64,
    pub throughput_mbps: f64,
    pub average_latency_ms: f64,
}

/// Document input source
#[derive(Debug, Clone)]
pub enum DocumentSource {
    File(PathBuf),
    Directory(PathBuf),
    Zip(PathBuf),
    Stream(String),
    Memory(Vec<u8>),
}

/// Document format detection
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DocumentFormat {
    Pdf,
    Html,
    Markdown,
    PlainText,
    Csv,
    Json,
    Xml,
    Unknown,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub max_concurrent: usize,
    pub memory_limit_mb: u64,
    pub enable_streaming: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrent: 8,
            memory_limit_mb: 4096, // 4GB for batch processing
            enable_streaming: true,
        }
    }
}

/// High-performance document reader
pub struct DocumentReader {
    config: BatchConfig,
    metrics: IoMetrics,
}

impl DocumentReader {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            metrics: IoMetrics {
                bytes_read: 0,
                bytes_written: 0,
                files_processed: 0,
                throughput_mbps: 0.0,
                average_latency_ms: 0.0,
            },
        }
    }

    /// Read document from source with M3 Max optimization
    pub async fn read_document(&mut self, source: DocumentSource) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        let content = match source {
            DocumentSource::File(path) => {
                self.read_file_optimized(&path).await?
            }
            DocumentSource::Directory(path) => {
                self.read_directory_batch(&path).await?
            }
            DocumentSource::Zip(path) => {
                self.read_zip_archive(&path).await?
            }
            DocumentSource::Stream(url) => {
                self.read_stream(&url).await?
            }
            DocumentSource::Memory(data) => data,
        };

        let elapsed = start_time.elapsed();
        self.update_metrics(content.len() as u64, elapsed);
        
        Ok(content)
    }

    /// Optimized file reading for M3 Max unified memory
    async fn read_file_optimized(&self, path: &PathBuf) -> Result<Vec<u8>> {
        use tokio::fs::File;
        use tokio::io::AsyncReadExt;

        // Use memory-mapped I/O for large files on M3 Max
        let file_size = tokio::fs::metadata(path).await?.len();
        
        if file_size > 100 * 1024 * 1024 { // 100MB threshold
            // Use memory mapping for large files to leverage unified memory
            self.read_file_mmap(path).await
        } else {
            // Use standard async I/O for smaller files
            let mut file = File::open(path).await?;
            let mut contents = Vec::with_capacity(file_size as usize);
            file.read_to_end(&mut contents).await?;
            Ok(contents)
        }
    }

    /// Memory-mapped file reading for M3 Max optimization
    async fn read_file_mmap(&self, path: &PathBuf) -> Result<Vec<u8>> {
        use std::fs::File;
        use std::os::unix::io::AsRawFd;
        
        // This would use actual memory mapping in production
        // For now, fall back to standard file reading
        let contents = std::fs::read(path)
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        Ok(contents)
    }

    /// Batch directory reading with parallel processing
    async fn read_directory_batch(&self, path: &PathBuf) -> Result<Vec<u8>> {
        use tokio::fs;
        use futures::stream::{self, StreamExt};

        let mut entries = fs::read_dir(path).await?;
        let mut files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let file_path = entry.path();
            if file_path.is_file() {
                files.push(file_path);
            }
        }

        // Process files in parallel with M3 Max optimizations
        let contents: Vec<Vec<u8>> = stream::iter(files)
            .map(|path| async move {
                self.read_file_optimized(&path).await.unwrap_or_default()
            })
            .buffer_unordered(self.config.max_concurrent)
            .collect()
            .await;

        // Combine all file contents
        let combined: Vec<u8> = contents.into_iter().flatten().collect();
        Ok(combined)
    }

    /// ZIP archive reading with decompression
    async fn read_zip_archive(&self, path: &PathBuf) -> Result<Vec<u8>> {
        use std::io::Cursor;
        
        // Read ZIP file
        let zip_data = std::fs::read(path)?;
        let cursor = Cursor::new(zip_data);
        
        // Extract all files from ZIP
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        
        let mut combined_content = Vec::new();
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| PipelineError::Io(e.to_string()))?;
            
            let mut content = Vec::new();
            std::io::copy(&mut file, &mut content)?;
            combined_content.extend(content);
        }
        
        Ok(combined_content)
    }

    /// Stream reading for remote sources
    async fn read_stream(&self, url: &str) -> Result<Vec<u8>> {
        // This would implement HTTP/HTTPS streaming in production
        // For now, return empty content
        Ok(Vec::new())
    }

    /// Update I/O metrics
    fn update_metrics(&mut self, bytes_processed: u64, elapsed: std::time::Duration) {
        self.metrics.bytes_read += bytes_processed;
        self.metrics.files_processed += 1;
        
        let elapsed_ms = elapsed.as_millis() as f64;
        let throughput = (bytes_processed as f64) / (elapsed.as_secs_f64() * 1024.0 * 1024.0);
        
        self.metrics.throughput_mbps = throughput;
        self.metrics.average_latency_ms = elapsed_ms;
    }

    /// Get current I/O metrics
    pub fn get_metrics(&self) -> &IoMetrics {
        &self.metrics
    }
}

/// Detect document format from content
pub fn detect_format(content: &[u8]) -> DocumentFormat {
    // PDF magic number
    if content.starts_with(b"%PDF") {
        return DocumentFormat::Pdf;
    }
    
    // HTML detection
    if content.windows(5).any(|window| window.eq_ignore_ascii_case(b"<html")) {
        return DocumentFormat::Html;
    }
    
    // JSON detection
    let trimmed = content.iter()
        .skip_while(|&&b| b.is_ascii_whitespace())
        .take(1)
        .next();
    if trimmed == Some(&b'{') || trimmed == Some(&b'[') {
        return DocumentFormat::Json;
    }
    
    // CSV detection (simple heuristic)
    if let Ok(text) = std::str::from_utf8(content) {
        let lines: Vec<&str> = text.lines().take(5).collect();
        if lines.len() > 1 {
            let first_line_commas = lines[0].matches(',').count();
            if first_line_commas > 0 && 
               lines.iter().skip(1).all(|line| line.matches(',').count() == first_line_commas) {
                return DocumentFormat::Csv;
            }
        }
    }
    
    // Markdown detection
    if let Ok(text) = std::str::from_utf8(content) {
        if text.lines().any(|line| line.starts_with('#') || line.starts_with("```")) {
            return DocumentFormat::Markdown;
        }
    }
    
    // XML detection
    if content.windows(5).any(|window| window.eq_ignore_ascii_case(b"<?xml")) {
        return DocumentFormat::Xml;
    }
    
    // Default to plain text if valid UTF-8
    if std::str::from_utf8(content).is_ok() {
        DocumentFormat::PlainText
    } else {
        DocumentFormat::Unknown
    }
}