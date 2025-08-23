use crate::config::IpcSettings;
use crate::types::*;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dashmap::DashMap;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Write, BufReader, BufWriter, Cursor};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// High-performance IPC manager for Rust-Python communication
pub struct IpcManager {
    config: IpcSettings,
    shared_memory: Arc<SharedMemoryManager>,
    named_pipes: Arc<NamedPipeManager>,
    request_tracker: Arc<RequestTracker>,
    connection_pool: Arc<ConnectionPool>,
    health_monitor: Arc<HealthMonitor>,
}

/// Shared memory manager for zero-copy data transfer
pub struct SharedMemoryManager {
    memory_pool: Arc<RwLock<MmapMut>>,
    allocation_map: DashMap<u64, MemoryAllocation>,
    next_offset: AtomicUsize,
    pool_size_bytes: usize,
    allocation_lock: Mutex<()>,
}

/// Named pipe manager for control messages
pub struct NamedPipeManager {
    pipe_path: PathBuf,
    writer_pool: Vec<Arc<Mutex<UnixStream>>>,
    reader_pool: Vec<Arc<Mutex<UnixStream>>>,
    round_robin_counter: AtomicUsize,
}

/// Request tracking for correlation and timeouts
pub struct RequestTracker {
    pending_requests: DashMap<Uuid, PendingRequest>,
    request_timeout: Duration,
}

/// Connection pooling for efficient resource usage
pub struct ConnectionPool {
    available_connections: Arc<Semaphore>,
    max_connections: usize,
    active_connections: AtomicUsize,
}

/// Health monitoring for process coordination
pub struct HealthMonitor {
    python_process_health: Arc<RwLock<ProcessHealth>>,
    last_heartbeat: Arc<RwLock<Instant>>,
    health_check_interval: Duration,
}

/// Memory allocation in shared pool
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub offset: u64,
    pub size: usize,
    pub allocated_at: Instant,
    pub request_id: Uuid,
}

/// Pending IPC request
#[derive(Debug)]
pub struct PendingRequest {
    pub response_sender: oneshot::Sender<MLProcessingResponse>,
    pub created_at: Instant,
    pub memory_allocation: Option<MemoryAllocation>,
}

/// Process health status
#[derive(Debug, Clone)]
pub struct ProcessHealth {
    pub is_alive: bool,
    pub last_response_time: Duration,
    pub error_count: usize,
    pub successful_requests: usize,
}

/// IPC message structure
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcMessage {
    pub message_id: Uuid,
    pub message_type: MessageType,
    pub shared_memory_offset: Option<u64>,
    pub data_size: Option<usize>,
    pub checksum: Option<u32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: IpcMetadata,
}

/// Types of IPC messages
#[derive(Debug, Serialize, Deserialize)]
pub enum MessageType {
    ProcessDocument,
    ProcessingComplete,
    ProcessingError,
    HealthCheck,
    HealthResponse,
    Shutdown,
}

/// IPC message metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcMetadata {
    pub priority: ProcessingPriority,
    pub expected_processing_time: Option<Duration>,
    pub model_preference: Option<ComplexityLevel>,
    pub batch_id: Option<String>,
}

/// IPC error types
#[derive(thiserror::Error, Debug)]
pub enum IpcError {
    #[error("Shared memory allocation failed: {0}")]
    SharedMemoryAllocation(String),
    
    #[error("Named pipe communication error: {0}")]
    NamedPipe(String),
    
    #[error("Request timeout: {0}")]
    Timeout(String),
    
    #[error("Python process not responding")]
    ProcessNotResponding,
    
    #[error("Checksum validation failed")]
    ChecksumMismatch,
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl IpcManager {
    /// Create new IPC manager with M3 Max optimization
    pub async fn new(config: &IpcSettings) -> Result<Self> {
        info!("Initializing IPC Manager");
        info!("Shared memory pool: {}GB", config.shared_memory_size_gb);
        info!("Max connections: {}", config.max_connections);
        
        // Initialize shared memory manager
        let shared_memory = Arc::new(
            SharedMemoryManager::new(config.shared_memory_size_gb * 1024 * 1024 * 1024)
                .context("Failed to initialize shared memory")?
        );
        
        // Initialize named pipe manager
        let named_pipes = Arc::new(
            NamedPipeManager::new(&PathBuf::from("/tmp/claude_flow_ipc"))
                .await
                .context("Failed to initialize named pipes")?
        );
        
        // Initialize request tracker
        let request_tracker = Arc::new(RequestTracker::new(
            Duration::from_secs(config.timeout_seconds)
        ));
        
        // Initialize connection pool
        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));
        
        // Initialize health monitor
        let health_monitor = Arc::new(HealthMonitor::new());
        
        // Start background tasks
        let manager = Self {
            config: config.clone(),
            shared_memory,
            named_pipes,
            request_tracker: request_tracker.clone(),
            connection_pool,
            health_monitor: health_monitor.clone(),
        };
        
        // Start health monitoring
        manager.start_health_monitoring().await?;
        
        // Start request cleanup
        manager.start_request_cleanup().await?;
        
        info!("IPC Manager initialized successfully");
        Ok(manager)
    }
    
    /// Send document for ML processing with zero-copy transfer
    pub async fn send_for_ml_processing(
        &self,
        processed_doc: ProcessedDocument
    ) -> Result<MLProcessingResponse, IpcError> {
        let request_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        debug!("Sending document for ML processing: {}", request_id);
        
        // Acquire connection permit
        let _connection_permit = self.connection_pool
            .acquire_connection()
            .await
            .map_err(|e| IpcError::NamedPipe(e.to_string()))?;
        
        // Serialize document data
        let serialized_data = serde_json::to_vec(&processed_doc)
            .map_err(IpcError::Serialization)?;
        
        // Allocate shared memory
        let memory_allocation = self.shared_memory
            .allocate(serialized_data.len(), request_id)
            .await
            .map_err(|e| IpcError::SharedMemoryAllocation(e.to_string()))?;
        
        // Write data to shared memory
        self.shared_memory
            .write_data(memory_allocation.offset, &serialized_data)
            .map_err(|e| IpcError::SharedMemoryAllocation(e.to_string()))?;
        
        // Calculate checksum for integrity
        let checksum = if self.config.enable_checksum_validation {
            Some(crc32fast::hash(&serialized_data))
        } else {
            None
        };
        
        // Create IPC message
        let ipc_message = IpcMessage {
            message_id: request_id,
            message_type: MessageType::ProcessDocument,
            shared_memory_offset: Some(memory_allocation.offset),
            data_size: Some(serialized_data.len()),
            checksum,
            timestamp: chrono::Utc::now(),
            metadata: IpcMetadata {
                priority: processed_doc.processing_hints.processing_priority.clone(),
                expected_processing_time: None,
                model_preference: Some(processed_doc.processing_hints.recommended_model.clone()),
                batch_id: None,
            },
        };
        
        // Set up response channel
        let (response_tx, response_rx) = oneshot::channel();
        let pending_request = PendingRequest {
            response_sender: response_tx,
            created_at: start_time,
            memory_allocation: Some(memory_allocation.clone()),
        };
        
        // Register request for tracking
        self.request_tracker.register_request(request_id, pending_request);
        
        // Send message via named pipe
        self.named_pipes
            .send_message(&ipc_message)
            .await
            .map_err(|e| IpcError::NamedPipe(e.to_string()))?;
        
        // Wait for response with timeout
        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            response_rx
        ).await;
        
        match response {
            Ok(Ok(ml_response)) => {
                // Clean up memory allocation
                self.shared_memory.deallocate(memory_allocation.offset);
                
                let processing_time = start_time.elapsed();
                debug!("ML processing completed in {:?}: {}", processing_time, request_id);
                
                // Update health metrics
                self.health_monitor.record_successful_request(processing_time).await;
                
                Ok(ml_response)
            }
            Ok(Err(_)) => {
                error!("Response channel closed for request: {}", request_id);
                self.cleanup_failed_request(request_id).await;
                Err(IpcError::ProcessNotResponding)
            }
            Err(_) => {
                error!("Request timeout for: {}", request_id);
                self.cleanup_failed_request(request_id).await;
                Err(IpcError::Timeout(format!("Request {} timed out", request_id)))
            }
        }
    }
    
    /// Handle incoming response from Python process
    pub async fn handle_response(&self, message: IpcMessage) -> Result<()> {
        match message.message_type {
            MessageType::ProcessingComplete => {
                self.handle_processing_complete(message).await?;
            }
            MessageType::ProcessingError => {
                self.handle_processing_error(message).await?;
            }
            MessageType::HealthResponse => {
                self.handle_health_response(message).await?;
            }
            _ => {
                warn!("Unexpected message type: {:?}", message.message_type);
            }
        }
        Ok(())
    }
    
    /// Handle successful processing response
    async fn handle_processing_complete(&self, message: IpcMessage) -> Result<()> {
        let request_id = message.message_id;
        
        if let Some((_, pending_request)) = self.request_tracker.pending_requests.remove(&request_id) {
            // Read response data from shared memory
            if let (Some(offset), Some(size)) = (message.shared_memory_offset, message.data_size) {
                let response_data = self.shared_memory
                    .read_data(offset, size)
                    .map_err(|e| anyhow::anyhow!("Failed to read response data: {}", e))?;
                
                // Verify checksum if enabled
                if let Some(expected_checksum) = message.checksum {
                    let actual_checksum = crc32fast::hash(&response_data);
                    if actual_checksum != expected_checksum {
                        error!("Checksum mismatch for request {}", request_id);
                        let _ = pending_request.response_sender.send(MLProcessingResponse::error(
                            request_id,
                            "Checksum validation failed".to_string()
                        ));
                        return Err(IpcError::ChecksumMismatch.into());
                    }
                }
                
                // Deserialize response
                let ml_response: MLProcessingResponse = serde_json::from_slice(&response_data)
                    .context("Failed to deserialize ML response")?;
                
                // Send response to waiting task
                let _ = pending_request.response_sender.send(ml_response);
            } else {
                error!("Missing shared memory information in response");
            }
        } else {
            warn!("Received response for unknown request: {}", request_id);
        }
        
        Ok(())
    }
    
    /// Handle processing error response
    async fn handle_processing_error(&self, message: IpcMessage) -> Result<()> {
        let request_id = message.message_id;
        
        if let Some((_, pending_request)) = self.request_tracker.pending_requests.remove(&request_id) {
            let error_response = MLProcessingResponse::error(
                request_id,
                "Python processing error".to_string()
            );
            let _ = pending_request.response_sender.send(error_response);
            
            self.health_monitor.record_error().await;
        }
        
        Ok(())
    }
    
    /// Handle health check response
    async fn handle_health_response(&self, message: IpcMessage) -> Result<()> {
        let processing_time = chrono::Utc::now()
            .signed_duration_since(message.timestamp)
            .to_std()
            .unwrap_or(Duration::from_millis(0));
        
        self.health_monitor.update_health(true, processing_time).await;
        debug!("Health check response received in {:?}", processing_time);
        
        Ok(())
    }
    
    /// Start health monitoring background task
    async fn start_health_monitoring(&self) -> Result<()> {
        let named_pipes = Arc::clone(&self.named_pipes);
        let health_monitor = Arc::clone(&self.health_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let health_check = IpcMessage {
                    message_id: Uuid::new_v4(),
                    message_type: MessageType::HealthCheck,
                    shared_memory_offset: None,
                    data_size: None,
                    checksum: None,
                    timestamp: chrono::Utc::now(),
                    metadata: IpcMetadata {
                        priority: ProcessingPriority::Low,
                        expected_processing_time: Some(Duration::from_secs(1)),
                        model_preference: None,
                        batch_id: None,
                    },
                };
                
                if let Err(e) = named_pipes.send_message(&health_check).await {
                    error!("Health check failed: {}", e);
                    health_monitor.update_health(false, Duration::from_secs(30)).await;
                }
            }
        });
        
        Ok(())
    }
    
    /// Start request cleanup background task
    async fn start_request_cleanup(&self) -> Result<()> {
        let request_tracker = Arc::clone(&self.request_tracker);
        let shared_memory = Arc::clone(&self.shared_memory);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                request_tracker.cleanup_expired_requests(&shared_memory).await;
            }
        });
        
        Ok(())
    }
    
    /// Clean up failed request
    async fn cleanup_failed_request(&self, request_id: Uuid) {
        if let Some((_, pending_request)) = self.request_tracker.pending_requests.remove(&request_id) {
            if let Some(allocation) = pending_request.memory_allocation {
                self.shared_memory.deallocate(allocation.offset);
            }
        }
        
        self.health_monitor.record_error().await;
    }
    
    /// Get current IPC statistics
    pub async fn get_statistics(&self) -> IpcStatistics {
        let health = self.health_monitor.get_health().await;
        let memory_usage = self.shared_memory.get_usage_statistics();
        let active_requests = self.request_tracker.pending_requests.len();
        let active_connections = self.connection_pool.active_connections.load(Ordering::SeqCst);
        
        IpcStatistics {
            active_requests,
            active_connections,
            python_process_healthy: health.is_alive,
            average_response_time: health.last_response_time,
            error_rate: if health.successful_requests > 0 {
                health.error_count as f64 / health.successful_requests as f64
            } else {
                0.0
            },
            memory_usage,
        }
    }
}

/// IPC usage statistics
#[derive(Debug)]
pub struct IpcStatistics {
    pub active_requests: usize,
    pub active_connections: usize,
    pub python_process_healthy: bool,
    pub average_response_time: Duration,
    pub error_rate: f64,
    pub memory_usage: MemoryUsageStats,
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryUsageStats {
    pub total_allocated_bytes: usize,
    pub active_allocations: usize,
    pub peak_usage_bytes: usize,
    pub fragmentation_ratio: f64,
}

impl SharedMemoryManager {
    fn new(pool_size_bytes: usize) -> Result<Self> {
        info!("Creating shared memory pool: {} bytes", pool_size_bytes);
        
        // Create temporary file for memory mapping
        let temp_file = tempfile::NamedTempFile::new()
            .context("Failed to create temporary file")?;
        
        // Set file size
        temp_file.as_file().set_len(pool_size_bytes as u64)
            .context("Failed to set file size")?;
        
        // Create memory map
        let memory_pool = unsafe {
            MmapOptions::new()
                .len(pool_size_bytes)
                .map_mut(temp_file.as_file())
                .context("Failed to create memory map")?
        };
        
        Ok(Self {
            memory_pool: Arc::new(RwLock::new(memory_pool)),
            allocation_map: DashMap::new(),
            next_offset: AtomicUsize::new(0),
            pool_size_bytes,
            allocation_lock: Mutex::new(()),
        })
    }
    
    async fn allocate(&self, size: usize, request_id: Uuid) -> Result<MemoryAllocation> {
        let _lock = self.allocation_lock.lock();
        
        // Align to 8-byte boundary for optimal performance
        let aligned_size = (size + 7) & !7;
        
        let current_offset = self.next_offset.load(Ordering::SeqCst);
        
        if current_offset + aligned_size > self.pool_size_bytes {
            // Try to find a free block (simple first-fit allocation)
            if let Some(reused_offset) = self.find_free_block(aligned_size) {
                let allocation = MemoryAllocation {
                    offset: reused_offset as u64,
                    size: aligned_size,
                    allocated_at: Instant::now(),
                    request_id,
                };
                
                self.allocation_map.insert(reused_offset as u64, allocation.clone());
                return Ok(allocation);
            } else {
                anyhow::bail!("Shared memory pool exhausted");
            }
        }
        
        let allocation = MemoryAllocation {
            offset: current_offset as u64,
            size: aligned_size,
            allocated_at: Instant::now(),
            request_id,
        };
        
        self.allocation_map.insert(current_offset as u64, allocation.clone());
        self.next_offset.store(current_offset + aligned_size, Ordering::SeqCst);
        
        debug!("Allocated {} bytes at offset {}", aligned_size, current_offset);
        Ok(allocation)
    }
    
    fn write_data(&self, offset: u64, data: &[u8]) -> Result<()> {
        let memory_pool = self.memory_pool.read();
        let start_idx = offset as usize;
        let end_idx = start_idx + data.len();
        
        if end_idx > self.pool_size_bytes {
            anyhow::bail!("Write would exceed memory pool bounds");
        }
        
        memory_pool[start_idx..end_idx].copy_from_slice(data);
        Ok(())
    }
    
    fn read_data(&self, offset: u64, size: usize) -> Result<Vec<u8>> {
        let memory_pool = self.memory_pool.read();
        let start_idx = offset as usize;
        let end_idx = start_idx + size;
        
        if end_idx > self.pool_size_bytes {
            anyhow::bail!("Read would exceed memory pool bounds");
        }
        
        Ok(memory_pool[start_idx..end_idx].to_vec())
    }
    
    fn deallocate(&self, offset: u64) {
        if let Some((_, allocation)) = self.allocation_map.remove(&offset) {
            debug!("Deallocated {} bytes at offset {}", allocation.size, offset);
        }
    }
    
    fn find_free_block(&self, required_size: usize) -> Option<usize> {
        // Simple first-fit allocation strategy
        // In production, could use more sophisticated algorithms
        
        let mut current_offset = 0;
        let mut allocated_blocks: Vec<_> = self.allocation_map.iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        
        allocated_blocks.sort_by_key(|(offset, _)| *offset);
        
        for (offset, allocation) in allocated_blocks {
            if current_offset + required_size <= offset as usize {
                return Some(current_offset);
            }
            current_offset = offset as usize + allocation.size;
        }
        
        None
    }
    
    fn get_usage_statistics(&self) -> MemoryUsageStats {
        let active_allocations = self.allocation_map.len();
        let total_allocated: usize = self.allocation_map
            .iter()
            .map(|entry| entry.value().size)
            .sum();
        
        let peak_usage = self.next_offset.load(Ordering::SeqCst);
        let fragmentation_ratio = if peak_usage > 0 {
            1.0 - (total_allocated as f64 / peak_usage as f64)
        } else {
            0.0
        };
        
        MemoryUsageStats {
            total_allocated_bytes: total_allocated,
            active_allocations,
            peak_usage_bytes: peak_usage,
            fragmentation_ratio,
        }
    }
}

impl NamedPipeManager {
    async fn new(base_path: &PathBuf) -> Result<Self> {
        let pipe_path = base_path.with_extension("pipe");
        
        // Create named pipe if it doesn't exist
        if !pipe_path.exists() {
            std::process::Command::new("mkfifo")
                .arg(&pipe_path)
                .output()
                .context("Failed to create named pipe")?;
        }
        
        info!("Named pipe created: {:?}", pipe_path);
        
        // Initialize connection pools
        let writer_pool = Vec::new();
        let reader_pool = Vec::new();
        
        Ok(Self {
            pipe_path,
            writer_pool,
            reader_pool,
            round_robin_counter: AtomicUsize::new(0),
        })
    }
    
    async fn send_message(&self, message: &IpcMessage) -> Result<()> {
        let serialized = serde_json::to_vec(message)
            .context("Failed to serialize IPC message")?;
        
        // Write message size first, then message data
        let mut data = Vec::with_capacity(4 + serialized.len());
        data.write_u32::<LittleEndian>(serialized.len() as u32)?;
        data.extend_from_slice(&serialized);
        
        // Open pipe for writing (blocking operation)
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.pipe_path)
            .context("Failed to open named pipe for writing")?;
        
        file.write_all(&data)
            .context("Failed to write to named pipe")?;
        
        debug!("Sent IPC message: {} bytes", data.len());
        Ok(())
    }
}

impl RequestTracker {
    fn new(request_timeout: Duration) -> Self {
        Self {
            pending_requests: DashMap::new(),
            request_timeout,
        }
    }
    
    fn register_request(&self, request_id: Uuid, pending_request: PendingRequest) {
        self.pending_requests.insert(request_id, pending_request);
    }
    
    async fn cleanup_expired_requests(&self, shared_memory: &SharedMemoryManager) {
        let now = Instant::now();
        let mut expired_requests = Vec::new();
        
        for entry in self.pending_requests.iter() {
            let (request_id, pending_request) = (entry.key(), entry.value());
            
            if now.duration_since(pending_request.created_at) > self.request_timeout {
                expired_requests.push(*request_id);
            }
        }
        
        for request_id in expired_requests {
            if let Some((_, pending_request)) = self.pending_requests.remove(&request_id) {
                warn!("Cleaning up expired request: {}", request_id);
                
                if let Some(allocation) = pending_request.memory_allocation {
                    shared_memory.deallocate(allocation.offset);
                }
                
                // Notify waiting task of timeout
                let _ = pending_request.response_sender.send(
                    MLProcessingResponse::error(request_id, "Request timeout".to_string())
                );
            }
        }
        
        if !expired_requests.is_empty() {
            info!("Cleaned up {} expired requests", expired_requests.len());
        }
    }
}

impl ConnectionPool {
    fn new(max_connections: usize) -> Self {
        Self {
            available_connections: Arc::new(Semaphore::new(max_connections)),
            max_connections,
            active_connections: AtomicUsize::new(0),
        }
    }
    
    async fn acquire_connection(&self) -> Result<ConnectionGuard> {
        let permit = self.available_connections.acquire().await
            .map_err(|e| anyhow::anyhow!("Failed to acquire connection: {}", e))?;
        
        self.active_connections.fetch_add(1, Ordering::SeqCst);
        
        Ok(ConnectionGuard {
            _permit: permit,
            active_connections: Arc::clone(&self.active_connections),
        })
    }
}

/// RAII guard for connection management
pub struct ConnectionGuard {
    _permit: tokio::sync::SemaphorePermit<'_>,
    active_connections: Arc<AtomicUsize>,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
    }
}

impl HealthMonitor {
    fn new() -> Self {
        Self {
            python_process_health: Arc::new(RwLock::new(ProcessHealth {
                is_alive: true,
                last_response_time: Duration::from_millis(0),
                error_count: 0,
                successful_requests: 0,
            })),
            last_heartbeat: Arc::new(RwLock::new(Instant::now())),
            health_check_interval: Duration::from_secs(30),
        }
    }
    
    async fn update_health(&self, is_alive: bool, response_time: Duration) {
        let mut health = self.python_process_health.write();
        health.is_alive = is_alive;
        health.last_response_time = response_time;
        
        *self.last_heartbeat.write() = Instant::now();
    }
    
    async fn record_successful_request(&self, processing_time: Duration) {
        let mut health = self.python_process_health.write();
        health.successful_requests += 1;
        health.last_response_time = processing_time;
        health.is_alive = true;
    }
    
    async fn record_error(&self) {
        let mut health = self.python_process_health.write();
        health.error_count += 1;
    }
    
    async fn get_health(&self) -> ProcessHealth {
        let health = self.python_process_health.read();
        health.clone()
    }
}

impl MLProcessingResponse {
    pub fn error(request_id: Uuid, error_message: String) -> Self {
        Self {
            request_id,
            qa_pairs: Vec::new(),
            semantic_quality: SemanticQuality {
                coherence_score: 0.0,
                relevance_score: 0.0,
                technical_accuracy_score: 0.0,
                diversity_score: 0.0,
                overall_score: 0.0,
            },
            processing_metadata: MLProcessingMetadata {
                model_name: "error".to_string(),
                model_version: "0.0.0".to_string(),
                inference_time: Duration::from_millis(0),
                tokens_processed: 0,
                memory_used_mb: 0,
                gpu_utilization: None,
            },
            model_used: "none".to_string(),
            processing_time: Duration::from_millis(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_shared_memory_allocation() {
        let manager = SharedMemoryManager::new(1024 * 1024).unwrap(); // 1MB
        let request_id = Uuid::new_v4();
        
        let allocation = manager.allocate(1000, request_id).await.unwrap();
        assert_eq!(allocation.size, 1000);
        assert_eq!(allocation.request_id, request_id);
        
        manager.deallocate(allocation.offset);
    }
    
    #[tokio::test]
    async fn test_memory_write_read() {
        let manager = SharedMemoryManager::new(1024 * 1024).unwrap();
        let request_id = Uuid::new_v4();
        let test_data = b"Hello, World!";
        
        let allocation = manager.allocate(test_data.len(), request_id).await.unwrap();
        manager.write_data(allocation.offset, test_data).unwrap();
        
        let read_data = manager.read_data(allocation.offset, test_data.len()).unwrap();
        assert_eq!(read_data, test_data);
        
        manager.deallocate(allocation.offset);
    }
    
    #[test]
    fn test_request_tracker() {
        let tracker = RequestTracker::new(Duration::from_secs(60));
        let request_id = Uuid::new_v4();
        
        let (tx, _rx) = oneshot::channel();
        let pending_request = PendingRequest {
            response_sender: tx,
            created_at: Instant::now(),
            memory_allocation: None,
        };
        
        tracker.register_request(request_id, pending_request);
        assert!(tracker.pending_requests.contains_key(&request_id));
    }
}