//! IPC Integration Tests for Rust-Python Communication
//!
//! Tests the Inter-Process Communication layer that enables zero-copy
//! data exchange between Rust core and Python ML components.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
    collections::HashMap,
    thread,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Mock IPC types and structures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IPCMessage {
    pub id: Uuid,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64,
    pub checksum: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    DocumentRequest,
    DocumentResponse, 
    MLProcessingRequest,
    MLProcessingResponse,
    StatusUpdate,
    Error,
    Heartbeat,
}

#[derive(Debug, Clone)]
pub struct SharedMemoryPool {
    pool_id: u32,
    size_bytes: usize,
    allocated_bytes: Arc<Mutex<usize>>,
    segments: Arc<Mutex<HashMap<String, MemorySegment>>>,
}

#[derive(Debug, Clone)]
pub struct MemorySegment {
    id: String,
    offset: usize,
    size_bytes: usize,
    in_use: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum IPCError {
    #[error("Memory pool exhausted: requested {requested}GB, available {available}GB")]
    MemoryPoolExhausted { requested: f64, available: f64 },
    
    #[error("Message serialization failed: {0}")]
    SerializationError(String),
    
    #[error("Communication timeout after {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
    
    #[error("Invalid message format: {0}")]
    InvalidMessage(String),
    
    #[error("IPC channel closed")]
    ChannelClosed,
    
    #[error("Memory segment not found: {segment_id}")]
    SegmentNotFound { segment_id: String },
    
    #[error("Checksum validation failed: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },
}

type IPCResult<T> = Result<T, IPCError>;

// Mock IPC Manager implementation
pub struct IPCManager {
    memory_pools: HashMap<u32, SharedMemoryPool>,
    message_queue: Arc<Mutex<Vec<IPCMessage>>>,
    stats: Arc<Mutex<IPCStats>>,
    config: IPCConfig,
}

#[derive(Debug, Clone)]
pub struct IPCConfig {
    pub pool_size_gb: f64,
    pub max_message_size_mb: usize,
    pub timeout_ms: u64,
    pub enable_checksum: bool,
    pub enable_compression: bool,
}

#[derive(Debug, Clone)]
pub struct IPCStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_transferred: u64,
    pub memory_allocated_bytes: u64,
    pub errors_count: u64,
    pub average_latency_ms: f64,
}

impl IPCManager {
    pub fn new(config: IPCConfig) -> IPCResult<Self> {
        let mut memory_pools = HashMap::new();
        
        // Create initial memory pool
        let pool = SharedMemoryPool::new(
            1,
            (config.pool_size_gb * 1024.0 * 1024.0 * 1024.0) as usize,
        )?;
        
        memory_pools.insert(1, pool);
        
        Ok(Self {
            memory_pools,
            message_queue: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(IPCStats::default())),
            config,
        })
    }
    
    pub fn send_message(&self, message: IPCMessage) -> IPCResult<()> {
        let start_time = Instant::now();
        
        // Validate message size
        if message.payload.len() > self.config.max_message_size_mb * 1024 * 1024 {
            return Err(IPCError::InvalidMessage(
                format!("Message size {}MB exceeds limit {}MB", 
                       message.payload.len() / 1024 / 1024,
                       self.config.max_message_size_mb)
            ));
        }
        
        // Validate checksum if enabled
        if self.config.enable_checksum {
            let calculated_checksum = self.calculate_checksum(&message.payload);
            if calculated_checksum != message.checksum {
                return Err(IPCError::ChecksumMismatch {
                    expected: message.checksum,
                    actual: calculated_checksum,
                });
            }
        }
        
        // Add to message queue
        {
            let mut queue = self.message_queue.lock()
                .map_err(|_| IPCError::ChannelClosed)?;
            queue.push(message.clone());
        }
        
        // Update statistics
        {
            let mut stats = self.stats.lock()
                .map_err(|_| IPCError::ChannelClosed)?;
            stats.messages_sent += 1;
            stats.bytes_transferred += message.payload.len() as u64;
            
            let latency = start_time.elapsed().as_millis() as f64;
            stats.average_latency_ms = (stats.average_latency_ms + latency) / 2.0;
        }
        
        Ok(())
    }
    
    pub fn receive_message(&self, timeout_ms: Option<u64>) -> IPCResult<IPCMessage> {
        let timeout = timeout_ms.unwrap_or(self.config.timeout_ms);
        let start_time = Instant::now();
        
        loop {
            // Check for timeout
            if start_time.elapsed().as_millis() > timeout as u128 {
                return Err(IPCError::TimeoutError { timeout_ms: timeout });
            }
            
            // Try to get a message from the queue
            {
                let mut queue = self.message_queue.lock()
                    .map_err(|_| IPCError::ChannelClosed)?;
                
                if let Some(message) = queue.pop() {
                    // Update statistics
                    {
                        let mut stats = self.stats.lock()
                            .map_err(|_| IPCError::ChannelClosed)?;
                        stats.messages_received += 1;
                    }
                    
                    return Ok(message);
                }
            }
            
            // Small sleep to prevent busy waiting
            thread::sleep(Duration::from_millis(1));
        }
    }
    
    pub fn allocate_memory(&self, pool_id: u32, size_bytes: usize) -> IPCResult<String> {
        let pool = self.memory_pools.get(&pool_id)
            .ok_or_else(|| IPCError::SegmentNotFound { 
                segment_id: format!("pool_{}", pool_id) 
            })?;
        
        pool.allocate_segment(size_bytes)
    }
    
    pub fn deallocate_memory(&self, pool_id: u32, segment_id: &str) -> IPCResult<()> {
        let pool = self.memory_pools.get(&pool_id)
            .ok_or_else(|| IPCError::SegmentNotFound { 
                segment_id: format!("pool_{}", pool_id) 
            })?;
        
        pool.deallocate_segment(segment_id)
    }
    
    pub fn get_stats(&self) -> IPCResult<IPCStats> {
        let stats = self.stats.lock()
            .map_err(|_| IPCError::ChannelClosed)?;
        Ok(stats.clone())
    }
    
    pub fn get_memory_utilization(&self, pool_id: u32) -> IPCResult<f64> {
        let pool = self.memory_pools.get(&pool_id)
            .ok_or_else(|| IPCError::SegmentNotFound { 
                segment_id: format!("pool_{}", pool_id) 
            })?;
        
        Ok(pool.get_utilization_percent())
    }
    
    fn calculate_checksum(&self, data: &[u8]) -> u32 {
        // Simple CRC32-like checksum for testing
        data.iter().enumerate().fold(0u32, |acc, (i, &byte)| {
            acc.wrapping_add((byte as u32) * (i as u32 + 1))
        })
    }
}

impl SharedMemoryPool {
    pub fn new(pool_id: u32, size_bytes: usize) -> IPCResult<Self> {
        Ok(Self {
            pool_id,
            size_bytes,
            allocated_bytes: Arc::new(Mutex::new(0)),
            segments: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    pub fn allocate_segment(&self, size_bytes: usize) -> IPCResult<String> {
        let mut allocated = self.allocated_bytes.lock()
            .map_err(|_| IPCError::ChannelClosed)?;
        let mut segments = self.segments.lock()
            .map_err(|_| IPCError::ChannelClosed)?;
        
        // Check if we have enough space
        if *allocated + size_bytes > self.size_bytes {
            let available_gb = (self.size_bytes - *allocated) as f64 / 1024.0 / 1024.0 / 1024.0;
            let requested_gb = size_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            
            return Err(IPCError::MemoryPoolExhausted {
                requested: requested_gb,
                available: available_gb,
            });
        }
        
        let segment_id = Uuid::new_v4().to_string();
        let segment = MemorySegment {
            id: segment_id.clone(),
            offset: *allocated,
            size_bytes,
            in_use: true,
        };
        
        segments.insert(segment_id.clone(), segment);
        *allocated += size_bytes;
        
        Ok(segment_id)
    }
    
    pub fn deallocate_segment(&self, segment_id: &str) -> IPCResult<()> {
        let mut allocated = self.allocated_bytes.lock()
            .map_err(|_| IPCError::ChannelClosed)?;
        let mut segments = self.segments.lock()
            .map_err(|_| IPCError::ChannelClosed)?;
        
        let segment = segments.remove(segment_id)
            .ok_or_else(|| IPCError::SegmentNotFound { 
                segment_id: segment_id.to_string() 
            })?;
        
        *allocated = allocated.saturating_sub(segment.size_bytes);
        
        Ok(())
    }
    
    pub fn get_utilization_percent(&self) -> f64 {
        let allocated = self.allocated_bytes.lock().unwrap();
        (*allocated as f64 / self.size_bytes as f64) * 100.0
    }
    
    pub fn get_available_bytes(&self) -> usize {
        let allocated = self.allocated_bytes.lock().unwrap();
        self.size_bytes.saturating_sub(*allocated)
    }
}

impl IPCStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    pub fn get_throughput_mb_per_sec(&self, duration_secs: f64) -> f64 {
        if duration_secs > 0.0 {
            (self.bytes_transferred as f64 / 1024.0 / 1024.0) / duration_secs
        } else {
            0.0
        }
    }
}

impl Default for IPCStats {
    fn default() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_transferred: 0,
            memory_allocated_bytes: 0,
            errors_count: 0,
            average_latency_ms: 0.0,
        }
    }
}

impl IPCMessage {
    pub fn new(message_type: MessageType, payload: Vec<u8>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let checksum = Self::calculate_checksum(&payload);
        
        Self {
            id: Uuid::new_v4(),
            message_type,
            payload,
            timestamp,
            checksum,
        }
    }
    
    fn calculate_checksum(data: &[u8]) -> u32 {
        data.iter().enumerate().fold(0u32, |acc, (i, &byte)| {
            acc.wrapping_add((byte as u32) * (i as u32 + 1))
        })
    }
    
    pub fn serialize(&self) -> IPCResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| IPCError::SerializationError(e.to_string()))
    }
    
    pub fn deserialize(data: &[u8]) -> IPCResult<Self> {
        serde_json::from_slice(data)
            .map_err(|e| IPCError::SerializationError(e.to_string()))
    }
}

// ============================================================================
// COMPREHENSIVE IPC TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn create_test_config() -> IPCConfig {
        IPCConfig {
            pool_size_gb: 1.0, // 1GB for testing
            max_message_size_mb: 100,
            timeout_ms: 5000,
            enable_checksum: true,
            enable_compression: false,
        }
    }

    fn create_test_message(size_bytes: usize) -> IPCMessage {
        let payload = vec![42u8; size_bytes];
        IPCMessage::new(MessageType::DocumentRequest, payload)
    }

    // ========================================================================
    // Basic IPC Manager Tests
    // ========================================================================

    #[test]
    fn test_ipc_manager_creation() {
        let config = create_test_config();
        let manager = IPCManager::new(config.clone());
        
        assert!(manager.is_ok());
        let mgr = manager.unwrap();
        assert_eq!(mgr.config.pool_size_gb, 1.0);
        assert_eq!(mgr.config.max_message_size_mb, 100);
        assert!(mgr.config.enable_checksum);
    }

    #[test]
    fn test_message_creation_and_serialization() {
        let payload = b"Hello, IPC World!".to_vec();
        let message = IPCMessage::new(MessageType::DocumentRequest, payload.clone());
        
        assert_eq!(message.message_type, MessageType::DocumentRequest);
        assert_eq!(message.payload, payload);
        assert!(!message.id.is_nil());
        assert!(message.timestamp > 0);
        assert_ne!(message.checksum, 0);
        
        // Test serialization round-trip
        let serialized = message.serialize().unwrap();
        let deserialized = IPCMessage::deserialize(&serialized).unwrap();
        
        assert_eq!(message.id, deserialized.id);
        assert_eq!(message.message_type, deserialized.message_type);
        assert_eq!(message.payload, deserialized.payload);
        assert_eq!(message.checksum, deserialized.checksum);
    }

    #[test]
    fn test_message_types() {
        let message_types = vec![
            MessageType::DocumentRequest,
            MessageType::DocumentResponse,
            MessageType::MLProcessingRequest,
            MessageType::MLProcessingResponse,
            MessageType::StatusUpdate,
            MessageType::Error,
            MessageType::Heartbeat,
        ];
        
        for msg_type in message_types {
            let message = IPCMessage::new(msg_type.clone(), b"test".to_vec());
            assert_eq!(message.message_type, msg_type);
        }
    }

    // ========================================================================
    // Message Passing Tests
    // ========================================================================

    #[test]
    fn test_send_and_receive_message() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        let message = create_test_message(1024);
        let message_id = message.id;
        
        // Send message
        let send_result = manager.send_message(message);
        assert!(send_result.is_ok());
        
        // Receive message
        let received = manager.receive_message(Some(1000)).unwrap();
        assert_eq!(received.id, message_id);
        assert_eq!(received.payload.len(), 1024);
        
        // Check stats were updated
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_transferred, 1024);
    }

    #[test]
    fn test_message_size_validation() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Create oversized message (101MB > 100MB limit)
        let oversized_message = create_test_message(101 * 1024 * 1024);
        let result = manager.send_message(oversized_message);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            IPCError::InvalidMessage(msg) => {
                assert!(msg.contains("exceeds limit"));
                assert!(msg.contains("101MB"));
            }
            _ => panic!("Expected InvalidMessage error"),
        }
    }

    #[test]
    fn test_checksum_validation() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Create message with invalid checksum
        let mut message = create_test_message(1024);
        message.checksum = 12345; // Wrong checksum
        
        let result = manager.send_message(message);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IPCError::ChecksumMismatch { expected, actual } => {
                assert_eq!(expected, 12345);
                assert_ne!(actual, 12345);
            }
            _ => panic!("Expected ChecksumMismatch error"),
        }
    }

    #[test]
    fn test_receive_timeout() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        let start_time = Instant::now();
        let result = manager.receive_message(Some(100)); // 100ms timeout
        let duration = start_time.elapsed();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            IPCError::TimeoutError { timeout_ms } => {
                assert_eq!(timeout_ms, 100);
                assert!(duration.as_millis() >= 100);
                assert!(duration.as_millis() < 200); // Should not take much longer
            }
            _ => panic!("Expected TimeoutError"),
        }
    }

    // ========================================================================
    // Shared Memory Tests
    // ========================================================================

    #[test]
    fn test_memory_pool_creation() {
        let pool = SharedMemoryPool::new(1, 1024 * 1024).unwrap(); // 1MB pool
        
        assert_eq!(pool.pool_id, 1);
        assert_eq!(pool.size_bytes, 1024 * 1024);
        assert_eq!(pool.get_utilization_percent(), 0.0);
        assert_eq!(pool.get_available_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_memory_allocation_and_deallocation() {
        let pool = SharedMemoryPool::new(1, 1024 * 1024).unwrap(); // 1MB pool
        
        // Allocate memory segment
        let segment_id = pool.allocate_segment(512 * 1024).unwrap(); // 512KB
        assert!(!segment_id.is_empty());
        assert_eq!(pool.get_utilization_percent(), 50.0);
        assert_eq!(pool.get_available_bytes(), 512 * 1024);
        
        // Deallocate memory segment
        let dealloc_result = pool.deallocate_segment(&segment_id);
        assert!(dealloc_result.is_ok());
        assert_eq!(pool.get_utilization_percent(), 0.0);
        assert_eq!(pool.get_available_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_memory_pool_exhaustion() {
        let pool = SharedMemoryPool::new(1, 1024).unwrap(); // 1KB pool
        
        // First allocation should succeed
        let segment1 = pool.allocate_segment(512);
        assert!(segment1.is_ok());
        
        // Second allocation should succeed
        let segment2 = pool.allocate_segment(512);
        assert!(segment2.is_ok());
        
        // Third allocation should fail (pool exhausted)
        let segment3 = pool.allocate_segment(1);
        assert!(segment3.is_err());
        
        match segment3.unwrap_err() {
            IPCError::MemoryPoolExhausted { requested, available } => {
                assert!(requested > 0.0);
                assert_eq!(available, 0.0);
            }
            _ => panic!("Expected MemoryPoolExhausted error"),
        }
    }

    #[test]
    fn test_memory_manager_integration() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Allocate memory through manager
        let segment_id = manager.allocate_memory(1, 1024 * 1024).unwrap(); // 1MB
        assert!(!segment_id.is_empty());
        
        // Check memory utilization
        let utilization = manager.get_memory_utilization(1).unwrap();
        assert!(utilization > 0.0);
        assert!(utilization < 100.0);
        
        // Deallocate memory
        let dealloc_result = manager.deallocate_memory(1, &segment_id);
        assert!(dealloc_result.is_ok());
        
        // Check utilization dropped
        let final_utilization = manager.get_memory_utilization(1).unwrap();
        assert!(final_utilization < utilization);
    }

    #[test]
    fn test_invalid_pool_access() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Try to allocate from non-existent pool
        let result = manager.allocate_memory(999, 1024);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IPCError::SegmentNotFound { segment_id } => {
                assert_eq!(segment_id, "pool_999");
            }
            _ => panic!("Expected SegmentNotFound error"),
        }
    }

    // ========================================================================
    // Concurrent Access Tests
    // ========================================================================

    #[test]
    fn test_concurrent_message_passing() {
        let config = create_test_config();
        let manager = Arc::new(IPCManager::new(config).unwrap());
        
        let num_threads = 4;
        let messages_per_thread = 10;
        
        // Spawn sender threads
        let sender_handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let mgr = Arc::clone(&manager);
                thread::spawn(move || {
                    for j in 0..messages_per_thread {
                        let payload = format!("Thread {} Message {}", i, j).into_bytes();
                        let message = IPCMessage::new(MessageType::DocumentRequest, payload);
                        mgr.send_message(message).unwrap();
                    }
                })
            })
            .collect();
        
        // Spawn receiver threads
        let receiver_handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mgr = Arc::clone(&manager);
                thread::spawn(move || {
                    let mut received_count = 0;
                    for _ in 0..messages_per_thread {
                        match mgr.receive_message(Some(5000)) {
                            Ok(_) => received_count += 1,
                            Err(_) => break, // Timeout or other error
                        }
                    }
                    received_count
                })
            })
            .collect();
        
        // Wait for all threads to complete
        for handle in sender_handles {
            handle.join().unwrap();
        }
        
        let mut total_received = 0;
        for handle in receiver_handles {
            total_received += handle.join().unwrap();
        }
        
        // Verify all messages were processed
        let expected_total = num_threads * messages_per_thread;
        assert_eq!(total_received, expected_total);
        
        // Check final stats
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.messages_sent as usize, expected_total);
        assert_eq!(stats.messages_received as usize, expected_total);
    }

    #[test]
    fn test_concurrent_memory_allocation() {
        let config = create_test_config();
        let manager = Arc::new(IPCManager::new(config).unwrap());
        
        let num_threads = 8;
        let allocations_per_thread = 10;
        let allocation_size = 1024; // 1KB per allocation
        
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mgr = Arc::clone(&manager);
                thread::spawn(move || {
                    let mut segment_ids = Vec::new();
                    
                    // Allocate memory
                    for _ in 0..allocations_per_thread {
                        match mgr.allocate_memory(1, allocation_size) {
                            Ok(segment_id) => segment_ids.push(segment_id),
                            Err(_) => break, // Pool exhausted
                        }
                    }
                    
                    // Deallocate memory
                    for segment_id in segment_ids {
                        let _ = mgr.deallocate_memory(1, &segment_id);
                    }
                })
            })
            .collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Memory pool should be mostly empty after cleanup
        let final_utilization = manager.get_memory_utilization(1).unwrap();
        assert!(final_utilization < 50.0); // Allow some fragmentation
    }

    // ========================================================================
    // Performance and Statistics Tests
    // ========================================================================

    #[test]
    fn test_statistics_tracking() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Initial stats should be zero
        let initial_stats = manager.get_stats().unwrap();
        assert_eq!(initial_stats.messages_sent, 0);
        assert_eq!(initial_stats.messages_received, 0);
        assert_eq!(initial_stats.bytes_transferred, 0);
        
        // Send and receive some messages
        for i in 0..5 {
            let message = create_test_message(100 * (i + 1)); // Variable sizes
            manager.send_message(message).unwrap();
            let _ = manager.receive_message(Some(1000)).unwrap();
        }
        
        // Check updated stats
        let final_stats = manager.get_stats().unwrap();
        assert_eq!(final_stats.messages_sent, 5);
        assert_eq!(final_stats.messages_received, 5);
        assert_eq!(final_stats.bytes_transferred, 1500); // 100+200+300+400+500
        assert!(final_stats.average_latency_ms >= 0.0);
    }

    #[test]
    fn test_throughput_calculation() {
        let mut stats = IPCStats::new();
        stats.bytes_transferred = 10 * 1024 * 1024; // 10MB
        
        let throughput = stats.get_throughput_mb_per_sec(2.0); // 2 seconds
        assert_eq!(throughput, 5.0); // 10MB / 2s = 5MB/s
        
        // Test zero duration
        let zero_throughput = stats.get_throughput_mb_per_sec(0.0);
        assert_eq!(zero_throughput, 0.0);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = IPCStats::new();
        stats.messages_sent = 100;
        stats.bytes_transferred = 5000;
        stats.average_latency_ms = 10.5;
        
        assert_ne!(stats.messages_sent, 0);
        
        stats.reset();
        
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.bytes_transferred, 0);
        assert_eq!(stats.average_latency_ms, 0.0);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_error_types_formatting() {
        let errors = vec![
            IPCError::MemoryPoolExhausted { requested: 1.5, available: 0.5 },
            IPCError::SerializationError("test error".to_string()),
            IPCError::TimeoutError { timeout_ms: 5000 },
            IPCError::InvalidMessage("bad format".to_string()),
            IPCError::ChannelClosed,
            IPCError::SegmentNotFound { segment_id: "seg_123".to_string() },
            IPCError::ChecksumMismatch { expected: 123, actual: 456 },
        ];
        
        for error in errors {
            let error_string = format!("{}", error);
            assert!(!error_string.is_empty());
            
            // Check that error contains relevant information
            match &error {
                IPCError::MemoryPoolExhausted { requested, available } => {
                    assert!(error_string.contains(&requested.to_string()));
                    assert!(error_string.contains(&available.to_string()));
                }
                IPCError::TimeoutError { timeout_ms } => {
                    assert!(error_string.contains(&timeout_ms.to_string()));
                }
                IPCError::SegmentNotFound { segment_id } => {
                    assert!(error_string.contains(segment_id));
                }
                IPCError::ChecksumMismatch { expected, actual } => {
                    assert!(error_string.contains(&expected.to_string()));
                    assert!(error_string.contains(&actual.to_string()));
                }
                _ => {} // Other errors don't have specific data to check
            }
        }
    }

    // ========================================================================
    // Real-world Scenario Tests
    // ========================================================================

    #[test]
    fn test_document_processing_workflow() {
        let config = create_test_config();
        let manager = Arc::new(IPCManager::new(config).unwrap());
        
        // Simulate Rust sending document to Python for ML processing
        let document_data = serde_json::to_vec(&serde_json::json!({
            "id": "doc_123",
            "content": "This is a test document with LTE parameters",
            "metadata": {
                "parameters": ["param1", "param2"],
                "counters": ["counter1"],
                "complexity": "balanced"
            }
        })).unwrap();
        
        let request = IPCMessage::new(MessageType::MLProcessingRequest, document_data);
        let request_id = request.id;
        
        // Send processing request
        manager.send_message(request).unwrap();
        
        // Simulate Python receiving and processing
        let received_request = manager.receive_message(Some(1000)).unwrap();
        assert_eq!(received_request.id, request_id);
        assert_eq!(received_request.message_type, MessageType::MLProcessingRequest);
        
        // Simulate Python sending response
        let response_data = serde_json::to_vec(&serde_json::json!({
            "request_id": request_id.to_string(),
            "qa_pairs": [
                {"question": "What is param1?", "answer": "Parameter 1 description"},
                {"question": "What is counter1?", "answer": "Counter 1 description"}
            ],
            "processing_time_ms": 1500,
            "model_used": "qwen3-30b"
        })).unwrap();
        
        let response = IPCMessage::new(MessageType::MLProcessingResponse, response_data);
        manager.send_message(response).unwrap();
        
        // Simulate Rust receiving response
        let received_response = manager.receive_message(Some(1000)).unwrap();
        assert_eq!(received_response.message_type, MessageType::MLProcessingResponse);
        
        // Verify workflow completed
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.messages_sent, 2);
        assert_eq!(stats.messages_received, 2);
    }

    #[test]
    fn test_heartbeat_monitoring() {
        let config = create_test_config();
        let manager = Arc::new(IPCManager::new(config).unwrap());
        
        // Send heartbeat messages
        for i in 0..5 {
            let heartbeat_data = format!("heartbeat_{}", i).into_bytes();
            let heartbeat = IPCMessage::new(MessageType::Heartbeat, heartbeat_data);
            manager.send_message(heartbeat).unwrap();
        }
        
        // Verify heartbeats can be received
        let mut heartbeat_count = 0;
        for _ in 0..5 {
            match manager.receive_message(Some(100)) {
                Ok(msg) if msg.message_type == MessageType::Heartbeat => {
                    heartbeat_count += 1;
                }
                Ok(_) => {} // Other message types
                Err(_) => break, // Timeout
            }
        }
        
        assert_eq!(heartbeat_count, 5);
    }

    #[test]
    fn test_error_message_handling() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        // Send error message
        let error_data = serde_json::to_vec(&serde_json::json!({
            "error_type": "ProcessingError",
            "message": "Document parsing failed",
            "document_id": "doc_456",
            "timestamp": 1234567890
        })).unwrap();
        
        let error_message = IPCMessage::new(MessageType::Error, error_data);
        manager.send_message(error_message).unwrap();
        
        // Receive and verify error message
        let received_error = manager.receive_message(Some(1000)).unwrap();
        assert_eq!(received_error.message_type, MessageType::Error);
        
        // Parse error content
        let error_json: serde_json::Value = serde_json::from_slice(&received_error.payload).unwrap();
        assert_eq!(error_json["error_type"], "ProcessingError");
        assert_eq!(error_json["document_id"], "doc_456");
    }

    // ========================================================================
    // Performance Benchmarks
    // ========================================================================

    #[test]
    fn test_message_passing_performance() {
        let config = create_test_config();
        let manager = IPCManager::new(config).unwrap();
        
        let message_count = 1000;
        let message_size = 1024; // 1KB messages
        
        // Benchmark sending
        let start_time = Instant::now();
        for _ in 0..message_count {
            let message = create_test_message(message_size);
            manager.send_message(message).unwrap();
        }
        let send_duration = start_time.elapsed();
        
        // Benchmark receiving
        let start_time = Instant::now();
        for _ in 0..message_count {
            let _ = manager.receive_message(Some(5000)).unwrap();
        }
        let receive_duration = start_time.elapsed();
        
        let send_throughput = message_count as f64 / send_duration.as_secs_f64();
        let receive_throughput = message_count as f64 / receive_duration.as_secs_f64();
        
        println!("Send throughput: {:.0} messages/sec", send_throughput);
        println!("Receive throughput: {:.0} messages/sec", receive_throughput);
        
        // Performance expectations (adjust based on hardware)
        assert!(send_throughput > 1000.0, "Send throughput too low: {}", send_throughput);
        assert!(receive_throughput > 1000.0, "Receive throughput too low: {}", receive_throughput);
    }

    #[test]
    fn test_memory_allocation_performance() {
        let pool = SharedMemoryPool::new(1, 100 * 1024 * 1024).unwrap(); // 100MB pool
        
        let allocation_count = 1000;
        let allocation_size = 1024; // 1KB allocations
        
        // Benchmark allocations
        let start_time = Instant::now();
        let mut segment_ids = Vec::new();
        
        for _ in 0..allocation_count {
            match pool.allocate_segment(allocation_size) {
                Ok(segment_id) => segment_ids.push(segment_id),
                Err(_) => break, // Pool exhausted
            }
        }
        
        let allocation_duration = start_time.elapsed();
        let allocation_throughput = segment_ids.len() as f64 / allocation_duration.as_secs_f64();
        
        // Benchmark deallocations
        let start_time = Instant::now();
        for segment_id in segment_ids {
            let _ = pool.deallocate_segment(&segment_id);
        }
        let deallocation_duration = start_time.elapsed();
        let deallocation_throughput = allocation_count as f64 / deallocation_duration.as_secs_f64();
        
        println!("Allocation throughput: {:.0} allocs/sec", allocation_throughput);
        println!("Deallocation throughput: {:.0} deallocs/sec", deallocation_throughput);
        
        // Performance expectations
        assert!(allocation_throughput > 10000.0, "Allocation throughput too low: {}", allocation_throughput);
        assert!(deallocation_throughput > 10000.0, "Deallocation throughput too low: {}", deallocation_throughput);
    }
}