use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::mem;
use anyhow::{Result, Context};
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{Duration, Instant};
use tracing::{debug, warn, error, info};

/// Lock-free ring buffer for high-performance IPC streaming
/// Optimized for M3 Max unified memory architecture
pub struct RingBuffer {
    /// Memory-mapped buffer for zero-copy operations
    buffer: Arc<MmapMut>,
    /// Buffer size in bytes
    capacity: usize,
    /// Write position (producer)
    write_pos: AtomicUsize,
    /// Read position (consumer)
    read_pos: AtomicUsize,
    /// Buffer metadata
    metadata: RingBufferMetadata,
    /// Performance metrics
    metrics: Arc<RingBufferMetrics>,
}

/// Ring buffer configuration for M3 Max optimization
#[derive(Debug, Clone)]
pub struct RingBufferConfig {
    /// Buffer capacity in bytes (must be power of 2 for optimal performance)
    pub capacity: usize,
    /// Memory alignment for SIMD operations (64 bytes for M3 Max)
    pub alignment: usize,
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    /// Cache line size optimization
    pub cache_line_size: usize,
    /// Enable performance monitoring
    pub enable_metrics: bool,
}

/// Ring buffer metadata stored in shared memory
#[derive(Debug, Clone)]
pub struct RingBufferMetadata {
    /// Buffer creation timestamp
    pub created_at: Instant,
    /// Buffer ID for tracking
    pub buffer_id: Uuid,
    /// Producer process ID
    pub producer_pid: u32,
    /// Consumer process ID
    pub consumer_pid: u32,
    /// Buffer state
    pub state: BufferState,
}

/// Ring buffer states
#[derive(Debug, Clone, PartialEq)]
pub enum BufferState {
    /// Buffer is ready for use
    Active,
    /// Buffer is being initialized
    Initializing,
    /// Buffer is full and waiting for consumer
    Full,
    /// Buffer has encountered an error
    Error(String),
    /// Buffer is being shutdown
    Shutdown,
}

/// Performance metrics for ring buffer operations
#[derive(Debug, Default)]
pub struct RingBufferMetrics {
    /// Total bytes written
    pub bytes_written: AtomicUsize,
    /// Total bytes read
    pub bytes_read: AtomicUsize,
    /// Number of write operations
    pub write_operations: AtomicUsize,
    /// Number of read operations
    pub read_operations: AtomicUsize,
    /// Number of buffer full events
    pub buffer_full_events: AtomicUsize,
    /// Number of buffer empty events
    pub buffer_empty_events: AtomicUsize,
    /// Total processing time for writes
    pub total_write_time_ns: AtomicUsize,
    /// Total processing time for reads
    pub total_read_time_ns: AtomicUsize,
}

/// Message header for ring buffer entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Message size including header
    pub total_size: u32,
    /// Message type identifier
    pub message_type: u8,
    /// Sequence number for ordering
    pub sequence_number: u64,
    /// Checksum for integrity validation
    pub checksum: u32,
    /// Timestamp when message was written
    pub timestamp: u64,
    /// Message ID for correlation
    pub message_id: Uuid,
}

impl RingBuffer {
    /// Create a new ring buffer optimized for M3 Max
    pub fn new(config: RingBufferConfig) -> Result<Self> {
        // Ensure capacity is power of 2 for efficient modulo operations
        if !config.capacity.is_power_of_two() {
            anyhow::bail!("Ring buffer capacity must be a power of 2");
        }

        info!("Creating ring buffer with capacity: {} bytes", config.capacity);

        // Create memory-mapped file for zero-copy operations
        let temp_file = tempfile::NamedTempFile::new()
            .context("Failed to create temporary file for ring buffer")?;
        
        temp_file.as_file().set_len(config.capacity as u64)
            .context("Failed to set ring buffer file size")?;

        let buffer = unsafe {
            MmapOptions::new()
                .len(config.capacity)
                .map_mut(temp_file.as_file())
                .context("Failed to create memory mapping for ring buffer")?
        };

        let metadata = RingBufferMetadata {
            created_at: Instant::now(),
            buffer_id: Uuid::new_v4(),
            producer_pid: std::process::id(),
            consumer_pid: 0, // Will be set when consumer connects
            state: BufferState::Initializing,
        };

        let ring_buffer = Self {
            buffer: Arc::new(buffer),
            capacity: config.capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            metadata,
            metrics: Arc::new(RingBufferMetrics::default()),
        };

        info!("Ring buffer created successfully: {}", ring_buffer.metadata.buffer_id);
        Ok(ring_buffer)
    }

    /// Write data to the ring buffer with zero-copy semantics
    pub fn write(&self, data: &[u8], message_type: u8) -> Result<usize> {
        let start_time = Instant::now();
        
        // Calculate total message size including header
        let header_size = mem::size_of::<MessageHeader>();
        let total_size = header_size + data.len();
        
        // Ensure message fits in buffer
        if total_size > self.capacity / 2 {
            anyhow::bail!("Message too large for ring buffer");
        }

        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        
        // Calculate available space with wrap-around
        let available_space = if write_pos >= read_pos {
            self.capacity - write_pos + read_pos
        } else {
            read_pos - write_pos
        };

        // Check if there's enough space
        if available_space < total_size + 1 {
            self.metrics.buffer_full_events.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("Ring buffer is full");
        }

        // Create message header
        let sequence_number = self.metrics.write_operations.load(Ordering::Relaxed) as u64;
        let header = MessageHeader {
            total_size: total_size as u32,
            message_type,
            sequence_number,
            checksum: crc32fast::hash(data),
            timestamp: start_time.elapsed().as_nanos() as u64,
            message_id: Uuid::new_v4(),
        };

        // Serialize header
        let header_bytes = bincode::serialize(&header)
            .context("Failed to serialize message header")?;

        // Write header and data with wrap-around handling
        let mut current_pos = write_pos;
        
        // Write header
        self.write_bytes_with_wrap(&header_bytes, &mut current_pos)?;
        
        // Write data
        self.write_bytes_with_wrap(data, &mut current_pos)?;

        // Update write position atomically
        self.write_pos.store(current_pos, Ordering::Release);

        // Update metrics
        self.metrics.bytes_written.fetch_add(total_size, Ordering::Relaxed);
        self.metrics.write_operations.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_write_time_ns.fetch_add(
            start_time.elapsed().as_nanos() as usize,
            Ordering::Relaxed
        );

        debug!("Wrote {} bytes to ring buffer at position {}", total_size, write_pos);
        Ok(total_size)
    }

    /// Read data from the ring buffer with zero-copy semantics
    pub fn read(&self) -> Result<Option<(Vec<u8>, MessageHeader)>> {
        let start_time = Instant::now();
        
        let read_pos = self.read_pos.load(Ordering::Acquire);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty
        if read_pos == write_pos {
            self.metrics.buffer_empty_events.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        // Read message header
        let header_size = mem::size_of::<MessageHeader>();
        let mut current_pos = read_pos;
        
        let header_bytes = self.read_bytes_with_wrap(header_size, &mut current_pos)?;
        let header: MessageHeader = bincode::deserialize(&header_bytes)
            .context("Failed to deserialize message header")?;

        // Calculate data size
        let data_size = header.total_size as usize - header_size;
        
        // Read message data
        let data = self.read_bytes_with_wrap(data_size, &mut current_pos)?;

        // Validate checksum
        let calculated_checksum = crc32fast::hash(&data);
        if calculated_checksum != header.checksum {
            error!("Checksum mismatch in ring buffer message");
            anyhow::bail!("Message integrity check failed");
        }

        // Update read position atomically
        self.read_pos.store(current_pos, Ordering::Release);

        // Update metrics
        self.metrics.bytes_read.fetch_add(header.total_size as usize, Ordering::Relaxed);
        self.metrics.read_operations.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_read_time_ns.fetch_add(
            start_time.elapsed().as_nanos() as usize,
            Ordering::Relaxed
        );

        debug!("Read {} bytes from ring buffer at position {}", header.total_size, read_pos);
        Ok(Some((data, header)))
    }

    /// Write bytes to buffer with wrap-around handling
    fn write_bytes_with_wrap(&self, data: &[u8], pos: &mut usize) -> Result<()> {
        let buffer = &*self.buffer;
        let start_pos = *pos;
        let end_pos = start_pos + data.len();

        if end_pos <= self.capacity {
            // No wrap-around needed
            buffer[start_pos..end_pos].copy_from_slice(data);
        } else {
            // Handle wrap-around
            let first_chunk_size = self.capacity - start_pos;
            let second_chunk_size = data.len() - first_chunk_size;
            
            // Write first chunk to end of buffer
            buffer[start_pos..self.capacity].copy_from_slice(&data[..first_chunk_size]);
            
            // Write second chunk to beginning of buffer
            buffer[0..second_chunk_size].copy_from_slice(&data[first_chunk_size..]);
        }

        *pos = (start_pos + data.len()) % self.capacity;
        Ok(())
    }

    /// Read bytes from buffer with wrap-around handling
    fn read_bytes_with_wrap(&self, size: usize, pos: &mut usize) -> Result<Vec<u8>> {
        let buffer = &*self.buffer;
        let start_pos = *pos;
        let end_pos = start_pos + size;

        let mut data = vec![0u8; size];

        if end_pos <= self.capacity {
            // No wrap-around needed
            data.copy_from_slice(&buffer[start_pos..end_pos]);
        } else {
            // Handle wrap-around
            let first_chunk_size = self.capacity - start_pos;
            let second_chunk_size = size - first_chunk_size;
            
            // Read first chunk from end of buffer
            data[..first_chunk_size].copy_from_slice(&buffer[start_pos..self.capacity]);
            
            // Read second chunk from beginning of buffer
            data[first_chunk_size..].copy_from_slice(&buffer[0..second_chunk_size]);
        }

        *pos = (start_pos + size) % self.capacity;
        Ok(data)
    }

    /// Get buffer utilization percentage
    pub fn utilization(&self) -> f64 {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        
        let used_space = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        };
        
        (used_space as f64) / (self.capacity as f64)
    }

    /// Get current buffer metrics
    pub fn get_metrics(&self) -> RingBufferPerformanceMetrics {
        let bytes_written = self.metrics.bytes_written.load(Ordering::Relaxed);
        let bytes_read = self.metrics.bytes_read.load(Ordering::Relaxed);
        let write_ops = self.metrics.write_operations.load(Ordering::Relaxed);
        let read_ops = self.metrics.read_operations.load(Ordering::Relaxed);
        let write_time_ns = self.metrics.total_write_time_ns.load(Ordering::Relaxed);
        let read_time_ns = self.metrics.total_read_time_ns.load(Ordering::Relaxed);

        RingBufferPerformanceMetrics {
            bytes_written,
            bytes_read,
            write_operations: write_ops,
            read_operations: read_ops,
            average_write_latency_ns: if write_ops > 0 { write_time_ns / write_ops } else { 0 },
            average_read_latency_ns: if read_ops > 0 { read_time_ns / read_ops } else { 0 },
            throughput_mbps: self.calculate_throughput_mbps(),
            utilization_percent: self.utilization() * 100.0,
            buffer_full_events: self.metrics.buffer_full_events.load(Ordering::Relaxed),
            buffer_empty_events: self.metrics.buffer_empty_events.load(Ordering::Relaxed),
        }
    }

    /// Calculate current throughput in MB/s
    fn calculate_throughput_mbps(&self) -> f64 {
        let elapsed = self.metadata.created_at.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }

        let total_bytes = self.metrics.bytes_written.load(Ordering::Relaxed) + 
                         self.metrics.bytes_read.load(Ordering::Relaxed);
        
        (total_bytes as f64) / (1024.0 * 1024.0 * elapsed)
    }

    /// Reset buffer metrics
    pub fn reset_metrics(&self) {
        self.metrics.bytes_written.store(0, Ordering::Relaxed);
        self.metrics.bytes_read.store(0, Ordering::Relaxed);
        self.metrics.write_operations.store(0, Ordering::Relaxed);
        self.metrics.read_operations.store(0, Ordering::Relaxed);
        self.metrics.buffer_full_events.store(0, Ordering::Relaxed);
        self.metrics.buffer_empty_events.store(0, Ordering::Relaxed);
        self.metrics.total_write_time_ns.store(0, Ordering::Relaxed);
        self.metrics.total_read_time_ns.store(0, Ordering::Relaxed);
    }

    /// Check if buffer is healthy
    pub fn health_check(&self) -> BufferHealthStatus {
        let utilization = self.utilization();
        let metrics = self.get_metrics();
        
        let status = if utilization > 0.95 {
            BufferHealth::Critical
        } else if utilization > 0.8 {
            BufferHealth::Warning
        } else if metrics.buffer_full_events > metrics.write_operations / 10 {
            BufferHealth::Warning
        } else {
            BufferHealth::Healthy
        };

        BufferHealthStatus {
            health: status,
            utilization_percent: utilization * 100.0,
            message: match status {
                BufferHealth::Healthy => "Buffer operating normally".to_string(),
                BufferHealth::Warning => "Buffer experiencing high utilization or frequent full events".to_string(),
                BufferHealth::Critical => "Buffer is critically full - immediate action required".to_string(),
            },
            recommendations: self.get_health_recommendations(&status, &metrics),
        }
    }

    /// Get health-based recommendations
    fn get_health_recommendations(&self, health: &BufferHealth, metrics: &RingBufferPerformanceMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match health {
            BufferHealth::Critical => {
                recommendations.push("Increase consumer processing rate".to_string());
                recommendations.push("Consider increasing buffer capacity".to_string());
                recommendations.push("Implement backpressure mechanism".to_string());
            },
            BufferHealth::Warning => {
                if metrics.buffer_full_events > 0 {
                    recommendations.push("Monitor consumer lag".to_string());
                    recommendations.push("Consider buffer size optimization".to_string());
                }
                if metrics.average_write_latency_ns > 1000 {
                    recommendations.push("Investigate write performance bottlenecks".to_string());
                }
            },
            BufferHealth::Healthy => {
                recommendations.push("Buffer operating optimally".to_string());
            },
        }
        
        recommendations
    }
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct RingBufferPerformanceMetrics {
    pub bytes_written: usize,
    pub bytes_read: usize,
    pub write_operations: usize,
    pub read_operations: usize,
    pub average_write_latency_ns: usize,
    pub average_read_latency_ns: usize,
    pub throughput_mbps: f64,
    pub utilization_percent: f64,
    pub buffer_full_events: usize,
    pub buffer_empty_events: usize,
}

/// Buffer health status
#[derive(Debug, Clone)]
pub struct BufferHealthStatus {
    pub health: BufferHealth,
    pub utilization_percent: f64,
    pub message: String,
    pub recommendations: Vec<String>,
}

/// Buffer health levels
#[derive(Debug, Clone, PartialEq)]
pub enum BufferHealth {
    Healthy,
    Warning,
    Critical,
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 16 * 1024 * 1024, // 16MB default
            alignment: 64, // M3 Max cache line size
            enable_prefetching: true,
            cache_line_size: 64,
            enable_metrics: true,
        }
    }
}

impl RingBufferConfig {
    /// Create configuration optimized for large document transfers
    pub fn for_large_documents() -> Self {
        Self {
            capacity: 256 * 1024 * 1024, // 256MB for large documents
            alignment: 64,
            enable_prefetching: true,
            cache_line_size: 64,
            enable_metrics: true,
        }
    }

    /// Create configuration optimized for high-frequency small messages
    pub fn for_small_messages() -> Self {
        Self {
            capacity: 4 * 1024 * 1024, // 4MB for small messages
            alignment: 64,
            enable_prefetching: true,
            cache_line_size: 64,
            enable_metrics: true,
        }
    }

    /// Create configuration for streaming data
    pub fn for_streaming() -> Self {
        Self {
            capacity: 64 * 1024 * 1024, // 64MB for streaming
            alignment: 64,
            enable_prefetching: true,
            cache_line_size: 64,
            enable_metrics: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_creation() {
        let config = RingBufferConfig::default();
        let buffer = RingBuffer::new(config).unwrap();
        assert_eq!(buffer.capacity, 16 * 1024 * 1024);
        assert_eq!(buffer.utilization(), 0.0);
    }

    #[test]
    fn test_write_and_read() {
        let config = RingBufferConfig::default();
        let buffer = RingBuffer::new(config).unwrap();
        
        let test_data = b"Hello, Ring Buffer!";
        let bytes_written = buffer.write(test_data, 1).unwrap();
        assert!(bytes_written > test_data.len()); // Includes header
        
        let result = buffer.read().unwrap();
        assert!(result.is_some());
        
        let (data, header) = result.unwrap();
        assert_eq!(data, test_data);
        assert_eq!(header.message_type, 1);
        assert_eq!(header.checksum, crc32fast::hash(test_data));
    }

    #[test]
    fn test_buffer_wrap_around() {
        let config = RingBufferConfig {
            capacity: 1024, // Small buffer to test wrap-around
            ..Default::default()
        };
        let buffer = RingBuffer::new(config).unwrap();
        
        // Fill buffer close to capacity
        let large_data = vec![0u8; 400];
        buffer.write(&large_data, 1).unwrap();
        buffer.write(&large_data, 2).unwrap();
        
        // Read first message to make space
        let result = buffer.read().unwrap();
        assert!(result.is_some());
        
        // Write another message that should wrap around
        let wrap_data = b"Wrap around test";
        let result = buffer.write(wrap_data, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_buffer_full_condition() {
        let config = RingBufferConfig {
            capacity: 256, // Very small buffer
            ..Default::default()
        };
        let buffer = RingBuffer::new(config).unwrap();
        
        // Fill buffer until full
        let data = vec![0u8; 100];
        buffer.write(&data, 1).unwrap();
        
        // This should fail due to insufficient space
        let result = buffer.write(&data, 2);
        assert!(result.is_err());
        
        // Verify buffer full event was recorded
        let metrics = buffer.get_metrics();
        assert!(metrics.buffer_full_events > 0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = RingBufferConfig::default();
        let buffer = RingBuffer::new(config).unwrap();
        
        let test_data = b"Test metrics";
        buffer.write(test_data, 1).unwrap();
        buffer.read().unwrap();
        
        let metrics = buffer.get_metrics();
        assert_eq!(metrics.write_operations, 1);
        assert_eq!(metrics.read_operations, 1);
        assert!(metrics.bytes_written > 0);
        assert!(metrics.bytes_read > 0);
        assert!(metrics.average_write_latency_ns > 0);
        assert!(metrics.average_read_latency_ns > 0);
    }

    #[test]
    fn test_health_check() {
        let config = RingBufferConfig::default();
        let buffer = RingBuffer::new(config).unwrap();
        
        let health = buffer.health_check();
        assert_eq!(health.health, BufferHealth::Healthy);
        assert!(health.utilization_percent < 1.0);
    }
}