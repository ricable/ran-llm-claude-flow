//! Shared memory module for zero-copy IPC operations
//! 
//! This module provides high-performance shared memory abstractions optimized
//! for M3 Max unified memory architecture with 15GB allocation strategy.

pub mod ring_buffer;
pub mod memory_pool;

pub use ring_buffer::{
    RingBuffer, 
    RingBufferConfig, 
    RingBufferPerformanceMetrics,
    BufferHealthStatus,
    BufferHealth,
    MessageHeader
};

pub use memory_pool::{
    SharedMemoryPool,
    PoolConfig,
    PoolUtilization,
    AllocationResult,
    BlockType,
    GarbageCollectionResult
};

use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;
use tracing::{info, error};

/// Unified shared memory manager combining ring buffer and memory pool
/// Provides comprehensive zero-copy IPC capabilities
pub struct SharedMemoryManager {
    /// Large document memory pool (15GB)
    memory_pool: Arc<SharedMemoryPool>,
    /// High-frequency streaming ring buffer
    ring_buffer: Arc<RingBuffer>,
    /// Manager configuration
    config: SharedMemoryConfig,
}

/// Configuration for shared memory manager
#[derive(Debug, Clone)]
pub struct SharedMemoryConfig {
    /// Memory pool configuration
    pub pool_config: PoolConfig,
    /// Ring buffer configuration  
    pub ring_buffer_config: RingBufferConfig,
    /// Enable automatic garbage collection
    pub auto_gc_enabled: bool,
    /// Garbage collection interval in seconds
    pub gc_interval_seconds: u64,
    /// Health monitoring enabled
    pub health_monitoring_enabled: bool,
}

/// Comprehensive memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Memory pool utilization
    pub pool_utilization: PoolUtilization,
    /// Ring buffer performance metrics
    pub ring_buffer_metrics: RingBufferPerformanceMetrics,
    /// Overall health status
    pub overall_health: OverallHealth,
}

/// Overall memory system health
#[derive(Debug, Clone, PartialEq)]
pub enum OverallHealth {
    /// All systems operating normally
    Healthy,
    /// Some subsystems showing warnings
    Warning,
    /// Critical issues requiring immediate attention
    Critical,
    /// System is degraded but operational
    Degraded,
}

/// Health status for the entire shared memory system
#[derive(Debug, Clone)]
pub struct SharedMemoryHealth {
    /// Overall health status
    pub health: OverallHealth,
    /// Detailed status message
    pub message: String,
    /// Pool-specific health
    pub pool_health: String,
    /// Ring buffer health
    pub buffer_health: BufferHealthStatus,
    /// Performance recommendations
    pub recommendations: Vec<String>,
    /// Critical alerts (if any)
    pub alerts: Vec<String>,
}

impl SharedMemoryManager {
    /// Create new shared memory manager optimized for M3 Max
    pub async fn new(config: SharedMemoryConfig) -> Result<Self> {
        info!("Initializing Shared Memory Manager");
        info!("Pool size: {} GB", config.pool_config.pool_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        info!("Ring buffer size: {} MB", config.ring_buffer_config.capacity as f64 / (1024.0 * 1024.0));

        // Initialize memory pool
        let memory_pool = Arc::new(SharedMemoryPool::new(config.pool_config.clone())?);
        
        // Initialize ring buffer
        let ring_buffer = Arc::new(RingBuffer::new(config.ring_buffer_config.clone())?);

        let manager = Self {
            memory_pool,
            ring_buffer,
            config,
        };

        // Start background maintenance tasks
        if manager.config.auto_gc_enabled {
            manager.start_garbage_collection_task().await?;
        }

        if manager.config.health_monitoring_enabled {
            manager.start_health_monitoring_task().await?;
        }

        info!("Shared Memory Manager initialized successfully");
        Ok(manager)
    }

    /// Allocate large document buffer (>1MB) using memory pool
    pub fn allocate_document_buffer(&self, size: usize, request_id: Uuid) -> Result<AllocationResult> {
        let block_type = if size > 1024 * 1024 {
            BlockType::LargeDocument
        } else if size > 64 * 1024 {
            BlockType::MediumDocument
        } else {
            BlockType::SmallDocument
        };

        self.memory_pool.allocate(size, request_id, block_type)
    }

    /// Deallocate document buffer
    pub fn deallocate_document_buffer(&self, allocation_id: u64) -> Result<()> {
        self.memory_pool.deallocate(allocation_id)
    }

    /// Write data to document buffer with zero-copy
    pub fn write_document_data(&self, offset: u64, data: &[u8]) -> Result<()> {
        self.memory_pool.write_data(offset, data)
    }

    /// Read data from document buffer with zero-copy
    pub fn read_document_data(&self, offset: u64, size: usize) -> Result<&[u8]> {
        self.memory_pool.read_data(offset, size)
    }

    /// Write streaming message to ring buffer
    pub fn write_stream_message(&self, data: &[u8], message_type: u8) -> Result<usize> {
        self.ring_buffer.write(data, message_type)
    }

    /// Read streaming message from ring buffer
    pub fn read_stream_message(&self) -> Result<Option<(Vec<u8>, MessageHeader)>> {
        self.ring_buffer.read()
    }

    /// Get comprehensive memory statistics
    pub fn get_statistics(&self) -> MemoryStatistics {
        let pool_utilization = self.memory_pool.get_utilization();
        let ring_buffer_metrics = self.ring_buffer.get_metrics();
        
        let overall_health = self.assess_overall_health(&pool_utilization, &ring_buffer_metrics);

        MemoryStatistics {
            pool_utilization,
            ring_buffer_metrics,
            overall_health,
        }
    }

    /// Perform manual garbage collection on memory pool
    pub fn garbage_collect(&self) -> Result<GarbageCollectionResult> {
        self.memory_pool.garbage_collect()
    }

    /// Get comprehensive health status
    pub fn health_check(&self) -> SharedMemoryHealth {
        let pool_utilization = self.memory_pool.get_utilization();
        let buffer_health = self.ring_buffer.health_check();
        let overall_health = self.assess_overall_health(&pool_utilization, &self.ring_buffer.get_metrics());

        let mut recommendations = Vec::new();
        let mut alerts = Vec::new();

        // Pool-specific health assessment
        let pool_health = if pool_utilization.utilization_percent > 95.0 {
            alerts.push("Memory pool utilization critical (>95%)".to_string());
            recommendations.push("Consider increasing pool size or implementing cleanup".to_string());
            "Critical: High utilization".to_string()
        } else if pool_utilization.utilization_percent > 80.0 {
            recommendations.push("Monitor memory pool usage closely".to_string());
            "Warning: High utilization".to_string()
        } else if pool_utilization.fragmentation.fragmentation_ratio > 0.3 {
            recommendations.push("Consider running garbage collection".to_string());
            "Warning: High fragmentation".to_string()
        } else {
            "Healthy".to_string()
        };

        // Add buffer recommendations
        recommendations.extend(buffer_health.recommendations.clone());

        // Overall health message
        let message = match overall_health {
            OverallHealth::Healthy => "All memory systems operating normally".to_string(),
            OverallHealth::Warning => "Some memory systems showing degraded performance".to_string(),
            OverallHealth::Critical => "Critical memory issues detected - immediate action required".to_string(),
            OverallHealth::Degraded => "Memory systems are operational but with reduced performance".to_string(),
        };

        SharedMemoryHealth {
            health: overall_health,
            message,
            pool_health,
            buffer_health,
            recommendations,
            alerts,
        }
    }

    /// Reset all performance metrics
    pub fn reset_metrics(&self) {
        self.ring_buffer.reset_metrics();
        // Note: Memory pool metrics are cumulative and typically not reset
    }

    /// Assess overall system health based on subsystem metrics
    fn assess_overall_health(&self, pool_util: &PoolUtilization, buffer_metrics: &RingBufferPerformanceMetrics) -> OverallHealth {
        let mut issues = 0;
        let mut critical_issues = 0;

        // Pool health assessment
        if pool_util.utilization_percent > 95.0 {
            critical_issues += 1;
        } else if pool_util.utilization_percent > 80.0 {
            issues += 1;
        }

        if pool_util.fragmentation.fragmentation_ratio > 0.5 {
            critical_issues += 1;
        } else if pool_util.fragmentation.fragmentation_ratio > 0.3 {
            issues += 1;
        }

        // Buffer health assessment
        if buffer_metrics.utilization_percent > 95.0 {
            critical_issues += 1;
        } else if buffer_metrics.utilization_percent > 80.0 {
            issues += 1;
        }

        if buffer_metrics.buffer_full_events > buffer_metrics.write_operations / 5 {
            issues += 1;
        }

        // Determine overall health
        if critical_issues > 0 {
            OverallHealth::Critical
        } else if issues > 2 {
            OverallHealth::Degraded
        } else if issues > 0 {
            OverallHealth::Warning
        } else {
            OverallHealth::Healthy
        }
    }

    /// Start background garbage collection task
    async fn start_garbage_collection_task(&self) -> Result<()> {
        let memory_pool = Arc::clone(&self.memory_pool);
        let gc_interval = self.config.gc_interval_seconds;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(gc_interval));
            
            loop {
                interval.tick().await;
                
                // Check if garbage collection is needed
                let utilization = memory_pool.get_utilization();
                if utilization.fragmentation.fragmentation_ratio > 0.15 {
                    if let Err(e) = memory_pool.garbage_collect() {
                        error!("Background garbage collection failed: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Start background health monitoring task
    async fn start_health_monitoring_task(&self) -> Result<()> {
        let manager_weak = Arc::downgrade(&Arc::new(self.clone()));

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                if let Some(manager) = manager_weak.upgrade() {
                    let health = manager.health_check();
                    
                    match health.health {
                        OverallHealth::Critical => {
                            error!("CRITICAL MEMORY HEALTH: {}", health.message);
                            for alert in &health.alerts {
                                error!("ALERT: {}", alert);
                            }
                        },
                        OverallHealth::Warning | OverallHealth::Degraded => {
                            tracing::warn!("Memory health warning: {}", health.message);
                        },
                        OverallHealth::Healthy => {
                            tracing::debug!("Memory systems healthy");
                        },
                    }
                } else {
                    // Manager has been dropped, exit monitoring
                    break;
                }
            }
        });

        Ok(())
    }
}

// We need to implement Clone for the manager to work with weak references
impl Clone for SharedMemoryManager {
    fn clone(&self) -> Self {
        Self {
            memory_pool: Arc::clone(&self.memory_pool),
            ring_buffer: Arc::clone(&self.ring_buffer),
            config: self.config.clone(),
        }
    }
}

impl Default for SharedMemoryConfig {
    fn default() -> Self {
        Self {
            pool_config: PoolConfig::default(),
            ring_buffer_config: RingBufferConfig::default(),
            auto_gc_enabled: true,
            gc_interval_seconds: 300, // 5 minutes
            health_monitoring_enabled: true,
        }
    }
}

impl SharedMemoryConfig {
    /// Create configuration optimized for M3 Max with 128GB unified memory
    pub fn for_m3_max_128gb() -> Self {
        Self {
            pool_config: PoolConfig::for_m3_max_128gb(),
            ring_buffer_config: RingBufferConfig::for_large_documents(),
            auto_gc_enabled: true,
            gc_interval_seconds: 180, // More frequent GC for high-performance use
            health_monitoring_enabled: true,
        }
    }

    /// Create configuration for high-throughput streaming
    pub fn for_streaming_workload() -> Self {
        Self {
            pool_config: PoolConfig::default(),
            ring_buffer_config: RingBufferConfig::for_streaming(),
            auto_gc_enabled: true,
            gc_interval_seconds: 60, // Very frequent GC for streaming
            health_monitoring_enabled: true,
        }
    }

    /// Create configuration for batch processing
    pub fn for_batch_processing() -> Self {
        Self {
            pool_config: PoolConfig::for_m3_max_128gb(),
            ring_buffer_config: RingBufferConfig::for_small_messages(),
            auto_gc_enabled: false, // Manual GC control for batch jobs
            gc_interval_seconds: 600, // Less frequent monitoring
            health_monitoring_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_shared_memory_manager_creation() {
        let config = SharedMemoryConfig {
            pool_config: PoolConfig {
                pool_size_bytes: 1024 * 1024, // 1MB for testing
                ..Default::default()
            },
            ring_buffer_config: RingBufferConfig {
                capacity: 64 * 1024, // 64KB for testing
                ..Default::default()
            },
            auto_gc_enabled: false, // Disable for testing
            health_monitoring_enabled: false,
            ..Default::default()
        };

        let manager = SharedMemoryManager::new(config).await.unwrap();
        let stats = manager.get_statistics();
        
        assert_eq!(stats.overall_health, OverallHealth::Healthy);
        assert_eq!(stats.pool_utilization.total_size_bytes, 1024 * 1024);
    }

    #[tokio::test]
    async fn test_document_buffer_operations() {
        let config = SharedMemoryConfig {
            pool_config: PoolConfig {
                pool_size_bytes: 1024 * 1024, // 1MB for testing
                ..Default::default()
            },
            ring_buffer_config: RingBufferConfig::default(),
            auto_gc_enabled: false,
            health_monitoring_enabled: false,
            ..Default::default()
        };

        let manager = SharedMemoryManager::new(config).await.unwrap();
        let request_id = Uuid::new_v4();
        
        // Allocate buffer
        let allocation = manager.allocate_document_buffer(1000, request_id).unwrap();
        assert!(allocation.size >= 1000);
        
        // Write and read data
        let test_data = b"Hello, shared memory manager!";
        manager.write_document_data(allocation.offset, test_data).unwrap();
        
        let read_data = manager.read_document_data(allocation.offset, test_data.len()).unwrap();
        assert_eq!(read_data, test_data);
        
        // Deallocate
        manager.deallocate_document_buffer(allocation.allocation_id).unwrap();
    }

    #[tokio::test]
    async fn test_stream_message_operations() {
        let config = SharedMemoryConfig {
            pool_config: PoolConfig::default(),
            ring_buffer_config: RingBufferConfig {
                capacity: 64 * 1024, // 64KB for testing
                ..Default::default()
            },
            auto_gc_enabled: false,
            health_monitoring_enabled: false,
            ..Default::default()
        };

        let manager = SharedMemoryManager::new(config).await.unwrap();
        
        // Write stream message
        let test_message = b"Stream message test";
        let bytes_written = manager.write_stream_message(test_message, 1).unwrap();
        assert!(bytes_written > test_message.len()); // Includes header
        
        // Read stream message
        let result = manager.read_stream_message().unwrap();
        assert!(result.is_some());
        
        let (data, header) = result.unwrap();
        assert_eq!(data, test_message);
        assert_eq!(header.message_type, 1);
    }

    #[tokio::test]
    async fn test_health_monitoring() {
        let config = SharedMemoryConfig {
            pool_config: PoolConfig {
                pool_size_bytes: 1024 * 1024, // 1MB for testing
                ..Default::default()
            },
            ring_buffer_config: RingBufferConfig::default(),
            auto_gc_enabled: false,
            health_monitoring_enabled: false,
            ..Default::default()
        };

        let manager = SharedMemoryManager::new(config).await.unwrap();
        let health = manager.health_check();
        
        assert_eq!(health.health, OverallHealth::Healthy);
        assert!(!health.message.is_empty());
        assert!(health.recommendations.len() >= 1);
    }
}