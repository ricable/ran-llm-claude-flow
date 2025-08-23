use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{Result, Context};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{RwLock, Mutex};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};

/// 15GB shared memory pool for zero-copy document transfers
/// Optimized for M3 Max unified memory architecture
pub struct SharedMemoryPool {
    /// Memory-mapped pool for allocations
    memory_pool: Arc<RwLock<MmapMut>>,
    /// Allocation tracking and free block management
    allocator: Arc<Mutex<PoolAllocator>>,
    /// Pool configuration
    config: PoolConfig,
    /// Performance metrics
    metrics: Arc<PoolMetrics>,
    /// Pool metadata
    metadata: PoolMetadata,
}

/// Memory pool configuration optimized for M3 Max
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Total pool size in bytes
    pub pool_size_bytes: usize,
    /// Minimum allocation size (aligned)
    pub min_allocation_size: usize,
    /// Maximum allocation size
    pub max_allocation_size: usize,
    /// Memory alignment for SIMD operations
    pub alignment: usize,
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    /// Garbage collection threshold
    pub gc_threshold_percent: f64,
    /// Enable fragmentation monitoring
    pub enable_fragmentation_tracking: bool,
}

/// Pool allocator with advanced free block management
struct PoolAllocator {
    /// Free blocks sorted by size (best-fit allocation)
    free_blocks: BTreeMap<usize, Vec<MemoryBlock>>,
    /// Allocated blocks for tracking and validation
    allocated_blocks: BTreeMap<u64, AllocatedBlock>,
    /// Next allocation ID
    next_allocation_id: u64,
    /// Fragmentation tracking
    fragmentation_stats: FragmentationStats,
}

/// Memory block descriptor
#[derive(Debug, Clone, PartialEq)]
struct MemoryBlock {
    /// Offset from pool start
    offset: u64,
    /// Block size in bytes
    size: usize,
    /// Creation timestamp
    created_at: Instant,
}

/// Allocated block with metadata
#[derive(Debug, Clone)]
struct AllocatedBlock {
    /// Memory block info
    block: MemoryBlock,
    /// Allocation request ID
    request_id: Uuid,
    /// Process ID that allocated this block
    process_id: u32,
    /// Block type for categorization
    block_type: BlockType,
    /// Allocation timestamp
    allocated_at: Instant,
}

/// Types of memory blocks for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum BlockType {
    /// Large document content (>1MB)
    LargeDocument,
    /// Medium document content (64KB-1MB)
    MediumDocument,
    /// Small document content (<64KB)
    SmallDocument,
    /// IPC message buffers
    MessageBuffer,
    /// Temporary processing buffers
    TemporaryBuffer,
    /// Streaming data buffers
    StreamingBuffer,
}

/// Pool metadata
#[derive(Debug, Clone)]
pub struct PoolMetadata {
    /// Pool creation timestamp
    pub created_at: Instant,
    /// Pool unique identifier
    pub pool_id: Uuid,
    /// Pool version for compatibility
    pub version: String,
    /// Owner process ID
    pub owner_pid: u32,
}

/// Performance metrics for the memory pool
#[derive(Debug, Default)]
pub struct PoolMetrics {
    /// Total allocations performed
    pub total_allocations: AtomicUsize,
    /// Total deallocations performed
    pub total_deallocations: AtomicUsize,
    /// Total bytes allocated
    pub total_bytes_allocated: AtomicUsize,
    /// Total bytes deallocated
    pub total_bytes_deallocated: AtomicUsize,
    /// Peak memory usage
    pub peak_usage_bytes: AtomicUsize,
    /// Current active allocations
    pub active_allocations: AtomicUsize,
    /// Failed allocation attempts
    pub allocation_failures: AtomicUsize,
    /// Garbage collection runs
    pub gc_runs: AtomicUsize,
    /// Total allocation time in nanoseconds
    pub total_allocation_time_ns: AtomicUsize,
    /// Total deallocation time in nanoseconds
    pub total_deallocation_time_ns: AtomicUsize,
}

/// Fragmentation statistics
#[derive(Debug, Default, Clone)]
struct FragmentationStats {
    /// Number of free blocks
    free_block_count: usize,
    /// Average free block size
    average_free_block_size: usize,
    /// Largest free block size
    largest_free_block_size: usize,
    /// Total fragmented space
    total_fragmented_bytes: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = completely fragmented)
    fragmentation_ratio: f64,
}

/// Allocation result with detailed information
#[derive(Debug)]
pub struct AllocationResult {
    /// Allocated block offset
    pub offset: u64,
    /// Allocated block size
    pub size: usize,
    /// Allocation ID for tracking
    pub allocation_id: u64,
    /// Block type
    pub block_type: BlockType,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Pool utilization statistics
#[derive(Debug, Clone)]
pub struct PoolUtilization {
    /// Total pool size in bytes
    pub total_size_bytes: usize,
    /// Currently allocated bytes
    pub allocated_bytes: usize,
    /// Available bytes
    pub available_bytes: usize,
    /// Utilization percentage
    pub utilization_percent: f64,
    /// Peak utilization percentage
    pub peak_utilization_percent: f64,
    /// Fragmentation statistics
    pub fragmentation: FragmentationStats,
    /// Performance metrics
    pub performance: PoolPerformanceSnapshot,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PoolPerformanceSnapshot {
    /// Average allocation time in microseconds
    pub avg_allocation_time_us: f64,
    /// Average deallocation time in microseconds
    pub avg_deallocation_time_us: f64,
    /// Allocations per second
    pub allocations_per_second: f64,
    /// Throughput in MB/s
    pub throughput_mbps: f64,
    /// Success rate percentage
    pub success_rate_percent: f64,
}

impl SharedMemoryPool {
    /// Create a new 15GB shared memory pool optimized for M3 Max
    pub fn new(config: PoolConfig) -> Result<Self> {
        info!("Creating shared memory pool: {} GB", config.pool_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

        // Create memory-mapped file for the pool
        let temp_file = tempfile::NamedTempFile::new()
            .context("Failed to create temporary file for memory pool")?;
        
        temp_file.as_file().set_len(config.pool_size_bytes as u64)
            .context("Failed to set memory pool file size")?;

        let memory_pool = unsafe {
            MmapOptions::new()
                .len(config.pool_size_bytes)
                .map_mut(temp_file.as_file())
                .context("Failed to create memory mapping for pool")?
        };

        // Initialize allocator with single large free block
        let mut allocator = PoolAllocator {
            free_blocks: BTreeMap::new(),
            allocated_blocks: BTreeMap::new(),
            next_allocation_id: 1,
            fragmentation_stats: FragmentationStats::default(),
        };

        // Add entire pool as initial free block
        let initial_block = MemoryBlock {
            offset: 0,
            size: config.pool_size_bytes,
            created_at: Instant::now(),
        };
        
        allocator.free_blocks
            .entry(config.pool_size_bytes)
            .or_insert_with(Vec::new)
            .push(initial_block);

        let metadata = PoolMetadata {
            created_at: Instant::now(),
            pool_id: Uuid::new_v4(),
            version: "2.0.0".to_string(),
            owner_pid: std::process::id(),
        };

        let pool = Self {
            memory_pool: Arc::new(RwLock::new(memory_pool)),
            allocator: Arc::new(Mutex::new(allocator)),
            config,
            metrics: Arc::new(PoolMetrics::default()),
            metadata,
        };

        info!("Shared memory pool created successfully: {}", pool.metadata.pool_id);
        Ok(pool)
    }

    /// Allocate memory block with optimal placement strategy
    pub fn allocate(&self, size: usize, request_id: Uuid, block_type: BlockType) -> Result<AllocationResult> {
        let start_time = Instant::now();
        
        // Validate allocation request
        if size == 0 {
            anyhow::bail!("Cannot allocate zero-sized block");
        }
        
        if size > self.config.max_allocation_size {
            anyhow::bail!("Allocation size {} exceeds maximum {}", size, self.config.max_allocation_size);
        }

        // Align size to configured alignment
        let aligned_size = self.align_size(size);
        
        let mut allocator = self.allocator.lock();
        
        // Find best-fit free block
        let (block_size, block_index) = self.find_best_fit_block(&allocator, aligned_size)?;
        
        // Remove block from free list
        let free_block = allocator.free_blocks
            .get_mut(&block_size)
            .and_then(|blocks| {
                if block_index < blocks.len() {
                    Some(blocks.remove(block_index))
                } else {
                    None
                }
            })
            .context("Failed to retrieve free block")?;

        // Clean up empty size entry
        if allocator.free_blocks.get(&block_size).map_or(false, |blocks| blocks.is_empty()) {
            allocator.free_blocks.remove(&block_size);
        }

        // Split block if it's significantly larger than needed
        let (allocated_block, remainder_block) = if free_block.size > aligned_size + self.config.min_allocation_size {
            let allocated = MemoryBlock {
                offset: free_block.offset,
                size: aligned_size,
                created_at: Instant::now(),
            };
            
            let remainder = MemoryBlock {
                offset: free_block.offset + aligned_size as u64,
                size: free_block.size - aligned_size,
                created_at: free_block.created_at,
            };
            
            (allocated, Some(remainder))
        } else {
            (free_block, None)
        };

        // Add remainder back to free blocks if it exists
        if let Some(remainder) = remainder_block {
            allocator.free_blocks
                .entry(remainder.size)
                .or_insert_with(Vec::new)
                .push(remainder);
        }

        // Create allocation record
        let allocation_id = allocator.next_allocation_id;
        allocator.next_allocation_id += 1;
        
        let allocated_block_info = AllocatedBlock {
            block: allocated_block.clone(),
            request_id,
            process_id: std::process::id(),
            block_type: block_type.clone(),
            allocated_at: Instant::now(),
        };
        
        allocator.allocated_blocks.insert(allocation_id, allocated_block_info);
        
        // Update fragmentation stats
        self.update_fragmentation_stats(&mut allocator);

        drop(allocator);

        // Update metrics
        self.metrics.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_bytes_allocated.fetch_add(aligned_size, Ordering::Relaxed);
        self.metrics.active_allocations.fetch_add(1, Ordering::Relaxed);
        
        let allocation_time = start_time.elapsed().as_nanos() as usize;
        self.metrics.total_allocation_time_ns.fetch_add(allocation_time, Ordering::Relaxed);
        
        // Update peak usage
        let current_allocated = self.metrics.total_bytes_allocated.load(Ordering::Relaxed) -
                               self.metrics.total_bytes_deallocated.load(Ordering::Relaxed);
        
        let mut peak_usage = self.metrics.peak_usage_bytes.load(Ordering::Relaxed);
        while current_allocated > peak_usage {
            match self.metrics.peak_usage_bytes.compare_exchange_weak(
                peak_usage,
                current_allocated,
                Ordering::Relaxed,
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(actual) => peak_usage = actual,
            }
        }

        let result = AllocationResult {
            offset: allocated_block.offset,
            size: allocated_block.size,
            allocation_id,
            block_type,
            allocated_at: allocated_block.created_at,
        };

        debug!("Allocated {} bytes at offset {} (ID: {})", aligned_size, allocated_block.offset, allocation_id);
        Ok(result)
    }

    /// Deallocate memory block and coalesce with adjacent free blocks
    pub fn deallocate(&self, allocation_id: u64) -> Result<()> {
        let start_time = Instant::now();
        
        let mut allocator = self.allocator.lock();
        
        // Remove from allocated blocks
        let allocated_block = allocator.allocated_blocks
            .remove(&allocation_id)
            .context("Allocation ID not found")?;

        // Create free block
        let free_block = allocated_block.block.clone();
        
        // Coalesce with adjacent free blocks
        let coalesced_block = self.coalesce_free_block(&mut allocator, free_block)?;
        
        // Add coalesced block to free list
        allocator.free_blocks
            .entry(coalesced_block.size)
            .or_insert_with(Vec::new)
            .push(coalesced_block);

        // Update fragmentation stats
        self.update_fragmentation_stats(&mut allocator);

        drop(allocator);

        // Update metrics
        self.metrics.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_bytes_deallocated.fetch_add(allocated_block.block.size, Ordering::Relaxed);
        self.metrics.active_allocations.fetch_sub(1, Ordering::Relaxed);
        
        let deallocation_time = start_time.elapsed().as_nanos() as usize;
        self.metrics.total_deallocation_time_ns.fetch_add(deallocation_time, Ordering::Relaxed);

        debug!("Deallocated {} bytes at offset {} (ID: {})", 
               allocated_block.block.size, allocated_block.block.offset, allocation_id);
        Ok(())
    }

    /// Write data to allocated memory block with zero-copy semantics
    pub fn write_data(&self, offset: u64, data: &[u8]) -> Result<()> {
        let memory_pool = self.memory_pool.read();
        let start_idx = offset as usize;
        let end_idx = start_idx + data.len();
        
        if end_idx > self.config.pool_size_bytes {
            anyhow::bail!("Write would exceed memory pool bounds");
        }

        // Perform zero-copy write
        memory_pool[start_idx..end_idx].copy_from_slice(data);
        
        // Memory prefetching for next potential access
        if self.config.enable_prefetching && end_idx + 4096 < self.config.pool_size_bytes {
            unsafe {
                let prefetch_ptr = memory_pool.as_ptr().add(end_idx);
                std::ptr::read_volatile(prefetch_ptr);
            }
        }

        debug!("Wrote {} bytes at offset {}", data.len(), offset);
        Ok(())
    }

    /// Read data from memory block with zero-copy semantics
    pub fn read_data(&self, offset: u64, size: usize) -> Result<&[u8]> {
        let memory_pool = self.memory_pool.read();
        let start_idx = offset as usize;
        let end_idx = start_idx + size;
        
        if end_idx > self.config.pool_size_bytes {
            anyhow::bail!("Read would exceed memory pool bounds");
        }

        // Return slice directly from memory pool (zero-copy)
        let data_slice = &memory_pool[start_idx..end_idx];
        
        debug!("Read {} bytes from offset {}", size, offset);
        Ok(data_slice)
    }

    /// Get current pool utilization statistics
    pub fn get_utilization(&self) -> PoolUtilization {
        let allocator = self.allocator.lock();
        let metrics = self.get_performance_snapshot();
        
        let allocated_bytes = self.metrics.total_bytes_allocated.load(Ordering::Relaxed) -
                             self.metrics.total_bytes_deallocated.load(Ordering::Relaxed);
        
        let available_bytes = self.config.pool_size_bytes - allocated_bytes;
        let utilization_percent = (allocated_bytes as f64 / self.config.pool_size_bytes as f64) * 100.0;
        
        let peak_usage = self.metrics.peak_usage_bytes.load(Ordering::Relaxed);
        let peak_utilization_percent = (peak_usage as f64 / self.config.pool_size_bytes as f64) * 100.0;

        PoolUtilization {
            total_size_bytes: self.config.pool_size_bytes,
            allocated_bytes,
            available_bytes,
            utilization_percent,
            peak_utilization_percent,
            fragmentation: allocator.fragmentation_stats.clone(),
            performance: metrics,
        }
    }

    /// Perform garbage collection to reduce fragmentation
    pub fn garbage_collect(&self) -> Result<GarbageCollectionResult> {
        let start_time = Instant::now();
        info!("Starting garbage collection");
        
        let mut allocator = self.allocator.lock();
        
        let initial_fragmentation = allocator.fragmentation_stats.fragmentation_ratio;
        let initial_free_blocks = allocator.fragmentation_stats.free_block_count;
        
        // Coalesce all adjacent free blocks
        let mut coalesced_count = 0;
        let mut size_keys: Vec<_> = allocator.free_blocks.keys().cloned().collect();
        
        for &size in &size_keys {
            if let Some(blocks) = allocator.free_blocks.get_mut(&size) {
                blocks.sort_by_key(|block| block.offset);
                
                let mut i = 0;
                while i < blocks.len() {
                    let mut j = i + 1;
                    while j < blocks.len() {
                        let block_i = &blocks[i];
                        let block_j = &blocks[j];
                        
                        // Check if blocks are adjacent
                        if block_i.offset + block_i.size as u64 == block_j.offset {
                            // Merge blocks
                            let merged_size = block_i.size + block_j.size;
                            let merged_block = MemoryBlock {
                                offset: block_i.offset,
                                size: merged_size,
                                created_at: block_i.created_at.min(block_j.created_at),
                            };
                            
                            // Remove both blocks and add merged block
                            blocks.remove(j);
                            blocks.remove(i);
                            
                            // Add merged block to appropriate size bucket
                            allocator.free_blocks
                                .entry(merged_size)
                                .or_insert_with(Vec::new)
                                .push(merged_block);
                            
                            coalesced_count += 1;
                            break;
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }
        }
        
        // Remove empty size entries
        allocator.free_blocks.retain(|_, blocks| !blocks.is_empty());
        
        // Update fragmentation stats
        self.update_fragmentation_stats(&mut allocator);
        
        let final_fragmentation = allocator.fragmentation_stats.fragmentation_ratio;
        let final_free_blocks = allocator.fragmentation_stats.free_block_count;
        
        drop(allocator);
        
        // Update metrics
        self.metrics.gc_runs.fetch_add(1, Ordering::Relaxed);
        
        let gc_time = start_time.elapsed();
        let result = GarbageCollectionResult {
            duration: gc_time,
            blocks_coalesced: coalesced_count,
            fragmentation_before: initial_fragmentation,
            fragmentation_after: final_fragmentation,
            free_blocks_before: initial_free_blocks,
            free_blocks_after: final_free_blocks,
            memory_reclaimed_bytes: 0, // Would need more sophisticated tracking
        };
        
        info!("Garbage collection completed in {:?}: {} blocks coalesced, fragmentation reduced from {:.2}% to {:.2}%",
              gc_time, coalesced_count, initial_fragmentation * 100.0, final_fragmentation * 100.0);
        
        Ok(result)
    }

    /// Find best-fit free block for allocation
    fn find_best_fit_block(&self, allocator: &PoolAllocator, size: usize) -> Result<(usize, usize)> {
        // Find smallest free block that can satisfy the request
        for (&block_size, blocks) in allocator.free_blocks.iter() {
            if block_size >= size && !blocks.is_empty() {
                return Ok((block_size, 0)); // Return first block of this size
            }
        }
        
        anyhow::bail!("No suitable free block found for allocation of {} bytes", size)
    }

    /// Coalesce free block with adjacent blocks
    fn coalesce_free_block(&self, allocator: &mut PoolAllocator, mut block: MemoryBlock) -> Result<MemoryBlock> {
        let mut coalesced = false;
        
        // Look for adjacent blocks to coalesce
        let mut blocks_to_remove = Vec::new();
        
        for (&size, blocks) in allocator.free_blocks.iter_mut() {
            let mut indices_to_remove = Vec::new();
            
            for (index, free_block) in blocks.iter().enumerate() {
                // Check if blocks are adjacent
                if free_block.offset + free_block.size as u64 == block.offset {
                    // Merge before current block
                    block.offset = free_block.offset;
                    block.size += free_block.size;
                    indices_to_remove.push(index);
                    coalesced = true;
                } else if block.offset + block.size as u64 == free_block.offset {
                    // Merge after current block
                    block.size += free_block.size;
                    indices_to_remove.push(index);
                    coalesced = true;
                }
            }
            
            // Remove coalesced blocks (in reverse order to maintain indices)
            for &index in indices_to_remove.iter().rev() {
                blocks.remove(index);
            }
            
            if blocks.is_empty() {
                blocks_to_remove.push(size);
            }
        }
        
        // Remove empty size entries
        for size in blocks_to_remove {
            allocator.free_blocks.remove(&size);
        }
        
        Ok(block)
    }

    /// Update fragmentation statistics
    fn update_fragmentation_stats(&self, allocator: &mut PoolAllocator) {
        let mut total_free_bytes = 0;
        let mut free_block_count = 0;
        let mut largest_block_size = 0;
        
        for (_, blocks) in &allocator.free_blocks {
            for block in blocks {
                total_free_bytes += block.size;
                free_block_count += 1;
                largest_block_size = largest_block_size.max(block.size);
            }
        }
        
        let average_free_block_size = if free_block_count > 0 {
            total_free_bytes / free_block_count
        } else {
            0
        };
        
        let fragmentation_ratio = if total_free_bytes > 0 && free_block_count > 1 {
            1.0 - (largest_block_size as f64 / total_free_bytes as f64)
        } else {
            0.0
        };
        
        allocator.fragmentation_stats = FragmentationStats {
            free_block_count,
            average_free_block_size,
            largest_free_block_size: largest_block_size,
            total_fragmented_bytes: total_free_bytes.saturating_sub(largest_block_size),
            fragmentation_ratio,
        };
    }

    /// Align size to configured alignment
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Get performance snapshot
    fn get_performance_snapshot(&self) -> PoolPerformanceSnapshot {
        let total_allocs = self.metrics.total_allocations.load(Ordering::Relaxed);
        let total_deallocs = self.metrics.total_deallocations.load(Ordering::Relaxed);
        let total_alloc_time = self.metrics.total_allocation_time_ns.load(Ordering::Relaxed);
        let total_dealloc_time = self.metrics.total_deallocation_time_ns.load(Ordering::Relaxed);
        let total_bytes_allocated = self.metrics.total_bytes_allocated.load(Ordering::Relaxed);
        let allocation_failures = self.metrics.allocation_failures.load(Ordering::Relaxed);
        
        let elapsed_secs = self.metadata.created_at.elapsed().as_secs_f64();
        
        let avg_allocation_time_us = if total_allocs > 0 {
            (total_alloc_time as f64) / (total_allocs as f64) / 1000.0
        } else {
            0.0
        };
        
        let avg_deallocation_time_us = if total_deallocs > 0 {
            (total_dealloc_time as f64) / (total_deallocs as f64) / 1000.0
        } else {
            0.0
        };
        
        let allocations_per_second = if elapsed_secs > 0.0 {
            total_allocs as f64 / elapsed_secs
        } else {
            0.0
        };
        
        let throughput_mbps = if elapsed_secs > 0.0 {
            (total_bytes_allocated as f64) / (1024.0 * 1024.0 * elapsed_secs)
        } else {
            0.0
        };
        
        let success_rate_percent = if total_allocs + allocation_failures > 0 {
            (total_allocs as f64 / (total_allocs + allocation_failures) as f64) * 100.0
        } else {
            100.0
        };
        
        PoolPerformanceSnapshot {
            avg_allocation_time_us,
            avg_deallocation_time_us,
            allocations_per_second,
            throughput_mbps,
            success_rate_percent,
        }
    }
}

/// Garbage collection results
#[derive(Debug, Clone)]
pub struct GarbageCollectionResult {
    pub duration: Duration,
    pub blocks_coalesced: usize,
    pub fragmentation_before: f64,
    pub fragmentation_after: f64,
    pub free_blocks_before: usize,
    pub free_blocks_after: usize,
    pub memory_reclaimed_bytes: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            pool_size_bytes: 15 * 1024 * 1024 * 1024, // 15GB
            min_allocation_size: 64,                    // 64 bytes minimum
            max_allocation_size: 512 * 1024 * 1024,    // 512MB maximum
            alignment: 64,                              // M3 Max cache line size
            enable_prefetching: true,
            gc_threshold_percent: 15.0,                 // GC when fragmentation > 15%
            enable_fragmentation_tracking: true,
        }
    }
}

impl PoolConfig {
    /// Create configuration optimized for M3 Max with 128GB unified memory
    pub fn for_m3_max_128gb() -> Self {
        Self {
            pool_size_bytes: 15 * 1024 * 1024 * 1024, // 15GB shared pool
            min_allocation_size: 64,
            max_allocation_size: 1024 * 1024 * 1024,   // 1GB max for very large documents
            alignment: 64,                              // Optimized for M3 Max
            enable_prefetching: true,
            gc_threshold_percent: 10.0,                 // More aggressive GC
            enable_fragmentation_tracking: true,
        }
    }

    /// Create configuration for high-frequency small allocations
    pub fn for_small_allocations() -> Self {
        Self {
            pool_size_bytes: 1024 * 1024 * 1024,       // 1GB pool
            min_allocation_size: 32,                    // Smaller minimum
            max_allocation_size: 1024 * 1024,          // 1MB maximum
            alignment: 32,                              // Smaller alignment
            enable_prefetching: false,                  // Less beneficial for small blocks
            gc_threshold_percent: 20.0,
            enable_fragmentation_tracking: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = PoolConfig {
            pool_size_bytes: 1024 * 1024, // 1MB for testing
            ..Default::default()
        };
        let pool = SharedMemoryPool::new(config).unwrap();
        
        let utilization = pool.get_utilization();
        assert_eq!(utilization.total_size_bytes, 1024 * 1024);
        assert_eq!(utilization.allocated_bytes, 0);
        assert!(utilization.utilization_percent < 1.0);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let config = PoolConfig {
            pool_size_bytes: 1024 * 1024, // 1MB for testing
            ..Default::default()
        };
        let pool = SharedMemoryPool::new(config).unwrap();
        
        let request_id = Uuid::new_v4();
        let allocation = pool.allocate(1000, request_id, BlockType::SmallDocument).unwrap();
        
        assert!(allocation.size >= 1000);
        assert_eq!(allocation.block_type, BlockType::SmallDocument);
        
        let result = pool.deallocate(allocation.allocation_id);
        assert!(result.is_ok());
        
        let utilization = pool.get_utilization();
        assert_eq!(utilization.allocated_bytes, 0);
    }

    #[test]
    fn test_write_and_read_data() {
        let config = PoolConfig {
            pool_size_bytes: 1024 * 1024, // 1MB for testing
            ..Default::default()
        };
        let pool = SharedMemoryPool::new(config).unwrap();
        
        let request_id = Uuid::new_v4();
        let allocation = pool.allocate(1000, request_id, BlockType::SmallDocument).unwrap();
        
        let test_data = b"Hello, shared memory pool!";
        pool.write_data(allocation.offset, test_data).unwrap();
        
        let read_data = pool.read_data(allocation.offset, test_data.len()).unwrap();
        assert_eq!(read_data, test_data);
        
        pool.deallocate(allocation.allocation_id).unwrap();
    }

    #[test]
    fn test_garbage_collection() {
        let config = PoolConfig {
            pool_size_bytes: 1024 * 1024, // 1MB for testing
            ..Default::default()
        };
        let pool = SharedMemoryPool::new(config).unwrap();
        
        // Create some allocations and deallocate them to create fragmentation
        let mut allocation_ids = Vec::new();
        for i in 0..10 {
            let request_id = Uuid::new_v4();
            let allocation = pool.allocate(1000 + i * 100, request_id, BlockType::SmallDocument).unwrap();
            allocation_ids.push(allocation.allocation_id);
        }
        
        // Deallocate every other allocation to create fragmentation
        for (i, &id) in allocation_ids.iter().enumerate() {
            if i % 2 == 0 {
                pool.deallocate(id).unwrap();
            }
        }
        
        let gc_result = pool.garbage_collect().unwrap();
        assert!(gc_result.duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_block_coalescing() {
        let config = PoolConfig {
            pool_size_bytes: 1024 * 1024, // 1MB for testing
            min_allocation_size: 100,
            ..Default::default()
        };
        let pool = SharedMemoryPool::new(config).unwrap();
        
        // Allocate adjacent blocks
        let request_id1 = Uuid::new_v4();
        let request_id2 = Uuid::new_v4();
        
        let allocation1 = pool.allocate(1000, request_id1, BlockType::SmallDocument).unwrap();
        let allocation2 = pool.allocate(1000, request_id2, BlockType::SmallDocument).unwrap();
        
        // Deallocate both - they should coalesce
        pool.deallocate(allocation1.allocation_id).unwrap();
        pool.deallocate(allocation2.allocation_id).unwrap();
        
        let utilization = pool.get_utilization();
        assert_eq!(utilization.allocated_bytes, 0);
        
        // Should be able to allocate a large block that spans both previous allocations
        let large_allocation = pool.allocate(2500, Uuid::new_v4(), BlockType::LargeDocument).unwrap();
        assert!(large_allocation.size >= 2500);
    }
}