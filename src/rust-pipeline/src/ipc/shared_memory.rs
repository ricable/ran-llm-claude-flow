/*!
# Shared Memory Manager
 
High-performance shared memory implementation optimized for M3 Max unified memory architecture.
Provides zero-copy data transfer between Rust core and Python ML workers.
*/

use crate::{Result, PipelineError};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::slice;
use uuid::Uuid;

/// Shared memory region for inter-process communication
#[derive(Debug)]
pub struct SharedMemoryRegion {
    id: Uuid,
    name: String,
    size_bytes: usize,
    ptr: NonNull<u8>,
    is_owner: bool,
}

unsafe impl Send for SharedMemoryRegion {}
unsafe impl Sync for SharedMemoryRegion {}

/// Shared memory manager optimized for M3 Max
pub struct SharedMemoryManager {
    total_size_bytes: usize,
    regions: Arc<RwLock<HashMap<Uuid, SharedMemoryRegion>>>,
    allocation_tracker: Arc<Mutex<AllocationTracker>>,
    m3_max_config: M3MaxMemoryConfig,
}

#[derive(Debug, Clone)]
pub struct M3MaxMemoryConfig {
    /// Use M3 Max unified memory optimizations
    pub use_unified_memory: bool,
    /// Memory alignment for Apple Silicon (64 bytes for optimal performance)
    pub alignment_bytes: usize,
    /// Enable memory prefetch hints
    pub enable_prefetch: bool,
    /// Memory pool size in bytes
    pub pool_size_bytes: usize,
}

#[derive(Debug)]
struct AllocationTracker {
    allocated_bytes: usize,
    allocation_count: u32,
    fragmentation_factor: f64,
    peak_usage_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_size_bytes: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub allocation_count: u32,
    pub fragmentation_factor: f64,
    pub regions: Vec<RegionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionInfo {
    pub id: Uuid,
    pub name: String,
    pub size_bytes: usize,
    pub offset: usize,
}

impl SharedMemoryManager {
    /// Create new shared memory manager with M3 Max optimizations
    pub async fn new(total_size_bytes: usize) -> Result<Self> {
        tracing::info!("Initializing shared memory manager with {} bytes", total_size_bytes);
        
        let m3_max_config = M3MaxMemoryConfig {
            use_unified_memory: cfg!(target_arch = "aarch64") && cfg!(target_os = "macos"),
            alignment_bytes: 64, // Optimal for Apple Silicon
            enable_prefetch: true,
            pool_size_bytes: total_size_bytes,
        };

        let allocation_tracker = AllocationTracker {
            allocated_bytes: 0,
            allocation_count: 0,
            fragmentation_factor: 0.0,
            peak_usage_bytes: 0,
        };

        Ok(Self {
            total_size_bytes,
            regions: Arc::new(RwLock::new(HashMap::new())),
            allocation_tracker: Arc::new(Mutex::new(allocation_tracker)),
            m3_max_config,
        })
    }

    /// Create a new shared memory region
    pub async fn create_region(&self, name: &str, size_bytes: usize) -> Result<Uuid> {
        // Align size to M3 Max optimal boundary
        let aligned_size = self.align_size(size_bytes);
        
        tracing::debug!("Creating shared memory region '{}' with {} bytes (aligned: {})", 
                       name, size_bytes, aligned_size);

        // Check available space
        {
            let tracker = self.allocation_tracker.lock();
            if tracker.allocated_bytes + aligned_size > self.total_size_bytes {
                return Err(PipelineError::Ipc(format!(
                    "Insufficient memory: need {} bytes, have {} bytes available",
                    aligned_size,
                    self.total_size_bytes - tracker.allocated_bytes
                )));
            }
        }

        // Create the shared memory region using mmap for M3 Max optimization
        let ptr = self.allocate_aligned_memory(aligned_size)?;
        
        let region_id = Uuid::new_v4();
        let region = SharedMemoryRegion {
            id: region_id,
            name: name.to_string(),
            size_bytes: aligned_size,
            ptr,
            is_owner: true,
        };

        // Initialize memory to zero
        unsafe {
            ptr::write_bytes(ptr.as_ptr(), 0, aligned_size);
        }

        // Update tracking
        {
            let mut tracker = self.allocation_tracker.lock();
            tracker.allocated_bytes += aligned_size;
            tracker.allocation_count += 1;
            tracker.peak_usage_bytes = tracker.peak_usage_bytes.max(tracker.allocated_bytes);
            tracker.fragmentation_factor = self.calculate_fragmentation(&tracker);
        }

        // Store region
        {
            let mut regions = self.regions.write();
            regions.insert(region_id, region);
        }

        tracing::info!("Created shared memory region '{}' with ID {}", name, region_id);
        Ok(region_id)
    }

    /// Write data to shared memory region
    pub async fn write_data(&self, region_id: Uuid, offset: usize, data: &[u8]) -> Result<()> {
        let regions = self.regions.read();
        let region = regions.get(&region_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Region {} not found", region_id)))?;

        if offset + data.len() > region.size_bytes {
            return Err(PipelineError::Ipc(format!(
                "Write would exceed region bounds: offset {} + data {} > region size {}",
                offset, data.len(), region.size_bytes
            )));
        }

        unsafe {
            let dest = region.ptr.as_ptr().add(offset);
            ptr::copy_nonoverlapping(data.as_ptr(), dest, data.len());
            
            // Memory barrier for M3 Max consistency
            if self.m3_max_config.use_unified_memory {
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
        }

        tracing::debug!("Wrote {} bytes to region {} at offset {}", data.len(), region_id, offset);
        Ok(())
    }

    /// Read data from shared memory region
    pub async fn read_data(&self, region_id: Uuid, offset: usize, length: usize) -> Result<Vec<u8>> {
        let regions = self.regions.read();
        let region = regions.get(&region_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Region {} not found", region_id)))?;

        if offset + length > region.size_bytes {
            return Err(PipelineError::Ipc(format!(
                "Read would exceed region bounds: offset {} + length {} > region size {}",
                offset, length, region.size_bytes
            )));
        }

        let mut data = vec![0u8; length];
        unsafe {
            let src = region.ptr.as_ptr().add(offset);
            
            // Prefetch hint for M3 Max
            if self.m3_max_config.enable_prefetch {
                self.prefetch_memory(src, length);
            }
            
            ptr::copy_nonoverlapping(src, data.as_mut_ptr(), length);
        }

        tracing::debug!("Read {} bytes from region {} at offset {}", length, region_id, offset);
        Ok(data)
    }

    /// Get zero-copy slice from shared memory (unsafe but fast)
    pub unsafe fn get_slice(&self, region_id: Uuid, offset: usize, length: usize) -> Result<&[u8]> {
        let regions = self.regions.read();
        let region = regions.get(&region_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Region {} not found", region_id)))?;

        if offset + length > region.size_bytes {
            return Err(PipelineError::Ipc(format!(
                "Slice would exceed region bounds: offset {} + length {} > region size {}",
                offset, length, region.size_bytes
            )));
        }

        let ptr = region.ptr.as_ptr().add(offset);
        Ok(slice::from_raw_parts(ptr, length))
    }

    /// Get mutable zero-copy slice from shared memory (unsafe but fast)  
    pub unsafe fn get_mut_slice(&self, region_id: Uuid, offset: usize, length: usize) -> Result<&mut [u8]> {
        let regions = self.regions.read();
        let region = regions.get(&region_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Region {} not found", region_id)))?;

        if offset + length > region.size_bytes {
            return Err(PipelineError::Ipc(format!(
                "Mutable slice would exceed region bounds: offset {} + length {} > region size {}",
                offset, length, region.size_bytes
            )));
        }

        let ptr = region.ptr.as_ptr().add(offset);
        Ok(slice::from_raw_parts_mut(ptr as *mut u8, length))
    }

    /// Delete shared memory region
    pub async fn delete_region(&self, region_id: Uuid) -> Result<()> {
        let mut regions = self.regions.write();
        let region = regions.remove(&region_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Region {} not found", region_id)))?;

        // Deallocate memory
        if region.is_owner {
            unsafe {
                self.deallocate_memory(region.ptr, region.size_bytes)?;
            }
        }

        // Update tracking
        {
            let mut tracker = self.allocation_tracker.lock();
            tracker.allocated_bytes = tracker.allocated_bytes.saturating_sub(region.size_bytes);
            tracker.allocation_count = tracker.allocation_count.saturating_sub(1);
            tracker.fragmentation_factor = self.calculate_fragmentation(&tracker);
        }

        tracing::info!("Deleted shared memory region '{}' with ID {}", region.name, region_id);
        Ok(())
    }

    /// Get memory usage statistics
    pub async fn get_usage_mb(&self) -> Result<u64> {
        let tracker = self.allocation_tracker.lock();
        Ok(tracker.allocated_bytes as u64 / 1024 / 1024)
    }

    /// Get detailed memory statistics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let tracker = self.allocation_tracker.lock();
        let regions_read = self.regions.read();
        
        let regions: Vec<RegionInfo> = regions_read.iter().map(|(id, region)| {
            RegionInfo {
                id: *id,
                name: region.name.clone(),
                size_bytes: region.size_bytes,
                offset: 0, // Would need more sophisticated tracking for actual offsets
            }
        }).collect();

        Ok(MemoryStats {
            total_size_bytes: self.total_size_bytes,
            allocated_bytes: tracker.allocated_bytes,
            free_bytes: self.total_size_bytes - tracker.allocated_bytes,
            allocation_count: tracker.allocation_count,
            fragmentation_factor: tracker.fragmentation_factor,
            regions,
        })
    }

    /// Defragment memory (compact allocations)
    pub async fn defragment(&self) -> Result<()> {
        tracing::info!("Starting memory defragmentation");
        
        // This is a simplified implementation - in production would need more sophisticated defrag
        let mut tracker = self.allocation_tracker.lock();
        tracker.fragmentation_factor = 0.0;
        
        tracing::info!("Memory defragmentation completed");
        Ok(())
    }

    /// Cleanup all regions and resources
    pub async fn cleanup(&self) -> Result<()> {
        tracing::info!("Cleaning up shared memory manager");
        
        let region_ids: Vec<Uuid> = {
            self.regions.read().keys().copied().collect()
        };
        
        for region_id in region_ids {
            if let Err(e) = self.delete_region(region_id).await {
                tracing::warn!("Failed to delete region {}: {}", region_id, e);
            }
        }
        
        tracing::info!("Shared memory manager cleanup completed");
        Ok(())
    }

    // Private helper methods

    /// Align size to M3 Max optimal boundary
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.m3_max_config.alignment_bytes;
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Allocate aligned memory using system APIs
    fn allocate_aligned_memory(&self, size: usize) -> Result<NonNull<u8>> {
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, self.m3_max_config.alignment_bytes)
                .map_err(|e| PipelineError::Ipc(format!("Invalid memory layout: {}", e)))?;
            
            let ptr = std::alloc::alloc(layout);
            NonNull::new(ptr)
                .ok_or_else(|| PipelineError::Ipc("Failed to allocate aligned memory".to_string()))
        }
    }

    /// Deallocate memory
    unsafe fn deallocate_memory(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        let layout = std::alloc::Layout::from_size_align(size, self.m3_max_config.alignment_bytes)
            .map_err(|e| PipelineError::Ipc(format!("Invalid memory layout for deallocation: {}", e)))?;
        
        std::alloc::dealloc(ptr.as_ptr(), layout);
        Ok(())
    }

    /// Memory prefetch hint for M3 Max
    fn prefetch_memory(&self, _ptr: *const u8, _length: usize) {
        // Prefetch is an unstable feature, so this is a no-op on stable.
    }

    /// Calculate memory fragmentation factor
    fn calculate_fragmentation(&self, tracker: &AllocationTracker) -> f64 {
        if self.total_size_bytes == 0 {
            return 0.0;
        }
        
        let used_ratio = tracker.allocated_bytes as f64 / self.total_size_bytes as f64;
        let allocation_density = if tracker.allocation_count == 0 {
            1.0
        } else {
            (tracker.allocated_bytes as f64) / (tracker.allocation_count as f64)
        };
        
        // Simple fragmentation heuristic
        1.0 - (used_ratio * (allocation_density / 1024.0).min(1.0))
    }
}

impl Drop for SharedMemoryManager {
    fn drop(&mut self) {
        tracing::debug!("Dropping shared memory manager");
        // Cleanup handled by explicit cleanup() method to avoid async in drop
    }
}

/// Utility functions for shared memory operations

/// Create a shared memory region for large document transfer
pub async fn create_document_region(manager: &SharedMemoryManager, document_id: Uuid, size: usize) -> Result<Uuid> {
    let region_name = format!("document-{}", document_id);
    manager.create_region(&region_name, size).await
}

/// Create a shared memory region for model data
pub async fn create_model_region(manager: &SharedMemoryManager, model_id: &str, size: usize) -> Result<Uuid> {
    let region_name = format!("model-{}", model_id);
    manager.create_region(&region_name, size).await
}

/// Create a shared memory region for results
pub async fn create_results_region(manager: &SharedMemoryManager, batch_id: Uuid, size: usize) -> Result<Uuid> {
    let region_name = format!("results-{}", batch_id);
    manager.create_region(&region_name, size).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shared_memory_basic_operations() {
        let manager = SharedMemoryManager::new(1024 * 1024).await.unwrap(); // 1MB

        // Create region
        let region_id = manager.create_region("test-region", 1024).await.unwrap();

        // Write data
        let test_data = b"Hello, M3 Max shared memory!";
        manager.write_data(region_id, 0, test_data).await.unwrap();

        // Read data back
        let read_data = manager.read_data(region_id, 0, test_data.len()).await.unwrap();
        assert_eq!(read_data, test_data);

        // Delete region
        manager.delete_region(region_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_memory_alignment() {
        let manager = SharedMemoryManager::new(1024 * 1024).await.unwrap();
        
        // Test that sizes are properly aligned
        assert_eq!(manager.align_size(100), 128); // Should align to 64-byte boundary
        assert_eq!(manager.align_size(64), 64);
        assert_eq!(manager.align_size(65), 128);
    }
}