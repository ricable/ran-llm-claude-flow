/*!
# M3 Max Unified Memory Management

Optimized memory management for MacBook Pro M3 Max with 128GB unified memory architecture.
Provides hardware-specific optimizations for CPU, GPU, and Neural Engine coordination.
*/

use crate::{Result, PipelineError, M3MaxConfig};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;
use uuid::Uuid;

/// M3 Max memory pool configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct M3MaxMemoryPool {
    pub pool_id: Uuid,
    pub name: String,
    pub base_address: u64,
    pub size_bytes: usize,
    #[serde(skip)]
    pub allocated_bytes: AtomicUsize,
    #[serde(skip)]
    pub allocation_count: AtomicU64,
    pub pool_type: PoolType,
    pub optimization_flags: OptimizationFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolType {
    /// System memory pool for general allocation
    System,
    /// GPU memory pool for Metal computations
    Gpu,
    /// Neural Engine memory pool for ML acceleration
    NeuralEngine,
    /// High-bandwidth memory pool for data processing
    HighBandwidth,
    /// Cache-optimized memory pool for frequent access
    Cache,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationFlags {
    /// Enable Metal Performance Shaders optimization
    pub metal_optimization: bool,
    /// Enable Neural Engine acceleration
    pub neural_engine_acceleration: bool,
    /// Use AMX (Apple Matrix Extension) coprocessor
    pub amx_acceleration: bool,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    /// Memory alignment (64 bytes optimal for Apple Silicon)
    pub alignment_bytes: usize,
}

impl Default for OptimizationFlags {
    fn default() -> Self {
        Self {
            metal_optimization: true,
            neural_engine_acceleration: true,
            amx_acceleration: true,
            enable_prefetch: true,
            alignment_bytes: 64,
        }
    }
}

/// M3 Max system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3MaxSystemInfo {
    pub total_memory_gb: u32,
    pub performance_cores: u8,
    pub efficiency_cores: u8,
    pub gpu_cores: u32,
    pub neural_engine_tops: f32,
    pub memory_bandwidth_gbps: f32,
    pub cache_hierarchy: CacheHierarchy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHierarchy {
    pub l1_instruction_kb: u32,
    pub l1_data_kb: u32,
    pub l2_cache_mb: u32,
    pub l3_cache_mb: u32,
    pub slc_cache_mb: u32, // System Level Cache
}

/// Memory allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_allocated_bytes: u64,
    pub peak_allocation_bytes: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub fragmentation_ratio: f64,
    pub pools: Vec<PoolStats>,
    pub system_metrics: SystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub pool_id: Uuid,
    pub name: String,
    pub pool_type: PoolType,
    pub total_size_bytes: usize,
    pub allocated_bytes: usize,
    pub utilization_percent: f64,
    pub allocation_count: u64,
    pub average_allocation_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_pressure: f64,
    pub gpu_utilization_percent: f64,
    pub neural_engine_utilization_percent: f64,
    pub memory_bandwidth_utilization_gbps: f64,
    pub thermal_state: ThermalState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalState {
    Normal,
    Fair,
    Serious,
    Critical,
}

/// M3 Max Memory Manager
pub struct M3MaxMemoryManager {
    config: M3MaxConfig,
    system_info: M3MaxSystemInfo,
    memory_pools: Arc<RwLock<HashMap<Uuid, M3MaxMemoryPool>>>,
    allocation_tracker: Arc<Mutex<AllocationTracker>>,
    performance_monitor: Arc<PerformanceMonitor>,
    system: Arc<Mutex<System>>,
}

#[derive(Debug)]
struct AllocationTracker {
    total_allocated: AtomicU64,
    peak_allocation: AtomicU64,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
    allocations: HashMap<u64, AllocationInfo>,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    pool_id: Uuid,
    timestamp: u64,
    alignment: usize,
}

#[derive(Debug)]
struct PerformanceMonitor {
    cpu_usage: AtomicU64, // Fixed point representation (x100)
    memory_pressure: AtomicU64, // Fixed point representation (x100)
    gpu_utilization: AtomicU64, // Fixed point representation (x100)
    neural_engine_utilization: AtomicU64, // Fixed point representation (x100)
    last_update: AtomicU64,
}

impl M3MaxMemoryManager {
    /// Initialize M3 Max memory manager
    pub async fn new(config: M3MaxConfig) -> Result<Self> {
        let system_info = Self::detect_m3_max_system().await?;
        
        tracing::info!("Initializing M3 Max Memory Manager");
        tracing::info!("Detected M3 Max: {}GB unified memory, {} P-cores, {} E-cores", 
                      system_info.total_memory_gb, 
                      system_info.performance_cores, 
                      system_info.efficiency_cores);

        let allocation_tracker = AllocationTracker {
            total_allocated: AtomicU64::new(0),
            peak_allocation: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            allocations: HashMap::new(),
        };

        let performance_monitor = PerformanceMonitor {
            cpu_usage: AtomicU64::new(0),
            memory_pressure: AtomicU64::new(0),
            gpu_utilization: AtomicU64::new(0),
            neural_engine_utilization: AtomicU64::new(0),
            last_update: AtomicU64::new(current_timestamp_ms()),
        };

        let manager = Self {
            config,
            system_info,
            memory_pools: Arc::new(RwLock::new(HashMap::new())),
            allocation_tracker: Arc::new(Mutex::new(allocation_tracker)),
            performance_monitor: Arc::new(performance_monitor),
            system: Arc::new(Mutex::new(System::new_all())),
        };

        // Create default memory pools
        manager.initialize_memory_pools().await?;
        
        // Start performance monitoring
        manager.start_performance_monitoring().await;

        tracing::info!("M3 Max Memory Manager initialized successfully");
        Ok(manager)
    }

    /// Create a memory pool with M3 Max optimizations
    pub async fn create_pool(
        &self,
        name: &str,
        size_bytes: usize,
        pool_type: PoolType,
        optimization_flags: OptimizationFlags,
    ) -> Result<Uuid> {
        let pool_id = Uuid::new_v4();
        
        // Align size to M3 Max optimal boundaries
        let aligned_size = self.align_to_m3_max_boundary(size_bytes, &optimization_flags);
        
        tracing::info!("Creating M3 Max memory pool '{}': {} bytes (aligned: {})", 
                      name, size_bytes, aligned_size);

        let pool = M3MaxMemoryPool {
            pool_id,
            name: name.to_string(),
            base_address: self.allocate_pool_memory(aligned_size, &pool_type).await?,
            size_bytes: aligned_size,
            allocated_bytes: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            pool_type,
            optimization_flags,
        };

        // Configure pool for M3 Max hardware
        self.configure_pool_for_m3_max(&pool).await?;
        
        {
            let mut pools = self.memory_pools.write();
            pools.insert(pool_id, pool);
        }

        tracing::info!("M3 Max memory pool '{}' created with ID: {}", name, pool_id);
        Ok(pool_id)
    }

    /// Allocate memory from a specific pool with M3 Max optimizations
    pub async fn allocate_from_pool(
        &self,
        pool_id: Uuid,
        size: usize,
        alignment: Option<usize>,
    ) -> Result<u64> {
        let pools = self.memory_pools.read();
        let pool = pools.get(&pool_id)
            .ok_or_else(|| PipelineError::Optimization(format!("Pool {} not found", pool_id)))?;

        let alignment = alignment.unwrap_or(pool.optimization_flags.alignment_bytes);
        let aligned_size = self.align_size(size, alignment);

        // Check if allocation fits
        let current_allocated = pool.allocated_bytes.load(Ordering::Relaxed);
        if current_allocated + aligned_size > pool.size_bytes {
            return Err(PipelineError::Optimization(format!(
                "Pool {} insufficient space: need {} bytes, have {} available",
                pool_id, aligned_size, pool.size_bytes - current_allocated
            )));
        }

        // Allocate memory with M3 Max optimizations
        let address = self.allocate_optimized_memory(
            pool.base_address,
            current_allocated,
            aligned_size,
            &pool.optimization_flags
        ).await?;

        // Update pool statistics
        pool.allocated_bytes.fetch_add(aligned_size, Ordering::Relaxed);
        pool.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Update global tracking
        {
            let mut tracker = self.allocation_tracker.lock();
            let total = tracker.total_allocated.fetch_add(aligned_size as u64, Ordering::Relaxed);
            let peak = tracker.peak_allocation.load(Ordering::Relaxed);
            if total > peak {
                tracker.peak_allocation.store(total, Ordering::Relaxed);
            }
            tracker.allocation_count.fetch_add(1, Ordering::Relaxed);
            
            tracker.allocations.insert(address, AllocationInfo {
                size: aligned_size,
                pool_id,
                timestamp: current_timestamp_ms(),
                alignment,
            });
        }

        tracing::debug!("Allocated {} bytes from pool {} at address 0x{:x}", 
                       aligned_size, pool_id, address);
        Ok(address)
    }

    /// Deallocate memory with M3 Max cleanup
    pub async fn deallocate(&self, address: u64) -> Result<()> {
        let allocation_info = {
            let mut tracker = self.allocation_tracker.lock();
            tracker.allocations.remove(&address)
        };

        if let Some(info) = allocation_info {
            // Find the pool and update statistics
            let pools = self.memory_pools.read();
            if let Some(pool) = pools.get(&info.pool_id) {
                pool.allocated_bytes.fetch_sub(info.size, Ordering::Relaxed);
                
                // Perform M3 Max specific cleanup
                self.cleanup_optimized_memory(address, info.size, &pool.optimization_flags).await?;
            }

            // Update global tracking
            {
                let tracker = self.allocation_tracker.lock();
                tracker.total_allocated.fetch_sub(info.size as u64, Ordering::Relaxed);
                tracker.deallocation_count.fetch_add(1, Ordering::Relaxed);
            }

            tracing::debug!("Deallocated {} bytes at address 0x{:x}", info.size, address);
        } else {
            return Err(PipelineError::Optimization(format!("Invalid address for deallocation: 0x{:x}", address)));
        }

        Ok(())
    }

    /// Get memory statistics with M3 Max metrics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        // Clone data immediately after acquiring locks to avoid holding them across async boundaries
        let (total_allocated, peak_allocation, allocation_count, deallocation_count) = {
            let tracker = self.allocation_tracker.lock();
            (
                tracker.total_allocated.load(Ordering::Relaxed),
                tracker.peak_allocation.load(Ordering::Relaxed),
                tracker.allocation_count.load(Ordering::Relaxed),
                tracker.deallocation_count.load(Ordering::Relaxed),
            )
        };
        
        let (pool_stats, total_capacity) = {
            let pools = self.memory_pools.read();
            let mut pool_stats = Vec::new();
            let mut total_capacity = 0usize;
            
            for (pool_id, pool) in pools.iter() {
                let allocated_bytes = pool.allocated_bytes.load(Ordering::Relaxed);
                let allocation_count = pool.allocation_count.load(Ordering::Relaxed);
                total_capacity += pool.size_bytes;
                
                pool_stats.push(PoolStats {
                    pool_id: *pool_id,
                    name: pool.name.clone(),
                    pool_type: pool.pool_type.clone(),
                    total_size_bytes: pool.size_bytes,
                    allocated_bytes,
                    utilization_percent: (allocated_bytes as f64 / pool.size_bytes as f64) * 100.0,
                    allocation_count,
                    average_allocation_size: if allocation_count > 0 {
                        allocated_bytes as f64 / allocation_count as f64
                    } else {
                        0.0
                    },
                });
            }
            
            (pool_stats, total_capacity)
        };

        let fragmentation_ratio = if total_capacity > 0 {
            1.0 - (total_allocated as f64 / total_capacity as f64)
        } else {
            0.0
        };

        let system_metrics = self.get_system_metrics().await?;

        Ok(MemoryStats {
            total_allocated_bytes: total_allocated,
            peak_allocation_bytes: peak_allocation,
            allocation_count,
            deallocation_count,
            fragmentation_ratio,
            pools: pool_stats,
            system_metrics,
        })
    }

    /// Optimize memory layout for M3 Max performance
    pub async fn optimize_layout(&self) -> Result<()> {
        tracing::info!("Starting M3 Max memory layout optimization");
        
        let pools = self.memory_pools.read();
        for (_pool_id, pool) in pools.iter() {
            self.optimize_pool_layout(pool).await?;
        }
        
        // Trigger system-wide optimization
        self.trigger_m3_max_optimization().await?;
        
        tracing::info!("M3 Max memory layout optimization completed");
        Ok(())
    }

    /// Get M3 Max system information
    pub fn get_system_info(&self) -> &M3MaxSystemInfo {
        &self.system_info
    }

    // Private helper methods

    /// Detect M3 Max system specifications
    async fn detect_m3_max_system() -> Result<M3MaxSystemInfo> {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Get system information
        let total_memory = sys.total_memory();
        let total_memory_gb = (total_memory / 1024 / 1024 / 1024) as u32;

        let cpus = sys.cpus();
        let processor_count = cpus.len() as u8;

        // M3 Max specifications (hardcoded since system detection is limited)
        let (performance_cores, efficiency_cores) = if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            // M3 Max has 8 performance cores + 4 efficiency cores
            (8u8, 4u8)
        } else {
            // For other architectures, distribute evenly
            (processor_count / 2, processor_count / 2)
        };

        Ok(M3MaxSystemInfo {
            total_memory_gb,
            performance_cores,
            efficiency_cores,
            gpu_cores: 40, // M3 Max GPU cores
            neural_engine_tops: 15.8, // M3 Max Neural Engine TOPS
            memory_bandwidth_gbps: 400.0, // M3 Max unified memory bandwidth
            cache_hierarchy: CacheHierarchy {
                l1_instruction_kb: 192, // 192KB per P-core
                l1_data_kb: 128,       // 128KB per P-core
                l2_cache_mb: 24,       // 24MB shared L2
                l3_cache_mb: 0,        // No L3 on Apple Silicon
                slc_cache_mb: 48,      // 48MB System Level Cache
            },
        })
    }

    /// Initialize default memory pools for M3 Max
    async fn initialize_memory_pools(&self) -> Result<()> {
        // Create system processing pool (largest allocation)
        self.create_pool(
            "System Processing",
            (self.config.memory_pools.processing as usize) * 1024 * 1024 * 1024,
            PoolType::System,
            OptimizationFlags::default(),
        ).await?;

        // Create GPU pool for Metal computations
        self.create_pool(
            "GPU Metal",
            8 * 1024 * 1024 * 1024, // 8GB for GPU operations
            PoolType::Gpu,
            OptimizationFlags {
                metal_optimization: true,
                neural_engine_acceleration: false,
                amx_acceleration: false,
                enable_prefetch: true,
                alignment_bytes: 256, // GPU-optimal alignment
            },
        ).await?;

        // Create Neural Engine pool
        self.create_pool(
            "Neural Engine",
            4 * 1024 * 1024 * 1024, // 4GB for Neural Engine
            PoolType::NeuralEngine,
            OptimizationFlags {
                metal_optimization: false,
                neural_engine_acceleration: true,
                amx_acceleration: true,
                enable_prefetch: true,
                alignment_bytes: 64,
            },
        ).await?;

        // Create high-bandwidth pool for data processing
        self.create_pool(
            "High Bandwidth",
            (self.config.memory_pools.ipc as usize) * 1024 * 1024 * 1024,
            PoolType::HighBandwidth,
            OptimizationFlags::default(),
        ).await?;
        
        Ok(())
    }

    /// Allocate base memory for a pool (platform-specific)
    async fn allocate_pool_memory(&self, _size: usize, pool_type: &PoolType) -> Result<u64> {
        // This is a placeholder. In a real implementation, we would use
        // platform-specific APIs (like `mmap` or `mach_vm_allocate`) to
        // allocate memory with appropriate flags for GPU/ANE access.
        let base_address = match pool_type {
            PoolType::Gpu => 0x2_0000_0000,
            PoolType::NeuralEngine => 0x3_0000_0000,
            _ => 0x1_0000_0000,
        };
        Ok(base_address)
    }

    /// Configure memory pool for M3 Max hardware
    async fn configure_pool_for_m3_max(&self, pool: &M3MaxMemoryPool) -> Result<()> {
        tracing::debug!("Configuring pool '{}' for M3 Max", pool.name);
        
        match pool.pool_type {
            PoolType::Gpu => self.configure_metal_optimization(pool).await?,
            PoolType::NeuralEngine => self.configure_neural_engine(pool).await?,
            PoolType::System | PoolType::HighBandwidth | PoolType::Cache => {
                self.configure_cpu_optimization(pool).await?
            }
        }
        
        Ok(())
    }

    /// Configure Metal optimizations for a GPU pool
    async fn configure_metal_optimization(&self, _pool: &M3MaxMemoryPool) -> Result<()> {
        // In a real implementation, this would interact with Metal APIs
        // to register the memory region for GPU access.
        tracing::debug!("Metal optimization configured for GPU pool");
        Ok(())
    }

    /// Configure Neural Engine optimizations for an ANE pool
    async fn configure_neural_engine(&self, _pool: &M3MaxMemoryPool) -> Result<()> {
        // In a real implementation, this would interact with Core ML APIs
        // to prepare the memory for ANE processing.
        tracing::debug!("Neural Engine optimization configured for ANE pool");
        Ok(())
    }
    
    /// Configure CPU-specific optimizations (AMX, prefetch)
    async fn configure_cpu_optimization(&self, pool: &M3MaxMemoryPool) -> Result<()> {
        if pool.optimization_flags.enable_prefetch {
            // Modern CPUs handle this well, but we could issue prefetch
            // hints for specific access patterns.
            tracing::debug!("CPU prefetching enabled for pool '{}'", pool.name);
        }
        if pool.optimization_flags.amx_acceleration {
            // AMX is accessed via specific instructions, no specific setup needed here
            tracing::debug!("AMX acceleration noted for pool '{}'", pool.name);
        }
        Ok(())
    }

    fn align_size(&self, size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    fn align_to_m3_max_boundary(&self, size: usize, flags: &OptimizationFlags) -> usize {
        self.align_size(size, flags.alignment_bytes)
    }

    /// Allocate memory with optimized flags
    async fn allocate_optimized_memory(
        &self,
        base_address: u64,
        offset: usize,
        size: usize,
        flags: &OptimizationFlags,
    ) -> Result<u64> {
        let addr = base_address + offset as u64;
        if flags.enable_prefetch {
            self.setup_memory_prefetch(addr, size).await?;
        }
        Ok(addr)
    }

    /// Setup memory prefetching for a memory region
    async fn setup_memory_prefetch(&self, _address: u64, _size: usize) -> Result<()> {
        // This is a placeholder for platform-specific prefetch setup
        Ok(())
    }

    /// Cleanup optimized memory
    async fn cleanup_optimized_memory(
        &self,
        _address: u64,
        _size: usize,
        _flags: &OptimizationFlags,
    ) -> Result<()> {
        // Placeholder for cleanup logic (e.g., unmapping memory)
        Ok(())
    }

    /// Start background task for performance monitoring
    async fn start_performance_monitoring(&self) {
        let monitor = self.performance_monitor.clone();
        let system = self.system.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                
                let mut sys = system.lock();
                sys.refresh_cpu();
                sys.refresh_memory();

                let cpu_usage = sys.cpus().iter()
                    .map(|c| c.cpu_usage() as u64)
                    .sum::<u64>() / sys.cpus().len() as u64;

                monitor.cpu_usage.store(cpu_usage * 100, Ordering::Relaxed);
                
                let memory_pressure = (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0;
                monitor.memory_pressure.store(memory_pressure as u64 * 100, Ordering::Relaxed);
                
                monitor.last_update.store(current_timestamp_ms(), Ordering::Relaxed);
            }
        });
    }

    /// Get current system metrics
    async fn get_system_metrics(&self) -> Result<SystemMetrics> {
        let monitor = &self.performance_monitor;
        let cpu_usage = monitor.cpu_usage.load(Ordering::Relaxed) as f64 / 100.0;
        let memory_pressure = monitor.memory_pressure.load(Ordering::Relaxed) as f64 / 100.0;
        let gpu_utilization = monitor.gpu_utilization.load(Ordering::Relaxed) as f64 / 100.0;
        
        let thermal_state = match (cpu_usage + gpu_utilization) / 2.0 {
            x if x > 90.0 => ThermalState::Critical,
            x if x > 80.0 => ThermalState::Serious,
            x if x > 60.0 => ThermalState::Fair,
            _ => ThermalState::Normal,
        };
        
        let memory_bandwidth_utilization = (cpu_usage + gpu_utilization) * self.system_info.memory_bandwidth_gbps as f64 / 2.0;

        Ok(SystemMetrics {
            cpu_usage_percent: cpu_usage,
            memory_pressure,
            gpu_utilization_percent: gpu_utilization,
            neural_engine_utilization_percent: 0.0, // Placeholder
            memory_bandwidth_utilization_gbps: memory_bandwidth_utilization,
            thermal_state,
        })
    }
    
    /// Optimize memory layout for a specific pool
    async fn optimize_pool_layout(&self, _pool: &M3MaxMemoryPool) -> Result<()> {
        // Placeholder for pool-specific optimization logic
        Ok(())
    }

    /// Trigger system-wide memory optimization
    async fn trigger_m3_max_optimization(&self) -> Result<()> {
        // Placeholder for system-level calls (e.g., `purge`)
        Ok(())
    }
}


/// Initialize the M3 Max memory manager globally
pub async fn initialize(config: &M3MaxConfig) -> Result<()> {
    let _manager = M3MaxMemoryManager::new(config.clone()).await?;
    // Store manager in a global static or return it for use in the pipeline
    // For now, it's created and dropped, which is not useful.
    // This part of the design needs to be completed.
    Ok(())
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_m3_max_system_detection() {
        let sys_info = M3MaxMemoryManager::detect_m3_max_system().await.unwrap();
        assert!(sys_info.total_memory_gb > 0);
        assert!(sys_info.performance_cores > 0);
        assert!(sys_info.efficiency_cores > 0);
    }

    #[tokio::test]
    async fn test_memory_alignment() {
        let config = M3MaxConfig::default();
        let manager = M3MaxMemoryManager::new(config).await.unwrap();
        assert_eq!(manager.align_size(100, 64), 128);
        assert_eq!(manager.align_size(64, 64), 64);
        assert_eq!(manager.align_size(65, 64), 128);
    }

    #[tokio::test]
    async fn test_memory_pool_creation() {
        let config = M3MaxConfig::default();
        let manager = M3MaxMemoryManager::new(config).await.unwrap();
        let pool_id = manager.create_pool("test-pool", 1024, PoolType::System, OptimizationFlags::default()).await.unwrap();
        
        let stats = manager.get_stats().await.unwrap();
        assert_eq!(stats.pools.len(), 5); // 4 default + 1 new
        assert_eq!(stats.pools.iter().find(|p| p.pool_id == pool_id).is_some(), true);
    }
}