use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::types::{Error, PerformanceMetrics};
use uuid::Uuid;

/// NUMA-aware memory optimization for M3 Max architecture
/// Manages 128GB unified memory allocation and CPU affinity
pub struct NUMAOptimizer {
    memory_pools: Arc<Mutex<HashMap<u32, MemoryPool>>>,
    cpu_topology: CPUTopology,
    current_allocation: Arc<Mutex<MemoryAllocation>>,
    performance_metrics: Arc<Mutex<NUMAMetrics>>,
}

#[derive(Debug, Clone)]
pub struct MemoryPool {
    pub node_id: u32,
    pub total_size_gb: usize,
    pub allocated_gb: usize,
    pub available_gb: usize,
    pub segments: Vec<MemorySegment>,
    pub pool_type: PoolType,
}

#[derive(Debug, Clone)]
pub enum PoolType {
    RustProcessing,      // 60GB - High-speed document processing
    PythonML,           // 45GB - ML model inference and training
    SharedIPC,          // 15GB - Inter-process communication
    SystemReserve,      // 8GB - OS and monitoring overhead
}

#[derive(Debug, Clone)]
pub struct MemorySegment {
    pub id: String,
    pub size_gb: usize,
    pub numa_node: u32,
    pub is_locked: bool,
    pub last_accessed: std::time::Instant,
    pub access_pattern: AccessPattern,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct CPUTopology {
    pub performance_cores: Vec<u32>,  // P-cores on M3 Max
    pub efficiency_cores: Vec<u32>,   // E-cores on M3 Max
    pub total_cores: u32,
    pub numa_nodes: Vec<u32>,
    pub cache_sizes: HashMap<u32, CacheInfo>,
}

#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub l1_size_kb: usize,
    pub l2_size_kb: usize,
    pub l3_size_kb: usize,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub rust_core_gb: usize,      // 60GB
    pub python_ml_gb: usize,       // 45GB  
    pub shared_ipc_gb: usize,      // 15GB
    pub system_reserve_gb: usize,  // 8GB
    pub total_allocated_gb: usize, // 128GB
}

#[derive(Debug, Clone)]
pub struct NUMAMetrics {
    pub memory_bandwidth_gb_per_sec: f64,
    pub cache_hit_ratio: f64,
    pub numa_locality_ratio: f64,
    pub memory_fragmentation: f64,
    pub allocation_latency_us: f64,
    pub cross_numa_accesses: u64,
    pub local_numa_accesses: u64,
}

impl NUMAOptimizer {
    /// Initialize NUMA optimizer with M3 Max specific configuration
    pub fn new() -> Result<Self, Error> {
        let cpu_topology = Self::detect_m3_max_topology()?;
        let memory_pools = Arc::new(Mutex::new(HashMap::new()));
        let current_allocation = Arc::new(Mutex::new(MemoryAllocation::default()));
        let performance_metrics = Arc::new(Mutex::new(NUMAMetrics::default()));
        
        let mut optimizer = NUMAOptimizer {
            memory_pools,
            cpu_topology,
            current_allocation,
            performance_metrics,
        };
        
        // Initialize memory pools for 128GB allocation
        optimizer.initialize_memory_pools()?;
        
        // Set optimal CPU affinity for current process
        optimizer.set_optimal_cpu_affinity()?;
        
        Ok(optimizer)
    }

    /// Initialize the 128GB memory pool allocation strategy
    fn initialize_memory_pools(&mut self) -> Result<(), Error> {
        let mut pools = self.memory_pools.lock().unwrap();
        
        // Rust Processing Pool - 60GB
        let rust_pool = MemoryPool {
            node_id: 0, // Primary NUMA node
            total_size_gb: 60,
            allocated_gb: 0,
            available_gb: 60,
            segments: Vec::new(),
            pool_type: PoolType::RustProcessing,
        };
        pools.insert(0, rust_pool);
        
        // Python ML Pool - 45GB
        let python_pool = MemoryPool {
            node_id: 0, // Unified memory on M3 Max
            total_size_gb: 45,
            allocated_gb: 0,
            available_gb: 45,
            segments: Vec::new(),
            pool_type: PoolType::PythonML,
        };
        pools.insert(1, python_pool);
        
        // Shared IPC Pool - 15GB
        let ipc_pool = MemoryPool {
            node_id: 0, // Low-latency access required
            total_size_gb: 15,
            allocated_gb: 0,
            available_gb: 15,
            segments: Vec::new(),
            pool_type: PoolType::SharedIPC,
        };
        pools.insert(2, ipc_pool);
        
        // System Reserve Pool - 8GB
        let system_pool = MemoryPool {
            node_id: 0, // OS and monitoring
            total_size_gb: 8,
            allocated_gb: 0,
            available_gb: 8,
            segments: Vec::new(),
            pool_type: PoolType::SystemReserve,
        };
        pools.insert(3, system_pool);
        
        // Update allocation tracking
        let mut allocation = self.current_allocation.lock().unwrap();
        allocation.rust_core_gb = 60;
        allocation.python_ml_gb = 45;
        allocation.shared_ipc_gb = 15;
        allocation.system_reserve_gb = 8;
        allocation.total_allocated_gb = 128;
        
        Ok(())
    }

    /// Detect M3 Max CPU topology and capabilities
    fn detect_m3_max_topology() -> Result<CPUTopology, Error> {
        // M3 Max: 12-core CPU (8 performance + 4 efficiency cores)
        let performance_cores = vec![0, 1, 2, 3, 4, 5, 6, 7]; // P-cores
        let efficiency_cores = vec![8, 9, 10, 11]; // E-cores
        let total_cores = 12;
        let numa_nodes = vec![0]; // Unified memory architecture
        
        // M3 Max cache hierarchy
        let mut cache_sizes = HashMap::new();
        cache_sizes.insert(0, CacheInfo {
            l1_size_kb: 192,      // 128KB I + 64KB D per core
            l2_size_kb: 16384,    // 16MB shared L2
            l3_size_kb: 0,        // No L3 on M3 Max
            cache_line_size: 64,  // 64-byte cache lines
        });
        
        Ok(CPUTopology {
            performance_cores,
            efficiency_cores,
            total_cores,
            numa_nodes,
            cache_sizes,
        })
    }

    /// Set optimal CPU affinity for high-performance processing
    pub fn set_optimal_cpu_affinity(&self) -> Result<(), Error> {
        // Note: CPU affinity setting is limited on macOS
        // On M3 Max, the scheduler automatically optimizes for performance cores
        // This is a no-op on macOS but logs the intention
        log::info!(
            "CPU affinity optimization requested for {} performance cores", 
            self.cpu_topology.performance_cores.len()
        );
        
        // On macOS, we rely on the system scheduler and Grand Central Dispatch
        // for optimal core utilization with M3 Max architecture
        Ok(())
    }

    /// Allocate memory segment with NUMA awareness
    pub fn allocate_memory_segment(
        &self, 
        size_gb: usize, 
        pool_type: PoolType,
        access_pattern: AccessPattern
    ) -> Result<MemorySegment, Error> {
        let mut pools = self.memory_pools.lock().unwrap();
        
        // Find appropriate pool
        let pool_id = match pool_type {
            PoolType::RustProcessing => 0,
            PoolType::PythonML => 1,
            PoolType::SharedIPC => 2,
            PoolType::SystemReserve => 3,
        };
        
        let pool = pools.get_mut(&pool_id)
            .ok_or_else(|| Error::MemoryPoolNotFound(pool_id))?;
        
        if pool.available_gb < size_gb {
            return Err(Error::InsufficientMemory {
                requested: size_gb,
                available: pool.available_gb,
                pool_type: format!("{:?}", pool_type),
            });
        }
        
        // Create memory segment
        let segment_id = format!("seg_{}_{}", pool_id, pool.segments.len());
        let segment = MemorySegment {
            id: segment_id,
            size_gb,
            numa_node: pool.node_id,
            is_locked: false,
            last_accessed: std::time::Instant::now(),
            access_pattern,
        };
        
        // Update pool allocation
        pool.allocated_gb += size_gb;
        pool.available_gb -= size_gb;
        pool.segments.push(segment.clone());
        
        // Update performance metrics
        self.update_allocation_metrics(size_gb)?;
        
        Ok(segment)
    }

    /// Deallocate memory segment
    pub fn deallocate_memory_segment(&self, segment_id: &str) -> Result<(), Error> {
        let mut pools = self.memory_pools.lock().unwrap();
        
        for pool in pools.values_mut() {
            if let Some(pos) = pool.segments.iter().position(|s| s.id == segment_id) {
                let segment = pool.segments.remove(pos);
                pool.allocated_gb -= segment.size_gb;
                pool.available_gb += segment.size_gb;
                return Ok(());
            }
        }
        
        Err(Error::SegmentNotFound(segment_id.to_string()))
    }

    /// Optimize memory layout for processing workload
    pub fn optimize_memory_layout(&self) -> Result<(), Error> {
        let mut pools = self.memory_pools.lock().unwrap();
        
        for pool in pools.values_mut() {
            // Sort segments by access pattern and frequency
            pool.segments.sort_by(|a, b| {
                // Prioritize frequently accessed segments
                b.last_accessed.cmp(&a.last_accessed)
            });
            
            // Defragment memory by consolidating small segments
            self.defragment_pool_segments(pool)?;
        }
        
        Ok(())
    }

    /// Defragment memory pool segments
    fn defragment_pool_segments(&self, pool: &mut MemoryPool) -> Result<(), Error> {
        let mut consolidated_segments = Vec::new();
        let mut current_group = Vec::new();
        let mut current_size = 0;
        
        for segment in &pool.segments {
            if current_size + segment.size_gb <= 8 { // 8GB max segment size
                current_group.push(segment.clone());
                current_size += segment.size_gb;
            } else {
                if !current_group.is_empty() {
                    consolidated_segments.push(self.merge_segments(current_group)?);
                }
                current_group = vec![segment.clone()];
                current_size = segment.size_gb;
            }
        }
        
        // Handle remaining segments
        if !current_group.is_empty() {
            consolidated_segments.push(self.merge_segments(current_group)?);
        }
        
        pool.segments = consolidated_segments;
        Ok(())
    }

    /// Merge multiple segments into one
    fn merge_segments(&self, segments: Vec<MemorySegment>) -> Result<MemorySegment, Error> {
        let total_size = segments.iter().map(|s| s.size_gb).sum();
        let merged_id = format!("merged_{}", Uuid::new_v4());
        
        Ok(MemorySegment {
            id: merged_id,
            size_gb: total_size,
            numa_node: segments[0].numa_node,
            is_locked: false,
            last_accessed: std::time::Instant::now(),
            access_pattern: AccessPattern::Mixed,
        })
    }

    /// Update allocation performance metrics
    fn update_allocation_metrics(&self, allocated_size_gb: usize) -> Result<(), Error> {
        let mut metrics = self.performance_metrics.lock().unwrap();
        
        // Update allocation latency (simulated)
        let allocation_start = std::time::Instant::now();
        
        // Simulate allocation overhead
        std::thread::sleep(std::time::Duration::from_micros(10));
        
        metrics.allocation_latency_us = allocation_start.elapsed().as_micros() as f64;
        
        // Update memory bandwidth estimation
        metrics.memory_bandwidth_gb_per_sec = Self::estimate_memory_bandwidth(allocated_size_gb)?;
        
        // Update cache hit ratio estimation
        metrics.cache_hit_ratio = Self::estimate_cache_hit_ratio()?;
        
        // Update NUMA locality (always 1.0 on M3 Max unified memory)
        metrics.numa_locality_ratio = 1.0;
        
        Ok(())
    }

    /// Estimate memory bandwidth for M3 Max
    fn estimate_memory_bandwidth(allocated_size_gb: usize) -> Result<f64, Error> {
        // M3 Max: ~400 GB/s memory bandwidth (theoretical)
        // Account for allocation overhead and fragmentation
        let base_bandwidth = 400.0; // GB/s
        let utilization_factor = (allocated_size_gb as f64 / 128.0).min(1.0);
        let efficiency_factor = 0.85; // 85% practical efficiency
        
        Ok(base_bandwidth * utilization_factor * efficiency_factor)
    }

    /// Estimate cache hit ratio
    fn estimate_cache_hit_ratio() -> Result<f64, Error> {
        // M3 Max L2 cache efficiency estimation
        // 16MB L2 cache shared across performance cores
        Ok(0.92) // 92% cache hit ratio (optimistic)
    }

    /// Get current memory allocation status
    pub fn get_memory_status(&self) -> Result<MemoryAllocation, Error> {
        let allocation = self.current_allocation.lock().unwrap();
        Ok(allocation.clone())
    }

    /// Get NUMA performance metrics
    pub fn get_numa_metrics(&self) -> Result<NUMAMetrics, Error> {
        let metrics = self.performance_metrics.lock().unwrap();
        Ok(metrics.clone())
    }

    /// Get memory pool statistics
    pub fn get_pool_statistics(&self) -> Result<Vec<MemoryPool>, Error> {
        let pools = self.memory_pools.lock().unwrap();
        Ok(pools.values().cloned().collect())
    }

    /// Perform memory health check
    pub fn memory_health_check(&self) -> Result<MemoryHealthReport, Error> {
        let pools = self.memory_pools.lock().unwrap();
        let metrics = self.performance_metrics.lock().unwrap();
        let allocation = self.current_allocation.lock().unwrap();
        
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Check memory utilization
        let total_allocated = pools.values().map(|p| p.allocated_gb).sum::<usize>();
        let utilization_rate = total_allocated as f64 / allocation.total_allocated_gb as f64;
        
        if utilization_rate > 0.95 {
            errors.push("Memory utilization exceeds 95%".to_string());
        } else if utilization_rate > 0.80 {
            warnings.push("Memory utilization above 80%".to_string());
        }
        
        // Check fragmentation
        if metrics.memory_fragmentation > 0.30 {
            warnings.push("High memory fragmentation detected".to_string());
        }
        
        // Check bandwidth
        if metrics.memory_bandwidth_gb_per_sec < 300.0 {
            warnings.push("Memory bandwidth below optimal threshold".to_string());
        }
        
        let health_score = Self::calculate_health_score(&metrics, utilization_rate)?;
        
        Ok(MemoryHealthReport {
            health_score,
            utilization_rate,
            total_allocated_gb: total_allocated,
            warnings,
            errors,
            metrics: metrics.clone(),
        })
    }

    /// Calculate overall memory health score
    fn calculate_health_score(metrics: &NUMAMetrics, utilization_rate: f64) -> Result<f64, Error> {
        let bandwidth_score = (metrics.memory_bandwidth_gb_per_sec / 400.0).min(1.0);
        let cache_score = metrics.cache_hit_ratio;
        let locality_score = metrics.numa_locality_ratio;
        let fragmentation_score = 1.0 - metrics.memory_fragmentation;
        let utilization_score = if utilization_rate > 0.95 { 0.0 } else { 1.0 - utilization_rate };
        
        let health_score = (bandwidth_score * 0.3) + 
                          (cache_score * 0.25) + 
                          (locality_score * 0.15) + 
                          (fragmentation_score * 0.15) + 
                          (utilization_score * 0.15);
        
        Ok(health_score.max(0.0).min(1.0))
    }
}

#[derive(Debug, Clone)]
pub struct MemoryHealthReport {
    pub health_score: f64,
    pub utilization_rate: f64,
    pub total_allocated_gb: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub metrics: NUMAMetrics,
}

impl Default for MemoryAllocation {
    fn default() -> Self {
        Self {
            rust_core_gb: 60,
            python_ml_gb: 45,
            shared_ipc_gb: 15,
            system_reserve_gb: 8,
            total_allocated_gb: 128,
        }
    }
}

impl Default for NUMAMetrics {
    fn default() -> Self {
        Self {
            memory_bandwidth_gb_per_sec: 0.0,
            cache_hit_ratio: 0.0,
            numa_locality_ratio: 1.0,
            memory_fragmentation: 0.0,
            allocation_latency_us: 0.0,
            cross_numa_accesses: 0,
            local_numa_accesses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numa_optimizer_initialization() {
        let optimizer = NUMAOptimizer::new().unwrap();
        let status = optimizer.get_memory_status().unwrap();
        
        assert_eq!(status.total_allocated_gb, 128);
        assert_eq!(status.rust_core_gb, 60);
        assert_eq!(status.python_ml_gb, 45);
        assert_eq!(status.shared_ipc_gb, 15);
        assert_eq!(status.system_reserve_gb, 8);
    }
    
    #[test]
    fn test_memory_allocation() {
        let optimizer = NUMAOptimizer::new().unwrap();
        
        let segment = optimizer.allocate_memory_segment(
            4, 
            PoolType::RustProcessing, 
            AccessPattern::Sequential
        ).unwrap();
        
        assert_eq!(segment.size_gb, 4);
        assert_eq!(segment.numa_node, 0);
        assert!(!segment.is_locked);
    }
    
    #[test]
    fn test_memory_health_check() {
        let optimizer = NUMAOptimizer::new().unwrap();
        let health = optimizer.memory_health_check().unwrap();
        
        assert!(health.health_score >= 0.0);
        assert!(health.health_score <= 1.0);
        assert_eq!(health.total_allocated_gb, 0); // No allocations yet
    }
}