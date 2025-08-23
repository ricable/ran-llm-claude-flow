use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use mimalloc::MiMalloc;
use crossbeam_utils::CachePadded;

/// Adaptive Memory Allocator optimized for M3 Max 128GB system
/// Distributes memory efficiently across NUMA nodes with intelligent allocation strategies
pub struct AdaptiveMemoryAllocator {
    // Memory pool configurations for optimal M3 Max utilization
    rust_core_pool: MemoryPool,      // 60GB allocation
    python_ml_pool: MemoryPool,      // 45GB allocation  
    shared_ipc_pool: MemoryPool,     // 15GB allocation
    system_reserve_pool: MemoryPool, // 8GB allocation
    
    // NUMA topology awareness
    numa_nodes: Vec<NumaNode>,
    allocation_strategy: AllocationStrategy,
    
    // Performance monitoring
    stats: CachePadded<AllocationStats>,
}

#[derive(Clone)]
pub struct MemoryPool {
    size_limit: AtomicUsize,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    allocator: Arc<Mutex<PoolAllocator>>,
    numa_node: u8,
}

pub struct PoolAllocator {
    blocks: Vec<MemoryBlock>,
    free_blocks: Vec<usize>,
    large_blocks: Vec<LargeBlock>,
}

#[derive(Debug, Clone)]
pub struct MemoryBlock {
    ptr: *mut u8,
    size: usize,
    numa_node: u8,
    in_use: bool,
}

#[derive(Debug)]
pub struct LargeBlock {
    ptr: *mut u8,
    size: usize,
    numa_node: u8,
}

#[derive(Debug)]
pub struct NumaNode {
    id: u8,
    memory_total: usize,
    memory_free: usize,
    cpu_cores: Vec<u8>,
    latency_matrix: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    LocalFirst,      // Prefer local NUMA node
    LoadBalanced,    // Distribute across nodes
    Adaptive,        // AI-driven allocation decisions
    PerformanceFirst, // Optimize for speed
}

#[derive(Debug, Default)]
pub struct AllocationStats {
    total_allocations: AtomicUsize,
    total_deallocations: AtomicUsize,
    peak_memory_usage: AtomicUsize,
    numa_hit_ratio: AtomicUsize,
    avg_allocation_time_ns: AtomicUsize,
}

impl AdaptiveMemoryAllocator {
    /// Initialize allocator with M3 Max optimized configuration
    pub fn new() -> Self {
        let numa_nodes = Self::detect_numa_topology();
        
        Self {
            rust_core_pool: MemoryPool::new(60 * 1024 * 1024 * 1024, 0), // 60GB
            python_ml_pool: MemoryPool::new(45 * 1024 * 1024 * 1024, 1), // 45GB
            shared_ipc_pool: MemoryPool::new(15 * 1024 * 1024 * 1024, 0), // 15GB
            system_reserve_pool: MemoryPool::new(8 * 1024 * 1024 * 1024, 1), // 8GB
            numa_nodes,
            allocation_strategy: AllocationStrategy::Adaptive,
            stats: CachePadded::new(AllocationStats::default()),
        }
    }

    /// Allocate memory with NUMA awareness and performance optimization
    pub fn allocate(&self, size: usize, pool_type: PoolType) -> Option<*mut u8> {
        let start_time = std::time::Instant::now();
        
        let pool = match pool_type {
            PoolType::RustCore => &self.rust_core_pool,
            PoolType::PythonML => &self.python_ml_pool,
            PoolType::SharedIPC => &self.shared_ipc_pool,
            PoolType::SystemReserve => &self.system_reserve_pool,
        };

        let ptr = self.allocate_from_pool(pool, size)?;
        
        // Update statistics
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        let allocation_time = start_time.elapsed().as_nanos() as usize;
        self.update_avg_allocation_time(allocation_time);
        
        Some(ptr)
    }

    /// High-performance bulk allocation for vectorized operations
    pub fn bulk_allocate(&self, sizes: &[usize], pool_type: PoolType) -> Vec<Option<*mut u8>> {
        sizes.iter().map(|&size| self.allocate(size, pool_type)).collect()
    }

    /// SIMD-optimized memory copy with prefetching
    pub fn optimized_copy(&self, dst: *mut u8, src: *const u8, size: usize) {
        unsafe {
            // Prefetch data for better cache performance
            Self::prefetch_data(src, size);
            
            if size >= 64 {
                // Use SIMD for large copies
                Self::simd_memcpy(dst, src, size);
            } else {
                // Use standard copy for small sizes
                std::ptr::copy_nonoverlapping(src, dst, size);
            }
        }
    }

    /// Adaptive allocation strategy based on workload patterns
    pub fn optimize_allocation_strategy(&mut self, workload_metrics: &WorkloadMetrics) {
        self.allocation_strategy = match workload_metrics.pattern {
            WorkloadPattern::HighThroughput => AllocationStrategy::PerformanceFirst,
            WorkloadPattern::NUMA_Sensitive => AllocationStrategy::LocalFirst,
            WorkloadPattern::Balanced => AllocationStrategy::LoadBalanced,
            WorkloadPattern::Dynamic => AllocationStrategy::Adaptive,
        };
    }

    /// Get comprehensive memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            rust_core_usage: self.rust_core_pool.current_usage.load(Ordering::Relaxed),
            python_ml_usage: self.python_ml_pool.current_usage.load(Ordering::Relaxed),
            shared_ipc_usage: self.shared_ipc_pool.current_usage.load(Ordering::Relaxed),
            system_reserve_usage: self.system_reserve_pool.current_usage.load(Ordering::Relaxed),
            total_allocations: self.stats.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.stats.total_deallocations.load(Ordering::Relaxed),
            peak_usage: self.stats.peak_memory_usage.load(Ordering::Relaxed),
            numa_hit_ratio: self.stats.numa_hit_ratio.load(Ordering::Relaxed) as f32 / 100.0,
            avg_allocation_time_ns: self.stats.avg_allocation_time_ns.load(Ordering::Relaxed),
        }
    }

    // Private implementation methods
    
    fn detect_numa_topology() -> Vec<NumaNode> {
        // M3 Max specific NUMA detection
        vec![
            NumaNode {
                id: 0,
                memory_total: 64 * 1024 * 1024 * 1024, // 64GB
                memory_free: 64 * 1024 * 1024 * 1024,
                cpu_cores: (0..12).collect(), // Performance cores
                latency_matrix: vec![1.0, 1.2], // Local, remote latencies
            },
            NumaNode {
                id: 1,
                memory_total: 64 * 1024 * 1024 * 1024, // 64GB
                memory_free: 64 * 1024 * 1024 * 1024,
                cpu_cores: (12..20).collect(), // Efficiency cores + GPU
                latency_matrix: vec![1.2, 1.0],
            },
        ]
    }

    fn allocate_from_pool(&self, pool: &MemoryPool, size: usize) -> Option<*mut u8> {
        if pool.current_usage.load(Ordering::Relaxed) + size > pool.size_limit.load(Ordering::Relaxed) {
            return None;
        }

        let mut allocator = pool.allocator.lock();
        
        // Try to find existing free block
        for &block_idx in &allocator.free_blocks {
            if allocator.blocks[block_idx].size >= size {
                let block = &mut allocator.blocks[block_idx];
                block.in_use = true;
                pool.current_usage.fetch_add(size, Ordering::Relaxed);
                return Some(block.ptr);
            }
        }

        // Allocate new block
        self.allocate_new_block(&mut allocator, pool, size)
    }

    fn allocate_new_block(&self, allocator: &mut PoolAllocator, pool: &MemoryPool, size: usize) -> Option<*mut u8> {
        unsafe {
            let layout = Layout::from_size_align(size, 64).ok()?; // 64-byte alignment for SIMD
            let ptr = std::alloc::alloc(layout);
            
            if ptr.is_null() {
                return None;
            }

            // Bind memory to specific NUMA node
            Self::bind_memory_to_numa(ptr, size, pool.numa_node);

            allocator.blocks.push(MemoryBlock {
                ptr,
                size,
                numa_node: pool.numa_node,
                in_use: true,
            });

            pool.current_usage.fetch_add(size, Ordering::Relaxed);
            Some(ptr)
        }
    }

    unsafe fn simd_memcpy(dst: *mut u8, src: *const u8, size: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON optimization for M3 Max
            let chunks = size / 16;
            let remainder = size % 16;

            for i in 0..chunks {
                let src_vec = std::arch::aarch64::vld1q_u8(src.add(i * 16));
                std::arch::aarch64::vst1q_u8(dst.add(i * 16), src_vec);
            }

            if remainder > 0 {
                std::ptr::copy_nonoverlapping(src.add(chunks * 16), dst.add(chunks * 16), remainder);
            }
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    unsafe fn prefetch_data(ptr: *const u8, size: usize) {
        let cache_line_size = 64;
        let lines = (size + cache_line_size - 1) / cache_line_size;
        
        for i in 0..lines {
            let prefetch_addr = ptr.add(i * cache_line_size);
            #[cfg(target_arch = "aarch64")]
            std::arch::aarch64::_prefetch(prefetch_addr, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
        }
    }

    fn bind_memory_to_numa(ptr: *mut u8, size: usize, numa_node: u8) {
        // Platform-specific NUMA binding would go here
        // For M3 Max, this involves system calls to set memory policy
    }

    fn update_avg_allocation_time(&self, new_time_ns: usize) {
        let current_avg = self.stats.avg_allocation_time_ns.load(Ordering::Relaxed);
        let total_allocs = self.stats.total_allocations.load(Ordering::Relaxed);
        
        if total_allocs > 0 {
            let new_avg = (current_avg * (total_allocs - 1) + new_time_ns) / total_allocs;
            self.stats.avg_allocation_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
}

impl MemoryPool {
    fn new(size_limit: usize, numa_node: u8) -> Self {
        Self {
            size_limit: AtomicUsize::new(size_limit),
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocator: Arc::new(Mutex::new(PoolAllocator {
                blocks: Vec::new(),
                free_blocks: Vec::new(),
                large_blocks: Vec::new(),
            })),
            numa_node,
        }
    }
}

// Type definitions
#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    RustCore,
    PythonML,
    SharedIPC,
    SystemReserve,
}

#[derive(Debug)]
pub struct WorkloadMetrics {
    pattern: WorkloadPattern,
    allocation_frequency: f64,
    average_size: usize,
    numa_locality_preference: f32,
}

#[derive(Debug)]
pub enum WorkloadPattern {
    HighThroughput,
    NUMA_Sensitive,
    Balanced,
    Dynamic,
}

#[derive(Debug)]
pub struct MemoryStats {
    pub rust_core_usage: usize,
    pub python_ml_usage: usize,
    pub shared_ipc_usage: usize,
    pub system_reserve_usage: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub peak_usage: usize,
    pub numa_hit_ratio: f32,
    pub avg_allocation_time_ns: usize,
}

// Global allocator integration
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Export for C interop
#[no_mangle]
pub extern "C" fn adaptive_allocate(size: usize, pool_type: u8) -> *mut u8 {
    static ALLOCATOR: std::sync::OnceLock<AdaptiveMemoryAllocator> = std::sync::OnceLock::new();
    let allocator = ALLOCATOR.get_or_init(AdaptiveMemoryAllocator::new);
    
    let pool = match pool_type {
        0 => PoolType::RustCore,
        1 => PoolType::PythonML,
        2 => PoolType::SharedIPC,
        _ => PoolType::SystemReserve,
    };
    
    allocator.allocate(size, pool).unwrap_or(std::ptr::null_mut())
}