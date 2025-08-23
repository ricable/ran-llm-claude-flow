"""
M3 Max Unified Memory Manager

Optimized memory management for MacBook Pro M3 Max with 128GB unified memory.
Implements intelligent allocation, Apple Silicon acceleration, and performance monitoring.
"""

import asyncio
import logging
import psutil
import gc
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import weakref

from ..interfaces.memory import (
    IMemoryManager, IMemoryPool, IResourceAllocator, IMemoryOptimizer,
    MemoryPoolType, AllocationStrategy, MemoryUsage, ResourceLimits, M3MaxResources
)


@dataclass
class AllocationRecord:
    """Record of memory allocation."""
    allocation_id: str
    size_gb: float
    requester_id: str
    allocated_at: float
    pool_type: MemoryPoolType
    priority: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def update_access_time(self):
        """Update last access time."""
        self.last_accessed = time.time()


class MemoryPool(IMemoryPool):
    """Implementation of memory pool with intelligent allocation."""
    
    def __init__(self, 
                 pool_type: MemoryPoolType, 
                 size_gb: float,
                 strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE):
        self._pool_type = pool_type
        self._size_gb = size_gb
        self._strategy = strategy
        self._allocations: Dict[str, AllocationRecord] = {}
        self._allocated_gb = 0.0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{pool_type.value}")
        self._allocation_counter = 0
    
    @property
    def pool_type(self) -> MemoryPoolType:
        return self._pool_type
    
    @property
    def size_gb(self) -> float:
        return self._size_gb
    
    @property
    def available_gb(self) -> float:
        return max(0, self._size_gb - self._allocated_gb)
    
    async def allocate(self, size_gb: float, requester_id: str) -> Optional[str]:
        """Allocate memory from pool."""
        async with self._lock:
            if self.available_gb < size_gb:
                # Try to free memory if using adaptive strategy
                if self._strategy == AllocationStrategy.ADAPTIVE:
                    freed_gb = await self._try_free_memory(size_gb)
                    if freed_gb < size_gb:
                        self._logger.warning(
                            f"Insufficient memory in {self._pool_type.value} pool. "
                            f"Requested: {size_gb}GB, Available: {self.available_gb}GB"
                        )
                        return None
                else:
                    return None
            
            # Create allocation record
            self._allocation_counter += 1
            allocation_id = f"{self._pool_type.value}_{self._allocation_counter}_{int(time.time())}"
            
            record = AllocationRecord(
                allocation_id=allocation_id,
                size_gb=size_gb,
                requester_id=requester_id,
                allocated_at=time.time(),
                pool_type=self._pool_type
            )
            
            self._allocations[allocation_id] = record
            self._allocated_gb += size_gb
            
            self._logger.debug(
                f"Allocated {size_gb}GB for {requester_id}. "
                f"Pool usage: {self._allocated_gb}/{self._size_gb}GB"
            )
            
            return allocation_id
    
    async def deallocate(self, allocation_id: str) -> bool:
        """Deallocate memory by allocation ID."""
        async with self._lock:
            if allocation_id not in self._allocations:
                self._logger.warning(f"Allocation ID {allocation_id} not found")
                return False
            
            record = self._allocations[allocation_id]
            self._allocated_gb -= record.size_gb
            del self._allocations[allocation_id]
            
            self._logger.debug(
                f"Deallocated {record.size_gb}GB from {record.requester_id}. "
                f"Pool usage: {self._allocated_gb}/{self._size_gb}GB"
            )
            
            return True
    
    async def resize_allocation(self, allocation_id: str, new_size_gb: float) -> bool:
        """Resize existing allocation."""
        async with self._lock:
            if allocation_id not in self._allocations:
                return False
            
            record = self._allocations[allocation_id]
            size_diff = new_size_gb - record.size_gb
            
            if size_diff > 0 and self.available_gb < size_diff:
                return False
            
            record.size_gb = new_size_gb
            record.update_access_time()
            self._allocated_gb += size_diff
            
            return True
    
    async def get_usage(self) -> MemoryUsage:
        """Get current pool usage statistics."""
        return MemoryUsage(
            total_gb=self._size_gb,
            used_gb=self._allocated_gb,
            available_gb=self.available_gb,
            reserved_gb=0.0,
            pool_usage={self._pool_type.value: self._allocated_gb},
            fragmentation_ratio=self._calculate_fragmentation()
        )
    
    async def compact(self) -> float:
        """Compact pool memory."""
        # For this implementation, we'll trigger garbage collection
        freed_before = self.available_gb
        gc.collect()
        freed_after = self.available_gb
        return max(0, freed_after - freed_before)
    
    async def _try_free_memory(self, needed_gb: float) -> float:
        """Try to free memory by removing old allocations."""
        if not self._allocations:
            return 0.0
        
        # Sort allocations by last access time (oldest first)
        sorted_allocations = sorted(
            self._allocations.values(),
            key=lambda x: x.last_accessed
        )
        
        freed_gb = 0.0
        allocations_to_remove = []
        
        for record in sorted_allocations:
            # Only free temporary allocations or old ones (> 1 hour)
            age_hours = (time.time() - record.last_accessed) / 3600
            if record.pool_type == MemoryPoolType.TEMPORARY or age_hours > 1.0:
                allocations_to_remove.append(record.allocation_id)
                freed_gb += record.size_gb
                
                if freed_gb >= needed_gb:
                    break
        
        # Remove selected allocations
        for allocation_id in allocations_to_remove:
            await self.deallocate(allocation_id)
        
        return freed_gb
    
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self._allocations:
            return 0.0
        
        # Simple fragmentation calculation based on number of allocations
        # Real implementation would analyze actual memory layout
        num_allocations = len(self._allocations)
        return min(1.0, num_allocations / 100.0)


class M3MaxMemoryManager(IMemoryManager):
    """M3 Max optimized unified memory manager."""
    
    def __init__(self, total_memory_gb: float = 128.0):
        self._total_memory_gb = total_memory_gb
        self._pools: Dict[MemoryPoolType, IMemoryPool] = {}
        self._pressure_callbacks: Dict[str, Callable[[MemoryUsage], None]] = {}
        self._callback_counter = 0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize default memory pools
        asyncio.create_task(self._initialize_default_pools())
    
    @property
    def total_memory_gb(self) -> float:
        return self._total_memory_gb
    
    @property
    def pools(self) -> Dict[MemoryPoolType, IMemoryPool]:
        return self._pools.copy()
    
    async def _initialize_default_pools(self):
        """Initialize default memory pools for M3 Max."""
        # Default pool allocation strategy for M3 Max
        pool_configs = {
            MemoryPoolType.MODELS: (40.0, AllocationStrategy.LAZY),      # Model storage
            MemoryPoolType.PROCESSING: (50.0, AllocationStrategy.ADAPTIVE),  # Document processing
            MemoryPoolType.CACHE: (30.0, AllocationStrategy.ADAPTIVE),       # Intelligent caching
            MemoryPoolType.SYSTEM: (8.0, AllocationStrategy.EAGER),          # System overhead
        }
        
        for pool_type, (size_gb, strategy) in pool_configs.items():
            pool = await self.create_pool(pool_type, size_gb, strategy)
            self._logger.info(f"Initialized {pool_type.value} pool with {size_gb}GB")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_memory_pressure())
    
    async def create_pool(self, 
                         pool_type: MemoryPoolType,
                         size_gb: float,
                         strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE) -> IMemoryPool:
        """Create memory pool with specified parameters."""
        async with self._lock:
            if pool_type in self._pools:
                raise ValueError(f"Pool {pool_type.value} already exists")
            
            pool = MemoryPool(pool_type, size_gb, strategy)
            self._pools[pool_type] = pool
            
            return pool
    
    async def allocate_from_pool(self,
                                pool_type: MemoryPoolType,
                                size_gb: float,
                                requester_id: str,
                                priority: int = 0) -> Optional[str]:
        """Allocate memory from specific pool."""
        if pool_type not in self._pools:
            self._logger.error(f"Pool {pool_type.value} not found")
            return None
        
        pool = self._pools[pool_type]
        allocation_id = await pool.allocate(size_gb, requester_id)
        
        if allocation_id:
            # Check for memory pressure after allocation
            await self._check_memory_pressure()
        
        return allocation_id
    
    async def smart_allocate(self,
                           size_gb: float,
                           requester_id: str,
                           usage_hint: str = "general",
                           priority: int = 0) -> Optional[str]:
        """Smart allocation across pools based on usage hint."""
        # Determine best pool based on usage hint
        pool_mapping = {
            "model": MemoryPoolType.MODELS,
            "processing": MemoryPoolType.PROCESSING,
            "cache": MemoryPoolType.CACHE,
            "temporary": MemoryPoolType.TEMPORARY,
            "general": MemoryPoolType.PROCESSING
        }
        
        preferred_pool = pool_mapping.get(usage_hint, MemoryPoolType.PROCESSING)
        
        # Try preferred pool first
        allocation_id = await self.allocate_from_pool(preferred_pool, size_gb, requester_id, priority)
        if allocation_id:
            return allocation_id
        
        # Try other pools in order of preference
        fallback_pools = [
            MemoryPoolType.PROCESSING,
            MemoryPoolType.CACHE,
            MemoryPoolType.MODELS,
            MemoryPoolType.SYSTEM
        ]
        
        for pool_type in fallback_pools:
            if pool_type == preferred_pool:
                continue
            
            allocation_id = await self.allocate_from_pool(pool_type, size_gb, requester_id, priority)
            if allocation_id:
                self._logger.info(f"Allocated {size_gb}GB from fallback pool {pool_type.value}")
                return allocation_id
        
        self._logger.error(f"Failed to allocate {size_gb}GB for {requester_id}")
        return None
    
    async def get_global_usage(self) -> MemoryUsage:
        """Get global memory usage statistics."""
        total_used = 0.0
        total_available = 0.0
        pool_usage = {}
        total_fragmentation = 0.0
        
        for pool_type, pool in self._pools.items():
            usage = await pool.get_usage()
            total_used += usage.used_gb
            total_available += usage.available_gb
            pool_usage[pool_type.value] = usage.used_gb
            total_fragmentation += usage.fragmentation_ratio
        
        # Add system memory information
        system_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return MemoryUsage(
            total_gb=self._total_memory_gb,
            used_gb=total_used,
            available_gb=total_available,
            reserved_gb=0.0,
            pool_usage=pool_usage,
            fragmentation_ratio=total_fragmentation / len(self._pools) if self._pools else 0.0,
            swap_usage_gb=swap_memory.used / (1024**3)
        )
    
    async def optimize_allocation(self) -> Dict[str, float]:
        """Optimize memory allocation across pools."""
        metrics = {}
        
        # Compact all pools
        for pool_type, pool in self._pools.items():
            freed_gb = await pool.compact()
            metrics[f"{pool_type.value}_compacted_gb"] = freed_gb
        
        # Force garbage collection
        gc_before = gc.get_count()
        gc.collect()
        gc_after = gc.get_count()
        
        metrics["gc_collected_objects"] = sum(gc_before) - sum(gc_after)
        metrics["optimization_timestamp"] = time.time()
        
        self._logger.info(f"Memory optimization completed: {metrics}")
        return metrics
    
    async def register_pressure_callback(self,
                                       callback: Callable[[MemoryUsage], None],
                                       threshold_percent: float = 90.0) -> str:
        """Register callback for memory pressure events."""
        self._callback_counter += 1
        callback_id = f"callback_{self._callback_counter}"
        
        # Store callback with threshold
        self._pressure_callbacks[callback_id] = {
            'callback': callback,
            'threshold': threshold_percent
        }
        
        return callback_id
    
    async def garbage_collect(self, force: bool = False) -> Dict[str, float]:
        """Trigger garbage collection."""
        before_usage = await self.get_global_usage()
        
        # Python garbage collection
        collected = gc.collect()
        
        # Optimize allocations if forced
        if force:
            await self.optimize_allocation()
        
        after_usage = await self.get_global_usage()
        
        freed_gb = before_usage.used_gb - after_usage.used_gb
        
        return {
            "freed_gb": max(0, freed_gb),
            "collected_objects": collected,
            "before_usage_gb": before_usage.used_gb,
            "after_usage_gb": after_usage.used_gb
        }
    
    async def _monitor_memory_pressure(self):
        """Monitor memory pressure and trigger callbacks."""
        while True:
            try:
                await self._check_memory_pressure()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in memory pressure monitoring: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _check_memory_pressure(self):
        """Check for memory pressure and trigger callbacks."""
        usage = await self.get_global_usage()
        
        for callback_info in self._pressure_callbacks.values():
            if usage.utilization_percent >= callback_info['threshold']:
                try:
                    callback_info['callback'](usage)
                except Exception as e:
                    self._logger.error(f"Error in memory pressure callback: {e}")
    
    def __del__(self):
        """Cleanup monitoring task on deletion."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()


class M3MaxResourceAllocator(IResourceAllocator):
    """M3 Max specific resource allocator with Apple Silicon optimization."""
    
    def __init__(self, memory_manager: IMemoryManager):
        self._memory_manager = memory_manager
        self._hardware_info = M3MaxResources()
        self._allocations: Dict[str, Dict[str, Any]] = {}
        self._allocation_counter = 0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Apple Silicon acceleration flags
        self._metal_enabled = False
        self._neural_engine_enabled = False
        self._amx_enabled = False
    
    @property
    def hardware_info(self) -> M3MaxResources:
        return self._hardware_info
    
    async def allocate_compute_resources(self,
                                       cpu_cores: int,
                                       memory_gb: float,
                                       gpu_memory_gb: float = 0.0,
                                       requester_id: str = None) -> Optional[str]:
        """Allocate comprehensive compute resources."""
        async with self._lock:
            # Validate resource availability
            if cpu_cores > (self._hardware_info.performance_cores + self._hardware_info.efficiency_cores):
                return None
            
            # Allocate memory
            memory_allocation = await self._memory_manager.smart_allocate(
                memory_gb, requester_id or "compute_resource", "processing"
            )
            
            if not memory_allocation:
                return None
            
            # Create resource allocation record
            self._allocation_counter += 1
            allocation_id = f"compute_{self._allocation_counter}_{int(time.time())}"
            
            self._allocations[allocation_id] = {
                'type': 'compute',
                'cpu_cores': cpu_cores,
                'memory_gb': memory_gb,
                'gpu_memory_gb': gpu_memory_gb,
                'memory_allocation': memory_allocation,
                'requester_id': requester_id,
                'created_at': time.time()
            }
            
            return allocation_id
    
    async def allocate_model_resources(self,
                                     model_size_gb: float,
                                     inference_memory_gb: float,
                                     model_id: str) -> Optional[str]:
        """Allocate resources optimized for model loading/inference."""
        total_memory = model_size_gb + inference_memory_gb
        
        # Allocate from models pool
        memory_allocation = await self._memory_manager.allocate_from_pool(
            MemoryPoolType.MODELS, total_memory, f"model_{model_id}"
        )
        
        if not memory_allocation:
            return None
        
        async with self._lock:
            self._allocation_counter += 1
            allocation_id = f"model_{self._allocation_counter}_{int(time.time())}"
            
            self._allocations[allocation_id] = {
                'type': 'model',
                'model_id': model_id,
                'model_size_gb': model_size_gb,
                'inference_memory_gb': inference_memory_gb,
                'memory_allocation': memory_allocation,
                'created_at': time.time(),
                'neural_engine_enabled': self._neural_engine_enabled,
                'metal_enabled': self._metal_enabled
            }
            
            return allocation_id
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get comprehensive resource usage statistics."""
        memory_usage = await self._memory_manager.get_global_usage()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory': {
                'total_gb': memory_usage.total_gb,
                'used_gb': memory_usage.used_gb,
                'utilization_percent': memory_usage.utilization_percent
            },
            'cpu': {
                'utilization_percent': cpu_percent,
                'performance_cores': self._hardware_info.performance_cores,
                'efficiency_cores': self._hardware_info.efficiency_cores
            },
            'gpu': {
                'cores': self._hardware_info.gpu_cores,
                'metal_enabled': self._metal_enabled
            },
            'neural_engine': {
                'tops': self._hardware_info.neural_engine_tops,
                'enabled': self._neural_engine_enabled
            },
            'active_allocations': len(self._allocations),
            'apple_silicon': {
                'amx_available': self._hardware_info.amx_coprocessor_available,
                'metal_shaders': self._hardware_info.metal_performance_shaders
            }
        }
    
    async def optimize_for_stage(self, stage_id: str, workload_type: str) -> ResourceLimits:
        """Optimize resource allocation for specific pipeline stage."""
        # Stage-specific optimization profiles
        stage_profiles = {
            'raw_input': ResourceLimits(
                max_memory_gb=20.0,
                max_cpu_cores=4,
                max_gpu_memory_gb=0.0,
                max_neural_engine_usage=0.0,
                max_concurrent_operations=8
            ),
            'document_conversion': ResourceLimits(
                max_memory_gb=30.0,
                max_cpu_cores=8,
                max_gpu_memory_gb=5.0,
                max_neural_engine_usage=0.2,
                max_concurrent_operations=4
            ),
            'langextract': ResourceLimits(
                max_memory_gb=50.0,
                max_cpu_cores=12,
                max_gpu_memory_gb=15.0,
                max_neural_engine_usage=0.8,
                max_concurrent_operations=3
            ),
            'conversation_generation': ResourceLimits(
                max_memory_gb=40.0,
                max_cpu_cores=10,
                max_gpu_memory_gb=10.0,
                max_neural_engine_usage=0.6,
                max_concurrent_operations=4
            )
        }
        
        return stage_profiles.get(stage_id.split('_')[1], ResourceLimits(
            max_memory_gb=25.0,
            max_cpu_cores=6,
            max_gpu_memory_gb=5.0,
            max_neural_engine_usage=0.3,
            max_concurrent_operations=5
        ))
    
    async def enable_apple_silicon_acceleration(self,
                                              enable_metal: bool = True,
                                              enable_neural_engine: bool = True,
                                              enable_amx: bool = True) -> bool:
        """Enable Apple Silicon specific accelerations."""
        self._metal_enabled = enable_metal and self._hardware_info.metal_performance_shaders
        self._neural_engine_enabled = enable_neural_engine
        self._amx_enabled = enable_amx and self._hardware_info.amx_coprocessor_available
        
        self._logger.info(
            f"Apple Silicon acceleration - Metal: {self._metal_enabled}, "
            f"Neural Engine: {self._neural_engine_enabled}, AMX: {self._amx_enabled}"
        )
        
        return True
    
    async def monitor_thermal_state(self) -> Dict[str, float]:
        """Monitor thermal state and throttling."""
        # Basic thermal monitoring (would need more sophisticated implementation)
        cpu_temp = 0.0
        gpu_temp = 0.0
        
        try:
            # On macOS, this would require additional tools or APIs
            sensors = psutil.sensors_temperatures()
            if sensors:
                for name, entries in sensors.items():
                    for entry in entries:
                        if 'cpu' in name.lower():
                            cpu_temp = max(cpu_temp, entry.current)
                        elif 'gpu' in name.lower():
                            gpu_temp = max(gpu_temp, entry.current)
        except Exception:
            # Thermal monitoring not available
            pass
        
        return {
            'cpu_temperature': cpu_temp,
            'gpu_temperature': gpu_temp,
            'throttling_detected': cpu_temp > 85.0 or gpu_temp > 90.0,
            'thermal_pressure': min(1.0, max(cpu_temp, gpu_temp) / 100.0)
        }