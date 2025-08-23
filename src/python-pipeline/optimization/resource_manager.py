"""
Dynamic Resource Manager for Python Pipeline Optimization
Intelligent allocation and management of M3 Max hardware resources
Coordinates CPU cores, GPU, Neural Engine, and unified memory
"""

import asyncio
import psutil
import threading
import time
import logging
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque, defaultdict
import os

class ResourceType(Enum):
    """Types of resources to manage"""
    CPU_PERFORMANCE = "cpu_performance"  # 8 P-cores
    CPU_EFFICIENCY = "cpu_efficiency"    # 4 E-cores
    GPU = "gpu"                          # 40-core GPU
    NEURAL_ENGINE = "neural_engine"      # 16-core Neural Engine
    MEMORY = "memory"                    # 128GB unified memory
    STORAGE_IO = "storage_io"            # NVMe SSD I/O

class ResourcePriority(Enum):
    """Resource allocation priorities"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ResourceRequest:
    """Resource allocation request"""
    id: str
    resource_type: ResourceType
    amount: float  # Amount of resource (cores, GB, percentage)
    priority: ResourcePriority
    estimated_duration_sec: float
    exclusive: bool = False  # Whether resource needs exclusive access
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class ResourceAllocation:
    """Allocated resource information"""
    request_id: str
    resource_type: ResourceType
    allocated_amount: float
    actual_amount: float  # Actual amount being used
    allocated_at: float
    estimated_release_at: float
    process_id: Optional[int] = None
    thread_id: Optional[int] = None

@dataclass
class ResourcePool:
    """Resource pool configuration and state"""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocated_capacity: float
    reserved_capacity: float  # Reserved for system/critical operations
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    utilization_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    peak_usage: float = 0.0

class ResourceManager:
    """
    Dynamic resource manager for M3 Max hardware optimization
    Provides intelligent allocation, monitoring, and optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Resource pools initialization
        self.resource_pools = self._initialize_resource_pools()
        
        # Allocation tracking
        self.active_allocations = {}
        self.allocation_history = deque(maxlen=10000)
        self.resource_locks = {rt: threading.RLock() for rt in ResourceType}
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'avg_allocation_time_ms': 0.0,
            'resource_efficiency': 0.0,
            'memory_pressure': 0.0,
            'cpu_saturation': 0.0,
            'gpu_utilization': 0.0
        }
        
        # Dynamic optimization
        self.optimization_enabled = True
        self.last_optimization = time.time()
        self.optimization_interval = 30.0  # seconds
        
        # Monitoring thread
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Start background monitoring
        self._start_monitoring()
        
        self.logger.info("ResourceManager initialized for M3 Max hardware")
    
    def _initialize_resource_pools(self) -> Dict[ResourceType, ResourcePool]:
        """Initialize resource pools based on M3 Max hardware specifications"""
        
        # Get system information
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024 ** 3)
        
        pools = {
            ResourceType.CPU_PERFORMANCE: ResourcePool(
                resource_type=ResourceType.CPU_PERFORMANCE,
                total_capacity=8.0,  # 8 P-cores
                available_capacity=7.0,  # Reserve 1 for system
                allocated_capacity=0.0,
                reserved_capacity=1.0
            ),
            
            ResourceType.CPU_EFFICIENCY: ResourcePool(
                resource_type=ResourceType.CPU_EFFICIENCY,
                total_capacity=4.0,  # 4 E-cores
                available_capacity=3.0,  # Reserve 1 for system
                allocated_capacity=0.0,
                reserved_capacity=1.0
            ),
            
            ResourceType.GPU: ResourcePool(
                resource_type=ResourceType.GPU,
                total_capacity=40.0,  # 40 GPU cores
                available_capacity=36.0,  # Reserve 10% for system
                allocated_capacity=0.0,
                reserved_capacity=4.0
            ),
            
            ResourceType.NEURAL_ENGINE: ResourcePool(
                resource_type=ResourceType.NEURAL_ENGINE,
                total_capacity=16.0,  # 16 Neural Engine cores
                available_capacity=14.0,  # Reserve 2 for system
                allocated_capacity=0.0,
                reserved_capacity=2.0
            ),
            
            ResourceType.MEMORY: ResourcePool(
                resource_type=ResourceType.MEMORY,
                total_capacity=total_memory_gb,
                available_capacity=total_memory_gb * 0.85,  # Reserve 15% for system
                allocated_capacity=0.0,
                reserved_capacity=total_memory_gb * 0.15
            ),
            
            ResourceType.STORAGE_IO: ResourcePool(
                resource_type=ResourceType.STORAGE_IO,
                total_capacity=100.0,  # 100% I/O bandwidth
                available_capacity=90.0,  # Reserve 10% for system
                allocated_capacity=0.0,
                reserved_capacity=10.0
            )
        }
        
        self.logger.info(f"Initialized resource pools: {[(rt.value, f'{pool.available_capacity:.1f}/{pool.total_capacity:.1f}') for rt, pool in pools.items()]}")
        return pools
    
    async def request_resource(
        self,
        request: ResourceRequest,
        timeout_seconds: float = 30.0
    ) -> ResourceAllocation:
        """
        Request resource allocation with intelligent scheduling
        Returns allocation or raises exception if unavailable
        """
        
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Validate request
            self._validate_resource_request(request)
            
            # Try immediate allocation
            allocation = self._try_immediate_allocation(request)
            if allocation:
                allocation_time = time.time() - start_time
                self._update_allocation_metrics(allocation_time, success=True)
                return allocation
            
            # Queue for future allocation if immediate allocation fails
            allocation = await self._queue_resource_request(request, timeout_seconds, start_time)
            return allocation
            
        except Exception as e:
            self.performance_metrics['failed_allocations'] += 1
            allocation_time = time.time() - start_time
            self._update_allocation_metrics(allocation_time, success=False)
            self.logger.error(f"Resource allocation failed for {request.id}: {e}")
            raise
    
    def _validate_resource_request(self, request: ResourceRequest):
        """Validate resource request parameters"""
        
        pool = self.resource_pools[request.resource_type]
        
        # Check if request amount is reasonable
        if request.amount <= 0:
            raise ValueError(f"Invalid resource amount: {request.amount}")
        
        if request.amount > pool.total_capacity:
            raise ValueError(f"Requested {request.amount} exceeds total capacity {pool.total_capacity}")
        
        # Check if exclusive request is possible
        if request.exclusive and request.amount < pool.total_capacity:
            # Exclusive requests must use full capacity or be smaller
            pass
    
    def _try_immediate_allocation(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Try to allocate resource immediately"""
        
        with self.resource_locks[request.resource_type]:
            pool = self.resource_pools[request.resource_type]
            
            # Check if sufficient resources are available
            if pool.available_capacity >= request.amount:
                # Create allocation
                allocation = ResourceAllocation(
                    request_id=request.id,
                    resource_type=request.resource_type,
                    allocated_amount=request.amount,
                    actual_amount=0.0,  # Will be updated during monitoring
                    allocated_at=time.time(),
                    estimated_release_at=time.time() + request.estimated_duration_sec,
                    process_id=os.getpid()
                )
                
                # Update pool state
                pool.allocated_capacity += request.amount
                pool.available_capacity -= request.amount
                pool.allocations[request.id] = allocation
                
                # Track allocation
                self.active_allocations[request.id] = allocation
                self.allocation_history.append({
                    'request_id': request.id,
                    'resource_type': request.resource_type.value,
                    'amount': request.amount,
                    'priority': request.priority.value,
                    'allocated_at': allocation.allocated_at,
                    'immediate': True
                })
                
                # Apply resource constraints
                self._apply_resource_constraints(allocation)
                
                self.logger.debug(f"Immediate allocation successful for {request.id}: {request.amount} {request.resource_type.value}")
                return allocation
        
        return None
    
    async def _queue_resource_request(
        self,
        request: ResourceRequest,
        timeout_seconds: float,
        start_time: float
    ) -> ResourceAllocation:
        """Queue resource request for future allocation"""
        
        end_time = start_time + timeout_seconds
        check_interval = 0.5  # Check every 500ms
        
        while time.time() < end_time:
            # Try allocation again
            allocation = self._try_immediate_allocation(request)
            if allocation:
                allocation_time = time.time() - start_time
                self._update_allocation_metrics(allocation_time, success=True)
                return allocation
            
            # Check if optimization can free up resources
            if self.optimization_enabled:
                await self._optimize_resource_allocation(request.resource_type)
            
            # Wait before trying again
            await asyncio.sleep(check_interval)
        
        # Timeout reached
        raise ResourceAllocationTimeoutException(f"Resource allocation timeout for {request.id}")
    
    def _apply_resource_constraints(self, allocation: ResourceAllocation):
        """Apply system-level resource constraints for allocation"""
        
        try:
            if allocation.resource_type == ResourceType.CPU_PERFORMANCE:
                # Set CPU affinity to P-cores (0-7)
                self._set_cpu_affinity(list(range(8)))
                
            elif allocation.resource_type == ResourceType.CPU_EFFICIENCY:
                # Set CPU affinity to E-cores (8-11)
                self._set_cpu_affinity(list(range(8, 12)))
                
            elif allocation.resource_type == ResourceType.MEMORY:
                # Set memory limits if supported
                self._set_memory_limit(allocation.allocated_amount)
                
            elif allocation.resource_type == ResourceType.GPU:
                # Configure GPU for optimal usage
                self._configure_gpu_allocation(allocation)
                
            elif allocation.resource_type == ResourceType.NEURAL_ENGINE:
                # Configure Neural Engine
                self._configure_neural_engine(allocation)
                
        except Exception as e:
            self.logger.debug(f"Could not apply resource constraints: {e}")
    
    def _set_cpu_affinity(self, cpu_list: List[int]):
        """Set CPU affinity for current process"""
        try:
            process = psutil.Process()
            if hasattr(process, 'cpu_affinity'):
                process.cpu_affinity(cpu_list)
        except Exception as e:
            self.logger.debug(f"Could not set CPU affinity: {e}")
    
    def _set_memory_limit(self, limit_gb: float):
        """Set memory limit for process"""
        try:
            import resource
            limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception as e:
            self.logger.debug(f"Could not set memory limit: {e}")
    
    def _configure_gpu_allocation(self, allocation: ResourceAllocation):
        """Configure GPU allocation settings"""
        # Set environment variables for MLX GPU optimization
        os.environ['MLX_GPU_MEMORY_FRACTION'] = str(allocation.allocated_amount / 40.0)
        os.environ['MLX_GPU_CORES'] = str(int(allocation.allocated_amount))
    
    def _configure_neural_engine(self, allocation: ResourceAllocation):
        """Configure Neural Engine allocation"""
        # Set Neural Engine utilization parameters
        os.environ['MLX_NEURAL_ENGINE_CORES'] = str(int(allocation.allocated_amount))
        os.environ['MLX_NEURAL_ENGINE_MEMORY_FRACTION'] = str(allocation.allocated_amount / 16.0)
    
    async def release_resource(self, allocation_id: str) -> bool:
        """Release allocated resource"""
        
        if allocation_id not in self.active_allocations:
            self.logger.warning(f"Allocation {allocation_id} not found for release")
            return False
        
        allocation = self.active_allocations[allocation_id]
        
        with self.resource_locks[allocation.resource_type]:
            pool = self.resource_pools[allocation.resource_type]
            
            # Update pool state
            pool.allocated_capacity -= allocation.allocated_amount
            pool.available_capacity += allocation.allocated_amount
            
            # Remove from tracking
            del pool.allocations[allocation_id]
            del self.active_allocations[allocation_id]
            
            # Record release in history
            self.allocation_history.append({
                'request_id': allocation_id,
                'resource_type': allocation.resource_type.value,
                'amount': allocation.allocated_amount,
                'released_at': time.time(),
                'duration': time.time() - allocation.allocated_at,
                'action': 'release'
            })
        
        # Store release metrics
        await self._store_resource_metrics(allocation, 'released')
        
        self.logger.debug(f"Released resource allocation {allocation_id}")
        return True
    
    async def _optimize_resource_allocation(self, resource_type: ResourceType):
        """Optimize resource allocation for specific resource type"""
        
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        self.last_optimization = current_time
        
        with self.resource_locks[resource_type]:
            pool = self.resource_pools[resource_type]
            
            # Find expired or low-priority allocations
            expired_allocations = []
            for alloc_id, allocation in pool.allocations.items():
                if current_time > allocation.estimated_release_at:
                    expired_allocations.append(alloc_id)
            
            # Release expired allocations
            for alloc_id in expired_allocations:
                await self.release_resource(alloc_id)
                self.logger.debug(f"Released expired allocation {alloc_id}")
            
            # Consolidate fragmented resources if needed
            await self._consolidate_resources(resource_type)
    
    async def _consolidate_resources(self, resource_type: ResourceType):
        """Consolidate fragmented resource allocations"""
        
        pool = self.resource_pools[resource_type]
        
        # Check if consolidation would be beneficial
        if len(pool.allocations) > 5 and pool.available_capacity < pool.total_capacity * 0.3:
            self.logger.debug(f"Considering consolidation for {resource_type.value}")
            
            # This is a placeholder for more sophisticated consolidation logic
            # In production, this might involve:
            # - Moving processes to different cores
            # - Defragmenting memory allocations
            # - Rebalancing GPU workloads
    
    def _start_monitoring(self):
        """Start background resource monitoring"""
        
        def monitoring_loop():
            while not self.shutdown_event.is_set():
                try:
                    # Update resource utilization
                    self._update_resource_utilization()
                    
                    # Check for optimization opportunities
                    asyncio.create_task(self._periodic_optimization())
                    
                    # Store performance metrics
                    asyncio.create_task(self._store_performance_metrics())
                    
                    # Clean up old history
                    self._cleanup_old_data()
                    
                    # Wait for next monitoring cycle
                    self.shutdown_event.wait(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    self.shutdown_event.wait(30)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started resource monitoring thread")
    
    def _update_resource_utilization(self):
        """Update current resource utilization metrics"""
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        if len(cpu_percent) >= 12:  # M3 Max has 12 cores
            p_core_avg = sum(cpu_percent[:8]) / 8.0 if len(cpu_percent) >= 8 else 0
            e_core_avg = sum(cpu_percent[8:12]) / 4.0 if len(cpu_percent) >= 12 else 0
            
            self.resource_pools[ResourceType.CPU_PERFORMANCE].utilization_history.append(p_core_avg)
            self.resource_pools[ResourceType.CPU_EFFICIENCY].utilization_history.append(e_core_avg)
            
            self.performance_metrics['cpu_saturation'] = max(p_core_avg, e_core_avg) / 100.0
        
        # Memory utilization
        memory = psutil.virtual_memory()
        memory_utilization = memory.percent
        self.resource_pools[ResourceType.MEMORY].utilization_history.append(memory_utilization)
        self.performance_metrics['memory_pressure'] = memory_utilization / 100.0
        
        # Update peak usage tracking
        for pool in self.resource_pools.values():
            current_usage = pool.allocated_capacity
            pool.peak_usage = max(pool.peak_usage, current_usage)
        
        # Calculate resource efficiency
        total_allocated = sum(pool.allocated_capacity for pool in self.resource_pools.values())
        total_capacity = sum(pool.total_capacity for pool in self.resource_pools.values())
        self.performance_metrics['resource_efficiency'] = total_allocated / max(1, total_capacity)
    
    async def _periodic_optimization(self):
        """Perform periodic resource optimization"""
        
        if not self.optimization_enabled:
            return
        
        # Optimize each resource type
        for resource_type in ResourceType:
            await self._optimize_resource_allocation(resource_type)
        
        # Global optimization
        await self._global_resource_optimization()
    
    async def _global_resource_optimization(self):
        """Perform global resource optimization across all types"""
        
        # Check for resource imbalances
        utilization_ratios = {}
        for rt, pool in self.resource_pools.items():
            if pool.total_capacity > 0:
                utilization_ratios[rt] = pool.allocated_capacity / pool.total_capacity
        
        # Find highly utilized resources
        high_util_resources = [rt for rt, ratio in utilization_ratios.items() if ratio > 0.8]
        low_util_resources = [rt for rt, ratio in utilization_ratios.items() if ratio < 0.3]
        
        if high_util_resources and low_util_resources:
            self.logger.debug(f"Resource imbalance detected: high={high_util_resources}, low={low_util_resources}")
            # Here we could implement cross-resource optimization strategies
    
    def _cleanup_old_data(self):
        """Clean up old historical data to prevent memory leaks"""
        
        # Keep last 24 hours of allocation history
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours ago
        
        # Clean allocation history
        while (self.allocation_history and 
               len(self.allocation_history) > 0 and
               self.allocation_history[0].get('allocated_at', current_time) < cutoff_time):
            self.allocation_history.popleft()
    
    def _update_allocation_metrics(self, allocation_time: float, success: bool):
        """Update allocation performance metrics"""
        
        if success:
            self.performance_metrics['successful_allocations'] += 1
        
        # Update average allocation time
        current_avg = self.performance_metrics['avg_allocation_time_ms']
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests > 1:
            new_avg = ((current_avg * (total_requests - 1)) + (allocation_time * 1000)) / total_requests
            self.performance_metrics['avg_allocation_time_ms'] = new_avg
        else:
            self.performance_metrics['avg_allocation_time_ms'] = allocation_time * 1000
    
    async def _store_resource_metrics(self, allocation: ResourceAllocation, action: str):
        """Store resource metrics in coordination memory"""
        try:
            metrics_data = {
                'allocation_id': allocation.request_id,
                'resource_type': allocation.resource_type.value,
                'allocated_amount': allocation.allocated_amount,
                'actual_amount': allocation.actual_amount,
                'action': action,
                'timestamp': time.time(),
                'duration': time.time() - allocation.allocated_at if action == 'released' else 0
            }
            
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"python-pipeline/performance/resource-{allocation.request_id}",
                "--data", json.dumps(metrics_data)
            ], capture_output=True, text=True, check=True)
            
        except Exception as e:
            self.logger.debug(f"Failed to store resource metrics: {e}")
    
    async def _store_performance_metrics(self):
        """Store overall performance metrics"""
        try:
            metrics_data = {
                'resource_manager_performance': self.performance_metrics.copy(),
                'resource_pools': {
                    rt.value: {
                        'total_capacity': pool.total_capacity,
                        'available_capacity': pool.available_capacity,
                        'allocated_capacity': pool.allocated_capacity,
                        'utilization_percent': (pool.allocated_capacity / pool.total_capacity) * 100,
                        'peak_usage': pool.peak_usage,
                        'active_allocations': len(pool.allocations)
                    }
                    for rt, pool in self.resource_pools.items()
                },
                'timestamp': time.time()
            }
            
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", "python-pipeline/performance/resource-manager",
                "--data", json.dumps(metrics_data)
            ], capture_output=True, text=True, check=True)
            
        except Exception as e:
            self.logger.debug(f"Failed to store performance metrics: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status report"""
        
        return {
            'resource_manager': {
                'performance_metrics': self.performance_metrics.copy(),
                'active_allocations': len(self.active_allocations),
                'resource_pools': {
                    rt.value: {
                        'total_capacity': pool.total_capacity,
                        'available_capacity': pool.available_capacity,
                        'allocated_capacity': pool.allocated_capacity,
                        'utilization_percent': (pool.allocated_capacity / pool.total_capacity) * 100,
                        'peak_usage': pool.peak_usage,
                        'active_allocations': len(pool.allocations),
                        'avg_utilization': statistics.mean(pool.utilization_history) if pool.utilization_history else 0.0
                    }
                    for rt, pool in self.resource_pools.items()
                },
                'optimization_enabled': self.optimization_enabled,
                'last_optimization': self.last_optimization
            }
        }
    
    async def shutdown(self):
        """Shutdown resource manager and clean up"""
        
        self.logger.info("Shutting down ResourceManager...")
        
        # Stop monitoring
        self.shutdown_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        # Release all active allocations
        allocation_ids = list(self.active_allocations.keys())
        for allocation_id in allocation_ids:
            await self.release_resource(allocation_id)
        
        # Store final metrics
        await self._store_performance_metrics()
        
        self.logger.info("ResourceManager shutdown completed")

# Custom exceptions
class ResourceAllocationTimeoutException(Exception):
    """Raised when resource allocation times out"""
    pass

class InsufficientResourcesException(Exception):
    """Raised when insufficient resources are available"""
    pass

# Utility functions
def create_resource_request(
    request_id: str,
    resource_type: ResourceType,
    amount: float,
    priority: ResourcePriority = ResourcePriority.MEDIUM,
    estimated_duration: float = 300.0,
    exclusive: bool = False,
    **metadata
) -> ResourceRequest:
    """Helper function to create resource request"""
    
    return ResourceRequest(
        id=request_id,
        resource_type=resource_type,
        amount=amount,
        priority=priority,
        estimated_duration_sec=estimated_duration,
        exclusive=exclusive,
        metadata=metadata
    )

async def allocate_processing_resources(
    task_id: str,
    cpu_cores: float = 2.0,
    memory_gb: float = 8.0,
    gpu_cores: float = 4.0,
    duration_sec: float = 300.0,
    resource_manager: ResourceManager = None
) -> Dict[str, ResourceAllocation]:
    """Allocate resources for document processing task"""
    
    if not resource_manager:
        raise ValueError("ResourceManager instance required")
    
    allocations = {}
    
    # Request CPU cores
    if cpu_cores > 0:
        cpu_request = create_resource_request(
            f"{task_id}_cpu",
            ResourceType.CPU_PERFORMANCE,
            cpu_cores,
            ResourcePriority.HIGH,
            duration_sec
        )
        allocations['cpu'] = await resource_manager.request_resource(cpu_request)
    
    # Request memory
    if memory_gb > 0:
        memory_request = create_resource_request(
            f"{task_id}_memory",
            ResourceType.MEMORY,
            memory_gb,
            ResourcePriority.HIGH,
            duration_sec
        )
        allocations['memory'] = await resource_manager.request_resource(memory_request)
    
    # Request GPU cores
    if gpu_cores > 0:
        gpu_request = create_resource_request(
            f"{task_id}_gpu",
            ResourceType.GPU,
            gpu_cores,
            ResourcePriority.MEDIUM,
            duration_sec
        )
        allocations['gpu'] = await resource_manager.request_resource(gpu_request)
    
    return allocations