"""
Memory management interfaces for M3 Max optimization.
Defines contracts for unified memory management and resource allocation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import asyncio


class MemoryPoolType(Enum):
    """Types of memory pools."""
    MODELS = "models"
    PROCESSING = "processing"
    CACHE = "cache"
    SYSTEM = "system"
    TEMPORARY = "temporary"


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    EAGER = "eager"
    LAZY = "lazy" 
    ADAPTIVE = "adaptive"
    BALANCED = "balanced"


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    total_gb: float
    used_gb: float
    available_gb: float
    reserved_gb: float
    pool_usage: Dict[str, float]  # pool_name -> usage_gb
    fragmentation_ratio: float = 0.0
    swap_usage_gb: float = 0.0
    
    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        return (self.used_gb / self.total_gb) * 100.0 if self.total_gb > 0 else 0.0


@dataclass
class ResourceLimits:
    """Resource allocation limits."""
    max_memory_gb: float
    max_cpu_cores: int
    max_gpu_memory_gb: float
    max_neural_engine_usage: float
    max_concurrent_operations: int
    timeout_seconds: int = 3600


@dataclass
class M3MaxResources:
    """M3 Max specific resource information."""
    unified_memory_gb: float = 128.0
    performance_cores: int = 8
    efficiency_cores: int = 4
    gpu_cores: int = 40
    neural_engine_tops: float = 15.8
    amx_coprocessor_available: bool = True
    metal_performance_shaders: bool = True


class IMemoryPool(ABC):
    """Interface for memory pool management."""
    
    @property
    @abstractmethod
    def pool_type(self) -> MemoryPoolType:
        """Type of memory pool."""
        pass
    
    @property
    @abstractmethod
    def size_gb(self) -> float:
        """Total pool size in GB."""
        pass
    
    @property
    @abstractmethod
    def available_gb(self) -> float:
        """Available memory in pool."""
        pass
    
    @abstractmethod
    async def allocate(self, size_gb: float, requester_id: str) -> Optional[str]:
        """Allocate memory from pool. Returns allocation ID or None if failed."""
        pass
    
    @abstractmethod
    async def deallocate(self, allocation_id: str) -> bool:
        """Deallocate memory by allocation ID."""
        pass
    
    @abstractmethod
    async def resize_allocation(self, allocation_id: str, new_size_gb: float) -> bool:
        """Resize existing allocation."""
        pass
    
    @abstractmethod
    async def get_usage(self) -> MemoryUsage:
        """Get current pool usage statistics."""
        pass
    
    @abstractmethod
    async def compact(self) -> float:
        """Compact pool memory. Returns GB freed."""
        pass


class IMemoryManager(ABC):
    """Interface for unified memory management."""
    
    @property
    @abstractmethod
    def total_memory_gb(self) -> float:
        """Total unified memory available."""
        pass
    
    @property
    @abstractmethod
    def pools(self) -> Dict[MemoryPoolType, IMemoryPool]:
        """Dictionary of memory pools by type."""
        pass
    
    @abstractmethod
    async def create_pool(
        self,
        pool_type: MemoryPoolType,
        size_gb: float,
        strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE
    ) -> IMemoryPool:
        """Create memory pool with specified parameters."""
        pass
    
    @abstractmethod
    async def allocate_from_pool(
        self,
        pool_type: MemoryPoolType,
        size_gb: float,
        requester_id: str,
        priority: int = 0
    ) -> Optional[str]:
        """Allocate memory from specific pool."""
        pass
    
    @abstractmethod
    async def smart_allocate(
        self,
        size_gb: float,
        requester_id: str,
        usage_hint: str = "general",
        priority: int = 0
    ) -> Optional[str]:
        """Smart allocation across pools based on usage hint."""
        pass
    
    @abstractmethod
    async def get_global_usage(self) -> MemoryUsage:
        """Get global memory usage statistics."""
        pass
    
    @abstractmethod
    async def optimize_allocation(self) -> Dict[str, float]:
        """Optimize memory allocation. Returns metrics."""
        pass
    
    @abstractmethod
    async def register_pressure_callback(
        self,
        callback: Callable[[MemoryUsage], None],
        threshold_percent: float = 90.0
    ) -> str:
        """Register callback for memory pressure events."""
        pass
    
    @abstractmethod
    async def garbage_collect(self, force: bool = False) -> Dict[str, float]:
        """Trigger garbage collection. Returns freed memory stats."""
        pass


class IResourceAllocator(ABC):
    """Interface for comprehensive resource allocation."""
    
    @property
    @abstractmethod
    def hardware_info(self) -> M3MaxResources:
        """M3 Max hardware information."""
        pass
    
    @abstractmethod
    async def allocate_compute_resources(
        self,
        cpu_cores: int,
        memory_gb: float,
        gpu_memory_gb: float = 0.0,
        requester_id: str = None
    ) -> Optional[str]:
        """Allocate comprehensive compute resources."""
        pass
    
    @abstractmethod
    async def allocate_model_resources(
        self,
        model_size_gb: float,
        inference_memory_gb: float,
        model_id: str
    ) -> Optional[str]:
        """Allocate resources optimized for model loading/inference."""
        pass
    
    @abstractmethod
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get comprehensive resource usage statistics."""
        pass
    
    @abstractmethod
    async def optimize_for_stage(self, stage_id: str, workload_type: str) -> ResourceLimits:
        """Optimize resource allocation for specific pipeline stage."""
        pass
    
    @abstractmethod
    async def enable_apple_silicon_acceleration(
        self,
        enable_metal: bool = True,
        enable_neural_engine: bool = True,
        enable_amx: bool = True
    ) -> bool:
        """Enable Apple Silicon specific accelerations."""
        pass
    
    @abstractmethod
    async def monitor_thermal_state(self) -> Dict[str, float]:
        """Monitor thermal state and throttling."""
        pass


class IMemoryOptimizer(ABC):
    """Interface for advanced memory optimization."""
    
    @abstractmethod
    async def analyze_allocation_patterns(
        self,
        historical_data: List[MemoryUsage]
    ) -> Dict[str, Any]:
        """Analyze allocation patterns for optimization opportunities."""
        pass
    
    @abstractmethod
    async def predict_memory_requirements(
        self,
        workload_description: Dict[str, Any]
    ) -> float:
        """Predict memory requirements for workload."""
        pass
    
    @abstractmethod
    async def optimize_pool_sizes(
        self,
        usage_history: List[MemoryUsage]
    ) -> Dict[MemoryPoolType, float]:
        """Optimize pool sizes based on usage history."""
        pass
    
    @abstractmethod
    async def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        pass
    
    @abstractmethod
    async def create_memory_profile(
        self,
        operation_id: str,
        duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """Create detailed memory profile for analysis."""
        pass