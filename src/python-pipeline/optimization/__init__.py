"""
Python Pipeline Optimization Package
M3 Max hardware-optimized performance components for 4-5x improvement
"""

from .mlx_accelerator import MLXAccelerator, create_mlx_accelerator, M3MaxHardwareSpecs
from .m3_max_optimizer import M3MaxOptimizer, WorkloadSpec, WorkloadType, CoreType
from .parallel_processor import ParallelProcessor, ProcessingTask, ProcessingResult, create_parallel_processor
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker_manager, circuit_breaker
from .resource_manager import ResourceManager, ResourceRequest, ResourceAllocation, ResourceType, ResourcePriority

__version__ = "1.0.0"
__author__ = "Performance Optimization Specialist"

# Performance targets for M3 Max hardware
PERFORMANCE_TARGETS = {
    'throughput_docs_per_hour': 25.0,
    'memory_efficiency_percent': 90.0,
    'processing_speed_docs_per_sec': 0.6,
    'model_switching_latency_sec': 5.0,
    'error_rate_percent': 2.0,
    'cpu_utilization_percent': 85.0,
    'gpu_utilization_percent': 70.0
}

# Hardware specifications for M3 Max
M3_MAX_SPECS = {
    'performance_cores': 8,
    'efficiency_cores': 4,
    'gpu_cores': 40,
    'neural_engine_cores': 16,
    'neural_engine_tops': 15.8,
    'unified_memory_gb': 128,
    'memory_bandwidth_gbps': 400.0
}

__all__ = [
    # Main classes
    'MLXAccelerator',
    'M3MaxOptimizer', 
    'ParallelProcessor',
    'CircuitBreaker',
    'ResourceManager',
    
    # Factory functions
    'create_mlx_accelerator',
    'create_parallel_processor',
    
    # Configuration classes
    'M3MaxHardwareSpecs',
    'WorkloadSpec',
    'ProcessingTask',
    'ProcessingResult',
    'CircuitBreakerConfig',
    'ResourceRequest',
    'ResourceAllocation',
    
    # Enums
    'WorkloadType',
    'CoreType',
    'ResourceType',
    'ResourcePriority',
    
    # Global instances
    'circuit_breaker_manager',
    
    # Decorators
    'circuit_breaker',
    
    # Constants
    'PERFORMANCE_TARGETS',
    'M3_MAX_SPECS'
]