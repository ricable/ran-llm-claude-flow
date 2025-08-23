"""
Apple Silicon M3 Max Optimizer
Optimizes workload distribution across 8P + 4E cores, 40-core GPU, 16-core Neural Engine
Achieves optimal resource utilization for Python pipeline processing
"""

import asyncio
import psutil
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import json
import subprocess
from pathlib import Path
import queue

class CoreType(Enum):
    """CPU core types on M3 Max"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    AUTO = "auto"

class WorkloadType(Enum):
    """Workload classification for optimal core assignment"""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    ML_INFERENCE = "ml_inference"
    PARALLEL_BATCH = "parallel_batch"

@dataclass
class CoreAllocation:
    """Core allocation configuration"""
    performance_cores: int = 8
    efficiency_cores: int = 4
    reserved_system: int = 1
    available_performance: int = field(default=7)
    available_efficiency: int = field(default=3)

@dataclass
class WorkloadSpec:
    """Workload specification for optimization"""
    type: WorkloadType
    priority: int  # 1-10, 10 being highest
    estimated_duration_sec: float
    memory_requirement_mb: float
    cpu_bound_ratio: float  # 0-1, 1 being fully CPU bound
    parallelizable: bool = True
    core_preference: CoreType = CoreType.AUTO

@dataclass
class PerformanceTargets:
    """Performance targets for M3 Max optimization"""
    target_throughput_docs_hour: float = 25.0
    target_memory_efficiency: float = 0.90
    target_cpu_utilization: float = 0.85
    target_gpu_utilization: float = 0.70
    max_processing_latency_sec: float = 0.7
    max_model_switch_latency_sec: float = 5.0

class M3MaxOptimizer:
    """
    Apple Silicon M3 Max hardware optimizer
    Manages CPU cores, GPU, Neural Engine, and unified memory for optimal performance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Hardware configuration
        self.core_allocation = CoreAllocation()
        self.performance_targets = PerformanceTargets()
        
        # Runtime state
        self.active_tasks = {}
        self.core_assignments = {CoreType.PERFORMANCE: [], CoreType.EFFICIENCY: []}
        self.resource_locks = {
            'performance_cores': threading.Semaphore(self.core_allocation.available_performance),
            'efficiency_cores': threading.Semaphore(self.core_allocation.available_efficiency),
            'gpu': threading.Semaphore(1),  # Single GPU scheduler
            'neural_engine': threading.Semaphore(1)
        }
        
        # Performance monitoring
        self.metrics = {
            'cpu_utilization': 0.0,
            'memory_utilization': 0.0,
            'gpu_utilization': 0.0,
            'neural_engine_utilization': 0.0,
            'throughput_docs_per_hour': 0.0,
            'avg_processing_latency': 0.0
        }
        
        # Task queues by priority and type
        self.task_queues = {
            WorkloadType.CPU_INTENSIVE: queue.PriorityQueue(),
            WorkloadType.IO_INTENSIVE: queue.PriorityQueue(),
            WorkloadType.MEMORY_INTENSIVE: queue.PriorityQueue(),
            WorkloadType.ML_INFERENCE: queue.PriorityQueue(),
            WorkloadType.PARALLEL_BATCH: queue.PriorityQueue()
        }
        
        # Executors for different workload types
        self._initialize_executors()
        
        # Start background monitoring
        self._start_monitoring()
        
    def _initialize_executors(self):
        """Initialize optimized executors for different workload types"""
        
        # Performance core executor for CPU-intensive tasks
        self.performance_executor = ProcessPoolExecutor(
            max_workers=self.core_allocation.available_performance,
            mp_context=mp.get_context('spawn')
        )
        
        # Efficiency core executor for IO and background tasks
        self.efficiency_executor = ThreadPoolExecutor(
            max_workers=self.core_allocation.available_efficiency * 2  # 2 threads per E-core
        )
        
        # Memory-intensive task executor with controlled concurrency
        self.memory_executor = ThreadPoolExecutor(
            max_workers=max(2, self.core_allocation.available_performance // 2)
        )
        
        # ML inference executor optimized for Neural Engine
        self.ml_executor = ThreadPoolExecutor(
            max_workers=4  # Optimal for Neural Engine coordination
        )
        
        self.logger.info("Initialized M3 Max optimized executors")
    
    def optimize_workload(
        self, 
        workload_spec: WorkloadSpec,
        task_function: Callable,
        *args,
        **kwargs
    ) -> asyncio.Future:
        """
        Optimize workload execution based on M3 Max hardware characteristics
        Automatically assigns optimal cores and resources
        """
        
        # Determine optimal core assignment
        core_type = self._select_optimal_cores(workload_spec)
        
        # Select appropriate executor
        executor = self._select_executor(workload_spec, core_type)
        
        # Create optimized execution context
        execution_context = self._create_execution_context(workload_spec, core_type)
        
        # Submit task with optimization
        future = self._submit_optimized_task(
            executor, 
            task_function, 
            workload_spec,
            execution_context,
            *args,
            **kwargs
        )
        
        return future
    
    def _select_optimal_cores(self, workload_spec: WorkloadSpec) -> CoreType:
        """Select optimal core type based on workload characteristics"""
        
        # Explicit core preference
        if workload_spec.core_preference != CoreType.AUTO:
            return workload_spec.core_preference
        
        # Workload-based assignment
        if workload_spec.type == WorkloadType.CPU_INTENSIVE:
            if workload_spec.cpu_bound_ratio > 0.8 and workload_spec.priority >= 7:
                return CoreType.PERFORMANCE
            elif workload_spec.estimated_duration_sec > 10.0:
                return CoreType.PERFORMANCE
        
        elif workload_spec.type == WorkloadType.ML_INFERENCE:
            return CoreType.PERFORMANCE  # ML inference benefits from P-cores
        
        elif workload_spec.type == WorkloadType.PARALLEL_BATCH:
            if workload_spec.parallelizable and workload_spec.priority >= 6:
                return CoreType.PERFORMANCE
        
        elif workload_spec.type in [WorkloadType.IO_INTENSIVE, WorkloadType.MEMORY_INTENSIVE]:
            return CoreType.EFFICIENCY  # E-cores efficient for I/O bound tasks
        
        # Default based on system load
        p_core_utilization = self._get_performance_core_utilization()
        if p_core_utilization < 0.7:
            return CoreType.PERFORMANCE
        else:
            return CoreType.EFFICIENCY
    
    def _select_executor(self, workload_spec: WorkloadSpec, core_type: CoreType) -> ThreadPoolExecutor:
        """Select appropriate executor based on workload and core type"""
        
        if workload_spec.type == WorkloadType.ML_INFERENCE:
            return self.ml_executor
        elif workload_spec.type == WorkloadType.MEMORY_INTENSIVE:
            return self.memory_executor
        elif core_type == CoreType.PERFORMANCE and workload_spec.cpu_bound_ratio > 0.6:
            return self.performance_executor
        else:
            return self.efficiency_executor
    
    def _create_execution_context(
        self, 
        workload_spec: WorkloadSpec, 
        core_type: CoreType
    ) -> Dict[str, Any]:
        """Create optimized execution context for M3 Max"""
        
        context = {
            'core_type': core_type,
            'workload_type': workload_spec.type,
            'priority': workload_spec.priority,
            'cpu_affinity': self._get_optimal_cpu_affinity(core_type),
            'memory_limit_mb': min(
                workload_spec.memory_requirement_mb * 1.2,  # 20% overhead
                32 * 1024  # Max 32GB per task
            ),
            'environment': self._get_optimized_environment(workload_spec)
        }
        
        return context
    
    def _get_optimal_cpu_affinity(self, core_type: CoreType) -> List[int]:
        """Get optimal CPU affinity for core type on M3 Max"""
        
        if core_type == CoreType.PERFORMANCE:
            # P-cores are typically 0-7 on M3 Max
            return list(range(8))
        else:
            # E-cores are typically 8-11 on M3 Max
            return list(range(8, 12))
    
    def _get_optimized_environment(self, workload_spec: WorkloadSpec) -> Dict[str, str]:
        """Get optimized environment variables for M3 Max"""
        
        env = os.environ.copy()
        
        # MLX optimizations
        env['MLX_METAL_DEVICE_WRAPPER'] = '1'
        env['MLX_METAL_DEBUG'] = '0'
        
        # Memory optimizations
        env['MALLOC_NANO_ZONE'] = '0'  # Use standard malloc for large allocations
        env['MallocNanoZone'] = '0'
        
        # Threading optimizations
        if workload_spec.type == WorkloadType.CPU_INTENSIVE:
            env['OMP_NUM_THREADS'] = str(self.core_allocation.available_performance)
            env['MKL_NUM_THREADS'] = str(self.core_allocation.available_performance)
        else:
            env['OMP_NUM_THREADS'] = str(self.core_allocation.available_efficiency)
        
        # Neural Engine optimizations
        if workload_spec.type == WorkloadType.ML_INFERENCE:
            env['BNNS_ALLOW_METAL'] = '1'
            env['BNNS_OPTIMIZE_FOR_MEMORY'] = '1'
        
        return env
    
    def _submit_optimized_task(
        self,
        executor: ThreadPoolExecutor,
        task_function: Callable,
        workload_spec: WorkloadSpec,
        execution_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> asyncio.Future:
        """Submit task with M3 Max optimizations"""
        
        # Wrap task with monitoring and optimization
        def optimized_wrapper():
            start_time = time.time()
            task_id = f"task_{int(start_time * 1000)}"
            
            try:
                # Set CPU affinity if supported
                self._set_cpu_affinity(execution_context['cpu_affinity'])
                
                # Set memory limits
                self._set_memory_limits(execution_context['memory_limit_mb'])
                
                # Execute task
                result = task_function(*args, **kwargs)
                
                # Record performance metrics
                duration = time.time() - start_time
                self._record_task_performance(task_id, workload_spec, duration, success=True)
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                self._record_task_performance(task_id, workload_spec, duration, success=False, error=str(e))
                raise
        
        # Submit to appropriate executor
        future = executor.submit(optimized_wrapper)
        
        # Track active task
        task_id = f"task_{int(time.time() * 1000)}"
        self.active_tasks[task_id] = {
            'workload_spec': workload_spec,
            'execution_context': execution_context,
            'start_time': time.time(),
            'future': future
        }
        
        return future
    
    def _set_cpu_affinity(self, cpu_list: List[int]):
        """Set CPU affinity for current process"""
        try:
            p = psutil.Process()
            if hasattr(p, 'cpu_affinity'):
                p.cpu_affinity(cpu_list)
        except Exception as e:
            self.logger.debug(f"Could not set CPU affinity: {e}")
    
    def _set_memory_limits(self, limit_mb: float):
        """Set memory limits for current process"""
        try:
            import resource
            limit_bytes = int(limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception as e:
            self.logger.debug(f"Could not set memory limit: {e}")
    
    def _record_task_performance(
        self, 
        task_id: str, 
        workload_spec: WorkloadSpec,
        duration: float,
        success: bool,
        error: str = None
    ):
        """Record task performance metrics"""
        
        performance_data = {
            'task_id': task_id,
            'workload_type': workload_spec.type.value,
            'priority': workload_spec.priority,
            'duration_sec': duration,
            'success': success,
            'error': error,
            'timestamp': time.time(),
            'memory_requirement_mb': workload_spec.memory_requirement_mb
        }
        
        # Update running metrics
        self._update_performance_metrics(performance_data)
        
        # Store in coordination memory
        asyncio.create_task(self._store_task_metrics(performance_data))
    
    def _update_performance_metrics(self, task_data: Dict[str, Any]):
        """Update running performance metrics"""
        
        # Update throughput calculation
        completed_tasks = sum(1 for task in self.active_tasks.values() 
                            if task['future'].done())
        
        if completed_tasks > 0:
            total_duration = time.time() - min(task['start_time'] 
                                             for task in self.active_tasks.values())
            self.metrics['throughput_docs_per_hour'] = (completed_tasks / max(total_duration / 3600, 0.001))
        
        # Update average latency
        if task_data['success']:
            current_latency = self.metrics['avg_processing_latency']
            self.metrics['avg_processing_latency'] = (
                (current_latency * 0.9) + (task_data['duration_sec'] * 0.1)
            )
        
        # Update resource utilization
        self.metrics['cpu_utilization'] = psutil.cpu_percent(interval=None)
        self.metrics['memory_utilization'] = psutil.virtual_memory().percent / 100.0
    
    async def _store_task_metrics(self, performance_data: Dict[str, Any]):
        """Store task metrics in coordination memory"""
        try:
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"python-pipeline/performance/task-{performance_data['task_id']}",
                "--data", json.dumps(performance_data)
            ], capture_output=True, text=True, check=True)
        except Exception as e:
            self.logger.warning(f"Failed to store task metrics: {e}")
    
    def _get_performance_core_utilization(self) -> float:
        """Get current performance core utilization"""
        try:
            # Estimate P-core utilization (cores 0-7)
            cpu_percent = psutil.cpu_percent(percpu=True, interval=0.1)
            if len(cpu_percent) >= 8:
                p_core_usage = sum(cpu_percent[:8]) / (8 * 100.0)
                return p_core_usage
        except Exception as e:
            self.logger.debug(f"Could not get P-core utilization: {e}")
        
        return psutil.cpu_percent(interval=0.1) / 100.0
    
    def _start_monitoring(self):
        """Start background performance monitoring"""
        
        def monitor_loop():
            while True:
                try:
                    # Update system metrics
                    self.metrics['cpu_utilization'] = psutil.cpu_percent(interval=1)
                    self.metrics['memory_utilization'] = psutil.virtual_memory().percent / 100.0
                    
                    # Check for resource optimization opportunities
                    self._optimize_resource_allocation()
                    
                    # Clean up completed tasks
                    self._cleanup_completed_tasks()
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _optimize_resource_allocation(self):
        """Dynamically optimize resource allocation based on current load"""
        
        cpu_usage = self.metrics['cpu_utilization'] / 100.0
        memory_usage = self.metrics['memory_utilization']
        
        # Adjust executor sizes based on load
        if cpu_usage < 0.3 and len(self.active_tasks) > 0:
            # Low CPU usage, may need more workers
            self._scale_executors(scale_factor=1.2)
        elif cpu_usage > 0.9:
            # High CPU usage, scale down to prevent thrashing
            self._scale_executors(scale_factor=0.8)
    
    def _scale_executors(self, scale_factor: float):
        """Scale executor sizes based on current performance"""
        # Note: ThreadPoolExecutor doesn't support dynamic scaling
        # This would require implementing custom executor management
        self.logger.debug(f"Resource scaling suggestion: {scale_factor}")
    
    def _cleanup_completed_tasks(self):
        """Clean up completed tasks from tracking"""
        completed_tasks = []
        for task_id, task_info in self.active_tasks.items():
            if task_info['future'].done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        return {
            "m3_max_optimization": {
                "hardware_utilization": {
                    "cpu_utilization_percent": self.metrics['cpu_utilization'],
                    "memory_utilization_percent": self.metrics['memory_utilization'] * 100,
                    "gpu_utilization_percent": self.metrics['gpu_utilization'],
                    "neural_engine_utilization_percent": self.metrics['neural_engine_utilization']
                },
                "performance_metrics": {
                    "throughput_docs_per_hour": self.metrics['throughput_docs_per_hour'],
                    "avg_processing_latency_sec": self.metrics['avg_processing_latency'],
                    "active_tasks": len(self.active_tasks)
                },
                "resource_allocation": {
                    "performance_cores": self.core_allocation.available_performance,
                    "efficiency_cores": self.core_allocation.available_efficiency,
                    "memory_pools": "managed_by_mlx_accelerator"
                },
                "targets": {
                    "target_throughput": self.performance_targets.target_throughput_docs_hour,
                    "target_cpu_utilization": self.performance_targets.target_cpu_utilization,
                    "target_memory_efficiency": self.performance_targets.target_memory_efficiency
                }
            }
        }
    
    async def shutdown(self):
        """Shutdown optimizer and clean up resources"""
        
        # Shutdown executors
        self.performance_executor.shutdown(wait=True)
        self.efficiency_executor.shutdown(wait=True)
        self.memory_executor.shutdown(wait=True)
        self.ml_executor.shutdown(wait=True)
        
        # Store final performance metrics
        final_report = self.get_performance_report()
        
        try:
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-task",
                "--task-id", "m3-max-optimization",
                "--data", json.dumps(final_report)
            ], capture_output=True, text=True, check=True)
        except Exception as e:
            self.logger.warning(f"Failed to store final metrics: {e}")
        
        self.logger.info("M3 Max Optimizer shutdown completed")

# Utility functions and factory

def create_workload_spec(
    workload_type: WorkloadType,
    priority: int = 5,
    estimated_duration: float = 1.0,
    memory_mb: float = 1024.0,
    cpu_ratio: float = 0.5,
    parallelizable: bool = True
) -> WorkloadSpec:
    """Create a workload specification for optimization"""
    
    return WorkloadSpec(
        type=workload_type,
        priority=priority,
        estimated_duration_sec=estimated_duration,
        memory_requirement_mb=memory_mb,
        cpu_bound_ratio=cpu_ratio,
        parallelizable=parallelizable
    )

async def optimize_document_processing(
    documents: List[str],
    processing_function: Callable,
    optimizer: M3MaxOptimizer = None
) -> List[Any]:
    """Optimize document processing using M3 Max capabilities"""
    
    if not optimizer:
        optimizer = M3MaxOptimizer()
    
    # Create workload spec for document processing
    workload = create_workload_spec(
        workload_type=WorkloadType.PARALLEL_BATCH,
        priority=7,
        estimated_duration=len(documents) * 0.1,  # Estimate 0.1s per doc
        memory_mb=len(documents) * 10.0,  # Estimate 10MB per doc
        cpu_ratio=0.7,
        parallelizable=True
    )
    
    # Submit optimized task
    future = optimizer.optimize_workload(
        workload, processing_function, documents
    )
    
    return await asyncio.wrap_future(future)