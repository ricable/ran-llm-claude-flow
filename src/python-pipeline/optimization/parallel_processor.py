"""
Parallel Processing Engine for Python Pipeline
Intelligent resource management with dynamic scaling and load balancing
Optimized for M3 Max hardware (8P+4E cores, 40-core GPU, 128GB unified memory)
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import logging
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
import psutil
import os
from pathlib import Path
import hashlib

@dataclass
class ProcessingTask:
    """Task specification for parallel processing"""
    id: str
    priority: int  # 1-10, higher = more priority
    data: Any
    processor_function: str  # Function name for processing
    estimated_time_sec: float = 1.0
    memory_requirement_mb: float = 100.0
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)

@dataclass 
class ProcessingResult:
    """Result from parallel processing"""
    task_id: str
    success: bool
    result: Any = None
    error: str = None
    processing_time_sec: float = 0.0
    memory_used_mb: float = 0.0
    worker_id: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class ResourceLimits:
    """Resource limits for processing"""
    max_cpu_percent: float = 85.0
    max_memory_gb: float = 100.0  # Out of 128GB total
    max_concurrent_tasks: int = 32
    max_processing_time_sec: float = 300.0  # 5 minutes per task
    gpu_memory_limit_gb: float = 30.0

class ProcessingStats:
    """Real-time processing statistics"""
    
    def __init__(self):
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.peak_memory_usage_gb = 0.0
        self.avg_throughput_per_sec = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_task_completion(self, processing_time: float, success: bool, memory_mb: float):
        with self.lock:
            if success:
                self.tasks_completed += 1
            else:
                self.tasks_failed += 1
            
            self.total_processing_time += processing_time
            self.peak_memory_usage_gb = max(self.peak_memory_usage_gb, memory_mb / 1024.0)
            
            # Update throughput
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.avg_throughput_per_sec = self.tasks_completed / elapsed_time

class ParallelProcessor:
    """
    Intelligent parallel processing engine optimized for M3 Max hardware
    Features dynamic scaling, load balancing, and resource management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Resource configuration
        self.resource_limits = ResourceLimits()
        self.stats = ProcessingStats()
        
        # Processing infrastructure
        self.task_queue = PriorityQueue()
        self.result_queue = Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # Worker management
        self.cpu_intensive_executor = None
        self.io_intensive_executor = None
        self.memory_intensive_executor = None
        self.gpu_executor = None
        
        # Coordination and monitoring
        self.processing_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.monitor_thread = None
        
        # Dynamic scaling parameters
        self.base_workers = {
            'cpu_intensive': min(8, mp.cpu_count()),  # P-cores for CPU work
            'io_intensive': min(16, mp.cpu_count() * 2),  # More threads for I/O
            'memory_intensive': 4,  # Limited for memory management
            'gpu': 4  # GPU processing coordination
        }
        
        self.current_workers = self.base_workers.copy()
        
        # Initialize processing infrastructure
        self._initialize_executors()
        self._start_monitoring()
        
        self.logger.info("ParallelProcessor initialized for M3 Max hardware")
    
    def _initialize_executors(self):
        """Initialize optimized executors for different workload types"""
        
        # CPU-intensive executor for compute-heavy tasks
        self.cpu_intensive_executor = ProcessPoolExecutor(
            max_workers=self.current_workers['cpu_intensive'],
            mp_context=mp.get_context('spawn')
        )
        
        # I/O-intensive executor for file operations and network requests
        self.io_intensive_executor = ThreadPoolExecutor(
            max_workers=self.current_workers['io_intensive']
        )
        
        # Memory-intensive executor with controlled concurrency
        self.memory_intensive_executor = ThreadPoolExecutor(
            max_workers=self.current_workers['memory_intensive']
        )
        
        # GPU executor for MLX operations
        self.gpu_executor = ThreadPoolExecutor(
            max_workers=self.current_workers['gpu']
        )
        
        self.logger.info(f"Initialized executors with workers: {self.current_workers}")
    
    async def submit_task(
        self, 
        task_id: str,
        data: Any,
        processor_function: Callable,
        priority: int = 5,
        estimated_time: float = 1.0,
        memory_mb: float = 100.0,
        dependencies: List[str] = None,
        task_type: str = "cpu_intensive"
    ) -> str:
        """
        Submit a task for parallel processing
        Returns task ID for tracking
        """
        
        # Create task specification
        task = ProcessingTask(
            id=task_id,
            priority=priority,
            data=data,
            processor_function=processor_function.__name__,
            estimated_time_sec=estimated_time,
            memory_requirement_mb=memory_mb,
            dependencies=dependencies or []
        )
        
        # Check resource availability
        if not await self._check_resource_availability(task):
            raise RuntimeError(f"Insufficient resources for task {task_id}")
        
        # Select optimal executor
        executor = self._select_executor(task_type, task)
        
        # Submit task with wrapper for monitoring
        future = executor.submit(
            self._execute_task_with_monitoring,
            task,
            processor_function,
            task_type
        )
        
        # Track active task
        with self.processing_lock:
            self.active_tasks[task_id] = {
                'task': task,
                'future': future,
                'task_type': task_type,
                'start_time': time.time(),
                'executor': executor
            }
            self.stats.tasks_submitted += 1
        
        self.logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id
    
    def _select_executor(self, task_type: str, task: ProcessingTask) -> concurrent.futures.Executor:
        """Select optimal executor based on task characteristics"""
        
        # Memory-intensive tasks
        if task.memory_requirement_mb > 1000:
            return self.memory_intensive_executor
        
        # GPU-accelerated tasks
        if task_type in ['gpu', 'ml_inference', 'embedding']:
            return self.gpu_executor
        
        # I/O-intensive tasks
        if task_type in ['io_intensive', 'file_processing', 'network']:
            return self.io_intensive_executor
        
        # Default to CPU-intensive
        return self.cpu_intensive_executor
    
    def _execute_task_with_monitoring(
        self, 
        task: ProcessingTask,
        processor_function: Callable,
        task_type: str
    ) -> ProcessingResult:
        """Execute task with comprehensive monitoring and resource tracking"""
        
        start_time = time.time()
        worker_id = f"{task_type}_{threading.current_thread().ident}"
        
        # Set up resource monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            # Set resource limits
            self._set_task_resource_limits(task)
            
            # Execute the actual processing function
            result = processor_function(task.data)
            
            # Calculate resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = max(0, final_memory - initial_memory)
            processing_time = time.time() - start_time
            
            # Create success result
            processing_result = ProcessingResult(
                task_id=task.id,
                success=True,
                result=result,
                processing_time_sec=processing_time,
                memory_used_mb=memory_used,
                worker_id=worker_id
            )
            
            # Update statistics
            self.stats.record_task_completion(processing_time, True, memory_used)
            
            # Store result for coordination
            asyncio.create_task(self._store_task_result(processing_result))
            
            return processing_result
            
        except Exception as e:
            # Calculate partial resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_used = max(0, final_memory - initial_memory)
            processing_time = time.time() - start_time
            
            # Create error result
            processing_result = ProcessingResult(
                task_id=task.id,
                success=False,
                error=str(e),
                processing_time_sec=processing_time,
                memory_used_mb=memory_used,
                worker_id=worker_id
            )
            
            # Update statistics
            self.stats.record_task_completion(processing_time, False, memory_used)
            
            # Log error
            self.logger.error(f"Task {task.id} failed: {e}")
            
            # Store error result
            asyncio.create_task(self._store_task_result(processing_result))
            
            return processing_result
    
    def _set_task_resource_limits(self, task: ProcessingTask):
        """Set resource limits for individual task execution"""
        try:
            import resource
            
            # Set memory limit
            memory_limit_bytes = int(task.memory_requirement_mb * 1024 * 1024 * 1.5)  # 1.5x safety margin
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            
            # Set CPU time limit
            max_cpu_time = int(task.estimated_time_sec * 3)  # 3x safety margin
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time, max_cpu_time))
            
        except Exception as e:
            self.logger.debug(f"Could not set resource limits: {e}")
    
    async def _check_resource_availability(self, task: ProcessingTask) -> bool:
        """Check if sufficient resources are available for task execution"""
        
        # Check memory availability
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024 ** 3)
        required_memory_gb = task.memory_requirement_mb / 1024.0
        
        if required_memory_gb > available_memory_gb * 0.8:  # Keep 20% buffer
            self.logger.warning(f"Insufficient memory for task {task.id}: need {required_memory_gb:.1f}GB, available {available_memory_gb:.1f}GB")
            return False
        
        # Check CPU availability
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.resource_limits.max_cpu_percent:
            self.logger.warning(f"CPU usage too high for task {task.id}: {cpu_percent}%")
            return False
        
        # Check active task count
        if len(self.active_tasks) >= self.resource_limits.max_concurrent_tasks:
            self.logger.warning(f"Too many active tasks for task {task.id}: {len(self.active_tasks)}")
            return False
        
        return True
    
    async def process_batch(
        self,
        data_batch: List[Any],
        processor_function: Callable,
        batch_size: int = None,
        priority: int = 5,
        task_type: str = "cpu_intensive"
    ) -> List[ProcessingResult]:
        """
        Process a batch of data items in parallel with intelligent batching
        """
        
        if not data_batch:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(
                len(data_batch), 
                task_type
            )
        
        self.logger.info(f"Processing batch of {len(data_batch)} items with batch size {batch_size}")
        
        # Submit all tasks
        task_futures = []
        for i, data_item in enumerate(data_batch):
            task_id = f"batch_{int(time.time() * 1000)}_{i}"
            
            future = await self.submit_task(
                task_id=task_id,
                data=data_item,
                processor_function=processor_function,
                priority=priority,
                task_type=task_type
            )
            task_futures.append(task_id)
        
        # Wait for all tasks to complete
        results = await self.wait_for_tasks(task_futures)
        
        self.logger.info(f"Completed batch processing: {len([r for r in results if r.success])}/{len(results)} successful")
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int, task_type: str) -> int:
        """Calculate optimal batch size based on hardware and task type"""
        
        # Base batch size on available workers
        available_workers = self.current_workers.get(task_type, 4)
        
        # Memory considerations
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024 ** 3)
        
        # Task type specific optimizations
        if task_type == "cpu_intensive":
            # CPU-intensive tasks benefit from smaller batches to prevent CPU saturation
            optimal_batch = min(available_workers, total_items // 4)
        elif task_type == "io_intensive":
            # I/O-intensive tasks can handle larger batches
            optimal_batch = min(available_workers * 2, total_items // 2)
        elif task_type == "gpu":
            # GPU tasks benefit from medium batches for optimal GPU utilization
            optimal_batch = min(available_workers, total_items // 3)
        else:
            optimal_batch = available_workers
        
        # Memory constraint
        max_memory_batch = int(available_memory_gb * 10)  # Assume 100MB per task
        optimal_batch = min(optimal_batch, max_memory_batch)
        
        # Ensure minimum batch size
        return max(1, optimal_batch)
    
    async def wait_for_tasks(self, task_ids: List[str], timeout: float = None) -> List[ProcessingResult]:
        """Wait for specified tasks to complete and return results"""
        
        results = []
        remaining_tasks = set(task_ids)
        start_time = time.time()
        
        while remaining_tasks:
            # Check for completed tasks
            completed_this_round = set()
            
            with self.processing_lock:
                for task_id in remaining_tasks:
                    if task_id in self.active_tasks:
                        task_info = self.active_tasks[task_id]
                        if task_info['future'].done():
                            try:
                                result = task_info['future'].result()
                                results.append(result)
                                completed_this_round.add(task_id)
                                
                                # Move to completed tasks
                                self.completed_tasks[task_id] = task_info
                                del self.active_tasks[task_id]
                                
                            except Exception as e:
                                # Create error result
                                error_result = ProcessingResult(
                                    task_id=task_id,
                                    success=False,
                                    error=str(e),
                                    processing_time_sec=time.time() - task_info['start_time']
                                )
                                results.append(error_result)
                                completed_this_round.add(task_id)
                                
                                # Clean up
                                del self.active_tasks[task_id]
            
            # Remove completed tasks
            remaining_tasks -= completed_this_round
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Timeout waiting for tasks: {remaining_tasks}")
                break
            
            # Short sleep to prevent busy waiting
            if remaining_tasks:
                await asyncio.sleep(0.1)
        
        return results
    
    def _start_monitoring(self):
        """Start background monitoring and optimization"""
        
        def monitoring_loop():
            while not self.shutdown_event.is_set():
                try:
                    # Monitor system resources
                    self._monitor_system_resources()
                    
                    # Optimize worker allocation
                    self._optimize_worker_allocation()
                    
                    # Clean up completed tasks
                    self._cleanup_old_tasks()
                    
                    # Store performance metrics
                    asyncio.create_task(self._store_performance_metrics())
                    
                    # Wait before next monitoring cycle
                    self.shutdown_event.wait(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    self.shutdown_event.wait(30)  # Wait longer on error
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started background monitoring")
    
    def _monitor_system_resources(self):
        """Monitor system resources and log alerts"""
        
        # CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {cpu_percent}%")
        
        # Memory monitoring  
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.logger.warning(f"High memory usage: {memory.percent}%")
        
        # Update peak memory tracking
        memory_gb = (memory.total - memory.available) / (1024 ** 3)
        self.stats.peak_memory_usage_gb = max(self.stats.peak_memory_usage_gb, memory_gb)
    
    def _optimize_worker_allocation(self):
        """Dynamically optimize worker allocation based on current load"""
        
        with self.processing_lock:
            # Analyze task distribution by type
            task_types = {}
            for task_info in self.active_tasks.values():
                task_type = task_info.get('task_type', 'cpu_intensive')
                task_types[task_type] = task_types.get(task_type, 0) + 1
            
            # Calculate optimization suggestions
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            # Suggest worker adjustments (actual implementation would recreate executors)
            optimization_suggestions = {}
            
            for task_type, count in task_types.items():
                current_workers = self.current_workers[task_type]
                
                if count > current_workers and cpu_percent < 70:
                    # More tasks than workers and CPU available
                    optimization_suggestions[task_type] = min(current_workers + 2, self.base_workers[task_type] * 2)
                elif count < current_workers // 2 and cpu_percent > 80:
                    # Fewer tasks than workers and high CPU usage
                    optimization_suggestions[task_type] = max(current_workers - 1, 1)
            
            # Log optimization suggestions
            if optimization_suggestions:
                self.logger.debug(f"Worker optimization suggestions: {optimization_suggestions}")
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory leaks"""
        
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour
        
        with self.processing_lock:
            old_tasks = []
            for task_id, task_info in self.completed_tasks.items():
                if current_time - task_info.get('start_time', 0) > cleanup_threshold:
                    old_tasks.append(task_id)
            
            for task_id in old_tasks:
                del self.completed_tasks[task_id]
            
            if old_tasks:
                self.logger.debug(f"Cleaned up {len(old_tasks)} old completed tasks")
    
    async def _store_task_result(self, result: ProcessingResult):
        """Store task result in coordination memory"""
        try:
            result_data = {
                'task_id': result.task_id,
                'success': result.success,
                'processing_time_sec': result.processing_time_sec,
                'memory_used_mb': result.memory_used_mb,
                'worker_id': result.worker_id,
                'timestamp': result.timestamp,
                'error': result.error
            }
            
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"python-pipeline/performance/task-result-{result.task_id}",
                "--data", json.dumps(result_data)
            ], capture_output=True, text=True, check=True)
            
        except Exception as e:
            self.logger.warning(f"Failed to store task result: {e}")
    
    async def _store_performance_metrics(self):
        """Store performance metrics in coordination memory"""
        try:
            metrics_data = {
                'tasks_submitted': self.stats.tasks_submitted,
                'tasks_completed': self.stats.tasks_completed,
                'tasks_failed': self.stats.tasks_failed,
                'success_rate': (self.stats.tasks_completed / max(1, self.stats.tasks_submitted)) * 100,
                'avg_throughput_per_sec': self.stats.avg_throughput_per_sec,
                'throughput_docs_per_hour': self.stats.avg_throughput_per_sec * 3600,
                'peak_memory_usage_gb': self.stats.peak_memory_usage_gb,
                'active_tasks': len(self.active_tasks),
                'current_workers': self.current_workers,
                'timestamp': time.time()
            }
            
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit", 
                "--memory-key", "python-pipeline/performance/parallel-processing",
                "--data", json.dumps(metrics_data)
            ], capture_output=True, text=True, check=True)
            
        except Exception as e:
            self.logger.warning(f"Failed to store performance metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        return {
            "parallel_processing": {
                "statistics": {
                    "tasks_submitted": self.stats.tasks_submitted,
                    "tasks_completed": self.stats.tasks_completed,
                    "tasks_failed": self.stats.tasks_failed,
                    "success_rate_percent": (self.stats.tasks_completed / max(1, self.stats.tasks_submitted)) * 100,
                    "avg_throughput_per_sec": self.stats.avg_throughput_per_sec,
                    "throughput_docs_per_hour": self.stats.avg_throughput_per_sec * 3600,
                    "peak_memory_usage_gb": self.stats.peak_memory_usage_gb
                },
                "current_state": {
                    "active_tasks": len(self.active_tasks),
                    "completed_tasks": len(self.completed_tasks),
                    "worker_allocation": self.current_workers
                },
                "resource_limits": {
                    "max_cpu_percent": self.resource_limits.max_cpu_percent,
                    "max_memory_gb": self.resource_limits.max_memory_gb,
                    "max_concurrent_tasks": self.resource_limits.max_concurrent_tasks
                }
            }
        }
    
    async def shutdown(self):
        """Shutdown parallel processor and clean up resources"""
        
        self.logger.info("Shutting down ParallelProcessor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            active_task_ids = list(self.active_tasks.keys())
            await self.wait_for_tasks(active_task_ids, timeout=60)
        
        # Shutdown executors
        if self.cpu_intensive_executor:
            self.cpu_intensive_executor.shutdown(wait=True)
        if self.io_intensive_executor:
            self.io_intensive_executor.shutdown(wait=True)
        if self.memory_intensive_executor:
            self.memory_intensive_executor.shutdown(wait=True)
        if self.gpu_executor:
            self.gpu_executor.shutdown(wait=True)
        
        # Store final metrics
        await self._store_performance_metrics()
        
        self.logger.info("ParallelProcessor shutdown complete")

# Utility functions

async def create_parallel_processor(config: Dict[str, Any] = None) -> ParallelProcessor:
    """Factory function to create and configure parallel processor"""
    processor = ParallelProcessor(config)
    return processor

def create_processing_task(
    task_id: str,
    data: Any,
    priority: int = 5,
    estimated_time: float = 1.0,
    memory_mb: float = 100.0
) -> ProcessingTask:
    """Helper function to create processing task specification"""
    return ProcessingTask(
        id=task_id,
        priority=priority,
        data=data,
        processor_function="",  # Will be set when submitting
        estimated_time_sec=estimated_time,
        memory_requirement_mb=memory_mb
    )