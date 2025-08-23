"""
Performance Optimization Strategies for Large-Scale Dataset Processing
Advanced optimization techniques for memory efficiency, speed, and scalability
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import time
import psutil
import gc
import mmap
import json
from typing import Dict, List, Optional, Any, Iterator, Callable, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from contextlib import contextmanager
import resource
import tempfile
import shutil

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring optimization effectiveness"""
    processing_start_time: float
    processing_end_time: float
    total_records_processed: int
    records_per_second: float
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    cpu_utilization_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    cache_hit_rate: float
    error_count: int
    
    @property
    def processing_duration(self) -> float:
        return self.processing_end_time - self.processing_start_time
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on speed and resource usage"""
        speed_score = min(1.0, self.records_per_second / 1000.0)  # Normalize to 1000 rps
        memory_efficiency = max(0.1, 1.0 - (self.peak_memory_usage_mb / 4096.0))  # Penalty for >4GB
        cpu_efficiency = max(0.1, 1.0 - (self.cpu_utilization_percent / 100.0))  # Penalty for 100% CPU
        
        return (speed_score + memory_efficiency + cpu_efficiency) / 3.0

class MemoryMonitor:
    """Monitor and optimize memory usage during processing"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0
        self.memory_samples = []
        self.monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring:
            current_memory = self.get_memory_usage_mb()
            self.memory_samples.append(current_memory)
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            
            # Force garbage collection if memory usage is high
            if current_memory > self.max_memory_mb * 0.8:
                gc.collect()
                
            time.sleep(0.5)  # Check every 500ms
            
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
        
    def get_average_memory_mb(self) -> float:
        """Get average memory usage"""
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples)
        
    @contextmanager
    def memory_limit_context(self):
        """Context manager for memory-limited operations"""
        initial_memory = self.get_memory_usage_mb()
        try:
            yield
        finally:
            # Force cleanup
            gc.collect()
            final_memory = self.get_memory_usage_mb()
            if final_memory > initial_memory + 100:  # 100MB increase
                logging.warning(f"Memory increased by {final_memory - initial_memory:.1f}MB")

class StreamingJSONProcessor:
    """Memory-efficient streaming processor for large JSONL files"""
    
    def __init__(self, buffer_size: int = 8192):
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)
        
    def stream_records(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Stream records from JSONL file with minimal memory footprint"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = ""
                
                while True:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break
                        
                    buffer += chunk
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                record = json.loads(line)
                                yield record
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON line: {e}")
                                continue
                                
                # Process remaining buffer
                if buffer.strip():
                    try:
                        record = json.loads(buffer)
                        yield record
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Error streaming file {file_path}: {e}")
            raise
            
    def stream_batch_records(self, file_path: Path, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Stream records in batches"""
        batch = []
        
        for record in self.stream_records(file_path):
            batch.append(record)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
                
        # Yield remaining batch
        if batch:
            yield batch

class ParallelProcessor:
    """Parallel processing engine with configurable strategies"""
    
    def __init__(self, max_workers: Optional[int] = None, processing_strategy: str = "thread"):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.processing_strategy = processing_strategy
        self.logger = logging.getLogger(__name__)
        
    def process_batches_parallel(self, batches: List[List[Dict]], 
                               processing_function: Callable[[List[Dict]], List[Dict]],
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        """Process batches in parallel"""
        
        all_results = []
        
        if self.processing_strategy == "thread":
            executor_class = ThreadPoolExecutor
        elif self.processing_strategy == "process":
            executor_class = ProcessPoolExecutor
        else:
            raise ValueError(f"Invalid processing strategy: {self.processing_strategy}")
            
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(processing_function, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_results.extend(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(batches))
                        
                except Exception as e:
                    batch_idx = future_to_batch[future]
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    
        return all_results
        
    async def process_batches_async(self, batches: List[List[Dict]],
                                  async_processing_function: Callable[[List[Dict]], List[Dict]]) -> List[Dict]:
        """Process batches asynchronously"""
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await async_processing_function(batch)
                
        tasks = [process_with_semaphore(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter out exceptions
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Async processing error: {result}")
            else:
                all_results.extend(result)
                
        return all_results

class CacheManager:
    """Intelligent caching system for processed data"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_cache_size_mb: int = 1024):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "dataset_processing_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_stats = {"hits": 0, "misses": 0}
        self.logger = logging.getLogger(__name__)
        
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        import hashlib
        
        if isinstance(data, dict):
            # Create deterministic hash from dictionary
            sorted_items = json.dumps(data, sort_keys=True)
            return hashlib.md5(sorted_items.encode()).hexdigest()
        elif isinstance(data, (list, tuple)):
            # Hash based on length and first few items
            preview = str(len(data)) + str(data[:5]) if len(data) > 5 else str(data)
            return hashlib.md5(preview.encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
            
    def get_cached(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_stats["hits"] += 1
                    return data
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")
                
        self.cache_stats["misses"] += 1
        return None
        
    def set_cached(self, cache_key: str, data: Any):
        """Store data in cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                
            # Check cache size and cleanup if necessary
            self._cleanup_cache_if_needed()
            
        except Exception as e:
            self.logger.warning(f"Error writing cache file {cache_file}: {e}")
            
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.json")) / (1024 * 1024)
        
        if total_size_mb > self.max_cache_size_mb:
            # Remove oldest files first
            cache_files = list(self.cache_dir.glob("*.json"))
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest 30% of files
            files_to_remove = cache_files[:len(cache_files) // 3]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Error removing cache file {file_path}: {e}")
                    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        cache_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.json")) / (1024 * 1024)
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "cache_size_mb": cache_size_mb,
            "cache_file_count": len(list(self.cache_dir.glob("*.json")))
        }
        
    def clear_cache(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Error removing cache file {cache_file}: {e}")
                
        self.cache_stats = {"hits": 0, "misses": 0}

class ResourceOptimizer:
    """Optimize system resources for large-scale processing"""
    
    def __init__(self):
        self.original_limits = {}
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def optimized_resources(self, max_memory_mb: int = 4096, max_cpu_percent: int = 80):
        """Context manager for optimized resource usage"""
        
        # Store original resource limits
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            self.original_limits['memory'] = (soft, hard)
            
            # Set memory limit (in bytes)
            new_memory_limit = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (new_memory_limit, hard))
            
        except Exception as e:
            self.logger.warning(f"Could not set memory limit: {e}")
            
        # Optimize garbage collection
        original_gc_thresholds = gc.get_threshold()
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        try:
            yield
        finally:
            # Restore original settings
            gc.set_threshold(*original_gc_thresholds)
            
            if 'memory' in self.original_limits:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, self.original_limits['memory'])
                except Exception as e:
                    self.logger.warning(f"Could not restore memory limit: {e}")

class OptimizedDatasetProcessor:
    """Main optimized processor combining all optimization strategies"""
    
    def __init__(self, 
                 max_memory_mb: int = 2048,
                 max_workers: Optional[int] = None,
                 processing_strategy: str = "thread",
                 cache_enabled: bool = True,
                 batch_size: int = 1000):
        
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.streaming_processor = StreamingJSONProcessor()
        self.parallel_processor = ParallelProcessor(max_workers, processing_strategy)
        self.cache_manager = CacheManager() if cache_enabled else None
        self.resource_optimizer = ResourceOptimizer()
        
        self.logger = logging.getLogger(__name__)
        
    def process_large_dataset(self, 
                            input_files: List[Path],
                            processing_function: Callable[[List[Dict]], List[Dict]],
                            output_path: Path,
                            progress_callback: Optional[Callable[[str, int, int], None]] = None) -> PerformanceMetrics:
        """
        Process large datasets with full optimization
        
        Args:
            input_files: List of input JSONL files
            processing_function: Function to process record batches
            output_path: Output file path
            progress_callback: Optional progress callback (stage, current, total)
            
        Returns:
            PerformanceMetrics: Performance statistics
        """
        
        start_time = time.time()
        total_records = 0
        error_count = 0
        
        with self.resource_optimizer.optimized_resources(self.max_memory_mb):
            # Start monitoring
            self.memory_monitor.start_monitoring()
            
            try:
                # Create output directory
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    
                    for file_idx, input_file in enumerate(input_files):
                        self.logger.info(f"Processing file {file_idx + 1}/{len(input_files)}: {input_file}")
                        
                        # Stream and batch records
                        batches = []
                        current_batch = []
                        
                        for record in self.streaming_processor.stream_records(input_file):
                            current_batch.append(record)
                            
                            if len(current_batch) >= self.batch_size:
                                batches.append(current_batch)
                                current_batch = []
                                
                                # Process batches when we have enough to parallelize
                                if len(batches) >= self.parallel_processor.max_workers * 2:
                                    processed_records = self._process_batches_with_caching(
                                        batches, processing_function
                                    )
                                    
                                    # Write results
                                    for record in processed_records:
                                        output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                                        total_records += 1
                                        
                                    batches = []
                                    
                                    if progress_callback:
                                        progress_callback(f"Processing {input_file.name}", 
                                                        file_idx + 1, len(input_files))
                        
                        # Add remaining batch
                        if current_batch:
                            batches.append(current_batch)
                            
                        # Process remaining batches
                        if batches:
                            processed_records = self._process_batches_with_caching(
                                batches, processing_function
                            )
                            
                            for record in processed_records:
                                output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                                total_records += 1
                                
            except Exception as e:
                self.logger.error(f"Error during processing: {e}")
                error_count += 1
                raise
                
            finally:
                # Stop monitoring
                self.memory_monitor.stop_monitoring()
                
        end_time = time.time()
        
        # Calculate performance metrics
        duration = end_time - start_time
        records_per_second = total_records / duration if duration > 0 else 0
        
        # Get system metrics
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        
        # Cache statistics
        cache_hit_rate = 0.0
        if self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
            cache_hit_rate = cache_stats["hit_rate"]
            
        metrics = PerformanceMetrics(
            processing_start_time=start_time,
            processing_end_time=end_time,
            total_records_processed=total_records,
            records_per_second=records_per_second,
            peak_memory_usage_mb=self.memory_monitor.peak_memory_mb,
            average_memory_usage_mb=self.memory_monitor.get_average_memory_mb(),
            cpu_utilization_percent=cpu_percent,
            disk_io_read_mb=0.0,  # Would need additional monitoring
            disk_io_write_mb=0.0,  # Would need additional monitoring
            cache_hit_rate=cache_hit_rate,
            error_count=error_count
        )
        
        self.logger.info(f"Processing completed: {total_records} records in {duration:.2f}s "
                        f"({records_per_second:.1f} rps)")
        
        return metrics
        
    def _process_batches_with_caching(self, batches: List[List[Dict]], 
                                    processing_function: Callable[[List[Dict]], List[Dict]]) -> List[Dict]:
        """Process batches with caching support"""
        
        if not self.cache_manager:
            # No caching - direct parallel processing
            return self.parallel_processor.process_batches_parallel(batches, processing_function)
            
        cached_results = []
        uncached_batches = []
        
        # Check cache for each batch
        for batch in batches:
            cache_key = self.cache_manager.get_cache_key(batch)
            cached_result = self.cache_manager.get_cached(cache_key)
            
            if cached_result:
                cached_results.extend(cached_result)
            else:
                uncached_batches.append((batch, cache_key))
                
        # Process uncached batches
        if uncached_batches:
            uncached_batch_list = [item[0] for item in uncached_batches]
            processed_batches = self.parallel_processor.process_batches_parallel(
                uncached_batch_list, processing_function
            )
            
            # Cache results and collect
            batch_start_idx = 0
            for i, (original_batch, cache_key) in enumerate(uncached_batches):
                batch_size = len(original_batch)
                batch_result = processed_batches[batch_start_idx:batch_start_idx + batch_size]
                
                # Cache the result
                self.cache_manager.set_cached(cache_key, batch_result)
                cached_results.extend(batch_result)
                
                batch_start_idx += batch_size
                
        return cached_results
        
    def benchmark_processing_strategies(self, sample_batches: List[List[Dict]], 
                                      processing_function: Callable[[List[Dict]], List[Dict]]) -> Dict[str, float]:
        """Benchmark different processing strategies"""
        
        strategies = ["thread", "process"]
        results = {}
        
        for strategy in strategies:
            # Create temporary processor with this strategy
            temp_processor = ParallelProcessor(processing_strategy=strategy)
            
            start_time = time.time()
            try:
                temp_processor.process_batches_parallel(sample_batches, processing_function)
                end_time = time.time()
                results[strategy] = end_time - start_time
            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed: {e}")
                results[strategy] = float('inf')
                
        return results
        
    def optimize_batch_size(self, sample_records: List[Dict],
                          processing_function: Callable[[List[Dict]], List[Dict]]) -> int:
        """Find optimal batch size for processing"""
        
        batch_sizes = [100, 500, 1000, 2000, 5000]
        performance_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(sample_records):
                continue
                
            # Create batches
            batches = []
            for i in range(0, len(sample_records), batch_size):
                batch = sample_records[i:i + batch_size]
                batches.append(batch)
                
            # Measure processing time
            start_time = time.time()
            with self.memory_monitor.memory_limit_context():
                try:
                    self.parallel_processor.process_batches_parallel(batches, processing_function)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    records_per_second = len(sample_records) / processing_time
                    performance_results[batch_size] = records_per_second
                    
                except Exception as e:
                    self.logger.warning(f"Batch size {batch_size} failed: {e}")
                    performance_results[batch_size] = 0
                    
        # Return batch size with best performance
        if performance_results:
            optimal_batch_size = max(performance_results.keys(), key=lambda x: performance_results[x])
            self.logger.info(f"Optimal batch size: {optimal_batch_size} "
                           f"({performance_results[optimal_batch_size]:.1f} rps)")
            return optimal_batch_size
        else:
            return 1000  # Default fallback

def analyze_performance_bottlenecks(metrics: PerformanceMetrics) -> Dict[str, Any]:
    """Analyze performance metrics to identify bottlenecks"""
    
    bottlenecks = []
    recommendations = []
    
    # Memory analysis
    if metrics.peak_memory_usage_mb > 3072:  # >3GB
        bottlenecks.append("high_memory_usage")
        recommendations.append("Consider reducing batch size or enabling more aggressive garbage collection")
        
    # Speed analysis
    if metrics.records_per_second < 100:
        bottlenecks.append("low_processing_speed")
        recommendations.append("Consider increasing parallelization or optimizing processing function")
        
    # CPU analysis
    if metrics.cpu_utilization_percent > 95:
        bottlenecks.append("cpu_bound")
        recommendations.append("Processing is CPU-bound - consider process-based parallelization")
    elif metrics.cpu_utilization_percent < 50:
        bottlenecks.append("cpu_underutilized")
        recommendations.append("CPU is underutilized - consider increasing worker threads")
        
    # Cache analysis
    if metrics.cache_hit_rate < 0.1:
        bottlenecks.append("poor_cache_performance")
        recommendations.append("Cache hit rate is low - review caching strategy or disable if not beneficial")
        
    # Error analysis
    if metrics.error_count > 0:
        bottlenecks.append("processing_errors")
        recommendations.append(f"Found {metrics.error_count} processing errors - review error handling")
        
    return {
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "efficiency_score": metrics.efficiency_score,
        "performance_summary": {
            "records_per_second": metrics.records_per_second,
            "peak_memory_mb": metrics.peak_memory_usage_mb,
            "processing_duration": metrics.processing_duration,
            "cache_hit_rate": metrics.cache_hit_rate
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Example processing function
    def sample_processing_function(batch: List[Dict]) -> List[Dict]:
        """Sample processing function that simulates work"""
        processed = []
        for record in batch:
            # Simulate some processing
            processed_record = record.copy()
            processed_record["processed"] = True
            processed_record["processing_time"] = time.time()
            processed.append(processed_record)
        return processed
    
    # Initialize optimized processor
    processor = OptimizedDatasetProcessor(
        max_memory_mb=1024,
        max_workers=4,
        batch_size=500,
        cache_enabled=True
    )
    
    # Sample data for testing
    sample_files = [Path("sample_dataset.jsonl")]  # Would need real files
    output_path = Path("processed_output.jsonl")
    
    try:
        # Process with optimization
        metrics = processor.process_large_dataset(
            sample_files,
            sample_processing_function,
            output_path
        )
        
        # Analyze performance
        analysis = analyze_performance_bottlenecks(metrics)
        
        print("Performance Analysis:")
        print(json.dumps(analysis, indent=2, default=str))
        
    except Exception as e:
        print(f"Processing failed: {e}")