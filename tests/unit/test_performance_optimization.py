"""
Unit tests for Performance Optimization
TDD London School testing patterns with comprehensive mocking
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from concurrent.futures import Future

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from performance_optimization import (
    PerformanceMetrics,
    MemoryMonitor,
    StreamingJSONProcessor,
    ParallelProcessor,
    CacheManager,
    ResourceOptimizer,
    OptimizedDatasetProcessor,
    analyze_performance_bottlenecks
)


class TestPerformanceMetrics:
    """Test performance metrics dataclass"""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics creation"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=1000,
            records_per_second=10.0,
            peak_memory_usage_mb=512.0,
            average_memory_usage_mb=256.0,
            cpu_utilization_percent=75.0,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            cache_hit_rate=0.8,
            error_count=2
        )
        
        assert metrics.processing_start_time == 1000.0
        assert metrics.processing_end_time == 1100.0
        assert metrics.total_records_processed == 1000
        assert metrics.records_per_second == 10.0
        assert metrics.peak_memory_usage_mb == 512.0
        assert metrics.average_memory_usage_mb == 256.0
        assert metrics.cpu_utilization_percent == 75.0
        assert metrics.disk_io_read_mb == 100.0
        assert metrics.disk_io_write_mb == 50.0
        assert metrics.cache_hit_rate == 0.8
        assert metrics.error_count == 2
    
    def test_processing_duration_property(self):
        """Test processing duration calculation"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1150.0,
            total_records_processed=100,
            records_per_second=1.0,
            peak_memory_usage_mb=100.0,
            average_memory_usage_mb=80.0,
            cpu_utilization_percent=50.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.5,
            error_count=0
        )
        
        assert metrics.processing_duration == 150.0
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=100.0,  # Good speed
            peak_memory_usage_mb=1024.0,  # Moderate memory
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=60.0,  # Reasonable CPU usage
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=0
        )
        
        efficiency = metrics.efficiency_score
        assert 0.0 <= efficiency <= 1.0
        
        # Test with poor performance
        poor_metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=10,
            records_per_second=1.0,  # Poor speed
            peak_memory_usage_mb=8192.0,  # High memory
            average_memory_usage_mb=4096.0,
            cpu_utilization_percent=100.0,  # Max CPU
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            cache_hit_rate=0.1,
            error_count=5
        )
        
        poor_efficiency = poor_metrics.efficiency_score
        assert poor_efficiency < efficiency


class TestMemoryMonitor:
    """Test memory monitoring functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.memory_monitor = MemoryMonitor(max_memory_mb=1024)
    
    def teardown_method(self):
        """Cleanup after tests"""
        if self.memory_monitor.monitoring:
            self.memory_monitor.stop_monitoring()
    
    @patch('psutil.Process')
    def test_get_memory_usage_mb(self, mock_process):
        """Test memory usage measurement"""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 512  # 512MB
        mock_process.return_value = mock_process_instance
        
        usage = self.memory_monitor.get_memory_usage_mb()
        
        assert usage == 512.0
        mock_process.assert_called_once()
        mock_process_instance.memory_info.assert_called_once()
    
    def test_start_stop_monitoring(self):
        """Test monitoring lifecycle"""
        assert self.memory_monitor.monitoring is False
        assert self.memory_monitor._monitor_thread is None
        
        self.memory_monitor.start_monitoring()
        
        assert self.memory_monitor.monitoring is True
        assert self.memory_monitor._monitor_thread is not None
        assert self.memory_monitor._monitor_thread.is_alive()
        
        self.memory_monitor.stop_monitoring()
        
        assert self.memory_monitor.monitoring is False
    
    @patch('psutil.Process')
    @patch('gc.collect')
    @patch('time.sleep')
    def test_monitor_loop_with_gc(self, mock_sleep, mock_gc, mock_process):
        """Test monitoring loop triggers garbage collection"""
        mock_process_instance = MagicMock()
        # Simulate high memory usage that triggers GC
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 900  # 900MB (>80% of 1024)
        mock_process.return_value = mock_process_instance
        
        # Stop monitoring after first iteration
        def stop_monitoring(*args):
            self.memory_monitor.monitoring = False
        
        mock_sleep.side_effect = stop_monitoring
        
        self.memory_monitor.start_monitoring()
        time.sleep(0.1)  # Give thread time to execute
        
        # Garbage collection should have been triggered
        mock_gc.assert_called()
    
    def test_get_average_memory_mb(self):
        """Test average memory calculation"""
        # Empty samples
        assert self.memory_monitor.get_average_memory_mb() == 0.0
        
        # Add some samples
        self.memory_monitor.memory_samples = [100.0, 200.0, 300.0]
        assert self.memory_monitor.get_average_memory_mb() == 200.0
    
    @patch('gc.collect')
    def test_memory_limit_context(self, mock_gc):
        """Test memory limit context manager"""
        with patch.object(self.memory_monitor, 'get_memory_usage_mb', 
                         side_effect=[100.0, 250.0]):  # 150MB increase
            
            with self.memory_monitor.memory_limit_context():
                pass  # Simulate some work
            
            mock_gc.assert_called_once()


class TestStreamingJSONProcessor:
    """Test streaming JSON processing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = StreamingJSONProcessor(buffer_size=1024)
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor.buffer_size == 1024
        assert self.processor.logger is not None
    
    @patch('builtins.open', new_callable=mock_open, 
           read_data='{"test": "data1"}\n{"test": "data2"}\n{"test": "data3"}\n')
    def test_stream_records_success(self, mock_file):
        """Test successful record streaming"""
        test_path = Path("test.jsonl")
        
        records = list(self.processor.stream_records(test_path))
        
        assert len(records) == 3
        assert all(isinstance(record, dict) for record in records)
        assert records[0]["test"] == "data1"
        assert records[1]["test"] == "data2"
        assert records[2]["test"] == "data3"
    
    @patch('builtins.open', new_callable=mock_open,
           read_data='{"valid": "record"}\n{"invalid": json}\n{"another": "valid"}\n')
    @patch('performance_optimization.logging.getLogger')
    def test_stream_records_with_invalid_json(self, mock_logger, mock_file):
        """Test streaming with invalid JSON lines"""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        test_path = Path("test.jsonl")
        
        records = list(self.processor.stream_records(test_path))
        
        # Should skip invalid JSON and continue
        assert len(records) == 2
        assert records[0]["valid"] == "record"
        assert records[1]["another"] == "valid"
        
        # Warning should have been logged
        mock_logger_instance.warning.assert_called()
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_stream_records_file_error(self, mock_file):
        """Test streaming with file error"""
        test_path = Path("nonexistent.jsonl")
        
        with pytest.raises(FileNotFoundError):
            list(self.processor.stream_records(test_path))
    
    @patch('builtins.open', new_callable=mock_open,
           read_data='{"batch": 1}\n{"batch": 2}\n{"batch": 3}\n{"batch": 4}\n{"batch": 5}\n')
    def test_stream_batch_records(self, mock_file):
        """Test batch streaming"""
        test_path = Path("test.jsonl")
        
        batches = list(self.processor.stream_batch_records(test_path, batch_size=2))
        
        assert len(batches) == 3  # 5 records / 2 batch size = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1  # Remaining record


class TestParallelProcessor:
    """Test parallel processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = ParallelProcessor(max_workers=2, processing_strategy="thread")
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor.max_workers == 2
        assert self.processor.processing_strategy == "thread"
        assert self.processor.logger is not None
    
    def test_process_batches_parallel_success(self):
        """Test successful parallel batch processing"""
        def mock_processing_function(batch):
            return [{"processed": True, **item} for item in batch]
        
        batches = [
            [{"id": 1}, {"id": 2}],
            [{"id": 3}, {"id": 4}],
            [{"id": 5}]
        ]
        
        results = self.processor.process_batches_parallel(
            batches, mock_processing_function
        )
        
        assert len(results) == 5
        assert all(result["processed"] is True for result in results)
        assert {result["id"] for result in results} == {1, 2, 3, 4, 5}
    
    def test_process_batches_parallel_with_progress(self):
        """Test parallel processing with progress callback"""
        def mock_processing_function(batch):
            return batch
        
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        batches = [
            [{"id": 1}],
            [{"id": 2}],
            [{"id": 3}]
        ]
        
        results = self.processor.process_batches_parallel(
            batches, mock_processing_function, progress_callback
        )
        
        assert len(results) == 3
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)  # Final call should be (total, total)
    
    def test_process_batches_parallel_with_exception(self):
        """Test parallel processing with exceptions"""
        def failing_processing_function(batch):
            if any(item.get("id") == 2 for item in batch):
                raise ValueError("Processing failed")
            return batch
        
        batches = [
            [{"id": 1}],
            [{"id": 2}],  # This will fail
            [{"id": 3}]
        ]
        
        # Should not raise exception, just log and continue
        results = self.processor.process_batches_parallel(
            batches, failing_processing_function
        )
        
        # Should have results from successful batches
        assert len(results) == 2
        assert {result["id"] for result in results} == {1, 3}
    
    @pytest.mark.asyncio
    async def test_process_batches_async(self):
        """Test async batch processing"""
        async def async_processing_function(batch):
            await asyncio.sleep(0.01)  # Simulate async work
            return [{"processed": True, **item} for item in batch]
        
        batches = [
            [{"id": 1}, {"id": 2}],
            [{"id": 3}]
        ]
        
        results = await self.processor.process_batches_async(
            batches, async_processing_function
        )
        
        assert len(results) == 3
        assert all(result["processed"] is True for result in results)
    
    def test_invalid_processing_strategy(self):
        """Test invalid processing strategy"""
        with pytest.raises(ValueError):
            processor = ParallelProcessor(processing_strategy="invalid")
            processor.process_batches_parallel([], lambda x: x)


class TestCacheManager:
    """Test caching functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_cache_dir = Path("/tmp/test_cache")
        self.cache_manager = CacheManager(
            cache_dir=self.temp_cache_dir,
            max_cache_size_mb=10
        )
    
    def teardown_method(self):
        """Cleanup after tests"""
        import shutil
        if self.temp_cache_dir.exists():
            shutil.rmtree(self.temp_cache_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test cache manager initialization"""
        assert self.cache_manager.cache_dir == self.temp_cache_dir
        assert self.cache_manager.max_cache_size_mb == 10
        assert self.cache_manager.cache_stats == {"hits": 0, "misses": 0}
    
    def test_get_cache_key_dict(self):
        """Test cache key generation for dictionary"""
        test_dict = {"key": "value", "number": 42}
        key1 = self.cache_manager.get_cache_key(test_dict)
        key2 = self.cache_manager.get_cache_key(test_dict)
        
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
        assert key1 == key2  # Same input should produce same key
        
        # Different dict should produce different key
        different_dict = {"key": "different", "number": 42}
        key3 = self.cache_manager.get_cache_key(different_dict)
        assert key1 != key3
    
    def test_get_cache_key_list(self):
        """Test cache key generation for list"""
        test_list = [1, 2, 3, 4, 5]
        key = self.cache_manager.get_cache_key(test_list)
        
        assert isinstance(key, str)
        assert len(key) == 32
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_get_cached_miss(self, mock_exists):
        """Test cache miss"""
        result = self.cache_manager.get_cached("nonexistent_key")
        
        assert result is None
        assert self.cache_manager.cache_stats["misses"] == 1
        assert self.cache_manager.cache_stats["hits"] == 0
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, 
           read_data='{"cached": "data"}')
    @patch('json.load', return_value={"cached": "data"})
    def test_get_cached_hit(self, mock_json_load, mock_file, mock_exists):
        """Test cache hit"""
        result = self.cache_manager.get_cached("existing_key")
        
        assert result == {"cached": "data"}
        assert self.cache_manager.cache_stats["hits"] == 1
        assert self.cache_manager.cache_stats["misses"] == 0
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch.object(CacheManager, '_cleanup_cache_if_needed')
    def test_set_cached(self, mock_cleanup, mock_json_dump, mock_file):
        """Test cache setting"""
        test_data = {"test": "data"}
        
        self.cache_manager.set_cached("test_key", test_data)
        
        mock_file.assert_called()
        mock_json_dump.assert_called_with(test_data, mock_file.return_value.__enter__(), ensure_ascii=False)
        mock_cleanup.assert_called_once()
    
    def test_get_cache_stats_empty(self):
        """Test cache statistics when empty"""
        with patch('pathlib.Path.glob', return_value=[]):
            stats = self.cache_manager.get_cache_stats()
            
            assert stats["hit_rate"] == 0.0
            assert stats["total_hits"] == 0
            assert stats["total_misses"] == 0
            assert stats["cache_size_mb"] == 0.0
            assert stats["cache_file_count"] == 0
    
    def test_get_cache_stats_with_data(self):
        """Test cache statistics with data"""
        # Simulate some cache activity
        self.cache_manager.cache_stats["hits"] = 8
        self.cache_manager.cache_stats["misses"] = 2
        
        # Mock file system
        mock_files = [MagicMock(), MagicMock()]
        mock_files[0].stat.return_value.st_size = 1024  # 1KB
        mock_files[1].stat.return_value.st_size = 2048  # 2KB
        
        with patch('pathlib.Path.glob', return_value=mock_files):
            stats = self.cache_manager.get_cache_stats()
            
            assert stats["hit_rate"] == 0.8  # 8/(8+2)
            assert stats["total_hits"] == 8
            assert stats["total_misses"] == 2
            assert stats["cache_size_mb"] == 3.0 / 1024  # 3KB in MB
            assert stats["cache_file_count"] == 2


class TestResourceOptimizer:
    """Test resource optimization functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = ResourceOptimizer()
    
    @patch('resource.getrlimit', return_value=(1024*1024*1024, 2048*1024*1024))
    @patch('resource.setrlimit')
    @patch('gc.get_threshold', return_value=(700, 10, 10))
    @patch('gc.set_threshold')
    def test_optimized_resources_context(self, mock_set_threshold, mock_get_threshold,
                                       mock_setrlimit, mock_getrlimit):
        """Test optimized resources context manager"""
        
        with self.optimizer.optimized_resources(max_memory_mb=2048, max_cpu_percent=80):
            # Inside context - verify limits were set
            mock_setrlimit.assert_called()
            mock_set_threshold.assert_called_with(700, 10, 10)
        
        # After context - verify cleanup
        assert mock_set_threshold.call_count == 2  # Once for setup, once for cleanup


class TestOptimizedDatasetProcessor:
    """Test complete optimized dataset processor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = OptimizedDatasetProcessor(
            max_memory_mb=1024,
            max_workers=2,
            processing_strategy="thread",
            cache_enabled=True,
            batch_size=10
        )
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor.max_memory_mb == 1024
        assert self.processor.batch_size == 10
        assert self.processor.memory_monitor is not None
        assert self.processor.streaming_processor is not None
        assert self.processor.parallel_processor is not None
        assert self.processor.cache_manager is not None
        assert self.processor.resource_optimizer is not None
    
    @patch.object(OptimizedDatasetProcessor, '_process_batches_with_caching')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('json.dumps', return_value='{"test": "output"}')
    def test_process_large_dataset_success(self, mock_json_dumps, mock_mkdir, 
                                         mock_file, mock_process_batches):
        """Test successful dataset processing"""
        # Setup mocks
        mock_process_batches.return_value = [{"processed": True, "id": 1}]
        
        # Mock streaming processor
        self.processor.streaming_processor.stream_records = Mock(
            return_value=iter([{"id": 1}, {"id": 2}, {"id": 3}])
        )
        
        # Mock resource optimizer context
        self.processor.resource_optimizer.optimized_resources = Mock()
        self.processor.resource_optimizer.optimized_resources.__enter__ = Mock()
        self.processor.resource_optimizer.optimized_resources.__exit__ = Mock()
        
        # Mock memory monitor
        self.processor.memory_monitor.start_monitoring = Mock()
        self.processor.memory_monitor.stop_monitoring = Mock()
        self.processor.memory_monitor.peak_memory_mb = 512.0
        self.processor.memory_monitor.get_average_memory_mb = Mock(return_value=256.0)
        
        def mock_processing_function(batch):
            return [{"processed": True, **item} for item in batch]
        
        input_files = [Path("test1.jsonl"), Path("test2.jsonl")]
        output_path = Path("output.jsonl")
        
        # Mock psutil
        with patch('psutil.Process') as mock_process:
            mock_process_instance = MagicMock()
            mock_process_instance.cpu_percent.return_value = 50.0
            mock_process.return_value = mock_process_instance
            
            metrics = self.processor.process_large_dataset(
                input_files,
                mock_processing_function,
                output_path
            )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_records_processed > 0
        assert metrics.records_per_second >= 0
        
        # Verify monitoring lifecycle
        self.processor.memory_monitor.start_monitoring.assert_called_once()
        self.processor.memory_monitor.stop_monitoring.assert_called_once()
    
    def test_process_batches_with_caching_no_cache(self):
        """Test batch processing without caching"""
        self.processor.cache_manager = None
        
        batches = [[{"id": 1}], [{"id": 2}]]
        
        def mock_processing_function(batch):
            return [{"processed": True, **item} for item in batch]
        
        with patch.object(self.processor.parallel_processor, 'process_batches_parallel',
                         return_value=[{"processed": True, "id": 1}, {"processed": True, "id": 2}]):
            
            results = self.processor._process_batches_with_caching(
                batches, mock_processing_function
            )
        
        assert len(results) == 2
        assert all(result["processed"] is True for result in results)
    
    def test_benchmark_processing_strategies(self):
        """Test benchmarking different processing strategies"""
        sample_batches = [[{"id": 1}], [{"id": 2}]]
        
        def mock_processing_function(batch):
            time.sleep(0.01)  # Simulate work
            return batch
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_thread_executor, \
             patch('concurrent.futures.ProcessPoolExecutor') as mock_process_executor:
            
            # Mock executor behavior
            mock_executor_instance = MagicMock()
            mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = Mock(return_value=None)
            mock_executor_instance.submit = Mock(return_value=Future())
            
            mock_thread_executor.return_value = mock_executor_instance
            mock_process_executor.return_value = mock_executor_instance
            
            # Mock futures
            mock_future = MagicMock()
            mock_future.result.return_value = [{"id": 1}]
            
            with patch('concurrent.futures.as_completed', return_value=[mock_future]):
                results = self.processor.benchmark_processing_strategies(
                    sample_batches, mock_processing_function
                )
            
            assert "thread" in results
            assert "process" in results
            assert all(isinstance(time_val, float) for time_val in results.values())
    
    def test_optimize_batch_size(self):
        """Test batch size optimization"""
        sample_records = [{"id": i} for i in range(100)]
        
        def mock_processing_function(batch):
            return batch
        
        with patch.object(self.processor.parallel_processor, 'process_batches_parallel',
                         return_value=sample_records), \
             patch.object(self.processor.memory_monitor, 'memory_limit_context'):
            
            optimal_size = self.processor.optimize_batch_size(
                sample_records, mock_processing_function
            )
        
        assert isinstance(optimal_size, int)
        assert optimal_size > 0


class TestAnalyzePerformanceBottlenecks:
    """Test performance bottleneck analysis"""
    
    def test_analyze_high_memory_bottleneck(self):
        """Test detection of high memory usage"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=50.0,
            peak_memory_usage_mb=4096.0,  # High memory
            average_memory_usage_mb=2048.0,
            cpu_utilization_percent=60.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=0
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "high_memory_usage" in analysis["bottlenecks"]
        assert any("reducing batch size" in rec for rec in analysis["recommendations"])
    
    def test_analyze_low_processing_speed(self):
        """Test detection of low processing speed"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=50.0,  # Low speed
            peak_memory_usage_mb=1024.0,
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=30.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=0
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "low_processing_speed" in analysis["bottlenecks"]
        assert any("increasing parallelization" in rec for rec in analysis["recommendations"])
    
    def test_analyze_cpu_bound(self):
        """Test detection of CPU-bound processing"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=200.0,
            peak_memory_usage_mb=1024.0,
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=98.0,  # Very high CPU
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=0
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "cpu_bound" in analysis["bottlenecks"]
        assert any("process-based parallelization" in rec for rec in analysis["recommendations"])
    
    def test_analyze_cpu_underutilized(self):
        """Test detection of underutilized CPU"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=200.0,
            peak_memory_usage_mb=1024.0,
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=30.0,  # Low CPU usage
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=0
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "cpu_underutilized" in analysis["bottlenecks"]
        assert any("increasing worker threads" in rec for rec in analysis["recommendations"])
    
    def test_analyze_poor_cache_performance(self):
        """Test detection of poor cache performance"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=200.0,
            peak_memory_usage_mb=1024.0,
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=60.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.05,  # Poor cache performance
            error_count=0
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "poor_cache_performance" in analysis["bottlenecks"]
        assert any("caching strategy" in rec for rec in analysis["recommendations"])
    
    def test_analyze_processing_errors(self):
        """Test detection of processing errors"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=200.0,
            peak_memory_usage_mb=1024.0,
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=60.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.8,
            error_count=10  # Errors present
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        assert "processing_errors" in analysis["bottlenecks"]
        assert any("10 processing errors" in rec for rec in analysis["recommendations"])
    
    def test_analyze_optimal_performance(self):
        """Test analysis of optimal performance metrics"""
        metrics = PerformanceMetrics(
            processing_start_time=1000.0,
            processing_end_time=1100.0,
            total_records_processed=100,
            records_per_second=500.0,  # Good speed
            peak_memory_usage_mb=1024.0,  # Reasonable memory
            average_memory_usage_mb=512.0,
            cpu_utilization_percent=75.0,  # Good CPU usage
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            cache_hit_rate=0.9,  # Good cache performance
            error_count=0  # No errors
        )
        
        analysis = analyze_performance_bottlenecks(metrics)
        
        # Should have no bottlenecks or minimal bottlenecks
        assert len(analysis["bottlenecks"]) <= 1
        assert analysis["efficiency_score"] > 0.5