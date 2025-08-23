"""
Unit tests for performance_monitor.py
Tests M3 MacBook Pro performance monitoring functionality
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
import time
import threading
import subprocess
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from performance_monitor import (
    M3PerformanceMetrics, M3PerformanceMonitor, M3Benchmarks
)


class TestM3PerformanceMetrics(unittest.TestCase):
    """Test M3PerformanceMetrics dataclass"""
    
    def test_metrics_creation_with_defaults(self):
        """Test creating M3PerformanceMetrics with default values"""
        metrics = M3PerformanceMetrics(
            timestamp="2023-01-01T00:00:00",
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        
        self.assertEqual(metrics.timestamp, "2023-01-01T00:00:00")
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_used_gb, 32.0)
        self.assertEqual(metrics.memory_percent, 25.0)
        self.assertEqual(metrics.gpu_utilization, 30.0)
        self.assertEqual(metrics.unified_memory_pressure, "normal")
        self.assertIsNone(metrics.tokens_per_second)
        self.assertIsNone(metrics.inference_latency_ms)
        self.assertEqual(metrics.concurrent_requests, 0)
        self.assertEqual(metrics.model_memory_usage_gb, 0.0)
        self.assertIsNone(metrics.memory_bandwidth_gbps)
    
    def test_metrics_creation_with_all_values(self):
        """Test creating M3PerformanceMetrics with all values"""
        metrics = M3PerformanceMetrics(
            timestamp="2023-01-01T00:00:00",
            cpu_percent=75.0,
            memory_used_gb=64.0,
            memory_percent=50.0,
            gpu_utilization=80.0,
            unified_memory_pressure="warning",
            tokens_per_second=150.0,
            inference_latency_ms=25.0,
            concurrent_requests=5,
            model_memory_usage_gb=16.0,
            memory_bandwidth_gbps=200.0
        )
        
        self.assertEqual(metrics.tokens_per_second, 150.0)
        self.assertEqual(metrics.inference_latency_ms, 25.0)
        self.assertEqual(metrics.concurrent_requests, 5)
        self.assertEqual(metrics.model_memory_usage_gb, 16.0)
        self.assertEqual(metrics.memory_bandwidth_gbps, 200.0)


class TestM3PerformanceMonitor(unittest.TestCase):
    """Test M3PerformanceMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = M3PerformanceMonitor(sampling_interval=0.1)
    
    def tearDown(self):
        """Clean up after tests"""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test M3PerformanceMonitor initialization"""
        self.assertEqual(self.monitor.sampling_interval, 0.1)
        self.assertEqual(self.monitor.metrics_history, [])
        self.assertFalse(self.monitor.monitoring)
        self.assertIsNone(self.monitor.monitor_thread)
    
    @patch('subprocess.run')
    def test_get_memory_pressure_normal(self, mock_run):
        """Test get_memory_pressure with normal pressure"""
        mock_run.return_value = MagicMock(stdout="System-wide memory free percentage: 85%\nThe system has normal memory pressure.")
        
        result = self.monitor.get_memory_pressure()
        self.assertEqual(result, "normal")
        mock_run.assert_called_once_with(['memory_pressure'], capture_output=True, text=True, timeout=5)
    
    @patch('subprocess.run')
    def test_get_memory_pressure_warning(self, mock_run):
        """Test get_memory_pressure with warning pressure"""
        mock_run.return_value = MagicMock(stdout="System-wide memory free percentage: 15%\nThe system has warn memory pressure.")
        
        result = self.monitor.get_memory_pressure()
        self.assertEqual(result, "warning")
    
    @patch('subprocess.run')
    def test_get_memory_pressure_critical(self, mock_run):
        """Test get_memory_pressure with critical pressure"""
        mock_run.return_value = MagicMock(stdout="System-wide memory free percentage: 5%\nThe system has critical memory pressure.")
        
        result = self.monitor.get_memory_pressure()
        self.assertEqual(result, "critical")
    
    @patch('subprocess.run')
    def test_get_memory_pressure_timeout(self, mock_run):
        """Test get_memory_pressure with timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired(['memory_pressure'], 5)
        
        result = self.monitor.get_memory_pressure()
        self.assertEqual(result, "unknown")
    
    @patch('subprocess.run')
    def test_get_memory_pressure_error(self, mock_run):
        """Test get_memory_pressure with subprocess error"""
        mock_run.side_effect = subprocess.CalledProcessError(1, ['memory_pressure'])
        
        result = self.monitor.get_memory_pressure()
        self.assertEqual(result, "unknown")
    
    @patch('subprocess.run')
    def test_get_gpu_utilization_success(self, mock_run):
        """Test get_gpu_utilization with successful parsing"""
        mock_run.return_value = MagicMock(stdout="GPU Activity: 75.5%\nOther data")
        
        result = self.monitor.get_gpu_utilization()
        self.assertEqual(result, 75.5)
    
    @patch('subprocess.run')
    def test_get_gpu_utilization_no_match(self, mock_run):
        """Test get_gpu_utilization with no percentage match"""
        mock_run.return_value = MagicMock(stdout="No GPU data available")
        
        result = self.monitor.get_gpu_utilization()
        self.assertEqual(result, 0.0)
    
    @patch('subprocess.run')
    def test_get_gpu_utilization_error(self, mock_run):
        """Test get_gpu_utilization with subprocess error"""
        mock_run.side_effect = Exception("Command failed")
        
        result = self.monitor.get_gpu_utilization()
        self.assertEqual(result, 0.0)
    
    @patch('subprocess.run')
    def test_get_memory_bandwidth_success(self, mock_run):
        """Test get_memory_bandwidth with successful calculation"""
        mock_run.return_value = MagicMock(stdout="""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             2000.
Pages inactive:                           1500.
Pages speculative:                         500.
Pages throttled:                             0.
Pages wired down:                         3000.
Pages purgeable:                           200.""")
        
        result = self.monitor.get_memory_bandwidth()
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 400.0)
    
    @patch('subprocess.run')
    def test_get_memory_bandwidth_error(self, mock_run):
        """Test get_memory_bandwidth with subprocess error"""
        mock_run.side_effect = Exception("Command failed")
        
        result = self.monitor.get_memory_bandwidth()
        self.assertIsNone(result)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch.object(M3PerformanceMonitor, 'get_gpu_utilization')
    @patch.object(M3PerformanceMonitor, 'get_memory_pressure')
    @patch.object(M3PerformanceMonitor, 'get_memory_bandwidth')
    def test_collect_metrics(self, mock_bandwidth, mock_pressure, mock_gpu, mock_cpu, mock_memory):
        """Test collect_metrics method"""
        # Mock return values
        mock_memory.return_value = MagicMock(used=64*1024**3, percent=50.0)  # 64GB used
        mock_cpu.return_value = 75.0
        mock_gpu.return_value = 60.0
        mock_pressure.return_value = "normal"
        mock_bandwidth.return_value = 150.0
        
        metrics = self.monitor.collect_metrics()
        
        self.assertIsInstance(metrics, M3PerformanceMetrics)
        self.assertEqual(metrics.cpu_percent, 75.0)
        self.assertEqual(metrics.memory_used_gb, 64.0)
        self.assertEqual(metrics.memory_percent, 50.0)
        self.assertEqual(metrics.gpu_utilization, 60.0)
        self.assertEqual(metrics.unified_memory_pressure, "normal")
        self.assertEqual(metrics.memory_bandwidth_gbps, 150.0)
        self.assertIsInstance(metrics.timestamp, str)
    
    def test_start_monitoring(self):
        """Test start_monitoring method"""
        self.assertFalse(self.monitor.monitoring)
        
        self.monitor.start_monitoring()
        
        self.assertTrue(self.monitor.monitoring)
        self.assertIsInstance(self.monitor.monitor_thread, threading.Thread)
        self.assertTrue(self.monitor.monitor_thread.daemon)
        
        # Clean up
        self.monitor.stop_monitoring()
    
    def test_start_monitoring_already_running(self):
        """Test start_monitoring when already running"""
        self.monitor.monitoring = True
        original_thread = self.monitor.monitor_thread
        
        self.monitor.start_monitoring()
        
        # Should not create new thread
        self.assertEqual(self.monitor.monitor_thread, original_thread)
    
    def test_stop_monitoring(self):
        """Test stop_monitoring method"""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring)
        
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.monitoring)
    
    @patch.object(M3PerformanceMonitor, 'collect_metrics')
    def test_monitor_loop(self, mock_collect):
        """Test _monitor_loop method"""
        # Create mock metrics
        mock_metrics = M3PerformanceMetrics(
            timestamp="2023-01-01T00:00:00",
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        mock_collect.return_value = mock_metrics
        
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect a few metrics
        self.monitor.stop_monitoring()
        
        # Check that metrics were collected
        self.assertGreater(len(self.monitor.metrics_history), 0)
        self.assertIsInstance(self.monitor.metrics_history[0], M3PerformanceMetrics)
    
    @patch('time.sleep')
    @patch.object(M3PerformanceMonitor, 'collect_metrics')
    def test_monitor_loop_history_limit(self, mock_collect, mock_sleep):
        """Test _monitor_loop history limit"""
        mock_metrics = M3PerformanceMetrics(
            timestamp="2023-01-01T00:00:00",
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        mock_collect.return_value = mock_metrics
        
        # Add more than 1000 metrics
        self.monitor.metrics_history = [mock_metrics] * 1005
        
        # Mock the monitoring flag to run only once
        call_count = 0
        def side_effect(sleep_time):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Run once, then stop
                self.monitor.monitoring = False
        
        mock_sleep.side_effect = side_effect
        
        # Start monitoring briefly to trigger the history limit logic
        self.monitor.monitoring = True
        self.monitor._monitor_loop()
        
        # Should be limited to 1000
        self.assertEqual(len(self.monitor.metrics_history), 1000)
    
    def test_get_recent_metrics(self):
        """Test get_recent_metrics method"""
        # Create test metrics with different timestamps
        now = datetime.now()
        old_time = now - timedelta(minutes=10)
        recent_time = now - timedelta(minutes=2)
        
        old_metrics = M3PerformanceMetrics(
            timestamp=old_time.isoformat(),
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        
        recent_metrics = M3PerformanceMetrics(
            timestamp=recent_time.isoformat(),
            cpu_percent=60.0,
            memory_used_gb=40.0,
            memory_percent=30.0,
            gpu_utilization=40.0,
            unified_memory_pressure="normal"
        )
        
        self.monitor.metrics_history = [old_metrics, recent_metrics]
        
        # Get recent metrics (last 5 minutes)
        result = self.monitor.get_recent_metrics(5)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].cpu_percent, 60.0)
    
    def test_get_recent_metrics_empty_history(self):
        """Test get_recent_metrics with empty history"""
        result = self.monitor.get_recent_metrics(5)
        self.assertEqual(result, [])
    
    def test_get_performance_summary_no_metrics(self):
        """Test get_performance_summary with no metrics"""
        result = self.monitor.get_performance_summary()
        self.assertEqual(result, {"error": "No metrics available"})
    
    def test_get_performance_summary_no_recent_metrics(self):
        """Test get_performance_summary with no recent metrics"""
        # Add old metrics
        old_time = datetime.now() - timedelta(minutes=30)
        old_metrics = M3PerformanceMetrics(
            timestamp=old_time.isoformat(),
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        self.monitor.metrics_history = [old_metrics]
        
        result = self.monitor.get_performance_summary()
        
        # Should use last 10 metrics instead
        self.assertIn("averages", result)
        self.assertEqual(result["averages"]["cpu_percent"], 50.0)
    
    def test_get_performance_summary_with_metrics(self):
        """Test get_performance_summary with valid metrics"""
        # Create test metrics
        now = datetime.now()
        metrics_list = []
        
        for i in range(5):
            metrics = M3PerformanceMetrics(
                timestamp=(now - timedelta(minutes=i)).isoformat(),
                cpu_percent=50.0 + i * 10,
                memory_used_gb=32.0 + i * 8,
                memory_percent=25.0 + i * 5,
                gpu_utilization=30.0 + i * 5,
                unified_memory_pressure="normal",
                memory_bandwidth_gbps=100.0 + i * 10
            )
            metrics_list.append(metrics)
        
        self.monitor.metrics_history = metrics_list
        
        result = self.monitor.get_performance_summary()
        
        # Verify structure
        self.assertIn("timestamp", result)
        self.assertIn("averages", result)
        self.assertIn("unified_memory", result)
        self.assertIn("m3_optimization", result)
        self.assertIn("performance_score", result)
        
        # Verify averages
        self.assertEqual(result["averages"]["cpu_percent"], 70.0)  # Average of 50,60,70,80,90
        self.assertEqual(result["averages"]["memory_used_gb"], 48.0)  # Average of 32,40,48,56,64
        self.assertEqual(result["unified_memory"]["pressure_status"], "normal")
        
        # Verify recommendations
        self.assertIn("recommendations", result["m3_optimization"])
        self.assertIsInstance(result["m3_optimization"]["recommendations"], list)
    
    def test_get_performance_summary_with_high_usage(self):
        """Test get_performance_summary with high resource usage"""
        now = datetime.now()
        high_usage_metrics = M3PerformanceMetrics(
            timestamp=now.isoformat(),
            cpu_percent=95.0,
            memory_used_gb=120.0,
            memory_percent=95.0,
            gpu_utilization=5.0,
            unified_memory_pressure="critical",
            memory_bandwidth_gbps=30.0
        )
        
        self.monitor.metrics_history = [high_usage_metrics]
        
        result = self.monitor.get_performance_summary()
        
        recommendations = result["m3_optimization"]["recommendations"]
        
        # Should have recommendations for high usage
        self.assertTrue(any("High memory usage" in rec for rec in recommendations))
        self.assertTrue(any("High CPU usage" in rec for rec in recommendations))
        self.assertTrue(any("Low GPU utilization" in rec for rec in recommendations))
        self.assertTrue(any("Memory pressure detected" in rec for rec in recommendations))
        self.assertTrue(any("Low memory bandwidth" in rec for rec in recommendations))
    
    def test_calculate_performance_score_perfect(self):
        """Test _calculate_performance_score with perfect conditions"""
        score = self.monitor._calculate_performance_score(50.0, 50.0, 50.0, "normal")
        self.assertEqual(score, 100)  # Base 100 + 10 for good GPU utilization
    
    def test_calculate_performance_score_high_cpu(self):
        """Test _calculate_performance_score with high CPU usage"""
        score = self.monitor._calculate_performance_score(95.0, 50.0, 50.0, "normal")
        self.assertEqual(score, 90)  # 100 + 10 (GPU) - 20 (high CPU)
    
    def test_calculate_performance_score_high_memory(self):
        """Test _calculate_performance_score with high memory usage"""
        score = self.monitor._calculate_performance_score(50.0, 95.0, 50.0, "normal")
        self.assertEqual(score, 85)  # 100 + 10 (GPU) - 25 (high memory)
    
    def test_calculate_performance_score_critical_pressure(self):
        """Test _calculate_performance_score with critical memory pressure"""
        score = self.monitor._calculate_performance_score(50.0, 50.0, 50.0, "critical")
        self.assertEqual(score, 80)  # 100 + 10 (GPU) - 30 (critical pressure)
    
    def test_calculate_performance_score_low_gpu(self):
        """Test _calculate_performance_score with low GPU utilization"""
        score = self.monitor._calculate_performance_score(50.0, 50.0, 5.0, "normal")
        self.assertEqual(score, 90)  # 100 - 10 (low GPU)
    
    def test_calculate_performance_score_minimum(self):
        """Test _calculate_performance_score minimum bound"""
        score = self.monitor._calculate_performance_score(95.0, 95.0, 5.0, "critical")
        # Score calculation: 100 - 20 (high CPU) - 25 (high memory) - 10 (low GPU) - 30 (critical) = 15
        self.assertEqual(score, 15)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_export_metrics(self, mock_json_dump, mock_file):
        """Test export_metrics method"""
        # Add test metrics
        test_metrics = M3PerformanceMetrics(
            timestamp="2023-01-01T00:00:00",
            cpu_percent=50.0,
            memory_used_gb=32.0,
            memory_percent=25.0,
            gpu_utilization=30.0,
            unified_memory_pressure="normal"
        )
        self.monitor.metrics_history = [test_metrics]
        
        self.monitor.export_metrics("test_metrics.json")
        
        # Verify file operations
        mock_file.assert_called_once_with("test_metrics.json", 'w')
        mock_json_dump.assert_called_once()
        
        # Verify data structure
        call_args = mock_json_dump.call_args[0]
        data = call_args[0]
        
        self.assertIn("export_time", data)
        self.assertIn("system_info", data)
        self.assertIn("metrics", data)
        self.assertEqual(len(data["metrics"]), 1)


class TestM3Benchmarks(unittest.TestCase):
    """Test M3Benchmarks class"""
    
    @patch('numpy.random.rand')
    @patch('numpy.dot')
    @patch('time.time')
    def test_benchmark_memory_bandwidth_success(self, mock_time, mock_dot, mock_rand):
        """Test benchmark_memory_bandwidth with successful execution"""
        # Mock numpy operations
        mock_array = MagicMock()
        mock_rand.return_value = mock_array
        mock_dot.return_value = mock_array
        
        # Mock timing
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        result = M3Benchmarks.benchmark_memory_bandwidth()
        
        self.assertIsInstance(result, dict)
        self.assertIn(1000, result)
        self.assertIn("duration_seconds", result[1000])
        self.assertIn("data_size_gb", result[1000])
        self.assertIn("bandwidth_gbps", result[1000])
    
    @patch('numpy.random.rand')
    def test_benchmark_memory_bandwidth_no_numpy(self, mock_rand):
        """Test benchmark_memory_bandwidth without numpy"""
        mock_rand.side_effect = ImportError("No module named 'numpy'")
        
        result = M3Benchmarks.benchmark_memory_bandwidth()
        
        self.assertEqual(result, {"error": "NumPy not available for benchmarking"})
    
    def test_benchmark_inference_performance(self):
        """Test benchmark_inference_performance method"""
        result = M3Benchmarks.benchmark_inference_performance()
        
        self.assertIsInstance(result, dict)
        self.assertIn("note", result)
        self.assertIn("recommended_tests", result)
        self.assertIsInstance(result["recommended_tests"], list)
        self.assertEqual(len(result["recommended_tests"]), 4)


if __name__ == '__main__':
    unittest.main()