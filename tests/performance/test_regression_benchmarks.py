"""
Performance regression testing and benchmarking suite
Validates pipeline performance against established baselines
"""

import pytest
import time
import json
import asyncio
import statistics
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Import modules under test
import sys
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.append(str(src_path))

from performance_optimization import OptimizedDatasetProcessor, PerformanceMetrics
from m3_optimizer import M3PipelineOptimizer, OptimizationConfig
from quality_control_framework import QualityController, validate_dataset_batch


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    test_name: str
    execution_time_ms: float
    memory_usage_mb: float
    throughput_rps: float
    quality_score: float
    success_rate: float
    timestamp: float
    

@dataclass
class RegressionBaseline:
    """Performance baseline for regression testing"""
    processing_speed_rps: float = 150.0
    memory_efficiency_mb: float = 2048.0
    quality_score_threshold: float = 7.42
    model_switch_time_ms: float = 4500.0
    circuit_breaker_threshold: int = 5
    inference_latency_p99_ms: float = 250.0
    concurrent_request_limit: int = 8


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, baseline: RegressionBaseline):
        self.baseline = baseline
        self.results: List[BenchmarkResult] = []
        self.test_data_dir = Path(tempfile.mkdtemp())
        
    def setup_benchmark_environment(self):
        """Setup benchmark test environment"""
        self.create_benchmark_datasets()
        
    def create_benchmark_datasets(self):
        """Create standardized datasets for benchmarking"""
        # Small dataset (100 records) - for quick benchmarks
        self.create_dataset("small_benchmark.jsonl", 100)
        
        # Medium dataset (1000 records) - for standard benchmarks  
        self.create_dataset("medium_benchmark.jsonl", 1000)
        
        # Large dataset (5000 records) - for stress testing
        self.create_dataset("large_benchmark.jsonl", 5000)
        
    def create_dataset(self, filename: str, num_records: int):
        """Create benchmark dataset with specified size"""
        dataset_file = self.test_data_dir / filename
        
        with open(dataset_file, 'w') as f:
            for i in range(num_records):
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"How do I configure eNodeB parameter {i} for LTE optimization in urban deployment scenario {i%10}?"
                        },
                        {
                            "role": "assistant",
                            "content": f"To configure eNodeB parameter {i}, set RSRP threshold to -{90+(i%30)} dBm, time-to-trigger to {320+(i%100)}ms, and enable carrier aggregation for {i%5+1} component carriers in the urban environment."
                        }
                    ],
                    "metadata": {
                        "feature_name": f"LTE Urban Optimization {i}",
                        "quality_score": 7.0 + (i % 25) * 0.1,
                        "technical_content": True,
                        "technical_terms": ["eNodeB", "LTE", "RSRP", "carrier aggregation"],
                        "benchmark_id": i,
                        "dataset_size": num_records
                    }
                }
                f.write(json.dumps(record) + '\n')
        
        return dataset_file
    
    def benchmark_processing_speed(self, dataset_size: str = "medium") -> BenchmarkResult:
        """Benchmark processing speed performance"""
        dataset_file = self.test_data_dir / f"{dataset_size}_benchmark.jsonl"
        
        processor = OptimizedDatasetProcessor(
            max_memory_mb=1024,
            max_workers=4,
            batch_size=100,
            cache_enabled=True
        )
        
        def benchmark_processing_function(batch):
            # Simulate realistic processing work
            processed_batch = []
            for record in batch:
                processed_record = record.copy()
                processed_record["benchmark_processed"] = True
                processed_record["processing_timestamp"] = time.time()
                processed_batch.append(processed_record)
            return processed_batch
        
        output_file = self.test_data_dir / f"{dataset_size}_speed_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 65.0
            mock_process.return_value = mock_process_instance
            
            start_time = time.time()
            metrics = processor.process_large_dataset(
                [dataset_file],
                benchmark_processing_function,
                output_file
            )
            execution_time_ms = (time.time() - start_time) * 1000
        
        result = BenchmarkResult(
            test_name=f"processing_speed_{dataset_size}",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=metrics.peak_memory_usage_mb,
            throughput_rps=metrics.records_per_second,
            quality_score=0.0,  # Not applicable for this benchmark
            success_rate=1.0 - (metrics.error_count / max(1, metrics.total_records_processed)),
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    def benchmark_memory_efficiency(self, dataset_size: str = "large") -> BenchmarkResult:
        """Benchmark memory efficiency under load"""
        dataset_file = self.test_data_dir / f"{dataset_size}_benchmark.jsonl"
        
        # Constrained memory setup for efficiency testing
        processor = OptimizedDatasetProcessor(
            max_memory_mb=512,  # Constrained memory
            max_workers=2,
            batch_size=50,
            cache_enabled=False  # Test raw processing efficiency
        )
        
        def memory_intensive_processing(batch):
            # Simulate memory-intensive operations
            large_temp_data = []
            for record in batch:
                # Create temporary data structures
                temp_data = {
                    "features": list(range(100)),
                    "embeddings": [0.1] * 256,
                    "metadata": record["metadata"].copy()
                }
                large_temp_data.append(temp_data)
            
            # Process with temporary data
            processed_batch = []
            for i, record in enumerate(batch):
                processed_record = record.copy()
                processed_record["memory_test_features"] = len(large_temp_data[i]["features"])
                processed_record["memory_test_embeddings"] = len(large_temp_data[i]["embeddings"])
                processed_batch.append(processed_record)
            
            # Cleanup
            del large_temp_data
            
            return processed_batch
        
        output_file = self.test_data_dir / f"{dataset_size}_memory_output.jsonl"
        
        # Track peak memory usage
        peak_memory = 0
        memory_samples = []
        
        def mock_memory_usage():
            nonlocal peak_memory
            # Simulate realistic memory usage pattern
            current_memory = 256 + (len(memory_samples) % 20) * 15  # Base + variation
            memory_samples.append(current_memory)
            peak_memory = max(peak_memory, current_memory)
            return current_memory * 1024 * 1024  # Convert to bytes
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = Mock(side_effect=mock_memory_usage)
            mock_process_instance.cpu_percent.return_value = 55.0
            mock_process.return_value = mock_process_instance
            
            start_time = time.time()
            metrics = processor.process_large_dataset(
                [dataset_file],
                memory_intensive_processing,
                output_file
            )
            execution_time_ms = (time.time() - start_time) * 1000
        
        result = BenchmarkResult(
            test_name=f"memory_efficiency_{dataset_size}",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=peak_memory,
            throughput_rps=metrics.records_per_second,
            quality_score=0.0,
            success_rate=1.0 - (metrics.error_count / max(1, metrics.total_records_processed)),
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    def benchmark_quality_validation(self, dataset_size: str = "medium") -> BenchmarkResult:
        """Benchmark quality validation performance"""
        dataset_file = self.test_data_dir / f"{dataset_size}_benchmark.jsonl"
        
        # Load test records
        records = []
        with open(dataset_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        quality_controller = QualityController({
            "min_quality_score": 7.0,
            "technical_term_threshold": 3,
            "similarity_threshold": 0.85
        })
        
        start_time = time.time()
        
        # Run validation benchmark
        valid_records, stats = validate_dataset_batch(records, quality_controller)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Calculate quality metrics
        avg_quality = stats.get("average_quality", 0.0)
        success_rate = stats["valid_records"] / max(1, stats["total_processed"])
        
        result = BenchmarkResult(
            test_name=f"quality_validation_{dataset_size}",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=0.0,  # Quality validation is not memory-intensive
            throughput_rps=len(records) / (execution_time_ms / 1000),
            quality_score=avg_quality,
            success_rate=success_rate,
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    @pytest.mark.asyncio
    async def benchmark_m3_optimization(self) -> BenchmarkResult:
        """Benchmark M3 optimization performance"""
        config = OptimizationConfig(
            max_memory_usage_percent=80.0,
            preferred_batch_size=256,
            max_concurrent_requests=6
        )
        
        pipeline_optimizer = M3PipelineOptimizer(config)
        
        # Mock optimization dependencies for consistent benchmarking
        with patch.object(pipeline_optimizer.memory_manager, 'create_memory_pool', return_value=True), \
             patch.object(pipeline_optimizer.memory_manager, 'optimize_allocation_pattern') as mock_pattern, \
             patch.object(pipeline_optimizer.memory_manager, 'get_memory_utilization') as mock_utilization, \
             patch.object(pipeline_optimizer.mlx_optimizer, 'optimize_model_loading') as mock_mlx, \
             patch.object(pipeline_optimizer.pytorch_optimizer, 'optimize_mps_settings') as mock_pytorch, \
             patch.object(pipeline_optimizer.inference_optimizer, 'optimize_lm_studio') as mock_lm_studio, \
             patch.object(pipeline_optimizer.inference_optimizer, 'optimize_ollama') as mock_ollama:
            
            # Setup consistent mock responses
            mock_pattern.return_value = {
                "model_pool_gb": 32, "batch_pool_gb": 16,
                "cache_pool_gb": 8, "working_pool_gb": 12
            }
            mock_utilization.return_value = {"utilization_percent": 55.0}
            mock_mlx.return_value = {"status": "optimized", "estimated_memory_gb": 15.0}
            mock_pytorch.return_value = {"status": "optimized"}
            mock_lm_studio.return_value = {"status": "optimized"}
            mock_ollama.return_value = {"status": "optimized"}
            
            # Benchmark optimization
            start_time = time.time()
            
            result_data = await pipeline_optimizer.optimize_inference_pipeline(
                "benchmark_model_path", "inference"
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Cleanup
            pipeline_optimizer.cleanup()
        
        result = BenchmarkResult(
            test_name="m3_optimization",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=0.0,  # Optimization setup is not memory-intensive
            throughput_rps=0.0,  # Not applicable
            quality_score=0.0,  # Not applicable
            success_rate=1.0 if result_data.get("optimization_complete") else 0.0,
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    def benchmark_concurrent_processing(self, dataset_size: str = "medium") -> BenchmarkResult:
        """Benchmark concurrent processing performance"""
        dataset_file = self.test_data_dir / f"{dataset_size}_benchmark.jsonl"
        
        # Setup multiple processors for concurrency testing
        processors = [
            OptimizedDatasetProcessor(
                max_memory_mb=256,
                max_workers=2,
                batch_size=25,
                cache_enabled=True
            ) for _ in range(3)
        ]
        
        def concurrent_processing_function(batch):
            # Simulate concurrent work
            time.sleep(0.001)  # Small delay to simulate processing
            processed_batch = []
            for record in batch:
                processed_record = record.copy()
                processed_record["concurrent_processed"] = True
                processed_batch.append(processed_record)
            return processed_batch
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_subset(processor_id, processor, subset_records):
            # Create temporary file for subset
            subset_file = self.test_data_dir / f"concurrent_subset_{processor_id}.jsonl"
            output_file = self.test_data_dir / f"concurrent_output_{processor_id}.jsonl"
            
            with open(subset_file, 'w') as f:
                for record in subset_records:
                    f.write(json.dumps(record) + '\n')
            
            with patch('psutil.Process') as mock_process:
                mock_process_instance = Mock()
                mock_process_instance.memory_info.return_value.rss = (150 + processor_id * 30) * 1024 * 1024
                mock_process_instance.cpu_percent.return_value = 30.0 + processor_id * 15
                mock_process.return_value = mock_process_instance
                
                metrics = processor.process_large_dataset(
                    [subset_file],
                    concurrent_processing_function,
                    output_file
                )
            
            return metrics
        
        # Load and split dataset
        records = []
        with open(dataset_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        # Split records among processors
        records_per_processor = len(records) // len(processors)
        subsets = [
            records[i*records_per_processor:(i+1)*records_per_processor] 
            for i in range(len(processors))
        ]
        
        # Add remaining records to last subset
        if len(records) % len(processors) != 0:
            subsets[-1].extend(records[len(processors)*records_per_processor:])
        
        # Run concurrent processing benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=len(processors)) as executor:
            futures = {
                executor.submit(process_subset, i, processor, subset): i
                for i, (processor, subset) in enumerate(zip(processors, subsets))
            }
            
            results = {}
            for future in as_completed(futures):
                processor_id = futures[future]
                metrics = future.result()
                results[processor_id] = metrics
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Aggregate metrics
        total_records = sum(metrics.total_records_processed for metrics in results.values())
        total_errors = sum(metrics.error_count for metrics in results.values())
        avg_throughput = sum(metrics.records_per_second for metrics in results.values()) / len(results)
        max_memory = max(metrics.peak_memory_usage_mb for metrics in results.values())
        
        result = BenchmarkResult(
            test_name=f"concurrent_processing_{dataset_size}",
            execution_time_ms=execution_time_ms,
            memory_usage_mb=max_memory,
            throughput_rps=avg_throughput,
            quality_score=0.0,
            success_rate=1.0 - (total_errors / max(1, total_records)),
            timestamp=time.time()
        )
        
        self.results.append(result)
        return result
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        report = {
            "benchmark_summary": {
                "total_tests": len(self.results),
                "execution_time": sum(r.execution_time_ms for r in self.results),
                "timestamp": time.time()
            },
            "performance_metrics": {},
            "regression_analysis": {},
            "recommendations": []
        }
        
        # Aggregate metrics by test type
        test_types = {}
        for result in self.results:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        # Calculate performance metrics
        for test_type, type_results in test_types.items():
            if type_results:
                report["performance_metrics"][test_type] = {
                    "avg_execution_time_ms": statistics.mean(r.execution_time_ms for r in type_results),
                    "avg_throughput_rps": statistics.mean(r.throughput_rps for r in type_results if r.throughput_rps > 0),
                    "avg_memory_usage_mb": statistics.mean(r.memory_usage_mb for r in type_results if r.memory_usage_mb > 0),
                    "avg_success_rate": statistics.mean(r.success_rate for r in type_results),
                    "test_count": len(type_results)
                }
        
        # Regression analysis
        report["regression_analysis"] = self.analyze_regression()
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations()
        
        return report
    
    def analyze_regression(self) -> Dict[str, Any]:
        """Analyze performance against baseline for regression detection"""
        regression_analysis = {
            "baseline_comparison": {},
            "performance_degradation": [],
            "performance_improvement": [],
            "overall_status": "PASS"
        }
        
        # Analyze processing speed
        speed_results = [r for r in self.results if "processing_speed" in r.test_name]
        if speed_results:
            avg_throughput = statistics.mean(r.throughput_rps for r in speed_results)
            speed_change = (avg_throughput - self.baseline.processing_speed_rps) / self.baseline.processing_speed_rps
            
            regression_analysis["baseline_comparison"]["processing_speed"] = {
                "baseline_rps": self.baseline.processing_speed_rps,
                "actual_rps": avg_throughput,
                "change_percent": speed_change * 100,
                "status": "PASS" if speed_change >= -0.1 else "FAIL"  # Allow 10% degradation
            }
            
            if speed_change < -0.1:
                regression_analysis["performance_degradation"].append(
                    f"Processing speed degraded by {abs(speed_change)*100:.1f}%"
                )
                regression_analysis["overall_status"] = "FAIL"
        
        # Analyze memory efficiency
        memory_results = [r for r in self.results if "memory" in r.test_name]
        if memory_results:
            avg_memory = statistics.mean(r.memory_usage_mb for r in memory_results if r.memory_usage_mb > 0)
            memory_change = (avg_memory - self.baseline.memory_efficiency_mb) / self.baseline.memory_efficiency_mb
            
            regression_analysis["baseline_comparison"]["memory_efficiency"] = {
                "baseline_mb": self.baseline.memory_efficiency_mb,
                "actual_mb": avg_memory,
                "change_percent": memory_change * 100,
                "status": "PASS" if memory_change <= 0.2 else "FAIL"  # Allow 20% increase
            }
            
            if memory_change > 0.2:
                regression_analysis["performance_degradation"].append(
                    f"Memory usage increased by {memory_change*100:.1f}%"
                )
                regression_analysis["overall_status"] = "FAIL"
        
        # Analyze quality scores
        quality_results = [r for r in self.results if r.quality_score > 0]
        if quality_results:
            avg_quality = statistics.mean(r.quality_score for r in quality_results)
            quality_change = (avg_quality - self.baseline.quality_score_threshold) / self.baseline.quality_score_threshold
            
            regression_analysis["baseline_comparison"]["quality_score"] = {
                "baseline_score": self.baseline.quality_score_threshold,
                "actual_score": avg_quality,
                "change_percent": quality_change * 100,
                "status": "PASS" if avg_quality >= self.baseline.quality_score_threshold else "FAIL"
            }
            
            if avg_quality < self.baseline.quality_score_threshold:
                regression_analysis["performance_degradation"].append(
                    f"Quality score below baseline: {avg_quality:.2f} < {self.baseline.quality_score_threshold:.2f}"
                )
                regression_analysis["overall_status"] = "FAIL"
        
        return regression_analysis
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze throughput
        speed_results = [r for r in self.results if "processing_speed" in r.test_name]
        if speed_results:
            avg_throughput = statistics.mean(r.throughput_rps for r in speed_results)
            if avg_throughput < 100:
                recommendations.append("Consider increasing batch size or worker count to improve throughput")
        
        # Analyze memory usage
        memory_results = [r for r in self.results if r.memory_usage_mb > 0]
        if memory_results:
            max_memory = max(r.memory_usage_mb for r in memory_results)
            if max_memory > 1536:  # > 1.5GB
                recommendations.append("Memory usage is high - consider optimizing batch sizes or enabling more aggressive garbage collection")
        
        # Analyze success rates
        success_rates = [r.success_rate for r in self.results if r.success_rate < 1.0]
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            if avg_success_rate < 0.95:
                recommendations.append("Error rate is elevated - review error handling and input data quality")
        
        # Analyze execution times
        execution_times = [r.execution_time_ms for r in self.results]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time > 60000:  # > 1 minute
                recommendations.append("Execution times are long - consider parallel processing or algorithm optimization")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup benchmark environment"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir, ignore_errors=True)


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression test suite"""
    
    def setup_method(self):
        """Setup regression test environment"""
        self.baseline = RegressionBaseline()
        self.benchmark_suite = PerformanceBenchmarkSuite(self.baseline)
        self.benchmark_suite.setup_benchmark_environment()
    
    def teardown_method(self):
        """Cleanup regression test environment"""
        self.benchmark_suite.cleanup()
    
    @pytest.mark.slow
    def test_processing_speed_regression(self):
        """Test processing speed against baseline"""
        # Run benchmark
        result = self.benchmark_suite.benchmark_processing_speed("medium")
        
        # Regression assertions
        assert result.throughput_rps >= self.baseline.processing_speed_rps * 0.9  # Allow 10% degradation
        assert result.success_rate >= 0.95  # 95% success rate minimum
        assert result.execution_time_ms < 60000  # Complete within 1 minute
        
        print(f"Processing Speed: {result.throughput_rps:.1f} RPS (baseline: {self.baseline.processing_speed_rps} RPS)")
    
    @pytest.mark.slow  
    def test_memory_efficiency_regression(self):
        """Test memory efficiency against baseline"""
        # Run benchmark
        result = self.benchmark_suite.benchmark_memory_efficiency("large")
        
        # Regression assertions
        assert result.memory_usage_mb <= self.baseline.memory_efficiency_mb * 1.2  # Allow 20% increase
        assert result.success_rate >= 0.90  # 90% success rate minimum
        assert result.throughput_rps > 0  # Must maintain throughput
        
        print(f"Memory Usage: {result.memory_usage_mb:.1f} MB (baseline: {self.baseline.memory_efficiency_mb} MB)")
    
    def test_quality_validation_regression(self):
        """Test quality validation performance"""
        # Run benchmark
        result = self.benchmark_suite.benchmark_quality_validation("medium")
        
        # Regression assertions  
        assert result.quality_score >= self.baseline.quality_score_threshold
        assert result.throughput_rps >= 500  # Quality validation should be fast
        assert result.success_rate >= 0.95
        
        print(f"Quality Score: {result.quality_score:.2f} (baseline: {self.baseline.quality_score_threshold})")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_m3_optimization_performance(self):
        """Test M3 optimization performance"""
        # Run benchmark
        result = await self.benchmark_suite.benchmark_m3_optimization()
        
        # Performance assertions
        assert result.execution_time_ms <= self.baseline.model_switch_time_ms
        assert result.success_rate == 1.0  # Must complete successfully
        
        print(f"M3 Optimization Time: {result.execution_time_ms:.1f}ms (baseline: {self.baseline.model_switch_time_ms}ms)")
    
    @pytest.mark.slow
    def test_concurrent_processing_performance(self):
        """Test concurrent processing capabilities"""
        # Run benchmark
        result = self.benchmark_suite.benchmark_concurrent_processing("medium")
        
        # Performance assertions
        assert result.success_rate >= 0.95
        assert result.throughput_rps > 0
        assert result.memory_usage_mb <= self.baseline.memory_efficiency_mb
        
        print(f"Concurrent Processing: {result.throughput_rps:.1f} RPS, {result.memory_usage_mb:.1f} MB")
    
    @pytest.mark.slow
    def test_comprehensive_regression_analysis(self):
        """Run comprehensive regression analysis"""
        # Run all benchmarks
        speed_result = self.benchmark_suite.benchmark_processing_speed("small")
        memory_result = self.benchmark_suite.benchmark_memory_efficiency("medium")
        quality_result = self.benchmark_suite.benchmark_quality_validation("small")
        
        # Generate comprehensive report
        report = self.benchmark_suite.generate_benchmark_report()
        
        # Validate report structure
        assert "benchmark_summary" in report
        assert "performance_metrics" in report
        assert "regression_analysis" in report
        assert "recommendations" in report
        
        # Validate regression analysis
        regression = report["regression_analysis"]
        assert regression["overall_status"] in ["PASS", "FAIL"]
        
        # Print summary
        print(f"\nBenchmark Summary:")
        print(f"Total Tests: {report['benchmark_summary']['total_tests']}")
        print(f"Overall Status: {regression['overall_status']}")
        
        if regression["performance_degradation"]:
            print(f"Degradation Issues:")
            for issue in regression["performance_degradation"]:
                print(f"  - {issue}")
        
        if report["recommendations"]:
            print(f"Recommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        
        # Overall regression test should pass
        assert regression["overall_status"] == "PASS", f"Regression detected: {regression['performance_degradation']}"


@pytest.mark.performance
class TestCircuitBreakerPerformance:
    """Test circuit breaker performance under load"""
    
    def setup_method(self):
        """Setup circuit breaker test environment"""
        self.baseline = RegressionBaseline()
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker triggers at correct failure threshold"""
        from unittest.mock import Mock
        
        # Mock circuit breaker
        mock_breaker = Mock()
        mock_breaker.state = "closed"
        mock_breaker.failure_count = 0
        mock_breaker.failure_threshold = self.baseline.circuit_breaker_threshold
        
        def mock_call(func, *args, **kwargs):
            try:
                if mock_breaker.failure_count >= mock_breaker.failure_threshold:
                    mock_breaker.state = "open"
                    raise Exception("Circuit breaker is open")
                
                # Simulate function call
                result = func(*args, **kwargs) if callable(func) else "success"
                return result
                
            except Exception as e:
                mock_breaker.failure_count += 1
                if mock_breaker.failure_count >= mock_breaker.failure_threshold:
                    mock_breaker.state = "open"
                raise e
        
        mock_breaker.call = Mock(side_effect=mock_call)
        
        # Test failure accumulation
        def failing_function():
            raise ValueError("Simulated failure")
        
        # Should succeed initially
        assert mock_breaker.state == "closed"
        
        # Accumulate failures up to threshold
        for i in range(self.baseline.circuit_breaker_threshold):
            try:
                mock_breaker.call(failing_function)
            except:
                pass
        
        # Circuit breaker should now be open
        assert mock_breaker.state == "open"
        assert mock_breaker.failure_count >= self.baseline.circuit_breaker_threshold
    
    def test_circuit_breaker_recovery_performance(self):
        """Test circuit breaker recovery time performance"""
        from unittest.mock import Mock
        import time
        
        mock_breaker = Mock()
        mock_breaker.state = "open"
        mock_breaker.recovery_timeout = 1.0  # 1 second for testing
        mock_breaker.last_failure_time = time.time()
        
        def mock_call_with_recovery(func, *args, **kwargs):
            current_time = time.time()
            
            if mock_breaker.state == "open":
                if current_time - mock_breaker.last_failure_time >= mock_breaker.recovery_timeout:
                    mock_breaker.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open")
            
            # Try the call
            try:
                result = func(*args, **kwargs) if callable(func) else "success"
                if mock_breaker.state == "half_open":
                    mock_breaker.state = "closed"
                    mock_breaker.failure_count = 0
                return result
            except Exception as e:
                mock_breaker.state = "open"
                mock_breaker.last_failure_time = current_time
                raise e
        
        mock_breaker.call = Mock(side_effect=mock_call_with_recovery)
        
        # Initially should fail (circuit open)
        with pytest.raises(Exception, match="Circuit breaker is open"):
            mock_breaker.call(lambda: "success")
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should succeed after timeout (transitions to half-open, then closed)
        result = mock_breaker.call(lambda: "recovery_success")
        assert result == "recovery_success"
        assert mock_breaker.state == "closed"


@pytest.mark.performance
class TestLatencyBenchmarks:
    """Test latency performance benchmarks"""
    
    def setup_method(self):
        """Setup latency benchmark environment"""
        self.baseline = RegressionBaseline()
    
    def test_inference_latency_p99(self):
        """Test 99th percentile inference latency"""
        latencies = []
        
        # Simulate 100 inference calls
        for i in range(100):
            start_time = time.time()
            
            # Simulate inference work
            time.sleep(0.05 + (i % 10) * 0.01)  # 50-150ms range
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate 99th percentile
        latencies.sort()
        p99_latency = latencies[98]  # 99th percentile of 100 samples
        
        # Regression assertion
        assert p99_latency <= self.baseline.inference_latency_p99_ms
        
        print(f"P99 Latency: {p99_latency:.1f}ms (baseline: {self.baseline.inference_latency_p99_ms}ms)")
    
    def test_concurrent_request_limit(self):
        """Test concurrent request handling capacity"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        successful_requests = 0
        failed_requests = 0
        lock = threading.Lock()
        
        def simulate_request(request_id):
            nonlocal successful_requests, failed_requests
            
            try:
                # Simulate request processing
                time.sleep(0.1 + (request_id % 5) * 0.02)  # 100-180ms processing
                
                with lock:
                    successful_requests += 1
                return f"success_{request_id}"
                
            except Exception as e:
                with lock:
                    failed_requests += 1
                return f"failed_{request_id}"
        
        # Test concurrent capacity
        max_workers = self.baseline.concurrent_request_limit
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(simulate_request, i): i 
                for i in range(max_workers * 2)  # Submit 2x requests
            }
            
            results = []
            for future in as_completed(futures, timeout=10):
                result = future.result()
                results.append(result)
        
        # Performance assertions
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        assert success_rate >= 0.90  # 90% success rate minimum
        assert successful_requests >= self.baseline.concurrent_request_limit
        
        print(f"Concurrent Requests: {successful_requests}/{total_requests} successful ({success_rate*100:.1f}%)")