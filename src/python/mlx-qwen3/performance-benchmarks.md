# Performance Benchmarks and Metrics for Qwen3 on M3 Max

## Executive Summary

Comprehensive performance benchmarking framework for Qwen3 model variants running on Apple M3 Max hardware with MLX optimization, providing detailed metrics, target performance goals, and monitoring strategies.

## Hardware Baseline Performance

### M3 Max Specifications
```yaml
Apple M3 Max Performance Characteristics:
  CPU: 12 cores (8 Performance + 4 Efficiency)
  GPU: 38 cores (Metal Performance Shaders)
  Neural Engine: 16-core (15.8 TOPS)
  Memory: Up to 128GB Unified Memory
  Memory Bandwidth: 400 GB/s
  
Theoretical Limits:
  Peak CPU Performance: ~50 GFLOPS
  Peak GPU Performance: ~13.6 TFLOPS (FP16)
  Peak Memory Throughput: 400 GB/s
  Neural Engine Throughput: 15.8 TOPS (INT8)
```

## Model-Specific Performance Targets

### Qwen3-1.7B Performance Benchmarks
```yaml
Qwen3-1.7B-4bit:
  Target Performance:
    Tokens/Second: 80-120 (single batch)
    Tokens/Second: 150-200 (optimal batch)
    First Token Latency: 80-150ms
    Memory Usage: 2.5-4GB
    Power Consumption: 15-25W
    
  Batch Processing:
    Batch Size 1: 80-120 tokens/sec
    Batch Size 4: 200-280 tokens/sec
    Batch Size 8: 300-400 tokens/sec
    Batch Size 16: 400-500 tokens/sec (peak)
    
  Use Case Performance:
    Document Classification: >1000 docs/minute
    Embedding Generation: >2000 chunks/minute
    Quick Responses: <200ms
    Real-time Chat: <100ms

Qwen3-1.7B-8bit:
  Target Performance:
    Tokens/Second: 60-90 (single batch)
    Tokens/Second: 120-180 (optimal batch)
    First Token Latency: 100-180ms
    Memory Usage: 4-6GB
    Power Consumption: 20-30W
```

### Qwen3-7B Performance Benchmarks
```yaml
Qwen3-7B-4bit:
  Target Performance:
    Tokens/Second: 35-55 (single batch)
    Tokens/Second: 70-100 (optimal batch)
    First Token Latency: 200-350ms
    Memory Usage: 6-10GB
    Power Consumption: 25-35W
    
  Batch Processing:
    Batch Size 1: 35-55 tokens/sec
    Batch Size 4: 80-120 tokens/sec
    Batch Size 8: 120-180 tokens/sec
    Batch Size 16: 150-220 tokens/sec (peak)

Qwen3-7B-8bit:
  Target Performance:
    Tokens/Second: 25-40 (single batch)
    Tokens/Second: 50-80 (optimal batch)
    First Token Latency: 250-400ms
    Memory Usage: 10-16GB
    Power Consumption: 30-45W
    
  Use Case Performance:
    Document Summarization: >200 docs/minute
    Code Generation: >50 functions/minute
    RAG Responses: 1-3 seconds
    Structured Extraction: >100 docs/minute
```

### Qwen3-14B Performance Benchmarks
```yaml
Qwen3-14B-8bit:
  Target Performance:
    Tokens/Second: 15-25 (single batch)
    Tokens/Second: 30-50 (optimal batch)
    First Token Latency: 400-600ms
    Memory Usage: 20-32GB
    Power Consumption: 40-60W
    
  Batch Processing:
    Batch Size 1: 15-25 tokens/sec
    Batch Size 4: 35-55 tokens/sec
    Batch Size 8: 50-75 tokens/sec (peak)
    
  Use Case Performance:
    Complex Analysis: >50 docs/minute
    Research Tasks: >20 queries/minute
    High-Quality Generation: >500 words/minute
    Technical Documentation: >25 docs/minute

Qwen3-14B-16bit (Fine-tuning):
  Target Performance:
    Tokens/Second: 8-15 (single batch)
    First Token Latency: 600-1000ms
    Memory Usage: 35-50GB
    Power Consumption: 50-70W
```

## Comprehensive Benchmark Suite

### Performance Testing Framework
```python
import time
import psutil
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics
import json

@dataclass
class BenchmarkResult:
    model_name: str
    test_name: str
    tokens_per_second: float
    first_token_latency_ms: float
    total_time_seconds: float
    memory_usage_gb: float
    power_consumption_w: Optional[float] = None
    error_rate: float = 0.0
    batch_size: int = 1
    sequence_length: int = 0
    timestamp: float = field(default_factory=time.time)

class Qwen3PerformanceBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_prompts = self._load_benchmark_prompts()
        
    def _load_benchmark_prompts(self) -> Dict[str, List[str]]:
        """Load standardized benchmark prompts"""
        return {
            "short_generation": [
                "Write a brief summary of artificial intelligence.",
                "Explain quantum computing in simple terms.",
                "Describe the benefits of renewable energy."
            ],
            "medium_generation": [
                "Write a comprehensive guide to machine learning algorithms, covering supervised, unsupervised, and reinforcement learning approaches.",
                "Create a detailed analysis of climate change impacts on global ecosystems and potential mitigation strategies.",
                "Develop a business plan for a sustainable technology startup focused on renewable energy solutions."
            ],
            "long_generation": [
                "Write a detailed technical documentation for a distributed system architecture, including microservices, database design, API specifications, security considerations, monitoring, and deployment strategies.",
                "Create a comprehensive research paper on the evolution of natural language processing, covering historical developments, current state-of-the-art models, evaluation metrics, and future directions.",
                "Develop a complete curriculum for teaching advanced machine learning, including theoretical foundations, practical implementations, project assignments, and assessment criteria."
            ],
            "code_generation": [
                "Write a Python function to implement a binary search algorithm.",
                "Create a RESTful API using FastAPI with authentication and database integration.",
                "Implement a neural network from scratch using only NumPy for image classification."
            ],
            "structured_extraction": [
                "Extract key information from this technical specification: [sample spec text]",
                "Parse the following contract and identify important clauses: [sample contract]",
                "Analyze this financial report and extract key metrics: [sample report]"
            ]
        }
    
    async def run_comprehensive_benchmark(self, model_client, model_name: str) -> Dict[str, Any]:
        """Run complete benchmark suite for a model"""
        
        benchmark_results = {
            "model_name": model_name,
            "timestamp": time.time(),
            "hardware_info": self._get_hardware_info(),
            "test_results": {}
        }
        
        # Run different benchmark categories
        for category, prompts in self.baseline_prompts.items():
            category_results = []
            
            for prompt in prompts:
                # Single batch performance
                result = await self._benchmark_single_inference(model_client, prompt, model_name, category)
                category_results.append(result)
                
                # Batch performance testing
                if category in ["short_generation", "code_generation"]:
                    for batch_size in [4, 8, 16]:
                        batch_result = await self._benchmark_batch_inference(
                            model_client, [prompt] * batch_size, model_name, f"{category}_batch_{batch_size}"
                        )
                        category_results.append(batch_result)
            
            benchmark_results["test_results"][category] = category_results
        
        # Memory stress test
        memory_test = await self._benchmark_memory_stress(model_client, model_name)
        benchmark_results["test_results"]["memory_stress"] = memory_test
        
        # Latency consistency test
        latency_test = await self._benchmark_latency_consistency(model_client, model_name)
        benchmark_results["test_results"]["latency_consistency"] = latency_test
        
        return benchmark_results
    
    async def _benchmark_single_inference(self, model_client, prompt: str, 
                                        model_name: str, test_name: str) -> BenchmarkResult:
        """Benchmark single inference performance"""
        
        # Warm up
        await model_client.generate(prompt[:100])
        
        # Memory before
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        # Measure performance
        start_time = time.time()
        first_token_time = None
        
        # Use streaming to measure first token latency
        token_count = 0
        async for token in model_client.stream_generation(prompt):
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
        
        end_time = time.time()
        
        # Memory after
        memory_after = psutil.virtual_memory().used / (1024**3)
        
        # Calculate metrics
        total_time = end_time - start_time
        first_token_latency = (first_token_time - start_time) * 1000 if first_token_time else 0
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        memory_usage = memory_after - memory_before
        
        return BenchmarkResult(
            model_name=model_name,
            test_name=test_name,
            tokens_per_second=tokens_per_second,
            first_token_latency_ms=first_token_latency,
            total_time_seconds=total_time,
            memory_usage_gb=memory_usage,
            batch_size=1,
            sequence_length=len(prompt)
        )
    
    async def _benchmark_batch_inference(self, model_client, prompts: List[str], 
                                       model_name: str, test_name: str) -> BenchmarkResult:
        """Benchmark batch inference performance"""
        
        batch_size = len(prompts)
        total_tokens = 0
        
        # Memory before
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        start_time = time.time()
        
        # Process batch
        results = await model_client.batch_generate(prompts)
        
        end_time = time.time()
        
        # Memory after
        memory_after = psutil.virtual_memory().used / (1024**3)
        
        # Calculate total tokens
        for result in results:
            total_tokens += len(result.get("text", "").split())
        
        # Calculate metrics
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        memory_usage = memory_after - memory_before
        avg_sequence_length = sum(len(p) for p in prompts) // len(prompts)
        
        return BenchmarkResult(
            model_name=model_name,
            test_name=test_name,
            tokens_per_second=tokens_per_second,
            first_token_latency_ms=0,  # Not applicable for batch
            total_time_seconds=total_time,
            memory_usage_gb=memory_usage,
            batch_size=batch_size,
            sequence_length=avg_sequence_length
        )
    
    async def _benchmark_memory_stress(self, model_client, model_name: str) -> Dict[str, Any]:
        """Test memory usage under stress"""
        
        stress_prompts = [
            "Write a detailed explanation of quantum mechanics " * 100,  # Long prompt
            "Generate a comprehensive technical manual " * 50,  # Medium prompt
            "Create code documentation " * 200  # Many repetitions
        ]
        
        memory_stats = []
        
        for i, prompt in enumerate(stress_prompts):
            memory_before = psutil.virtual_memory()
            
            try:
                result = await model_client.generate(prompt)
                memory_after = psutil.virtual_memory()
                
                memory_stats.append({
                    "test_id": i,
                    "memory_before_gb": memory_before.used / (1024**3),
                    "memory_after_gb": memory_after.used / (1024**3),
                    "memory_delta_gb": (memory_after.used - memory_before.used) / (1024**3),
                    "success": True
                })
                
            except Exception as e:
                memory_stats.append({
                    "test_id": i,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "model_name": model_name,
            "test_type": "memory_stress",
            "results": memory_stats,
            "max_memory_usage": max([s.get("memory_delta_gb", 0) for s in memory_stats]),
            "average_memory_usage": statistics.mean([s.get("memory_delta_gb", 0) for s in memory_stats if s["success"]])
        }
    
    async def _benchmark_latency_consistency(self, model_client, model_name: str) -> Dict[str, Any]:
        """Test latency consistency over multiple runs"""
        
        test_prompt = "Explain the concept of machine learning in detail."
        latencies = []
        
        # Run 20 iterations
        for _ in range(20):
            start_time = time.time()
            await model_client.generate(test_prompt)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "model_name": model_name,
            "test_type": "latency_consistency",
            "latencies_ms": latencies,
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "std_dev_ms": statistics.stdev(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "p99_latency_ms": sorted(latencies)[int(0.99 * len(latencies))]
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": "darwin",  # macOS
            "architecture": "arm64",  # M3 Max
            "timestamp": time.time()
        }
    
    def generate_performance_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        
        report = f"""
# Qwen3 Performance Benchmark Report

## Model: {benchmark_results['model_name']}
## Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(benchmark_results['timestamp']))}

## Hardware Configuration
- CPU Cores: {benchmark_results['hardware_info']['cpu_count']}
- Total Memory: {benchmark_results['hardware_info']['memory_total_gb']:.1f} GB
- Platform: {benchmark_results['hardware_info']['platform']}
- Architecture: {benchmark_results['hardware_info']['architecture']}

## Performance Summary

### Single Inference Performance
"""
        
        # Analyze single inference results
        single_results = []
        for category, results in benchmark_results['test_results'].items():
            if 'batch' not in category and category not in ['memory_stress', 'latency_consistency']:
                for result in results:
                    if isinstance(result, BenchmarkResult):
                        single_results.append(result)
        
        if single_results:
            avg_tokens_per_sec = statistics.mean([r.tokens_per_second for r in single_results])
            avg_first_token_latency = statistics.mean([r.first_token_latency_ms for r in single_results])
            avg_memory_usage = statistics.mean([r.memory_usage_gb for r in single_results])
            
            report += f"""
- Average Tokens/Second: {avg_tokens_per_sec:.1f}
- Average First Token Latency: {avg_first_token_latency:.1f} ms
- Average Memory Usage: {avg_memory_usage:.2f} GB
"""
        
        # Add batch performance analysis
        batch_results = []
        for category, results in benchmark_results['test_results'].items():
            if 'batch' in category:
                for result in results:
                    if isinstance(result, BenchmarkResult):
                        batch_results.append(result)
        
        if batch_results:
            report += f"""
### Batch Processing Performance
- Best Batch Throughput: {max([r.tokens_per_second for r in batch_results]):.1f} tokens/sec
- Optimal Batch Size: {batch_results[max(range(len(batch_results)), key=lambda i: batch_results[i].tokens_per_second)].batch_size}
"""
        
        # Add latency consistency analysis
        if 'latency_consistency' in benchmark_results['test_results']:
            latency_data = benchmark_results['test_results']['latency_consistency']
            report += f"""
### Latency Consistency
- Mean Latency: {latency_data['mean_latency_ms']:.1f} ms
- Standard Deviation: {latency_data['std_dev_ms']:.1f} ms
- P95 Latency: {latency_data['p95_latency_ms']:.1f} ms
- P99 Latency: {latency_data['p99_latency_ms']:.1f} ms
"""
        
        # Add memory stress test results
        if 'memory_stress' in benchmark_results['test_results']:
            memory_data = benchmark_results['test_results']['memory_stress']
            report += f"""
### Memory Performance
- Maximum Memory Usage: {memory_data['max_memory_usage']:.2f} GB
- Average Memory Usage: {memory_data['average_memory_usage']:.2f} GB
- Memory Stress Tests Passed: {sum([1 for r in memory_data['results'] if r['success']])}/{len(memory_data['results'])}
"""
        
        return report
```

## Real-World Performance Scenarios

### Document Processing Pipeline Benchmarks
```yaml
Document Classification (1000 docs):
  Qwen3-1.7B-4bit: 45-60 seconds
  Qwen3-7B-8bit: 60-90 seconds
  Target: <120 seconds

Document Summarization (100 docs):
  Qwen3-7B-8bit: 180-240 seconds
  Qwen3-14B-8bit: 300-420 seconds
  Target: <300 seconds (7B), <600 seconds (14B)

Structured Data Extraction (500 docs):
  Qwen3-7B-8bit: 600-900 seconds
  Qwen3-14B-8bit: 900-1200 seconds
  Target: <1200 seconds (7B), <1800 seconds (14B)

Code Generation (50 functions):
  Qwen3-7B-8bit: 300-450 seconds
  Qwen3-14B-8bit: 450-600 seconds
  Target: <600 seconds (7B), <900 seconds (14B)
```

### RAG Pipeline Performance
```yaml
Query Processing + Retrieval:
  Embedding Generation: <2 seconds (1000 chunks)
  Vector Search: <500ms
  Context Assembly: <200ms
  
Response Generation:
  Qwen3-7B-8bit: 2-4 seconds
  Qwen3-14B-8bit: 4-8 seconds
  
End-to-End RAG Response:
  Simple Query: 3-6 seconds
  Complex Query: 8-15 seconds
  Target: <10 seconds average
```

## Continuous Performance Monitoring

### Real-Time Performance Dashboard
```python
import asyncio
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceDashboard:
    def __init__(self, benchmark_suite: Qwen3PerformanceBenchmark):
        self.benchmark_suite = benchmark_suite
        self.performance_history = []
        self.monitoring_active = False
        
    async def start_continuous_monitoring(self, model_client, model_name: str, 
                                        interval_minutes: int = 30):
        """Start continuous performance monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Run quick benchmark
                quick_benchmark = await self._run_quick_benchmark(model_client, model_name)
                self.performance_history.append(quick_benchmark)
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.performance_history = [
                    entry for entry in self.performance_history 
                    if entry['timestamp'] > cutoff_time.timestamp()
                ]
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _run_quick_benchmark(self, model_client, model_name: str) -> Dict:
        """Run quick performance benchmark"""
        test_prompt = "Explain machine learning concepts briefly."
        
        start_time = time.time()
        result = await model_client.generate(test_prompt)
        end_time = time.time()
        
        # Get system metrics
        memory = psutil.virtual_memory()
        
        return {
            "timestamp": time.time(),
            "model_name": model_name,
            "response_time_seconds": end_time - start_time,
            "tokens_generated": len(result.get("text", "").split()),
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3)
        }
    
    def generate_performance_charts(self, output_dir: str = "./performance_charts"):
        """Generate performance visualization charts"""
        
        if not self.performance_history:
            print("No performance data available for charting")
            return
        
        # Extract data for plotting
        timestamps = [datetime.fromtimestamp(entry['timestamp']) for entry in self.performance_history]
        response_times = [entry['response_time_seconds'] for entry in self.performance_history]
        memory_usage = [entry['memory_usage_percent'] for entry in self.performance_history]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Response time chart
        ax1.plot(timestamps, response_times, 'b-', linewidth=2)
        ax1.set_title('Response Time Over Time')
        ax1.set_ylabel('Response Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage chart
        ax2.plot(timestamps, memory_usage, 'r-', linewidth=2)
        ax2.set_title('Memory Usage Over Time')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance distribution histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(response_times, bins=20, alpha=0.7, color='blue')
        ax1.set_title('Response Time Distribution')
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        
        ax2.hist(memory_usage, bins=20, alpha=0.7, color='red')
        ax2.set_title('Memory Usage Distribution')
        ax2.set_xlabel('Memory Usage (%)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
```

## Performance Regression Detection

### Automated Performance Regression Testing
```python
class PerformanceRegressionDetector:
    def __init__(self, baseline_file: str):
        self.baseline_results = self._load_baseline(baseline_file)
        self.regression_thresholds = {
            "tokens_per_second": 0.05,  # 5% decrease is regression
            "first_token_latency_ms": 0.10,  # 10% increase is regression
            "memory_usage_gb": 0.15  # 15% increase is regression
        }
    
    def detect_regressions(self, current_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline"""
        regressions = []
        
        for test_category in current_results['test_results']:
            if test_category in ['memory_stress', 'latency_consistency']:
                continue
                
            current_metrics = self._extract_metrics(current_results['test_results'][test_category])
            baseline_metrics = self._extract_metrics(self.baseline_results.get(test_category, []))
            
            # Compare metrics
            for metric_name, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric_name)
                if baseline_value is None:
                    continue
                
                # Calculate percentage change
                if metric_name == "tokens_per_second":
                    # Lower is worse
                    change = (baseline_value - current_value) / baseline_value
                else:
                    # Higher is worse (latency, memory)
                    change = (current_value - baseline_value) / baseline_value
                
                threshold = self.regression_thresholds.get(metric_name, 0.10)
                
                if change > threshold:
                    regressions.append({
                        "test_category": test_category,
                        "metric": metric_name,
                        "baseline_value": baseline_value,
                        "current_value": current_value,
                        "change_percent": change * 100,
                        "threshold_percent": threshold * 100,
                        "severity": "critical" if change > threshold * 2 else "warning"
                    })
        
        return regressions
    
    def _load_baseline(self, baseline_file: str) -> Dict:
        """Load baseline performance results"""
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _extract_metrics(self, results: List) -> Dict[str, float]:
        """Extract average metrics from results"""
        if not results:
            return {}
        
        # Filter to only BenchmarkResult objects
        benchmark_results = [r for r in results if isinstance(r, BenchmarkResult)]
        
        if not benchmark_results:
            return {}
        
        return {
            "tokens_per_second": statistics.mean([r.tokens_per_second for r in benchmark_results]),
            "first_token_latency_ms": statistics.mean([r.first_token_latency_ms for r in benchmark_results]),
            "memory_usage_gb": statistics.mean([r.memory_usage_gb for r in benchmark_results])
        }
```

## Expected Performance Achievements

### Target Performance Matrix
```yaml
Performance Targets by Model and Use Case:

Interactive Applications (< 500ms response):
  - Qwen3-1.7B-4bit: ✅ Achievable
  - Qwen3-7B-4bit: ⚠️ Marginal
  - Qwen3-14B: ❌ Not suitable

Batch Processing (> 100 items/minute):
  - Qwen3-1.7B: ✅ 500+ items/minute
  - Qwen3-7B: ✅ 150-300 items/minute
  - Qwen3-14B: ✅ 75-150 items/minute

Memory Constrained (< 16GB):
  - Qwen3-1.7B-4bit: ✅ ~4GB
  - Qwen3-7B-4bit: ✅ ~8GB
  - Qwen3-14B-8bit: ❌ ~24GB

Power Efficient (< 30W):
  - Qwen3-1.7B: ✅ 15-25W
  - Qwen3-7B-4bit: ✅ 25-35W
  - Qwen3-14B: ❌ 40-60W
```

## Conclusion

These comprehensive benchmarks provide a framework for measuring and optimizing Qwen3 performance on M3 Max hardware, enabling data-driven decisions for model selection and deployment strategies based on specific performance requirements and constraints.