#!/usr/bin/env python3
"""
M3 MacBook Pro Performance Monitor
Specialized monitoring for Apple Silicon M3 with 128GB unified memory
"""

import time
import psutil
import subprocess
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import threading
from datetime import datetime, timedelta

@dataclass
class M3PerformanceMetrics:
    """Performance metrics optimized for M3 architecture"""
    timestamp: str
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    gpu_utilization: float
    unified_memory_pressure: str
    tokens_per_second: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    concurrent_requests: int = 0
    model_memory_usage_gb: float = 0.0
    memory_bandwidth_gbps: Optional[float] = None

class M3PerformanceMonitor:
    """
    Performance monitoring system optimized for M3 MacBook Pro
    Tracks unified memory, CPU/GPU coordination, and inference performance
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_history: List[M3PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def get_memory_pressure(self) -> str:
        """Get macOS memory pressure status"""
        try:
            result = subprocess.run(['memory_pressure'], 
                                 capture_output=True, text=True, timeout=5)
            if 'normal' in result.stdout.lower():
                return 'normal'
            elif 'warn' in result.stdout.lower():
                return 'warning'
            elif 'critical' in result.stdout.lower():
                return 'critical'
            else:
                return 'unknown'
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return 'unknown'
    
    def get_gpu_utilization(self) -> float:
        """
        Estimate GPU utilization on M3
        Note: macOS doesn't expose detailed GPU metrics easily
        """
        try:
            # Use powermetrics to get GPU utilization (requires sudo)
            # This is a simplified approach - real implementation might need
            # to parse Activity Monitor data or use private frameworks
            result = subprocess.run([
                'powermetrics', '-n', '1', '-i', '500', '--samplers', 'gpu_power'
            ], capture_output=True, text=True, timeout=3)
            
            # Parse GPU power/utilization from output
            # This is a placeholder - actual parsing would be more complex
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line and '%' in line:
                    # Extract percentage if available
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)%', line)
                    if match:
                        return float(match.group(1))
            
            return 0.0  # Default if can't parse
        except:
            return 0.0  # Fallback if powermetrics not available
    
    def get_memory_bandwidth(self) -> Optional[float]:
        """
        Estimate memory bandwidth utilization
        M3 Max has ~400GB/s theoretical bandwidth
        """
        try:
            # This is a simplified estimation
            # Real implementation would need hardware performance counters
            vm_stat = subprocess.run(['vm_stat'], capture_output=True, text=True)
            
            # Parse vm_stat output for memory activity
            # This is a placeholder for actual bandwidth calculation
            lines = vm_stat.stdout.split('\n')
            total_activity = 0
            
            for line in lines:
                if 'Pages' in line and ':' in line:
                    try:
                        count = int(line.split(':')[1].strip().replace('.', ''))
                        total_activity += count
                    except ValueError:
                        continue
            
            # Rough estimation: convert page activity to GB/s
            # This is highly simplified and would need calibration
            page_size = 16384  # 16KB pages on M3
            bandwidth_estimate = (total_activity * page_size) / (1024**3)
            return min(bandwidth_estimate, 400.0)  # Cap at theoretical max
            
        except:
            return None
    
    def collect_metrics(self) -> M3PerformanceMetrics:
        """Collect comprehensive M3 performance metrics"""
        memory = psutil.virtual_memory()
        
        metrics = M3PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_used_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            gpu_utilization=self.get_gpu_utilization(),
            unified_memory_pressure=self.get_memory_pressure(),
            memory_bandwidth_gbps=self.get_memory_bandwidth()
        )
        
        return metrics
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("M3 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("M3 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries to prevent memory bloat
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def get_recent_metrics(self, minutes: int = 5) -> List[M3PerformanceMetrics]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = []
        for metric in reversed(self.metrics_history):
            metric_time = datetime.fromisoformat(metric.timestamp)
            if metric_time >= cutoff_time:
                recent_metrics.append(metric)
            else:
                break
        
        return list(reversed(recent_metrics))
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary optimized for M3"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.get_recent_metrics(5)
        if not recent_metrics:
            recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate averages and trends
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory_gb = sum(m.memory_used_gb for m in recent_metrics) / len(recent_metrics)
        avg_memory_percent = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Memory pressure analysis
        pressure_counts = {}
        for metric in recent_metrics:
            pressure = metric.unified_memory_pressure
            pressure_counts[pressure] = pressure_counts.get(pressure, 0) + 1
        
        most_common_pressure = max(pressure_counts, key=pressure_counts.get)
        
        # Performance optimization recommendations
        recommendations = []
        
        if avg_memory_percent > 90:
            recommendations.append("High memory usage (>90%) - consider model quantization or reducing batch size")
        elif avg_memory_percent > 80:
            recommendations.append("Moderate memory usage (>80%) - monitor for potential optimization")
        
        if most_common_pressure != 'normal':
            recommendations.append(f"Memory pressure detected: {most_common_pressure} - consider reducing memory usage")
        
        if avg_cpu > 90:
            recommendations.append("High CPU usage - consider distributing workload or optimizing algorithms")
        
        if avg_gpu < 10:
            recommendations.append("Low GPU utilization - ensure MPS/MLX acceleration is enabled")
        
        # Memory bandwidth efficiency
        bandwidth_metrics = [m for m in recent_metrics if m.memory_bandwidth_gbps is not None]
        avg_bandwidth = None
        if bandwidth_metrics:
            avg_bandwidth = sum(m.memory_bandwidth_gbps for m in bandwidth_metrics) / len(bandwidth_metrics)
            if avg_bandwidth < 50:  # Less than 50 GB/s utilization
                recommendations.append("Low memory bandwidth utilization - consider larger batch sizes or memory-intensive operations")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sampling_period_minutes": 5,
            "metrics_count": len(recent_metrics),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_used_gb": round(avg_memory_gb, 2),
                "memory_percent": round(avg_memory_percent, 2),
                "gpu_utilization": round(avg_gpu, 2),
                "memory_bandwidth_gbps": round(avg_bandwidth, 2) if avg_bandwidth else None
            },
            "unified_memory": {
                "total_gb": 128,
                "used_gb": round(avg_memory_gb, 2),
                "available_gb": round(128 - avg_memory_gb, 2),
                "pressure_status": most_common_pressure
            },
            "m3_optimization": {
                "unified_memory_efficiency": round((avg_memory_gb / 128) * 100, 1),
                "cpu_gpu_balance": "balanced" if abs(avg_cpu - avg_gpu) < 30 else "imbalanced",
                "recommendations": recommendations
            },
            "performance_score": self._calculate_performance_score(avg_cpu, avg_memory_percent, avg_gpu, most_common_pressure)
        }
    
    def _calculate_performance_score(self, cpu: float, memory_percent: float, gpu: float, pressure: str) -> int:
        """Calculate overall performance score (0-100)"""
        score = 100
        
        # Penalize high CPU usage
        if cpu > 90:
            score -= 20
        elif cpu > 80:
            score -= 10
        
        # Penalize high memory usage
        if memory_percent > 90:
            score -= 25
        elif memory_percent > 80:
            score -= 15
        
        # Reward good GPU utilization
        if 20 <= gpu <= 80:
            score += 10
        elif gpu < 10:
            score -= 10
        
        # Penalize memory pressure
        if pressure == 'critical':
            score -= 30
        elif pressure == 'warning':
            score -= 15
        
        return max(0, min(100, score))
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            "export_time": datetime.now().isoformat(),
            "system_info": {
                "model": "MacBook Pro M3 Max",
                "unified_memory_gb": 128,
                "cpu_cores": 16
            },
            "metrics": [asdict(metric) for metric in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics exported to {filepath}")

# Benchmark functions for M3 optimization
class M3Benchmarks:
    """Benchmarking suite optimized for M3 MacBook Pro"""
    
    @staticmethod
    def benchmark_memory_bandwidth():
        """Benchmark unified memory bandwidth"""
        try:
            import numpy as np
            
            print("Benchmarking M3 unified memory bandwidth...")
            
            # Test different array sizes
            sizes = [1000, 5000, 10000, 50000, 100000]
            results = {}
            
            for size in sizes:
                # Create large arrays to test memory bandwidth
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                start_time = time.time()
                c = np.dot(a, b)  # Matrix multiplication
                end_time = time.time()
                
                duration = end_time - start_time
                data_size_gb = (3 * size * size * 4) / (1024**3)  # 3 arrays, float32
                bandwidth_gbps = data_size_gb / duration
                
                results[size] = {
                    "duration_seconds": duration,
                    "data_size_gb": data_size_gb,
                    "bandwidth_gbps": bandwidth_gbps
                }
                
                print(f"Size {size}x{size}: {bandwidth_gbps:.2f} GB/s")
            
            return results
            
        except ImportError:
            return {"error": "NumPy not available for benchmarking"}
    
    @staticmethod
    def benchmark_inference_performance():
        """Benchmark inference performance patterns"""
        print("M3 Inference benchmark placeholder")
        print("Install PyTorch/MLX to run actual inference benchmarks")
        
        return {
            "note": "Requires PyTorch MPS or MLX installation",
            "recommended_tests": [
                "Matrix multiplication throughput",
                "Model loading time",
                "Batch inference latency",
                "Memory allocation efficiency"
            ]
        }

if __name__ == "__main__":
    # Example usage
    monitor = M3PerformanceMonitor(sampling_interval=2.0)
    
    try:
        monitor.start_monitoring()
        
        # Run for 30 seconds
        print("Monitoring M3 performance for 30 seconds...")
        time.sleep(30)
        
        # Get summary
        summary = monitor.get_performance_summary()
        print("\nPerformance Summary:")
        print(json.dumps(summary, indent=2))
        
        # Export metrics
        monitor.export_metrics("m3_performance_metrics.json")
        
    finally:
        monitor.stop_monitoring()
    
    # Run benchmarks
    print("\nRunning M3 benchmarks...")
    benchmarks = M3Benchmarks()
    memory_results = benchmarks.benchmark_memory_bandwidth()
    inference_results = benchmarks.benchmark_inference_performance()
    
    print("\nBenchmark Results:")
    print("Memory Bandwidth:")
    print(json.dumps(memory_results, indent=2))
    print("\nInference Performance:")
    print(json.dumps(inference_results, indent=2))