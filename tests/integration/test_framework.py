#!/usr/bin/env python3
"""
Integration Test Framework for RAN-LLM Claude Flow Hybrid Pipeline
Comprehensive testing suite for Rust-Python hybrid architecture
"""

import asyncio
import json
import os
import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@dataclass
class TestResult:
    """Test result with comprehensive metrics"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration_ms: float
    memory_used_mb: float
    cpu_percent: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance metrics for M3 Max optimization validation"""
    documents_per_hour: float
    memory_efficiency: float  # % of 128GB used efficiently
    cpu_utilization: float
    throughput_improvement: float  # Multiple vs baseline
    latency_p95_ms: float
    error_rate: float

class IntegrationTestFramework:
    """Comprehensive integration test framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results: List[TestResult] = []
        self.logger = self._setup_logging()
        self.start_time = time.time()
        
        # M3 Max specific constants
        self.M3_MAX_MEMORY_GB = 128
        self.M3_MAX_CORES = 16  # Performance cores + efficiency cores
        self.TARGET_DOCS_PER_HOUR = 25  # 4-5x improvement target
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            "rust_pipeline_path": "src/rust-pipeline",
            "python_pipeline_path": "src/python-pipeline", 
            "test_data_path": "tests/fixtures",
            "timeout_seconds": 300,
            "stress_test_duration": 60,
            "max_concurrent_processes": 8,
            "performance_targets": {
                "docs_per_hour_min": 20,
                "memory_efficiency_min": 0.7,
                "throughput_improvement_min": 4.0
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("IntegrationTest")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_path = Path("tests/integration/test_results.log")
        log_path.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def run_test(self, test_func: Callable, test_name: str, **kwargs) -> TestResult:
        """Run individual test with metrics collection"""
        self.logger.info(f"Starting test: {test_name}")
        start_time = time.time()
        
        # Get initial system metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run the test
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(**kwargs)
            else:
                result = test_func(**kwargs)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            cpu_percent = process.cpu_percent()
            
            test_result = TestResult(
                test_name=test_name,
                status="PASS",
                duration_ms=duration_ms,
                memory_used_mb=memory_used,
                cpu_percent=cpu_percent,
                metrics=result if isinstance(result, dict) else None
            )
            
            self.logger.info(f"Test PASSED: {test_name} ({duration_ms:.2f}ms)")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            test_result = TestResult(
                test_name=test_name,
                status="FAIL",
                duration_ms=duration_ms,
                memory_used_mb=memory_used,
                cpu_percent=0,
                error_message=str(e)
            )
            
            self.logger.error(f"Test FAILED: {test_name} - {str(e)}")
        
        self.results.append(test_result)
        return test_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": mp.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "is_m3_max": psutil.virtual_memory().total > 100 * (1024**3),  # >100GB indicates M3 Max
            "timestamp": time.time()
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        
        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration_ms": total_duration,
                "avg_duration_ms": avg_duration
            },
            "system_info": self.get_system_info(),
            "test_results": [r.to_dict() for r in self.results],
            "performance_analysis": self._analyze_performance(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        performance_tests = [r for r in self.results if r.metrics and "performance" in r.test_name.lower()]
        
        if not performance_tests:
            return {"status": "no_performance_tests"}
        
        # Extract performance metrics
        docs_per_hour = []
        memory_efficiency = []
        throughput_improvement = []
        
        for test in performance_tests:
            if test.metrics:
                docs_per_hour.append(test.metrics.get("docs_per_hour", 0))
                memory_efficiency.append(test.metrics.get("memory_efficiency", 0))
                throughput_improvement.append(test.metrics.get("throughput_improvement", 0))
        
        return {
            "avg_docs_per_hour": sum(docs_per_hour) / len(docs_per_hour) if docs_per_hour else 0,
            "avg_memory_efficiency": sum(memory_efficiency) / len(memory_efficiency) if memory_efficiency else 0,
            "avg_throughput_improvement": sum(throughput_improvement) / len(throughput_improvement) if throughput_improvement else 0,
            "meets_performance_targets": {
                "docs_per_hour": max(docs_per_hour) >= self.config["performance_targets"]["docs_per_hour_min"] if docs_per_hour else False,
                "memory_efficiency": max(memory_efficiency) >= self.config["performance_targets"]["memory_efficiency_min"] if memory_efficiency else False,
                "throughput_improvement": max(throughput_improvement) >= self.config["performance_targets"]["throughput_improvement_min"] if throughput_improvement else False
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.results if r.status == "FAIL"]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests before production deployment")
        
        high_memory_tests = [r for r in self.results if r.memory_used_mb > 1000]  # >1GB
        if high_memory_tests:
            recommendations.append("Consider memory optimization for high-memory consuming tests")
        
        slow_tests = [r for r in self.results if r.duration_ms > 30000]  # >30s
        if slow_tests:
            recommendations.append("Optimize performance for slow-running tests")
        
        performance_analysis = self._analyze_performance()
        if "meets_performance_targets" in performance_analysis:
            targets = performance_analysis["meets_performance_targets"]
            if not all(targets.values()):
                recommendations.append("Performance targets not met - review M3 Max optimization and Rust-Python IPC")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup test resources"""
        self.logger.info("Cleaning up test resources...")
        # Add cleanup logic here
        pass

# Global test framework instance
test_framework = IntegrationTestFramework()

def get_test_framework() -> IntegrationTestFramework:
    """Get global test framework instance"""
    return test_framework

# Pytest fixtures
@pytest.fixture
def framework():
    """Pytest fixture for test framework"""
    return test_framework

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()